# -*- coding: UTF-8 -*-
"""ComfyUI nodes for Personalize Anything (native FLUX DiT stack).

Architecture follows comfy's own conventions - this plugin does **not** load or touch
the CLIP and the VAE at all:

    Load Diffusion Model -> MODEL ──────────────┐
    Load CLIP + CLIPTextEncode("") -> CONDITIONING (source branch) ────┐
    Load CLIP + CLIPTextEncode + FluxGuidance -> CONDITIONING (target) ─┤
    Load Image -> VAEEncode -> LATENT (source image) ───────────────────┤
                                                                        ▼
                                              PersonalizeAnythingInvert
                                     -> inverted_latent + source_latent
                                                                        ▼
                                        PersonalizeAnythingSample -> LATENT
                                                                        ▼
                                                          VAEDecode -> IMAGE

Because CLIP / VAE / DiT live in different nodes, comfy's model management can unload
each of them when the next one needs the VRAM - the old wrapper kept all of them in one
process, which is what blew up the 12 GB card.

All LATENT inputs/outputs follow the stock convention: **raw (unscaled) latents**, the
same space VAEEncode / EmptySD3LatentImage / KSampler / VAEDecode use.  The scaling into
the DiT's sampling space happens inside the nodes via ``model.process_latent_in/out``.
"""

import torch
import os
import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.utils

from . import pa_core
from . import pa_sampling

MODES = [
    "inpainting",
    "outpainting",
    "single_personalize",
    "subject_reconstruction",
]

# modes where the comfy MASK means "region to replace" and therefore has to be
# inverted to obtain the upstream "region to copy" mask
INVERT_MASK_MODES = ("inpainting", "outpainting")


def _activation_budget(patcher, latent_shape):
    """VRAM budget to keep free for the activations of one forward.

    comfy's own estimate (BaseModel.memory_required) is tuned for the stock KSampler
    and is far too small for our batch-2/3 parallel forward on a 12 GB card, where the
    11.9 GB fp8 flux weights and the activations have to share the card.  We keep
    comfy's formula (scaled) with a 2 GB floor.
    """
    base_model = patcher.model
    batch = max(1, int(latent_shape[0]))
    area = batch * int(latent_shape[-2]) * int(latent_shape[-1])
    dtype_size = comfy.model_management.dtype_size(base_model.get_dtype_inference())
    mem_factor = getattr(base_model, "memory_usage_factor", 3.1) or 3.1
    est = int(area * dtype_size * 0.01 * mem_factor * (1024 * 1024))
    floor = int(float(os.environ.get("PA_VRAM_BUDGET", "2")) * (1024 ** 3))
    return max(est * 2, floor)


def _prepare_model(patcher, latent_shape):
    """Load the model onto the GPU and return (BaseModel, device).

    Memory rules learned on this machine (DiffuEraser / PiD notes):
      * ``free_memory(need)`` only *partially* unloads - flux would keep ~10 GB
        resident.  ``unload_all_models()`` detaches everything first (the CLIP and the
        VAE reload on demand when their own nodes run).
      * the amount of model kept in VRAM is ``free - minimum_memory_required``, so the
        budget passed here directly decides how much of the fp8 flux weights stay
        resident and how much streams from RAM per layer.
    """
    need = _activation_budget(patcher, latent_shape)
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()
    comfy.model_management.load_models_gpu([patcher], memory_required=need,
                                           minimum_memory_required=need)
    patcher.pre_run()
    free, total = torch.cuda.mem_get_info()
    print(f"[PersonalizeAnything] model loaded: budget={need/2**30:.2f}GB "
          f"alloc={torch.cuda.memory_allocated()/2**30:.2f}GB "
          f"dev_free={free/2**30:.2f}/{total/2**30:.2f}GB")
    return patcher.model, patcher.load_device


def _cond_rows(conditioning):
    """comfy CONDITIONING ([[emb, dict], ...]) -> cond dict for ``model.extra_conds``."""
    if conditioning is None or len(conditioning) == 0:
        raise ValueError("empty conditioning input")
    if len(conditioning) > 1:
        print(f"[PersonalizeAnything] warning: conditioning has {len(conditioning)} entries, "
              "only the first one is used")
    emb, extra = conditioning[0][0], conditioning[0][1]
    row = dict(extra)
    row["cross_attn"] = emb
    return row


def _rows_to_model_conds(base_model, rows, noise, device):
    """One cond dict per batch row -> merged kwargs for ``apply_model``.

    Uses the model's own ``extra_conds`` (exactly what ``process_conds`` does for the
    stock sampler) so every conditioning key (pooled_output, guidance, attention_mask,
    reference_latents, ...) keeps its standard meaning, then batches the rows with
    comfy's own ``cond_cat``.
    """
    per_row = []
    for row in rows:
        d = dict(row)
        d.setdefault("noise", noise)
        d.setdefault("device", device)
        per_row.append(base_model.extra_conds(**d))
    return comfy.samplers.cond_cat(per_row)


def _make_apply_fn(real_model, transformer_options, model_conds):
    """Build the ``x, t -> raw velocity`` closure used by both ODEs.

    The guidance embedding is *not* passed here - it lives in the conditioning
    (FluxGuidance node / the invert node's `guidance` input) exactly like the stock
    sampler, so it survives the cond_cat batching.
    """

    def apply_fn(x, t):
        t_tensor = torch.full((x.shape[0],), float(t), device=x.device, dtype=torch.float32)
        out = real_model.apply_model(
            x,
            t_tensor,
            transformer_options=transformer_options,
            **model_conds,
        )
        return out.float()

    return apply_fn


def _run_reverse(model, latents, y_0, rows, inject, target_row, seed, steps, tau, eta,
                 mode, concept_process=False):
    """Shared driver of the controlled reverse ODE for Sample / Compose.

    ``latents`` is in the *scaled* space, the LAST row being the target branch and the
    preceding rows the source branches.  ``rows`` is one cond dict per batch row, in
    the same order.  Returns the target row (scaled space).
    """
    device = model.load_device
    base_model = model.model
    h_tok, w_tok = pa_core.token_grid(latents.shape)

    sigmas, mu = pa_sampling.official_sigmas(steps, h_tok * w_tok, device=device)

    state = pa_core.PAState()
    state.inject = inject
    state.target_row = target_row
    state.tau = float(tau) / 100.0
    state.concept_process = bool(concept_process)
    transformer_options = pa_core.build_transformer_options(model, state)

    real_model, device = _prepare_model(model, latents.shape)
    # comfy's prepare_noise is intentionally generated on CPU -> move it onto the
    # latents' device (the stock sampler does the same move inside inner_sample)
    noise = comfy.sample.prepare_noise(torch.zeros_like(latents[-1:]), seed).to(
        device=latents.device, dtype=torch.float32)
    model_conds = _rows_to_model_conds(real_model, rows, noise, device)
    apply_fn = _make_apply_fn(real_model, transformer_options, model_conds)

    pbar = comfy.utils.ProgressBar(steps)
    state.enabled = True

    with comfy.model_management.cuda_device_context(device):
        with pa_core.raw_velocity(real_model):
            latents = pa_core.parallel_reverse(
                apply_fn, latents, y_0, sigmas, eta=eta, start=0,
                stop=len(sigmas) - 1, state=state,
                progress=lambda i, n: pbar.update_absolute(i, n),
            )

    target = latents[-1:].contiguous()
    n_subj = sum(int(m.numel()) for _, m, _ in (state.inject or []))
    print(f"[PersonalizeAnything] {mode} | {pa_sampling.describe(steps, h_tok * w_tok, mu)} | "
          f"tau={tau} eta={eta} seed={seed} | subject tokens={n_subj} | "
          f"patch fired {state.fire_count}x")
    return target


class PersonalizeAnythingInvert:
    """RF-Inversion forward ODE: source image latent -> noise latent (source branch)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING", {"tooltip": "Prompt of the source branch. "
                                                             "The original paper uses an empty prompt."}),
                "latent_image": ("LATENT", {"tooltip": "Raw latent of the source image "
                                                       "(e.g. from VAEEncode)."}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 100, "step": 1}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "guidance": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.1,
                                       "tooltip": "Inversion guidance; the original pipeline "
                                                  "hardcodes source_guidance_scale = 0.0."}),
            },
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("inverted_latent", "source_latent")
    FUNCTION = "run"
    CATEGORY = "PersonalizeAnything"
    DESCRIPTION = (
        "RF-Inversion (https://arxiv.org/pdf/2410.10792) of the source image on comfy's native "
        "FLUX DiT. Produces the 'source branch' latent that the sampler parallelises with the "
        "fresh noise latent. Keep `steps` equal to the sampler's `steps` (the original algorithm "
        "uses the same discretisation for both directions). No CLIP / VAE inside - they stay in "
        "their own nodes so comfy can unload them."
    )

    def run(self, model, conditioning, latent_image, steps, gamma, guidance):
        device = model.load_device
        base_model = model.model

        raw = latent_image["samples"].to(device=device, dtype=torch.float32)
        scaled = base_model.process_latent_in(raw)

        h_tok, w_tok = pa_core.token_grid(scaled.shape)
        sigmas, mu = pa_sampling.official_sigmas(steps, h_tok * w_tok, device=device)

        row = _cond_rows(conditioning)
        row["guidance"] = float(guidance)  # upstream: source_guidance_scale = 0.0
        noise = torch.zeros_like(scaled)

        state = pa_core.PAState()
        transformer_options = pa_core.build_transformer_options(model, state)

        real_model, device = _prepare_model(model, scaled.shape)
        model_conds = _rows_to_model_conds(real_model, [row], noise, device)
        apply_fn = _make_apply_fn(real_model, transformer_options, model_conds)

        pbar = comfy.utils.ProgressBar(steps)
        state.enabled = False  # the inversion itself is *not* personalized

        with comfy.model_management.cuda_device_context(device):
            with pa_core.raw_velocity(real_model):
                inverted = pa_core.rf_invert(
                    apply_fn, scaled, sigmas, gamma=gamma,
                    progress=lambda i, n: pbar.update_absolute(i, n),
                )

        print(f"[PersonalizeAnything] inverted {tuple(scaled.shape)} | "
              f"{pa_sampling.describe(steps, h_tok * w_tok, mu)} | gamma={gamma}")
        # LATENT convention: raw (unscaled) latents, like VAEEncode / KSampler / VAEDecode
        return ({"samples": base_model.process_latent_out(inverted)},
                {"samples": raw})


class PersonalizeAnythingSample:
    """Controlled reverse ODE with the timestep-adaptive token replacement (1 source branch)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING", {"tooltip": "Prompt of the generated image "
                                                             "(target branch, last row)."}),
                "source_conditioning": ("CONDITIONING", {"tooltip": "Prompt of the source branch "
                                                                    "(row 0, the inverted latent). "
                                                                    "Must match the Invert node's "
                                                                    "conditioning (upstream: empty)."}),
                "inverted_latent": ("LATENT", {"tooltip": "From PersonalizeAnythingInvert."}),
                "source_latent": ("LATENT", {"tooltip": "From PersonalizeAnythingInvert."}),
                "mask": ("MASK",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "display": "number"}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 100, "step": 1}),
                "tau": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 100.0, "step": 1.0,
                                  "tooltip": "Timestep (0-100) above which the subject tokens are "
                                             "replaced. Smaller tau = stronger local consistency, "
                                             "less global harmony. 50 = upstream default."}),
                "shift": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1,
                                  "tooltip": "Horizontal shift of the target region, in image tokens "
                                             "(1 token = 16 px). Positive moves the subject left."}),
                "eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                  "tooltip": "Strength of the source branch (0 = ignore the source, "
                                             "1 = fully follow it). 1.0 = upstream default."}),
                "mode": (MODES,),
            },
            "optional": {
                "concept_process": ("BOOLEAN", {"default": False,
                                                "tooltip": "Also concatenate the subject tokens to the "
                                                           "sequence (upstream `concept_process`). "
                                                           "The released upstream app runs with it OFF."}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "run"
    CATEGORY = "PersonalizeAnything"
    DESCRIPTION = (
        "Personalize Anything: the source branch (inverted latent) and the target branch (fresh "
        "noise) are denoised in parallel; while the timestep is above `tau`, the subject tokens of "
        "the source branch replace the target tokens selected by the mask.\n"
        "mask semantics: for inpainting / outpainting the MASK is the region to REPLACE and is "
        "inverted internally (the kept region is what gets copied, like upstream); for the "
        "personalization modes the MASK is the SUBJECT region itself.\n"
        "Plug the output into VAEDecode. Guidance comes from the conditioning (insert a "
        "FluxGuidance node for flux-dev; default 3.5).\n"
        "For multi-subject composition (two source images) use PersonalizeAnythingCompose."
    )

    def run(self, model, conditioning, source_conditioning, inverted_latent, source_latent,
            mask, seed, steps, tau, shift, eta, mode, concept_process=False):
        device = model.load_device
        base_model = model.model

        inv = base_model.process_latent_in(
            inverted_latent["samples"].to(device=device, dtype=torch.float32))
        src = base_model.process_latent_in(
            source_latent["samples"].to(device=device, dtype=torch.float32))
        h_tok, w_tok = pa_core.token_grid(inv.shape)

        grid = pa_core.mask_to_grid(mask, h_tok, w_tok, invert=(mode in INVERT_MASK_MODES))
        shift_grid = pa_core.shift_grid(grid, shift)
        # index tensors must live on the model device (used by index_copy_ in the patch)
        mask_idx = pa_core.grid_to_indices(grid).to(device)
        shift_idx = pa_core.grid_to_indices(shift_grid).to(device)

        noise = comfy.sample.prepare_noise(torch.zeros_like(inv), seed).to(
            device=inv.device, dtype=torch.float32)
        latents = torch.cat([inv, noise], dim=0)      # row 0 = source, row 1 = target
        y_0 = src
        rows = [_cond_rows(source_conditioning), _cond_rows(conditioning)]

        target = _run_reverse(model, latents, y_0, rows,
                              inject=[(0, mask_idx, shift_idx)], target_row=1,
                              seed=seed, steps=steps, tau=tau, eta=eta, mode=mode,
                              concept_process=concept_process)
        # LATENT convention: raw (unscaled) latent -> plug straight into VAEDecode
        return ({"samples": base_model.process_latent_out(target)},)


class PersonalizeAnythingCompose:
    """Scene composition: two inverted branches (foreground + background) -> one image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING", {"tooltip": "Prompt of the composed image "
                                                             "(target branch, last row)."}),
                "fg_conditioning": ("CONDITIONING", {"tooltip": "Prompt of the foreground branch "
                                                                "(row 0). Must match the foreground "
                                                                "Invert node's conditioning."}),
                "bg_conditioning": ("CONDITIONING", {"tooltip": "Prompt of the background branch "
                                                                "(row 1). Must match the background "
                                                                "Invert node's conditioning."}),
                "fg_inverted_latent": ("LATENT", {"tooltip": "Invert node #1 (foreground image)."}),
                "fg_source_latent": ("LATENT", {"tooltip": "Invert node #1 (foreground image)."}),
                "bg_inverted_latent": ("LATENT", {"tooltip": "Invert node #2 (background image)."}),
                "bg_source_latent": ("LATENT", {"tooltip": "Invert node #2 (background image)."}),
                "fg_mask": ("MASK", {"tooltip": "Subject region of the FOREGROUND image; the "
                                                "background branch fills the complement."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "display": "number"}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 100, "step": 1}),
                "tau": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "concept_process": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "run"
    CATEGORY = "PersonalizeAnything"
    DESCRIPTION = (
        "Personalize Anything scene composition (upstream MultiPersonalizeAnythingAttnProcessor): "
        "TWO inverted branches run in parallel with the target - batch = [foreground, background, "
        "target]. Foreground subject tokens are copied into the target at the fg_mask positions, "
        "background tokens at the complement. Wire one PersonalizeAnythingInvert per source image "
        "(fg and bg), keep the prompt of each branch identical between its Invert node and here."
    )

    def run(self, model, conditioning, fg_conditioning, bg_conditioning,
            fg_inverted_latent, fg_source_latent, bg_inverted_latent, bg_source_latent,
            fg_mask, seed, steps, tau, eta, concept_process=False):
        device = model.load_device
        base_model = model.model

        inv_fg = base_model.process_latent_in(
            fg_inverted_latent["samples"].to(device=device, dtype=torch.float32))
        src_fg = base_model.process_latent_in(
            fg_source_latent["samples"].to(device=device, dtype=torch.float32))
        inv_bg = base_model.process_latent_in(
            bg_inverted_latent["samples"].to(device=device, dtype=torch.float32))
        src_bg = base_model.process_latent_in(
            bg_source_latent["samples"].to(device=device, dtype=torch.float32))
        h_tok, w_tok = pa_core.token_grid(inv_fg.shape)

        fg_grid = pa_core.mask_to_grid(fg_mask, h_tok, w_tok, invert=False)
        bg_grid = ~fg_grid
        fg_idx = pa_core.grid_to_indices(fg_grid).to(device)
        bg_idx = pa_core.grid_to_indices(bg_grid).to(device)

        noise = comfy.sample.prepare_noise(torch.zeros_like(inv_fg), seed).to(
            device=inv_fg.device, dtype=torch.float32)
        latents = torch.cat([inv_fg, inv_bg, noise], dim=0)   # rows: fg, bg, target
        y_0 = torch.cat([src_fg, src_bg], dim=0)
        rows = [_cond_rows(fg_conditioning), _cond_rows(bg_conditioning),
                _cond_rows(conditioning)]

        target = _run_reverse(model, latents, y_0, rows,
                              inject=[(0, fg_idx, fg_idx), (1, bg_idx, bg_idx)],
                              target_row=2, seed=seed, steps=steps, tau=tau, eta=eta,
                              mode="scene_composition", concept_process=concept_process)
        return ({"samples": base_model.process_latent_out(target)},)
