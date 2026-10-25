# -*- coding: UTF-8 -*-
"""Personalize Anything, re-implemented on comfy's *native* FLUX DiT stack.

Upstream (https://github.com/fenghora/personalize-anything) is a diffusers wrapper that
re-implements three things:

  1. a custom FLUX attention processor - "timestep-adaptive token replacement"
     (the paper's core trick: the subject tokens of the source image are copied into
      the target image's tokens while the timestep is still high),
  2. the RF-Inversion *forward* ODE that turns the source image into a noise latent
     (Eq. 8 of https://arxiv.org/pdf/2410.10792),
  3. the RF-Inversion *controlled reverse* ODE that denoises a fresh noise latent
     while a second, parallel batch row carries the source branch (Algorithm 2).

This module re-implements exactly those three pieces against comfy's own
``comfy.ldm.flux`` implementation, so the plugin needs **no diffusers, no transformers,
no peft** and is immune to upstream version drift (the old wrapper dies on
``transformers>=5`` because ``diffusers.loaders.peft`` imports ``HybridCache``).

Why the attention trick survives the port
-----------------------------------------
Upstream replaces rows of the *hidden states* (before to_q/to_k/to_v) and then projects:

    concept = hidden[src_row, mask]              # pre-projection
    hidden[dst_row, shift_mask] = concept        # in-place
    q, k, v = to_q(hidden), to_k(hidden), to_v(hidden)

comfy's FLUX blocks hand us q/k/v *after* the fused qkv projection and after QKNorm.
Because the projection is linear and QKNorm is per-token elementwise (RMSNorm over
head_dim), replacing the projected rows is *exactly* equivalent to replacing the
hidden rows and projecting afterwards:

    to_q(hidden[dst_row, shift_mask] = concept) == to_q(hidden)[dst_row, shift_mask]

so we can do the whole trick on the q/k/v tensors comfy's own attention consumes.

The token layout is identical between the two stacks, too:
  * diffusers FLUX: ``_pack_latents`` -> view(B,C,h/2,2,w/2,2).permute(0,2,4,1,3,5)
    -> token = 16 latent channels x 2x2 spatial patch, token grid (h/16, w/16) row major.
  * comfy FLUX    : ``rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)")`` with
    patch_size=2 -> the very same token grid (h/16, w/16), row major, same 64 values.
  * the RoPE ids are row major over that same grid in both stacks.
Therefore a mask built on the (h/16, w/16) grid indexes the same spatial positions in
comfy as it does upstream - no index remapping needed.
"""

import math

import torch

import comfy.model_management
import comfy.utils
from comfy.ldm.flux.math import apply_rope as _flux_apply_rope

# --------------------------------------------------------------------------------------
# attention patch
# --------------------------------------------------------------------------------------


class PAState:
    """Mutable per-step state read by the attention patch.

    ``inject`` is a list of ``(src_row, mask_idx, shift_idx)``:
      * ``src_row``   - batch row that owns the subject tokens (0 = source branch)
      * ``mask_idx``  - flat token indices (on the h_tok x w_tok grid) of the subject
      * ``shift_idx`` - flat token indices where the subject is written
    ``target_row`` is the row that receives the replacement.
    """

    def __init__(self):
        self.enabled = False
        self.tau = 0.5            # replacement is active while model_t > tau
        self.model_t = 0.0        # model input timestep of the current step (sigma)
        self.inject = None
        self.target_row = 1
        self.concept_process = False
        self.token_len = 512
        self.fire_count = 0       # diagnostics: how often the replacement actually ran

    def reset(self):
        self.enabled = False
        self.model_t = 0.0
        self.inject = None
        self.target_row = 1
        self.concept_process = False
        self.fire_count = 0


def make_attn_patch(state: PAState):
    """comfy ``transformer_options["patches"]["attn1_patch"]`` callable.

    Called by every flux block (19 double + 38 single) with
    ``q, k, v`` of shape ``[B, heads, seq, head_dim]`` and ``pe`` of shape
    ``[1, 1, seq, head_dim/2, 2, 2]``.
    """

    def patch(q, k, v, pe=None, attn_mask=None, extra_options=None):
        if not state.enabled or not state.inject:
            return {}
        # timestep-adaptive gate: upstream compares the *model* timestep (sigma) with tau
        if not (state.model_t > state.tau):
            return {}
        extra_options = extra_options or {}
        img_slice = extra_options.get("img_slice", None)
        if img_slice is None:
            return {}
        txt_len = int(img_slice[0])

        tgt = state.target_row
        for src_row, mask_idx, shift_idx in state.inject:
            src = txt_len + mask_idx
            dst = txt_len + shift_idx
            for t in (q, k, v):
                t[tgt].index_copy_(1, dst, t[src_row].index_select(1, src))
        state.fire_count += 1

        if not state.concept_process:
            return {"q": q, "k": k, "v": v}

        # ---- token concatenation (upstream `concept_process`) ----------------------
        # the subject tokens are appended to the sequence so that *every* token can
        # attend to them; they keep the rope of their original position.
        c_q, c_k, c_v, c_pe = [], [], [], []
        for src_row, mask_idx, _ in state.inject:
            src = txt_len + mask_idx
            c_q.append(q[src_row].index_select(1, src))
            c_k.append(k[src_row].index_select(1, src))
            c_v.append(v[src_row].index_select(1, src))
            if pe is not None:
                c_pe.append(pe[0, 0].index_select(0, src))
        c_q = torch.cat(c_q, dim=1)
        c_k = torch.cat(c_k, dim=1)
        c_v = torch.cat(c_v, dim=1)
        b = q.shape[0]
        q = torch.cat([q, c_q.unsqueeze(0).expand(b, -1, -1, -1)], dim=2)
        k = torch.cat([k, c_k.unsqueeze(0).expand(b, -1, -1, -1)], dim=2)
        v = torch.cat([v, c_v.unsqueeze(0).expand(b, -1, -1, -1)], dim=2)

        if pe is not None:
            pe = torch.cat([pe, torch.cat(c_pe, dim=0).unsqueeze(0).unsqueeze(0)], dim=2)
            q, k = _flux_apply_rope(q, k, pe)
            # rope already applied -> hand the block a null pe so it does not rope twice
            return {"q": q, "k": k, "v": v, "pe": None}
        return {"q": q, "k": k, "v": v}

    return patch


def make_out_patch(state: PAState):
    """comfy ``transformer_options["patches"]["attn1_output_patch"]`` callable.

    Only needed when the sequence was extended by the token concatenation: the block
    expects the attention output to have exactly ``txt + img`` rows.
    """

    def patch(attn, extra_options=None):
        if not state.enabled or not state.concept_process:
            return attn
        extra_options = extra_options or {}
        img_slice = extra_options.get("img_slice", None)
        if img_slice is None:
            return attn
        seq = int(img_slice[1])
        if attn.shape[1] > seq:
            return attn[:, :seq]
        return attn

    return patch


def build_transformer_options(patcher, state: PAState):
    """transformer_options for the DiT call: the patcher's own options + our patches.

    The patcher's options are kept so that hooks installed by other plugins (LoRA
    patches, rope options, controlnet hooks, ...) keep working.
    """
    base = {}
    model_options = getattr(patcher, "model_options", None) or {}
    own = model_options.get("transformer_options", None) or {}
    if own:
        base = dict(own)
    patches = dict(base.get("patches", {}) or {})
    patches["attn1_patch"] = list(patches.get("attn1_patch", []) or []) + [make_attn_patch(state)]
    patches["attn1_output_patch"] = list(patches.get("attn1_output_patch", []) or []) + [make_out_patch(state)]
    base["patches"] = patches
    return base


# --------------------------------------------------------------------------------------
# raw velocity proxy
# --------------------------------------------------------------------------------------


class RawVelocitySampling(torch.nn.Module):
    """Proxy over ``BaseModel.model_sampling`` that returns the *raw* velocity.

    comfy's ``BaseModel._apply_model`` ends with::

        return self.model_sampling.calculate_denoised(sigma, model_output, x)

    and for FLUX (CONST mixin) that is ``x - sigma * v`` - an x0 prediction.  RF-Inversion
    needs the velocity ``v`` itself, and the inversion loop evaluates the model at
    ``sigma = 0`` where x0 and v cannot be separated any more.  So we swap that one
    conversion out and delegate everything else (calculate_input / timestep / sigma
    tables) unchanged.
    """

    def __init__(self, base):
        super().__init__()
        self.base = base

    def calculate_input(self, sigma, noise):
        return self.base.calculate_input(sigma, noise)

    def timestep(self, sigma):
        return self.base.timestep(sigma)

    def calculate_denoised(self, sigma, model_output, model_input):
        return model_output.float()

    def percent_to_sigma(self, percent):
        return self.base.percent_to_sigma(percent)

    def sigma(self, timestep):
        return self.base.sigma(timestep)

    @property
    def sigma_min(self):
        return self.base.sigma_min

    @property
    def sigma_max(self):
        return self.base.sigma_max


class raw_velocity:
    """Context manager: swap the model's model_sampling for the raw-velocity proxy."""

    def __init__(self, base_model):
        self.base_model = base_model
        self.orig = None

    def __enter__(self):
        self.orig = self.base_model.model_sampling
        self.base_model.model_sampling = RawVelocitySampling(self.orig)
        return self.base_model

    def __exit__(self, *args):
        self.base_model.model_sampling = self.orig
        self.orig = None
        return False


# --------------------------------------------------------------------------------------
# mask helpers
# --------------------------------------------------------------------------------------


def token_grid(latent_shape, patch_size=2):
    """(h_tok, w_tok) of the image-token grid for a comfy latent [B, C, h, w]."""
    h = int(latent_shape[-2])
    w = int(latent_shape[-1])
    return ((h + (patch_size // 2)) // patch_size, (w + (patch_size // 2)) // patch_size)


def mask_to_grid(mask, h_tok, w_tok, invert=False, threshold=0.5):
    """comfy MASK ([B,H,W] in [0,1]) -> boolean grid [h_tok, w_tok].

    Upstream resized the PIL mask with NEAREST to (w/16, h/16) and kept the *pure
    white* pixels.  We use the same nearest resize but threshold at 0.5 so that soft
    masks also work.
    """
    if mask is None:
        return torch.zeros((h_tok, w_tok), dtype=torch.bool)
    m = mask
    if m.ndim == 2:
        m = m.unsqueeze(0)
    m = m[0].detach().float().cpu()
    if invert:
        m = 1.0 - m
    m = m.unsqueeze(0).unsqueeze(0)
    m = comfy.utils.common_upscale(m, w_tok, h_tok, "nearest-exact", "center")
    return m[0, 0] > threshold


def grid_to_indices(grid):
    return torch.nonzero(grid.reshape(-1), as_tuple=False).reshape(-1).long()


def shift_grid(grid, shift):
    """Horizontal shift of the token grid (upstream ``shift_tensor``)."""
    if shift == 0:
        return grid
    out = torch.zeros_like(grid)
    w = grid.shape[1]
    if shift > 0:
        s = min(shift, w)
        out[:, s:] = grid[:, : w - s]
    else:
        s = min(-shift, w)
        out[:, : w - s] = grid[:, s:]
    return out


# --------------------------------------------------------------------------------------
# the two ODEs
# --------------------------------------------------------------------------------------


def rf_invert(apply_fn, image_latents, sigmas, gamma=1.0, progress=None):
    """RF-Inversion forward ODE (Eq. 8 of https://arxiv.org/pdf/2410.10792).

    ``apply_fn(x, t) -> velocity`` must return the *raw* velocity (the guidance
    embedding travels inside the conditioning, as in the stock sampler).

    Note the upstream asymmetry (reproduced verbatim):
      * the model is evaluated at the **raw** time ``t_i = i / (steps + 1)`` (NOT the
        shifted sigma), while the ODE step uses the **shifted** sigmas;
      * the source branch uses guidance 0.0 (set on the conditioning by the node).
    """
    y_t = image_latents
    y_1 = torch.randn_like(y_t)
    n = len(sigmas) - 1  # number of inversion steps
    for i in range(n):
        t_i = i / (n + 1)
        u_t_i = apply_fn(y_t, t_i)
        u_t_i_cond = (y_1 - y_t) / (1.0 - t_i)
        u_hat = u_t_i + gamma * (u_t_i_cond - u_t_i)
        y_t = y_t + u_hat * (sigmas[i] - sigmas[i + 1])
        if progress is not None:
            progress(i + 1, n)
    return y_t


def parallel_reverse(apply_fn, latents, y_0, sigmas, eta=1.0, start=0, stop=None,
                     state: PAState = None, progress=None):
    """Controlled reverse ODE (Algorithm 2 of RF-Inversion).

    ``latents`` is ``[source_branch, target_branch]`` (or ``[fg, bg, target]``):
      * every row but the last one follows the *conditional* field towards ``y_0``
        (the encoded source image) with strength ``eta``;
      * the last row is the plain euler step of the target image.

    The model is evaluated at ``sigma`` (the shifted sigma) - exactly like upstream
    (``timestep / 1000`` with ``timestep = sigma * 1000``).
    """
    if stop is None:
        stop = len(sigmas) - 1
    n = len(sigmas) - 1
    for i in range(n):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        if state is not None:
            state.model_t = float(sigma)
        noise_pred = apply_fn(latents, float(sigma))

        v_t = -noise_pred[:-1]
        v_t_cond = (y_0 - latents[:-1]) / sigma
        eta_t = eta if start <= i < stop else 0.0
        v_hat = v_t + eta_t * (v_t_cond - v_t)
        latents[:-1] = latents[:-1] + v_hat * (sigma - sigma_next)
        latents[-1] = latents[-1] + (sigma_next - sigma) * noise_pred[-1]
        if progress is not None:
            progress(i + 1, n)
    return latents


# --------------------------------------------------------------------------------------
# prompt / image helpers
# --------------------------------------------------------------------------------------


def encode_prompt(clip, text, min_length=512, device=None):
    """comfy CLIP -> (t5 context [1, T, 4096], clip_l pooled [1, 768]).

    ``min_length`` = 512 reproduces the official diffusers padding
    (``padding="max_length", max_length=512``); comfy's flux t5xxl tokenizer defaults
    to 256, which would put the model off its training distribution.
    """
    options = {}
    if min_length:
        options["t5xxl_min_length"] = int(min_length)
    tokens = clip.tokenize(text, tokenizer_options=options)
    out = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
    cond = out["cond"]
    pooled = out.get("pooled_output", None)
    if device is not None:
        cond = cond.to(device)
        if pooled is not None:
            pooled = pooled.to(device)
    return cond, pooled


def encode_prompts(clip, texts, min_length=512, device=None):
    """Encode several prompts, one call each (keeps the per-prompt padding exact)."""
    ctxs, pools = [], []
    for t in texts:
        c, p = encode_prompt(clip, t, min_length=min_length, device=device)
        ctxs.append(c)
        pools.append(p)
    return torch.cat(ctxs, dim=0), (torch.cat(pools, dim=0) if pools[0] is not None else None)
