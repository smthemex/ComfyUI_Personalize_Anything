# -*- coding: UTF-8 -*-
"""Official FLUX sigma grid, replayed on comfy's native stack.

Source of truth = the scheduler config that shipped with the original
personalize-anything repo (``src/FLUX.1-dev/scheduler/scheduler_config.json``)::

    FlowMatchEulerDiscreteScheduler
    base_image_seq_len 256 / max_image_seq_len 4096
    base_shift 0.5     / max_shift 1.15
    use_dynamic_shifting = True

diffusers 0.37 ``FlowMatchEulerDiscreteScheduler.set_timesteps(sigmas=..., mu=...)``::

    sigmas = linspace(1.0, 1/steps, steps)          # raw grid
    sigmas = time_shift(mu, 1.0, sigmas)             # exp(mu)/(exp(mu)+(1/t-1))
    sigmas = cat([sigmas, 0.0])                      # terminal

``mu = calculate_shift(image_seq_len, 256, 4096, 0.5, 1.15)`` - i.e. the shift is
*resolution dependent* (256 tokens -> mu 0.5, 4096 tokens -> mu 1.15).

comfy's own ``comfy.model_sampling.flux_time_shift`` has the identical formula, but
comfy's ``ModelSamplingFlux`` hard-codes ``shift=1.15`` for *every* resolution, so the
stock KSampler is NOT equivalent to the official pipeline below 1024px.  The whole
Personalize-Anything algebra (RF-Inversion) is written against the *official* grid, so
we recompute it here and drive the model ourselves.
"""

import math

import numpy as np
import torch

# scheduler_config.json of the original repo (FLUX.1-dev)
BASE_SEQ = 256
MAX_SEQ = 4096
BASE_SHIFT = 0.5
MAX_SHIFT = 1.15


def calculate_shift(seq_len, base_seq=BASE_SEQ, max_seq=MAX_SEQ,
                    base_shift=BASE_SHIFT, max_shift=MAX_SHIFT):
    """diffusers.pipelines.flux.pipeline_flux.calculate_shift (identical in comfy)."""
    m = (max_shift - base_shift) / (max_seq - base_seq)
    b = base_shift - m * base_seq
    return seq_len * m + b


def time_shift(mu, sigma, t):
    """FlowMatchEulerDiscreteScheduler.time_shift (== comfy's flux_time_shift)."""
    return math.exp(mu) / (math.exp(mu) + (1.0 / t - 1.0) ** sigma)


def official_sigmas(steps, tokens, device=None, dtype=torch.float32):
    """Full sigma table (len = steps+1, terminal 0), diffusers-exact.

    `tokens` = image_seq_len = (latent_h/2) * (latent_w/2) = number of image tokens.
    """
    raw = np.linspace(1.0, 1.0 / steps, steps).astype(np.float32)
    mu = calculate_shift(tokens)
    shifted = np.array([time_shift(mu, 1.0, float(t)) for t in raw], dtype=np.float32)
    sig = np.concatenate([shifted, np.zeros(1, dtype=np.float32)])
    out = torch.from_numpy(sig).to(dtype=dtype)
    if device is not None:
        out = out.to(device)
    return out, mu


def describe(steps, tokens, mu):
    return (f"steps={steps} tokens={tokens} mu={mu:.4f} "
            f"shift={math.exp(mu):.3f} sigma[0]={1.0:.4f}")
