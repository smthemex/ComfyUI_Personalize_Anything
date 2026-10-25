# -*- coding: UTF-8 -*-
"""Personalize Anything on comfy's **native** FLUX DiT stack.

The legacy diffusers wrapper that used to live here (``Personalize_Anything_node.py``,
``src/``, ``gradio_demo.py``) is kept in the folder for archaeology but is *not*
imported any more: it dies on transformers>=5 (``diffusers.loaders.peft`` imports
``HybridCache``), which used to make the whole node package vanish from the comfy
menu.  The new implementation needs nothing but torch + comfy itself.
"""

from .pa_nodes import (PersonalizeAnythingInvert, PersonalizeAnythingSample,
                       PersonalizeAnythingCompose)

NODE_CLASS_MAPPINGS = {
    "PersonalizeAnythingInvert": PersonalizeAnythingInvert,
    "PersonalizeAnythingSample": PersonalizeAnythingSample,
    "PersonalizeAnythingCompose": PersonalizeAnythingCompose,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PersonalizeAnythingInvert": "Personalize Anything Invert (RF-Inversion)",
    "PersonalizeAnythingSample": "Personalize Anything Sample",
    "PersonalizeAnythingCompose": "Personalize Anything Compose (fg+bg)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
