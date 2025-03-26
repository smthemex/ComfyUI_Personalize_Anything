import os
#import gradio as gr
from PIL import Image
import numpy as np
from .src.utils import *
from .scripts.grounding_sam import *
from diffusers.models.attention_processor import FluxAttnProcessor2_0
from .src.attn_processor import (
    PersonalizeAnythingAttnProcessor,MultiPersonalizeAnythingAttnProcessor,
    set_flux_transformer_attn_processor,
)
from .node_utils import cf_prompt_clip,cleanup,cf_unload


def generate_image_out(pipe,prompt,new_prompt, seed, timestep, tau, init_image,mask,height,width,img_dims,device,shift,cf_clip=None):
    shift_mask = shift_tensor(mask, shift) 
    generator = torch.Generator(device=device).manual_seed(seed)

    if cf_clip is not None:
 
        inv_prompt_embeds,inv_pooled_prompt_embeds,inv_text_ids=cf_prompt_clip(cf_clip,"")
        inf_prompt_embeds_,inf_pooled_prompt_embeds_,_= cf_prompt_clip(cf_clip,prompt)
        new_prompt_embeds,new_pooled_prompt_embeds,_=cf_prompt_clip(cf_clip,new_prompt)

        pipe_prompt_embeds=torch.cat([inf_prompt_embeds_,new_prompt_embeds],dim=0)# batch size 2
        pipe_pooled_prompt_embeds=torch.cat([inf_pooled_prompt_embeds_,new_pooled_prompt_embeds],dim=0)
       
        cf_clip=None
        cf_unload()
        cleanup()
        
        set_flux_transformer_attn_processor(pipe.transformer, set_attn_proc_func=lambda name, dh, nh, ap:FluxAttnProcessor2_0())
        inverted_latents, image_latents, latent_image_ids = pipe.invert( 
            source_prompt="", 
            image=init_image, 
            height=height,
            width=width,
            num_inversion_steps=timestep, 
            gamma=1.0,
            prompt_embeds= inv_prompt_embeds,
            pooled_prompt_embeds= inv_pooled_prompt_embeds,
            text_ids= inv_text_ids,
            )

        set_flux_transformer_attn_processor(
            pipe.transformer,
            set_attn_proc_func=lambda name, dh, nh, ap: PersonalizeAnythingAttnProcessor(
                name=name, tau=tau/100, mask=mask, shift_mask=shift_mask, device=device, img_dims=img_dims),
        )

        image = pipe(
            None, 
            inverted_latents=inverted_latents,
            image_latents=image_latents,
            latent_image_ids=latent_image_ids,
            height = height,
            width = width,
            start_timestep=0.0, 
            stop_timestep=0.99,
            num_inference_steps=timestep,
            eta=1.0, 
            generator=generator,
            prompt_embeds= pipe_prompt_embeds,
            pooled_prompt_embeds= pipe_pooled_prompt_embeds,
            text_ids= inv_text_ids,
        ).images[1]

    else:
        set_flux_transformer_attn_processor(pipe.transformer, set_attn_proc_func=lambda name, dh, nh, ap:FluxAttnProcessor2_0())
        inverted_latents, image_latents, latent_image_ids = pipe.invert( 
            source_prompt="", 
            image=init_image, 
            height=height,
            width=width,
            num_inversion_steps=timestep, 
            gamma=1.0)

        set_flux_transformer_attn_processor(
            pipe.transformer,
            set_attn_proc_func=lambda name, dh, nh, ap: PersonalizeAnythingAttnProcessor(
                name=name, tau=tau/100, mask=mask, shift_mask=shift_mask, device=device, img_dims=img_dims),
        )

        image = pipe(
            [prompt, new_prompt], 
            inverted_latents=inverted_latents,
            image_latents=image_latents,
            latent_image_ids=latent_image_ids,
            height = height,
            width = width,
            start_timestep=0.0, 
            stop_timestep=0.99,
            num_inference_steps=timestep,
            eta=1.0, 
            generator=generator,
        ).images[1]

    return image

def generate_image_in(pipe,prompt, seed, timestep, tau, init_image,mask,height,width,img_dims,device,shift,cf_clip=None):
    shift_mask = shift_tensor(mask, shift) 
    generator = torch.Generator(device=device).manual_seed(seed)

    if cf_clip is not None:
 
        inv_prompt_embeds,inv_pooled_prompt_embeds,inv_text_ids=cf_prompt_clip(cf_clip,"")
        inf_prompt_embeds_,inf_pooled_prompt_embeds_,_= cf_prompt_clip(cf_clip,prompt)
        
        pipe_prompt_embeds=torch.cat([inf_prompt_embeds_,inv_prompt_embeds],dim=0)# batch size 2
        pipe_pooled_prompt_embeds=torch.cat([inf_pooled_prompt_embeds_,inv_pooled_prompt_embeds],dim=0)
       
        cf_clip=None
        cf_unload()
        cleanup()
        
        inverted_latents, image_latents, latent_image_ids = pipe.invert(
            source_prompt="",
            image=init_image,
            height=height,
            width=width,
            num_inversion_steps=timestep,
            gamma=1.0,
            prompt_embeds= inv_prompt_embeds,
            pooled_prompt_embeds= inv_pooled_prompt_embeds,
            text_ids= inv_text_ids,
        )

        set_flux_transformer_attn_processor(
        pipe.transformer,
        set_attn_proc_func=lambda name, dh, nh, ap: PersonalizeAnythingAttnProcessor(
            name=name, tau=tau/100, mask=mask, shift_mask=shift_mask, device=device, img_dims=img_dims,token_len=inf_prompt_embeds_.shape[1]),
        )

        image = pipe(
            None,
            inverted_latents=inverted_latents,
            image_latents=image_latents,
            latent_image_ids=latent_image_ids,
            height=height,
            width=width,
            start_timestep=0.0,
            stop_timestep=0.99,
            num_inference_steps=timestep,
            eta=1.0,
            generator=generator,
            prompt_embeds= pipe_prompt_embeds,
            pooled_prompt_embeds= pipe_pooled_prompt_embeds,
            text_ids= inv_text_ids,
        ).images[1]
    
    else:
        
        inverted_latents, image_latents, latent_image_ids = pipe.invert(
            source_prompt="",
            image=init_image,
            height=height,
            width=width,
            num_inversion_steps=timestep,
            gamma=1.0,

        )

        set_flux_transformer_attn_processor(
        pipe.transformer,
        set_attn_proc_func=lambda name, dh, nh, ap: PersonalizeAnythingAttnProcessor(
            name=name, tau=tau/100, mask=mask, shift_mask=shift_mask, device=device, img_dims=img_dims),
        )

        image = pipe(
            ["", prompt],
            inverted_latents=inverted_latents,
            image_latents=image_latents,
            latent_image_ids=latent_image_ids,
            height=height,
            width=width,
            start_timestep=0.0,
            stop_timestep=0.99,
            num_inference_steps=timestep,
            eta=1.0,
            generator=generator,
        ).images[1]
        

    return image




def generate_image_personalize_single(pipe,prompt,personalize_prompt, seed, timestep, tau, init_image,mask,height,width,img_dims,device,shift,cf_clip=None):

    shift_mask = shift_tensor(mask, shift) 
    generator = torch.Generator(device=device).manual_seed(seed)

    if cf_clip is not None:
        inv_prompt_embeds,inv_pooled_prompt_embeds,inv_text_ids=cf_prompt_clip(cf_clip,"")
        inf_prompt_embeds_,inf_pooled_prompt_embeds_,_= cf_prompt_clip(cf_clip,prompt)
        new_prompt_embeds,new_pooled_prompt_embeds,_=cf_prompt_clip(cf_clip,personalize_prompt)

        pipe_prompt_embeds=torch.cat([inf_prompt_embeds_,new_prompt_embeds],dim=0)# batch size 2
        pipe_pooled_prompt_embeds=torch.cat([inf_pooled_prompt_embeds_,new_pooled_prompt_embeds],dim=0)
        

        cf_clip=None
        cf_unload()
        cleanup()

        set_flux_transformer_attn_processor(pipe.transformer, set_attn_proc_func=lambda name, dh, nh, ap:FluxAttnProcessor2_0())
        inverted_latents, image_latents, latent_image_ids = pipe.invert( 
            source_prompt="", 
            image=init_image, 
            height=height,
            width=width,
            num_inversion_steps=timestep, 
            gamma=1.0,
            prompt_embeds= inv_prompt_embeds,
            pooled_prompt_embeds= inv_pooled_prompt_embeds,
            text_ids= inv_text_ids,
        )

        set_flux_transformer_attn_processor(
            pipe.transformer,
            set_attn_proc_func=lambda name, dh, nh, ap: PersonalizeAnythingAttnProcessor(
                name=name, tau=tau/100, mask=mask, shift_mask=shift_mask, device=device, img_dims=img_dims),
        )

        image = pipe(
            None, 
            inverted_latents=inverted_latents,
            image_latents=image_latents,
            latent_image_ids=latent_image_ids,
            height = height,
            width = width,
            start_timestep=0.0, 
            stop_timestep=0.99,
            num_inference_steps=timestep,
            eta=1.0, 
            generator=generator,
            prompt_embeds= pipe_prompt_embeds,
            pooled_prompt_embeds= pipe_pooled_prompt_embeds,
            text_ids= inv_text_ids,
        ).images[1]
    else:
        set_flux_transformer_attn_processor(pipe.transformer, set_attn_proc_func=lambda name, dh, nh, ap:FluxAttnProcessor2_0())
        inverted_latents, image_latents, latent_image_ids = pipe.invert( 
            source_prompt="", 
            image=init_image, 
            height=height,
            width=width,
            num_inversion_steps=timestep, 
            gamma=1.0,
        )

        set_flux_transformer_attn_processor(
            pipe.transformer,
            set_attn_proc_func=lambda name, dh, nh, ap: PersonalizeAnythingAttnProcessor(
                name=name, tau=tau/100, mask=mask, shift_mask=shift_mask, device=device, img_dims=img_dims),
        )

        image = pipe(
            [prompt, personalize_prompt], 
            inverted_latents=inverted_latents,
            image_latents=image_latents,
            latent_image_ids=latent_image_ids,
            height = height,
            width = width,
            start_timestep=0.0, 
            stop_timestep=0.99,
            num_inference_steps=timestep,
            eta=1.0, 
            generator=generator,
        ).images[1]
    return image


def generate_image_reconstruction(pipe,prompt,new_prompt, seed, timestep, tau, init_image,mask,height,width,img_dims,device,shift,cf_clip=None):

    shift_mask = shift_tensor(mask, shift) 
    generator = torch.Generator(device=device).manual_seed(seed)

    if cf_clip is not None:
        inv_prompt_embeds,inv_pooled_prompt_embeds,inv_text_ids=cf_prompt_clip(cf_clip,"")
        inf_prompt_embeds_,inf_pooled_prompt_embeds_,_= cf_prompt_clip(cf_clip,prompt)
        new_prompt_embeds,new_pooled_prompt_embeds,_=cf_prompt_clip(cf_clip,new_prompt)

        pipe_prompt_embeds=torch.cat([inf_prompt_embeds_,new_prompt_embeds],dim=0)# batch size 2
        pipe_pooled_prompt_embeds=torch.cat([inf_pooled_prompt_embeds_,new_pooled_prompt_embeds],dim=0)
        

        cf_clip=None
        cf_unload()
        cleanup()

        set_flux_transformer_attn_processor(pipe.transformer, set_attn_proc_func=lambda name, dh, nh, ap:FluxAttnProcessor2_0())
        inverted_latents, image_latents, latent_image_ids = pipe.invert( 
            source_prompt="", 
            image=init_image, 
            height=height,
            width=width,
            num_inversion_steps=timestep, 
            gamma=1.0,
            prompt_embeds= inv_prompt_embeds,
            pooled_prompt_embeds= inv_pooled_prompt_embeds,
            text_ids= inv_text_ids,
            )

        
        shift_mask = shift_tensor(mask, shift)
        set_flux_transformer_attn_processor(
            pipe.transformer,
            set_attn_proc_func=lambda name, dh, nh, ap: PersonalizeAnythingAttnProcessor(
                name=name, 
                mask=mask, 
                shift_mask=shift_mask, 
                tau=tau/100, 
                device=device, 
                img_dims=img_dims,
                concept_process=False),
        )

        image = pipe(
            None, 
            inverted_latents=inverted_latents,
            image_latents=image_latents,
            latent_image_ids=latent_image_ids,
            height = height,
            width = width,
            start_timestep=0.0, 
            stop_timestep=0.99,
            num_inference_steps=timestep,
            eta=1.0, 
            generator=generator,
            prompt_embeds= pipe_prompt_embeds,
            pooled_prompt_embeds= pipe_pooled_prompt_embeds,
            text_ids= inv_text_ids,
        ).images[1]
    else:

        set_flux_transformer_attn_processor(pipe.transformer, set_attn_proc_func=lambda name, dh, nh, ap:FluxAttnProcessor2_0())
        inverted_latents, image_latents, latent_image_ids = pipe.invert( 
            source_prompt="", 
            image=init_image, 
            height=height,
            width=width,
            num_inversion_steps=timestep, 
            gamma=1.0)

        
        shift_mask = shift_tensor(mask, shift)
        set_flux_transformer_attn_processor(
            pipe.transformer,
            set_attn_proc_func=lambda name, dh, nh, ap: PersonalizeAnythingAttnProcessor(
                name=name, 
                mask=mask, 
                shift_mask=shift_mask, 
                tau=tau/100, 
                device=device, 
                img_dims=img_dims,
                concept_process=False),
        )

        image = pipe(
            [prompt, new_prompt], 
            inverted_latents=inverted_latents,
            image_latents=image_latents,
            latent_image_ids=latent_image_ids,
            height = height,
            width = width,
            start_timestep=0.0, 
            stop_timestep=0.99,
            num_inference_steps=timestep,
            eta=1.0, 
            generator=generator,
        ).images[1]

    return image

def generate_image_Composition(pipe,prompt,new_prompt, seed, timestep, tau, fg_image,bg_image,fg_mask,bg_mask,height,width,img_dims,device,shift,cf_clip=None):

    fg_prompt, bg_prompt = prompt[0], prompt[1]
    generator = torch.Generator(device=device).manual_seed(seed)

    
    if cf_clip is not None:
        #inv_prompt_embeds,inv_pooled_prompt_embeds,inv_text_ids=cf_prompt_clip(cf_clip,"")
        fg_prompt_embeds_,fg_pooled_prompt_embeds_,inv_text_ids= cf_prompt_clip(cf_clip,fg_prompt)
        bg_prompt_embeds_,bg_pooled_prompt_embeds_,_= cf_prompt_clip(cf_clip,bg_prompt)
        new_prompt_embeds,new_pooled_prompt_embeds,_=cf_prompt_clip(cf_clip,new_prompt)


        pipe_prompt_embeds=torch.cat([bg_prompt_embeds_,fg_prompt_embeds_,new_prompt_embeds],dim=0)# batch size 2
        pipe_pooled_prompt_embeds=torch.cat([bg_pooled_prompt_embeds_,fg_pooled_prompt_embeds_,new_pooled_prompt_embeds],dim=0)
        

        cf_clip=None
        cf_unload()
        cleanup()

        set_flux_transformer_attn_processor(pipe.transformer, set_attn_proc_func=lambda name, dh, nh, ap:FluxAttnProcessor2_0())
        inverted_latents_fg, image_latents_fg, latent_image_ids = pipe.invert(
            source_prompt="",
            image=fg_image,
            height=height,
            width=width,
            num_inversion_steps=timestep,
            gamma=1.0,
            prompt_embeds= fg_prompt_embeds_,
            pooled_prompt_embeds= fg_pooled_prompt_embeds_,
            text_ids= inv_text_ids,
        )

        inverted_latents_bg, image_latents_bg, latent_image_ids = pipe.invert(
            source_prompt="",
            image=bg_image,
            height=height,
            width=width,
            num_inversion_steps=timestep,
            gamma=1.0,
            prompt_embeds= bg_prompt_embeds_,
            pooled_prompt_embeds= bg_pooled_prompt_embeds_,
            text_ids= inv_text_ids,
        )

        inverted_latents = torch.cat([inverted_latents_fg, inverted_latents_bg], dim=0)
        image_latents = torch.cat([image_latents_fg, image_latents_bg], dim=0)
        masks = [fg_mask, bg_mask]

        set_flux_transformer_attn_processor(
            pipe.transformer,
            set_attn_proc_func=lambda name, dh, nh, ap: MultiPersonalizeAnythingAttnProcessor(
                name=name, tau=tau/100, masks=masks, shift_masks=None, device=device, img_dims=img_dims),
        )

        image = pipe(
            None,
            inverted_latents=inverted_latents,
            image_latents=image_latents,
            latent_image_ids=latent_image_ids,
            height = height,
            width = width,
            start_timestep=0.0,
            stop_timestep=0.99,
            num_inference_steps=timestep,
            eta=1.0,
            generator=generator,
            prompt_embeds= pipe_prompt_embeds,
            pooled_prompt_embeds= pipe_pooled_prompt_embeds,
            text_ids= inv_text_ids,
        ).images[-1]
    else:

        set_flux_transformer_attn_processor(pipe.transformer, set_attn_proc_func=lambda name, dh, nh, ap:FluxAttnProcessor2_0())
        inverted_latents_fg, image_latents_fg, latent_image_ids = pipe.invert(
            source_prompt="",
            image=fg_image,
            height=height,
            width=width,
            num_inversion_steps=timestep,
            gamma=1.0,
        )

        inverted_latents_bg, image_latents_bg, latent_image_ids = pipe.invert(
            source_prompt="",
            image=bg_image,
            height=height,
            width=width,
            num_inversion_steps=timestep,
            gamma=1.0,
        )

        inverted_latents = torch.cat([inverted_latents_fg, inverted_latents_bg], dim=0)
        image_latents = torch.cat([image_latents_fg, image_latents_bg], dim=0)
        masks = [fg_mask, bg_mask]

        set_flux_transformer_attn_processor(
            pipe.transformer,
            set_attn_proc_func=lambda name, dh, nh, ap: MultiPersonalizeAnythingAttnProcessor(
                name=name, tau=tau/100, masks=masks, shift_masks=None, device=device, img_dims=img_dims),
        )

        image = pipe(
            [fg_prompt, bg_prompt, new_prompt],
            inverted_latents=inverted_latents,
            image_latents=image_latents,
            latent_image_ids=latent_image_ids,
            height = height,
            width = width,
            start_timestep=0.0,
            stop_timestep=0.99,
            num_inference_steps=timestep,
            eta=1.0,
            generator=generator,
        ).images[-1]

    return image
