# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
from PIL import Image
import torch
import numpy as np
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig,FluxTransformer2DModel, AutoencoderKL
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig,T5EncoderModel

from .src.utils import *
from .scripts.grounding_sam import *
from .node_utils import cleanup,pil2narry,process_image_with_mask,tensor2pil_upscale
from .src.pipeline import RFInversionParallelFluxPipeline
from .src.pipeline_wrapper import RFInversionParallelFluxPipeline as RFInversionParallelFluxPipeline_Wrapper
from .gradio_demo import generate_image_in_out,generate_image_personalize_single,generate_image_reconstruction,generate_image_Composition

import folder_paths

MAX_SEED = np.iinfo(np.int32).max
current_node_path = os.path.dirname(os.path.abspath(__file__))

device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device(
    "mps") if torch.backends.mps.is_available() else torch.device(
    "cpu")


class Personalize_Anything_Load:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "transformer": (["none"]+folder_paths.get_filename_list("diffusion_models"),),
                "flux_repo": ("STRING", {"default": "F:/test/ComfyUI/models/diffusers/black-forest-labs/FLUX.1-dev"},),
                "quantization":(["none","fp8","nf4"],),
                "quantize_T5":("BOOLEAN",{"default":True}),
                },
            "optional": { "model":("MODEL",),
                         "vae":("VAE",),
                         }
            }

    RETURN_TYPES = ("MODEL_PERSONALIZE_ANYTHING", )
    RETURN_NAMES = ("model",)
    FUNCTION = "main"
    CATEGORY = "Personalize_Anything"

    def main(self, transformer,flux_repo,quantization,quantize_T5,**kwargs):
        cf_model=kwargs.get("model",None)
        cf_vae=kwargs.get("vae",None)
 
        flux_repo_local=os.path.join(current_node_path, 'src/FLUX.1-dev') 
       
        if flux_repo:
            
            if quantization=="none":
                if quantize_T5:
                    quant_config = TransformersBitsAndBytesConfig(load_in_8bit=True,) # 8bit default
                    text_encoder_2_8bit = T5EncoderModel.from_pretrained(
                    flux_repo,
                    subfolder="text_encoder_2",
                    quantization_config=quant_config,
                    torch_dtype=torch.bfloat16,
                )
                    pipeline = RFInversionParallelFluxPipeline.from_pretrained(flux_repo,text_encoder_2=text_encoder_2_8bit,torch_dtype=torch.bfloat16,)
                else:
                    pipeline = RFInversionParallelFluxPipeline.from_pretrained(flux_repo,torch_dtype=torch.bfloat16,)
            else:
                
                if quantization=="fp8":
                    if quantize_T5:
                        quant_config = TransformersBitsAndBytesConfig(load_in_8bit=True,) # 8bit 
                        text_encoder_2_8bit = T5EncoderModel.from_pretrained(
                        flux_repo,
                        subfolder="text_encoder_2",
                        quantization_config=quant_config,
                        torch_dtype=torch.bfloat16,
                        )    
                    transformer = FluxTransformer2DModel.from_pretrained(
                        flux_repo,
                        subfolder="transformer",
                        quantization_config=DiffusersBitsAndBytesConfig(load_in_8bit=True,),
                        torch_dtype=torch.bfloat16,
                    )
                    if quantize_T5:
                        pipeline =RFInversionParallelFluxPipeline.from_pretrained(flux_repo,transformer=transformer,text_encoder_2=text_encoder_2_8bit,torch_dtype=torch.bfloat16,)
                    else:
                        pipeline = RFInversionParallelFluxPipeline.from_pretrained(flux_repo,transformer=transformer, torch_dtype=torch.bfloat16,)
                else: #nf4
                    if quantize_T5:
                        quant_config = TransformersBitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_quant_type="nf4",
                            )

                        text_encoder_2_4bit = T5EncoderModel.from_pretrained(
                                flux_repo,
                                subfolder="text_encoder_2",
                                quantization_config=quant_config,
                                torch_dtype=torch.bfloat16,
                            )

                    transformer = FluxTransformer2DModel.from_pretrained(
                        flux_repo,
                        subfolder="transformer",
                        quantization_config=DiffusersBitsAndBytesConfig(
                            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
                        ),
                        torch_dtype=torch.bfloat16,)
                    if quantize_T5:
                        pipeline =RFInversionParallelFluxPipeline.from_pretrained(flux_repo,transformer=transformer,text_encoder_2=text_encoder_2_4bit,torch_dtype=torch.bfloat16,)
                    else:
                        pipeline = RFInversionParallelFluxPipeline.from_pretrained(flux_repo,transformer=transformer, torch_dtype=torch.bfloat16,)
        else:
            if cf_model is None and cf_vae is None:
                if transformer != "none":
                    flux_transformer_path = folder_paths.get_full_path("diffusion_models", transformer)
                    pipeline = RFInversionParallelFluxPipeline.from_single_file(
                             flux_transformer_path,config=flux_repo_local, torch_dtype=torch.bfloat16,)
                else:
                    raise Exception("Please select a transformer model")
            else:
                ae_dic=cf_vae.get_sd()
                # vae_path = folder_paths.get_full_path("vae", vae)
                vae_config=os.path.join(flux_repo_local, 'vae')
                ae = AutoencoderKL.from_single_file(ae_dic,config=vae_config, torch_dtype=torch.bfloat16)

                config_file = os.path.join(flux_repo_local,"transformer/config.json")
                if transformer != "none":
                    flux_transformer_path = folder_paths.get_full_path("diffusion_models", transformer)
                    transformer_ = FluxTransformer2DModel.from_single_file(
                            flux_transformer_path,
                            config=config_file,
                            torch_dtype=torch.bfloat16,
                            )
                else:
                    if cf_model is not None:
                       
                        cf_state_dict = cf_model.model.diffusion_model.state_dict()
                        del cf_model
                        cleanup()
                        transformer_=FluxTransformer2DModel.from_single_file(cf_state_dict,config=config_file,torch_dtype=torch.bfloat16)
                        # unet_config = FluxTransformer2DModel.load_config(config_file)
                        # transformer_ = FluxTransformer2DModel.from_config(unet_config).to(torch.bfloat16)
                        # transformer_.load_state_dict(cf_state_dict, strict=False)
                        del cf_state_dict
                        cleanup()
                    else:
                        raise ValueError("No transformer model found")
                pipeline = RFInversionParallelFluxPipeline_Wrapper.from_pretrained(
                                flux_repo_local,vae=ae,transformer=transformer_, torch_dtype=torch.bfloat16,)
        cleanup()
        pipeline.enable_model_cpu_offload()
        return (pipeline,)


class Personalize_Anything_Sampler:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL_PERSONALIZE_ANYTHING",),
                "iamge": ("IMAGE",), # B H W C
                "mask": ("MASK",),  # B H W 
                "prompt": ("STRING", {"default": "A teddy bear", "multiline": True,}),
                "personalize_prompt": ("STRING", {"default": "A teddy bear waving its right hand on a nighttime street, positioned on the left side of the frame, with an empty road on the right.", "multiline": True,}),
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED, "step": 1, "display": "number"}),
                "width": ("INT", {"default": 512, "min": 256, "max": 4096, "step": 16, "display": "number"}),
                "height": ("INT", {"default": 512, "min": 256, "max": 4096, "step": 16, "display": "number"}),
                "timestep": ("INT", {"default": 28, "min": 1, "max": 1024, "step": 1, "display": "number"}),
                "tau": ("INT", {"default": 50, "min": 0, "max": 100, "step": 1, "display": "number"}),
                "shift": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1, "display": "number"}),
                "infer_mode":(["inpainting","outpainting","single_personalize","multi_personalize","subject_reconstruction","scene_composition"],),
                },
            "optional": { "bg_image":("IMAGE",),
                         "clip": ("CLIP",),
                         },
            }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image",)
    FUNCTION = "main"
    CATEGORY = "Personalize_Anything"

    def main(self,model, iamge,mask,prompt,personalize_prompt,seed,width,height,timestep,tau,shift,infer_mode,**kwargs):
        bg_image=kwargs.get("bg_image")
        cf_clip=kwargs.get("clip")


        if infer_mode=="single_personalize" or infer_mode=="subject_reconstruction" or infer_mode=="scene_composition":
            init_image,mask_pil=process_image_with_mask(iamge,mask,width,height,False) #false
        else:
            init_image,mask_pil=process_image_with_mask(iamge,mask,width,height,True) # outpainting and inpainting need Ture

        if infer_mode=="scene_composition":
            if "[" in prompt:
                prompt=prompt.split("[")
            else:
                raise ValueError("scene_composition need a [ in prompt")
        
            if isinstance(bg_image, torch.Tensor) :
                bg_image=tensor2pil_upscale(bg_image,width,height)
            else:
                raise ValueError("bg_image must link a image")
                
            
        latent_h = height // 16
        latent_w = width // 16
        img_dims = latent_h * latent_w

        mask = create_mask(mask_pil, latent_w, latent_h)
        bg_mask=1-mask
        # inverse
  
        if infer_mode=="inpainting":
            img=generate_image_in_out(model,prompt,personalize_prompt, seed, timestep, tau,init_image,mask,height,width,img_dims,device,shift,True,cf_clip)
        elif infer_mode=="outpainting":
            img=generate_image_in_out(model,prompt,personalize_prompt, seed, timestep, tau,init_image,mask,height,width,img_dims,device,shift,False,cf_clip)
        elif infer_mode=="single_personalize":
            img=generate_image_personalize_single(model,prompt,personalize_prompt, seed, timestep, tau, init_image,mask,height,width,img_dims,device,shift,cf_clip)
        elif infer_mode=="multi_personalize":
            pass
        elif infer_mode=="scene_composition":
            img=generate_image_Composition(model,prompt,personalize_prompt, seed, timestep, tau, init_image,bg_image,mask,bg_mask,height,width,img_dims,device,shift,cf_clip)
        else: # subject_reconstruction
            img=generate_image_reconstruction(model,prompt,personalize_prompt, seed, timestep, tau, init_image,mask,height,width,img_dims,device,shift,cf_clip)
        cleanup()
        return (pil2narry(img),)


NODE_CLASS_MAPPINGS = {
    "Personalize_Anything_Load": Personalize_Anything_Load,
    "Personalize_Anything_Sampler":Personalize_Anything_Sampler,

}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Personalize_Anything_Load": "Personalize_Anything_Load",
    "Personalize_Anything_Sampler":"Personalize_Anything_Sampler",
}
