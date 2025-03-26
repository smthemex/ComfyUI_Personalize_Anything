# ComfyUI_Personalize_Anything
[Personalize Anything](https://github.com/fenghora/personalize-anything) for Free with Diffusion Transformer,use it in comfyUI with wrapper mode

# Update
* fix inpainting bug, inpainting only use prompt line one /修复内绘的bug，注意内绘只用第一行的prompt

# 1. Installation

In the ./ComfyUI/custom_node directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_Personalize_Anything
```
---

# 2. Requirements  

```
pip install -r requirements.txt
```

# 3.Model
**3.1 flux dev repo （preference/推荐）**
* 3.1 use flux dev [diffusser repo](https://huggingface.co/black-forest-labs/FLUX.1-dev/tree/main)
if OOM chocie nf4 or fp8 type 如果OOM，开启nf4或者fp8量化

**3.2 single model**
* 3.1.1 download  kijai 'flux1-dev.safetensors' fp8 from [here](https://huggingface.co/Kijai/flux-fp8/tree/main)  下载kijai的flux单体模型模型,文件结构如下图
* 3.1.2 download 'clip_l.safetensors' and 't5xxl_fp8_e4m3fn.safetensors' from [here](https://huggingface.co/comfyanonymous/flux_text_encoders/tree/main)下载T5和clip l单体模型模型,文件结构如下图

```
--   ComfyUI/models/diffusion_models  # or unet
    ├── flux1-dev-fp8-e4m3fn.safetensors  #11G
--   ComfyUI/models/clip
    ├── clip_l.safetensors
    ├── t5xxl_fp8_e4m3fn.safetensorsvae
--   ComfyUI/models/vae
    ├── ae.safetensor
```


# 4.Example
* Inpainting , use up prompt only
![](https://github.com/smthemex/ComfyUI_Personalize_Anything/blob/main/assets/in.png)
* Outpainting
![](https://github.com/smthemex/ComfyUI_Personalize_Anything/blob/main/assets/outpainting.png)
* Reconstruction
![](https://github.com/smthemex/ComfyUI_Personalize_Anything/blob/main/assets/reconstruction.png)
* Single_personalize
![](https://github.com/smthemex/ComfyUI_Personalize_Anything/blob/main/assets/single_personalize.png)
* comfy single model
![](https://github.com/smthemex/ComfyUI_Personalize_Anything/blob/main/assets/example_cf.png)

# 5.Tips
* 使用方法请参考示例图片/Please refer to the example image for usage instructions

# 6. Citation
```
@article{feng2025personalize,
  title={Personalize Anything for Free with Diffusion Transformer},
  author={Feng, Haoran and Huang, Zehuan and Li, Lin and Lv, Hairong and Sheng, Lu},
  journal={arXiv preprint arXiv:2503.12590},
  year={2025}
}
```
