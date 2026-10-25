# ComfyUI_Personalize_Anything
[Personalize Anything](https://github.com/fenghora/personalize-anything) for Free with Diffusion Transformer,use it in comfyUI with wrapper mode

# Update
* 改用comfyUI内核复现方法，去掉diffuser和transformer的版本限制/ Use comfyUI core replay method, remove the version limit of diffuser and transformer

# 1. Installation

In the ./ComfyUI/custom_nodes directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_Personalize_Anything
```
---

# 3.Model

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
* Inpainting 
![](https://github.com/smthemex/ComfyUI_Personalize_Anything/blob/main/example_workflows/inpaint.png)
* subject
![](https://github.com/smthemex/ComfyUI_Personalize_Anything/blob/main/example_workflows/subject.png)
* Reconstruction
![](https://github.com/smthemex/ComfyUI_Personalize_Anything/blob/main/example_workflows/subject_reconstruction.png)


# 5. Citation
```
@article{feng2025personalize,
  title={Personalize Anything for Free with Diffusion Transformer},
  author={Feng, Haoran and Huang, Zehuan and Li, Lin and Lv, Hairong and Sheng, Lu},
  journal={arXiv preprint arXiv:2503.12590},
  year={2025}
}
```
