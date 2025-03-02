import torch
from diffusers import DiffusionPipeline
from diffusers.utils import pt_to_pil
import streamlit as st

CACHE_DIR = "/home/local/QCRI/kisufaj/.cache/"

@st.cache_resource
def load_model(device=None):
    stage_1 = DiffusionPipeline.from_pretrained(
        "DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.bfloat16, cache_dir=CACHE_DIR
    )
    # stage_1.enable_model_cpu_offload()
    
    stage_2 = DiffusionPipeline.from_pretrained(
        "DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.bfloat16, cache_dir=CACHE_DIR
    )
    # stage_2.enable_model_cpu_offload()
    
    stage_3 = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-x4-upscaler", torch_dtype=torch.bfloat16, cache_dir=CACHE_DIR
    )
    # stage_3.enable_model_cpu_offload()

    if device: 
        stage_1.to(device)
        stage_2.to(device)
        stage_3.to(device)

    return stage_1, stage_2, stage_3

def generate_images(models, prompt):
    stage_1, stage_2, stage_3 = models
    generator = torch.manual_seed(0)

    # Stage 1
    prompt_embeds, negative_embeds = stage_1.encode_prompt(prompt)
    image = stage_1(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator, output_type="pt").images
    image = pt_to_pil(image)[0]

    # Stage 2
    image = stage_2(image=image, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator, output_type="pt").images
    image = pt_to_pil(image)[0]

    # Stage 3
    image = stage_3(prompt=prompt, image=image, generator=generator, noise_level=100).images
    return image
