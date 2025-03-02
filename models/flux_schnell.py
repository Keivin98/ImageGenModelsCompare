import torch
from diffusers import FluxPipeline
import streamlit as st

CACHE_DIR = "/home/local/QCRI/kisufaj/.cache/"

@st.cache_resource
def load_model(device=None):
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.bfloat16,
        cache_dir=CACHE_DIR,
    )
    if device: 
        pipe.to(device)
    # pipe.enable_sequential_cpu_offload()
    return pipe

def generate_images(pipe, prompt, num_inference_steps, guidance_scale, max_sequence_length, number_of_images):
    generator = torch.manual_seed(0)
    return pipe(
        [prompt] * number_of_images,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        max_sequence_length=max_sequence_length,
        generator=generator
    ).images
