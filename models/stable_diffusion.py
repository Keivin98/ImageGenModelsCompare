import torch
from diffusers import StableDiffusion3Pipeline
import streamlit as st

CACHE_DIR = "/home/local/QCRI/kisufaj/.cache/"

@st.cache_resource
def load_model(device=None):
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium",
        cache_dir=CACHE_DIR,
    )
    if device: 
        pipe.to(device)
    # pipe.enable_model_cpu_offload()
    return pipe

def generate_images(pipe, prompt, num_inference_steps, guidance_scale, max_sequence_length, number_of_images):
    return pipe(
        prompt=[prompt] * number_of_images,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        max_sequence_length=max_sequence_length
    ).images
