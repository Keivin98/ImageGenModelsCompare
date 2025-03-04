import torch
from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline, StableCascadeUNet
import streamlit as st

CACHE_DIR = "/home/local/QCRI/kisufaj/.cache/"

@st.cache_resource
def load_model(device=None):
    prior_unet = StableCascadeUNet.from_single_file(
        "/image-generation/StableCascade-training/checkpoints/mixed6/generator.safetensors",
        torch_dtype=torch.bfloat16
    )
    prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", prior=prior_unet, variant="bf16", torch_dtype=torch.bfloat16, cache_dir = CACHE_DIR)
    decoder = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade", variant="bf16", torch_dtype=torch.bfloat16, cache_dir = CACHE_DIR)

    if device: 
        prior.to(device)
        decoder.to(device)

    return prior, decoder

def generate_images(models, prompt, num_inference_steps, negative_prompt=""):
    prior, decoder = models

    prior_output = prior(
        prompt=prompt,
        height=1024,
        width=1024,
        negative_prompt=negative_prompt,
        guidance_scale=4.0,
        num_images_per_prompt=1,
        num_inference_steps=num_inference_steps
    )

    decoder_output = decoder(
        image_embeddings=prior_output.image_embeddings,
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=0.0,
        output_type="pil",
        num_inference_steps=num_inference_steps//2
    ).images
    
    return decoder_output
