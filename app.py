import os
import json
import hashlib
import datetime
import streamlit as st
import torch
from PIL import Image
from models import stable_diffusion, deep_floyd, flux_schnell, stable_cascade, fanar

OUTPUT_FOLDER = "output"
METADATA_FILE = os.path.join(OUTPUT_FOLDER, "metadata.json")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
st.set_page_config(layout="wide")

if os.path.exists(METADATA_FILE):
    with open(METADATA_FILE, "r") as f:
        metadata = json.load(f)
else:
    metadata = {}

def hash_prompt(prompt):
    return hashlib.md5(prompt.encode()).hexdigest()


device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
device2 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") 

sd_model = stable_diffusion.load_model(device1)
df_models = deep_floyd.load_model(device1)
flux_model = flux_schnell.load_model(device2)
cascade_model = stable_cascade.load_model(device2)
fanar_model = fanar.load_model(device2)

st.title("Stable Diffusion, Deep Floyd, and Flux Image Generator")

# User inputs
with st.form("generation_form"):
    prompt = st.text_area("Enter your prompt:", "Qatari family posing in front of a camera")
    guidance_scale = st.number_input("Guidance Scale:", min_value=1.0, max_value=20.0, value=8.0, step=0.1)
    max_sequence_length = st.number_input("Max Sequence Length:", min_value=64, max_value=512, value=512, step=64)
    num_inference_steps = st.slider("Num Inference Steps:", min_value=10, max_value=100, value=28, step=1)
    number_of_images = st.slider("Num of output images:", min_value=1, max_value=4, value=1, step=1)
    generate_button = st.form_submit_button("Generate Images")

def save_images(images, prefix):
    filenames = []
    for i, img in enumerate(images):
        filename = os.path.join(OUTPUT_FOLDER, f"{prompt_hash}_{prefix}_{timestamp}_{i+1}.png")
        img.save(filename)
        filenames.append(filename)
    return filenames

if generate_button:
    with st.status("Generating images... Please wait", expanded=True) as status:
        # try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            prompt_hash = hash_prompt(prompt)
            
            cols = st.columns(5)
            
            st.write("Generating with Stable Cascade...")
            cascade_images = stable_cascade.generate_images(cascade_model, prompt, num_inference_steps)
            cascade_filenames = save_images(cascade_images, "cascade")

            with cols[0]:
                st.write("### Stable Cascade")
                for filename in cascade_filenames:
                    st.image(Image.open(filename), caption="Stable Cascade", use_container_width=True)
            
            st.write("Generating with FLUX Schnell...")
            flux_images = flux_schnell.generate_images(flux_model, prompt, num_inference_steps, guidance_scale, max_sequence_length, number_of_images)
            flux_filenames = save_images(flux_images, "flux")

            with cols[1]:
                st.write("### FLUX Schnell")
                for filename in flux_filenames:
                    st.image(Image.open(filename), caption="FLUX Schnell", use_container_width=True)

            st.write("Generating with Stable Diffusion...")
            sd_images = stable_diffusion.generate_images(sd_model, prompt, num_inference_steps, guidance_scale, max_sequence_length, number_of_images)
            sd_filenames = save_images(sd_images, "sd")

            with cols[2]:
                st.write("### Stable Diffusion")
                for filename in sd_filenames:
                    st.image(Image.open(filename), caption="Stable Diffusion", use_container_width=True)

            st.write("Generating with Deep Floyd IF (3 Stages)...")
            df_images = deep_floyd.generate_images(df_models, prompt)
            df_filenames = save_images(df_images, "df")

            with cols[3]:
                st.write("### Deep Floyd")
                for filename in df_filenames:
                    st.image(Image.open(filename), caption="Deep Floyd", use_container_width=True)
            
            st.write("Generating with Fanar...")
            fanar_images = fanar.generate_images(fanar_model, prompt, num_inference_steps)
            fanar_filenames = save_images(fanar_images, "fanar")

            with cols[4]:
                st.write("### Fanar")
                for filename in fanar_filenames:
                    st.image(Image.open(filename), caption="Fanar", use_container_width=True)
            
            

            metadata[prompt_hash] = prompt
            with open(METADATA_FILE, "w") as f:
                json.dump(metadata, f, indent=4)

            status.update(label=" Image generation complete!", state="complete")

        # except Exception as e:
        #     status.update(label="Error occurred during generation!", state="error")
        #     st.error(f"An error occurred: {e}")


st.subheader("Previously Generated Images")

previous_images = [f for f in os.listdir(OUTPUT_FOLDER) if f.endswith(".png")]

sd_images = sorted([f for f in previous_images if "_sd_" in f])
df_images = sorted([f for f in previous_images if "_df_" in f])
flux_images = sorted([f for f in previous_images if "_flux_" in f])
cascade_images = sorted([f for f in previous_images if "_cascade_" in f])
fanar_images = sorted([f for f in previous_images if "_fanar_" in f])

unique_prompts = sorted(list(set(metadata.values())))

def find_images_for_prompt(prompt):
    prompt_hash = None
    for key, value in metadata.items():
        if value == prompt:
            prompt_hash = key
            break
    
    if not prompt_hash:
        return None, None, None, None, None

    sd_img = next((f for f in sd_images if f.startswith(prompt_hash)), None)
    df_img = next((f for f in df_images if f.startswith(prompt_hash)), None)
    flux_img = next((f for f in flux_images if f.startswith(prompt_hash)), None)
    cascade_img = next((f for f in cascade_images if f.startswith(prompt_hash)), None)
    fanar_img = next((f for f in fanar_images if f.startswith(prompt_hash)), None)
    
    return sd_img, df_img, flux_img, cascade_img, fanar_img

if unique_prompts:
    st.write("### Comparison Table")

    cols = st.columns([2, 3, 3, 3, 3, 3])
    model_names = ["Fanar", "Stable Cascade", "Flux Schnell", "Stable Diffusion", "Deep Floyd"]

    with cols[0]:
        st.write("**Prompt**")
    with cols[1]:
        st.write(f"**{model_names[0]}**")
    with cols[2]:
        st.write(f"**{model_names[1]}**")
    with cols[3]:
        st.write(f"**{model_names[2]}**")
    with cols[4]:
        st.write(f"**{model_names[3]}**")
    with cols[5]:
        st.write(f"**{model_names[4]}**")

    for prompt in unique_prompts:
        sd_img, df_img, flux_img, cascade_img, fanar_img = find_images_for_prompt(prompt)

        cols = st.columns([2, 3, 3, 3, 3, 3])
        with cols[0]:
            st.write(prompt)
        
        for i, (img_path, col) in enumerate(zip([fanar_img, cascade_img, flux_img, sd_img, df_img], cols[1:])):
            if img_path:
                img_full_path = os.path.join(OUTPUT_FOLDER, img_path)
                img = Image.open(img_full_path)
                with col:
                    st.image(img, caption=model_names[i], use_container_width=True)
            else:
                with col:
                    st.write("❌ No Image")

else:
    st.write("No previous images found.")