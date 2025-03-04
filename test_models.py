import os
import json
import hashlib
import datetime
import torch
from PIL import Image
from models import stable_diffusion, deep_floyd, flux_schnell, stable_cascade
import sys

OUTPUT_FOLDER = "output"
METADATA_FILE = os.path.join(OUTPUT_FOLDER, "metadata.json")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

if os.path.exists(METADATA_FILE):
    with open(METADATA_FILE, "r") as f:
        metadata = json.load(f)
else:
    metadata = {}

def hash_prompt(prompt):
    return hashlib.md5(prompt.encode()).hexdigest()


device = torch.device(f"cuda:{sys.argv[2]}" if torch.cuda.is_available() else "cpu")

models = {
    "sd": stable_diffusion, 
    "df": deep_floyd, 
    "flux": flux_schnell,
    "cascade": stable_cascade
}

def save_images(images, prefix):
    filenames = []
    for i, img in enumerate(images):
        filename = os.path.join(OUTPUT_FOLDER, f"{prompt_hash}_{prefix}_{timestamp}_{i+1}.png")
        img.save(filename)
        filenames.append(filename)
    return filenames

prompts = []
with open("test_prompts.txt", "r") as f:
    for line in f: 
        prompts.append(line.rstrip("\n"))


num_inference_steps = 28
guidance_scale = 6.00
max_sequence_length = 512
number_of_images = 1

for i, (model_name, model) in enumerate(models.items()):
    if i == int(sys.argv[1]):
        model_loaded = model.load_model(device)

        for prompt in prompts:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            prompt_hash = hash_prompt(prompt)

            if model_name in ["df"]:
                images = model.generate_images(model_loaded, prompt)
            elif model_name in ['cascade']:
                images = model.generate_images(model_loaded, prompt, num_inference_steps)
            else:
                images = model.generate_images(model_loaded, prompt, num_inference_steps, guidance_scale, max_sequence_length, number_of_images)

            filenames = save_images(images, model_name)

            metadata[prompt_hash] = prompt
        torch.cuda.empty_cache()

# with open(METADATA_FILE, "a") as f:
#     json.dump(metadata, f, indent=4)
