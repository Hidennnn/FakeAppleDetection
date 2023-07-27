import torch
from diffusers import StableDiffusionPipeline

# "runwayml/stable-diffusion-v1-5"
# "CompVis/stable-diffusion-v1-4"
# "stabilityai/stable-diffusion-2-1"

model_id = "SG161222/Realistic_Vision_V1.4"
device = "cuda"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)
# used only for stable diffusion 2.1 due to low VRAM.
# pipe.enable_attention_slicing()

fruits = ["apple"]
positions = [" on floor", " on grass", " on table", " on tree", " on hand", " plain background", " many"]
for fruit in fruits:
    for position in positions:
        for number in range(50):
            prompt = f"{fruit} fruit{position}, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3"
            image = pipe(prompt).images[0]
            image.save(f"C:\\Users\\PC\\OneDrive\\Pulpit\\baza danych\\owoce\\apple_realistic_1_4\\{fruit}_{position}_{number}.png")

#%%
