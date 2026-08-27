import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import matplotlib.pyplot as plt

# -----------------------------------
# 1. Load Stable Diffusion model
# -----------------------------------

model_id = "runwayml/stable-diffusion-v1-5"

print("Loading Stable Diffusion model...")
print("Please wait...")

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    safety_checker=None
)

# -----------------------------------
# 2. Select device
# -----------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = pipe.to(device)

print("Using device:", device)


# -----------------------------------
# 3. Load existing WEBP image
# -----------------------------------

try:
    input_image = Image.open("input_image.webp").convert("RGB")
except FileNotFoundError:
    print("\nERROR: input_image.webp not found!")
    print("Make sure input_image.webp is in the same folder as exp9.py")
    exit()

# Resize image
input_image = input_image.resize((512, 512))

print("Input image loaded successfully.")


# -----------------------------------
# 4. Transformation prompt
# -----------------------------------

prompt = """
Transform the background of this photograph into a futuristic smart city.

Keep the original people, their faces, hairstyles, clothing,
body positions, poses, proportions and group composition
as close to the original image as possible.

Do not replace or redesign the people.

Add futuristic skyscrapers, flying cars, green buildings,
robots assisting people and advanced city technology
in the background.

Photorealistic, cinematic lighting, highly detailed,
realistic photography, natural human faces.
"""


# -----------------------------------
# 5. Negative prompt
# -----------------------------------

negative_prompt = """
distorted face, deformed face, alien face, different person,
extra people, missing people, duplicate person,
extra arms, extra hands, extra fingers, deformed hands,
bad anatomy, unrealistic body, cartoon, anime,
blurry face, distorted eyes, mutated face,
changed hairstyle, changed clothing
"""


# -----------------------------------
# 6. Generate image
# -----------------------------------

print("\nGenerating futuristic image...")
print("Please wait...")

result = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=input_image,
    strength=0.25,
    guidance_scale=7.0,
    num_inference_steps=30
)

generated_image = result.images[0]


# -----------------------------------
# 7. Save generated image
# -----------------------------------

generated_image.save("generated_image.png")

print("\nImage successfully generated!")
print("Saved as: generated_image.png")


# -----------------------------------
# 8. Display Original and Generated
# -----------------------------------

plt.figure(figsize=(12, 6))

# Original image
plt.subplot(1, 2, 1)
plt.imshow(input_image)
plt.axis("off")
plt.title("Original Image")

# Generated image
plt.subplot(1, 2, 2)
plt.imshow(generated_image)
plt.axis("off")
plt.title("Generated Image")

plt.tight_layout()
plt.show()
