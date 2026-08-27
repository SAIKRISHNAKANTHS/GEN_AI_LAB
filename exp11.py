import torch
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from diffusers import StableDiffusionPipeline


# --------------------------------------------------
# 1. SELECT DEVICE
# --------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)


# --------------------------------------------------
# 2. LOAD TEXT GENERATION MODEL
# --------------------------------------------------

print("\nLoading FLAN-T5 text generation model...")
print("Please wait...")

text_model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(text_model_name)

text_model = AutoModelForSeq2SeqLM.from_pretrained(
    text_model_name
).to(device)

print("Text model loaded successfully!")


# --------------------------------------------------
# 3. LOAD IMAGE GENERATION MODEL
# --------------------------------------------------

print("\nLoading Stable Diffusion image generation model...")
print("This may take some time on the first run...")

image_generator = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if device == "cuda"
    else torch.float32,
    safety_checker=None
)

image_generator = image_generator.to(device)

print("Image generation model loaded successfully!")


# --------------------------------------------------
# 4. GET TOPIC FROM USER
# --------------------------------------------------

topic = input("\nEnter a content topic: ")


# --------------------------------------------------
# 5. GENERATE TEXT
# --------------------------------------------------

text_prompt = f"""
Write a short article of approximately 120 words
on the topic: {topic}

Include:
1. Introduction
2. Importance
3. Applications
"""

inputs = tokenizer(
    text_prompt,
    return_tensors="pt",
    truncation=True
)

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}

print("\nGenerating text...")

with torch.no_grad():

    output_ids = text_model.generate(
        **inputs,
        max_new_tokens=180,
        do_sample=False
    )

generated_text = tokenizer.decode(
    output_ids[0],
    skip_special_tokens=True
)


# --------------------------------------------------
# 6. GENERATE IMAGE
# --------------------------------------------------

image_prompt = f"""
A realistic high-quality illustration representing
{topic},
professional digital art,
highly detailed,
futuristic technology,
cinematic lighting,
4K quality.
"""

print("\nGenerating image...")
print("Please wait...")

generated_image = image_generator(
    prompt=image_prompt,
    num_inference_steps=20,
    guidance_scale=7.5
).images[0]


# --------------------------------------------------
# 7. SAVE IMAGE
# --------------------------------------------------

generated_image.save(
    "generated_content_image.png"
)


# --------------------------------------------------
# 8. DISPLAY GENERATED TEXT
# --------------------------------------------------

print("\n")
print("GENERATED TEXT")
print("-" * 60)

print(generated_text)


# --------------------------------------------------
# 9. DISPLAY GENERATED IMAGE
# --------------------------------------------------

plt.figure(figsize=(8, 8))

plt.imshow(generated_image)

plt.axis("off")

plt.title("AI Generated Image")

plt.show()


# --------------------------------------------------
# 10. FINAL MESSAGE
# --------------------------------------------------

print("\nImage saved as generated_content_image.png")
