import torch
import matplotlib.pyplot as plt
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

# -----------------------------------------
# 1. Select device
# -----------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)


# -----------------------------------------
# 2. Load BLIP VQA model
# -----------------------------------------

model_name = "Salesforce/blip-vqa-base"

print("\nLoading BLIP VQA model...")
print("Please wait...")

processor = BlipProcessor.from_pretrained(model_name)

model = BlipForQuestionAnswering.from_pretrained(
    model_name
).to(device)

print("Model loaded successfully!")


# -----------------------------------------
# 3. Load car parking image
# -----------------------------------------

image_path = "carparking.jpg"

try:
    image = Image.open(image_path).convert("RGB")

except FileNotFoundError:
    print("\nERROR: carparking.jpeg not found!")
    print("Make sure the image is in the same folder as exp10.py.")
    exit()


print("\nImage loaded successfully!")


# -----------------------------------------
# 4. Ask a question
# -----------------------------------------

print("\nImage is ready for analysis.")

question = input(
    "\nEnter a question about the image: "
)


# -----------------------------------------
# 5. Process image and question
# -----------------------------------------

inputs = processor(
    images=image,
    text=question,
    return_tensors="pt"
)

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}


# -----------------------------------------
# 6. Generate answer
# -----------------------------------------

print("\nAnalyzing image...")

with torch.no_grad():

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=30
    )


# -----------------------------------------
# 7. Decode answer
# -----------------------------------------

answer = processor.decode(
    generated_ids[0],
    skip_special_tokens=True
)


# -----------------------------------------
# 8. Display image
# -----------------------------------------

plt.figure(figsize=(8, 6))

plt.imshow(image)

plt.axis("off")

plt.title("Car Parking Image")

plt.show()


# -----------------------------------------
# 9. Display result
# -----------------------------------------

print("\nMULTIMODAL AI RESULT")
print("-" * 50)

print("Question:", question)

print("Answer:", answer)
