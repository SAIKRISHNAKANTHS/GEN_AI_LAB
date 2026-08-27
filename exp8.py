from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("Loading AI model... Please wait.")

# Load instruction-following code model
model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32
)

print("Model loaded successfully!")


def generate_response(prompt, max_tokens=250):

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return response.strip()


# -----------------------------------------
# 1. CODE GENERATION
# -----------------------------------------

code_task = """
Write a Python program to calculate the factorial of a number
using a recursive function.

Requirements:
1. Accept a number from the user.
2. Use recursion.
3. Display the factorial.
4. Handle negative input.
"""

generation_prompt = f"""
Generate Python code for the following task.

{code_task}

Return only the complete Python program.
Do not provide explanations.
"""

generated_code = generate_response(
    generation_prompt,
    max_tokens=250
)


# -----------------------------------------
# 2. DEBUGGING ASSISTANT
# -----------------------------------------

f_code = """
def calculate_average(numbers):
    total = sum(numbers)
    average = total / len(number)
    return average

values = [10, 20, 30, 40]
print("Average:", calculate_average(values))
"""

debugging_prompt = f"""
Analyze the following Python program.

Code:
{f_code}

Identify the error and explain why it occurs.

Then provide the corrected program.

Use this format:

Error:
Explanation:
Corrected Code:
"""

debugging_result = generate_response(
    debugging_prompt,
    max_tokens=300
)


# -----------------------------------------
# 3. DISPLAY RESULTS
# -----------------------------------------

print("\nAI-POWERED CODE GENERATION AND DEBUGGING ASSISTANT")
print("=" * 60)

print("\n1. GENERATED CODE")
print("-" * 60)
print(generated_code)

print("\n2. DEBUGGING RESULT")
print("-" * 60)
print(debugging_result)
