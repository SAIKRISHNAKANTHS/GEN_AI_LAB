from transformers import pipeline

# ---------------------------------------------
# Load Pretrained Instruction-Following Model
# ---------------------------------------------
generator = pipeline(
    task="text-generation",
    model="google/flan-t5-base"
)

# ---------------------------------------------
# Function to Generate Response
# ---------------------------------------------
def generate_response(prompt, max_tokens=200):
    result = generator(
        prompt,
        max_new_tokens=max_tokens,
        do_sample=False,
        truncation=True
    )
    return result[0]["generated_text"]


print("=" * 60)
print("PROMPT ENGINEERING APPLICATION")
print("=" * 60)


# =====================================================
# 1. CONTENT GENERATION
# =====================================================

content_prompt = """
Role: You are an experienced Artificial Intelligence teacher.

Task:
Write a simple introduction to Generative Artificial Intelligence.

Target audience:
First-year engineering students.

Requirements:
1. Use simple language.
2. Limit the response to five sentences.
3. Include two real-world applications.
4. Avoid highly technical terms.
"""

content_output = generate_response(content_prompt)

print("\n1. CONTENT GENERATION")
print("-" * 60)
print(content_output)


# =====================================================
# 2. REASONING TASK
# =====================================================

reasoning_prompt = """
Solve the following problem.

A college conducted a Generative AI workshop for 120 students.
Eighty-five students completed the workshop successfully.

Instructions:
1. Identify the total number of students.
2. Identify the number of students who completed the workshop.
3. Calculate the number of students who did not complete it.
4. Provide a brief explanation and the final answer.
"""

reasoning_output = generate_response(reasoning_prompt)

print("\n2. REASONING TASK")
print("-" * 60)
print(reasoning_output)


# =====================================================
# 3. EMAIL AUTOMATION
# =====================================================

email_prompt = """
Role:
You are a professional academic coordinator.

Task:
Write a formal email to students.

Context:
A Generative AI laboratory session is scheduled for Friday at
10:00 AM in AI Laboratory 2.

Students must bring their laptops and complete their
Hugging Face account registration before attending.

Requirements:
1. Include a suitable subject.
2. Use a professional tone.
3. Mention the time and venue.
4. Clearly state the instructions.
5. Keep the email concise.
"""

email_output = generate_response(email_prompt)

print("\n3. TASK AUTOMATION - EMAIL GENERATION")
print("-" * 60)
print(email_output)


# =====================================================
# 4. ACTION ITEM EXTRACTION
# =====================================================

meeting_notes = """
The project team reviewed the development of the college chatbot.

Arun will prepare the training dataset by Monday.

Priya will test the chatbot responses by Wednesday.

Rahul will prepare the final demonstration and presentation.

The team will meet again on Friday to review the progress.
"""

action_prompt = f"""
Extract the action items from the following meeting notes.

Meeting Notes:
{meeting_notes}

For each action item provide:
1. Person Responsible
2. Assigned Task
3. Deadline

Present the output as a numbered list.
"""

action_output = generate_response(action_prompt)

print("\n4. TASK AUTOMATION - ACTION ITEM EXTRACTION")
print("-" * 60)
print(action_output)

print("\n" + "=" * 60)
print("PROGRAM EXECUTED SUCCESSFULLY")
print("=" * 60)