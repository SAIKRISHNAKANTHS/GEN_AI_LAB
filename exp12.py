import os
import re
import time
import getpass
import gradio as gr
from groq import Groq

# ---------------- API KEY ----------------

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    api_key = getpass.getpass("Enter your Groq API key: ")

if not api_key:
    raise ValueError("Groq API key was not provided.")

client = Groq(api_key=api_key)


# ---------------- RELEVANCE ----------------

def calculate_relevance(prompt, response):

    stop_words = {
        "the", "a", "an", "is", "are", "was", "were",
        "to", "of", "in", "on", "for", "and", "or",
        "with", "what", "how", "why", "write", "explain",
        "describe", "give", "about"
    }

    prompt_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", prompt.lower()))
    response_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", response.lower()))

    important_words = prompt_words - stop_words

    if not important_words:
        return 100.0

    matched = important_words.intersection(response_words)

    return round(len(matched) / len(important_words) * 100, 2)


# ---------------- GENERATE ----------------

def generate_and_evaluate(prompt, temperature, max_tokens):

    if not prompt.strip():
        return "Please enter a valid prompt.", {"Status": "No prompt provided"}

    try:

        start = time.perf_counter()

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful Generative AI assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=float(temperature),
            max_tokens=int(max_tokens)
        )

        end = time.perf_counter()

        response = completion.choices[0].message.content.strip()

        evaluation = {
            "Model": "llama-3.1-8b-instant",
            "Response Time (seconds)": round(end - start, 3),
            "Generated Word Count": len(response.split()),
            "Generated Character Count": len(response),
            "Keyword Relevance Score (%)": calculate_relevance(prompt, response),
            "Temperature": float(temperature),
            "Maximum Tokens": int(max_tokens),
            "Status": "Successfully generated"
        }

        return response, evaluation

    except Exception as error:

        return (
            "The application could not generate a response.",
            {
                "Status": "Error",
                "Error Type": type(error).__name__,
                "Error Message": str(error)
            }
        )


# ---------------- GRADIO UI ----------------

with gr.Blocks() as application:

    gr.Markdown("""
    # Cloud-Based Generative AI Application

    Enter a prompt to generate content and evaluate the response.
    """)

    with gr.Row():

        with gr.Column():

            prompt_input = gr.Textbox(
                label="Enter Prompt",
                lines=6
            )

            temperature_input = gr.Slider(
                0, 1, value=0.3, step=0.1,
                label="Temperature"
            )

            max_tokens_input = gr.Slider(
                50, 500, value=250, step=50,
                label="Maximum Tokens"
            )

            generate_button = gr.Button("Generate and Evaluate")

        with gr.Column():

            response_output = gr.Textbox(
                label="Generated Response",
                lines=14
            )

            evaluation_output = gr.JSON(
                label="Evaluation Metrics"
            )

    # Clear button AFTER outputs are created
    clear_button = gr.ClearButton(
        [prompt_input, response_output, evaluation_output]
    )

    generate_button.click(
        fn=generate_and_evaluate,
        inputs=[
            prompt_input,
            temperature_input,
            max_tokens_input
        ],
        outputs=[
            response_output,
            evaluation_output
        ]
    )


application.launch(
    share=True,
    debug=True
)
