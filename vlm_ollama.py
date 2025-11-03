import ollama

# --- Configuration ---
PROMPT = "What is in this image? Describe it in detail."
TEMPERATURE = 0.4

def interpret_page(local_model, screenshot, prompt=PROMPT, temperature=TEMPERATURE) -> str:
    """
    Interprets a webpage screenshot using a local vision language model via Ollama.
    """

    # Inference time parameters
    options = {
        'temperature': temperature,
    }

    # Get Ollama response
    response = ollama.chat(
        model=local_model,
        messages=[
            {
                'role': 'user',
                'content': prompt,
                'images': [screenshot],
            }
        ],
        options=options,
    )

    # Extract and return the relevant information from the response
    return response.message.content