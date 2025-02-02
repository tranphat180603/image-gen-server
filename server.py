from flask import Flask, request, jsonify
import requests
import os
import replicate
import threading
import time

app = Flask(__name__)

REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")
REPLICATE_VERSION = "a3409648730239101538d4cf79f2fdb0e068a5c7e6509ad86ab3fae09c4d6ef8"

def process_image_generation(user_prompt, aspect_ratio, num_outputs, response_url):
    TMAI_prefix = """
    TMAI, a yellow robot which has a rounded rectangular head with glossy black eyes. 
    TMAI’s proportions are balanced, avoiding an overly exaggerated head-to-body ratio. TMAI’s size is equal to a 7-year-old kid.
    """
    full_prompt = TMAI_prefix + "\n" + user_prompt
    try:
        output = replicate.run(
            "token-metrics/tmai-imagegen-iter3:" + REPLICATE_VERSION,
            input={
                "prompt": full_prompt,
                "model": "dev",
                "go_fast": False,
                "lora_scale": 1,
                "megapixels": "1",
                "num_outputs": num_outputs,
                "aspect_ratio": aspect_ratio,
                "output_format": "png",
                "guidance_scale": 3,
                "output_quality": 80,
                "prompt_strength": 0.8,
                "extra_lora_scale": 1,
                "num_inference_steps": 28,
                "disable_safety_checker": True
            }
        )
        image_url = output[0] if output else None
        if image_url:
            slack_message = {
                "response_type": "in_channel",
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"Here is your generated image with aspect ratio *{aspect_ratio}*:"
                        }
                    },
                    {
                        "type": "image",
                        "image_url": image_url,
                        "alt_text": "Generated Image"
                    }
                ]
            }
        else:
            slack_message = {
                "response_type": "ephemeral",
                "text": "Failed to get the generated image."
            }
    except Exception as e:
        slack_message = {
            "response_type": "ephemeral",
            "text": f"An error occurred: {str(e)}"
        }
    
    # Update Slack using the response_url
    requests.post(response_url, json=slack_message)

@app.route('/slack/command', methods=['POST'])
def slack_command():
    # Immediately extract response_url and user input
    response_url = request.form.get("response_url")
    text = request.form.get('text', '')
    
    # Default parameter values
    aspect_ratio = "1:1"
    num_outputs = 1

    # Split input into prompt and parameter parts (assuming parameters are appended at the end)
    if "--" in text:
        parts = text.split("--", 1)
        user_prompt = parts[0].strip()
        params_text = "--" + parts[1]
    else:
        user_prompt = text.strip()
        params_text = ""

    tokens = params_text.split()
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token == "--aspect_ratio":
            if idx + 1 < len(tokens):
                ar = tokens[idx + 1]
                if ar in ["1:1", "16:9", "9:16", "21:9", "9:21"]:
                    aspect_ratio = ar
                else:
                    aspect_ratio = "1:1"
            idx += 2
        elif token == "--num_outputs":
            if idx + 1 < len(tokens):
                try:
                    no = int(tokens[idx + 1])
                    if no in [1, 4]:
                        num_outputs = no
                    else:
                        num_outputs = 1
                except ValueError:
                    num_outputs = 1
            idx += 2
        else:
            idx += 1

    # Immediately respond to Slack to acknowledge receipt
    ack_response = {
        "response_type": "ephemeral",
        "text": "Processing your image... This might take a moment."
    }
    
    # Start a background thread to process image generation
    thread = threading.Thread(target=process_image_generation, args=(user_prompt, aspect_ratio, num_outputs, response_url))
    thread.start()
    
    return jsonify(ack_response)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
