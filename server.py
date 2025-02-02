from flask import Flask, request, jsonify
import requests
import os
import json

app = Flask(__name__)

# Replace with your actual token
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")

# The Replicate API version you are using
REPLICATE_VERSION = "a3409648730239101538d4cf79f2fdb0e068a5c7e6509ad86ab3fae09c4d6ef8"

@app.route('/slack/command', methods=['POST'])
def slack_command():
    # Parse text from Slack command payload
    text = request.form.get('text', '')
    # Default parameter values
    aspect_ratio = "1:1"
    num_outputs = 1
    
    # Very simple parsing: expect input like "aspect_ratio=16:9"
    for token in text.split():
        if token.startswith("--aspect_ratio="):
            aspect_ratio = token.split("=", 1)[1]
        elif token.startswith("--num_outputs="):
            num_outputs = int(token.split("=", 1)[1])

    TMAI_prefix = """
    TMAI, a yellow robot which has a rounded rectangular head with glossy black eyes. 
    TMAI’s proportions are balanced, avoiding an overly exaggerated head-to-body ratio. TMAI’s size is equal to a 7-year-old kid.
    """
    # Build the Replicate API payload
    payload = {
        "version": REPLICATE_VERSION,
        "input": {
            "prompt": TMAI_prefix + text,
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
    }
    headers = {
        "Authorization": f"Bearer {REPLICATE_API_TOKEN}",
        "Content-Type": "application/json",
        "Prefer": "wait"
    }
    
    # Call the Replicate API
    replicate_url = "https://api.replicate.com/v1/predictions"
    replicate_response = requests.post(replicate_url, headers=headers, json=payload)
    
    if replicate_response.status_code != 201:
        return jsonify({
            "response_type": "ephemeral",
            "text": "There was an error generating the image."
        })
    
    # Extract the image URL from the response (this depends on Replicate's API response structure)
    result = replicate_response.json()
    image_url = result.get("output", [None])[0]
    if not image_url:
        return jsonify({
            "response_type": "ephemeral",
            "text": "Failed to get the generated image."
        })
    
    # Build a Slack message with an image block
    slack_response = {
        "response_type": "in_channel",  # makes it visible to the channel
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
    
    return jsonify(slack_response)

if __name__ == '__main__':
    app.run(debug=True)
