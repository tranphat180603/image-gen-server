from flask import Flask, request, jsonify
import requests
import os
import replicate
import threading
import json
import time

app = Flask(__name__)

# Replicate settings
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")
REPLICATE_VERSION = "a3409648730239101538d4cf79f2fdb0e068a5c7e6509ad86ab3fae09c4d6ef8"

# Slack settings
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")

def upload_images_to_slack(image_data_list, channel_id, user_prompt):
    """
    Upload each image (file-like object) directly to Slack using files.upload.
    Returns a list of Slack file permalinks if successful, or a dict with "error".
    """
    if not SLACK_BOT_TOKEN:
        return {"error": "SLACK_BOT_TOKEN not set or invalid."}

    slack_upload_url = "https://slack.com/api/files.upload"
    headers = {
        "Authorization": f"Bearer {SLACK_BOT_TOKEN}"
    }

    file_links = []
    for idx, image_data in enumerate(image_data_list):
        # Read bytes from the Replicate image-like object
        image_bytes = image_data.read()

        # We’ll name it something like TMAI_1675459200_1.png
        file_name = f"TMAI_{int(time.time())}_{idx+1}.png"

        # Slack requires a multipart/form-data upload with 'file' param:
        files = {
            "file": (file_name, image_bytes, "image/png")
        }
        # Send to the same channel that invoked the slash command,
        # or you can set a different channel if you prefer.
        data = {
            "channels": channel_id,
            # This text appears as the file's initial comment
            "initial_comment": f"Generated image {idx+1} for prompt: {user_prompt[:50]}..."
        }

        resp = requests.post(slack_upload_url, headers=headers, files=files, data=data)
        resp_json = resp.json()
        if not resp_json.get("ok"):
            return {"error": f"Slack file upload failed: {resp_json}"}
        
        # Slack returns file metadata. We'll store the permalink to display in final message.
        file_permalink = resp_json["file"].get("permalink")
        file_links.append(file_permalink or "No permalink found")

    return file_links


def process_image_generation(user_prompt, aspect_ratio, num_outputs, response_url, channel_id):
    print("Starting image generation with Replicate...")
    TMAI_prefix = """
TMAI, a yellow robot which has a rounded rectangular head with glossy black eyes.
TMAI’s proportions are balanced, avoiding an overly exaggerated head-to-body ratio. TMAI’s size is equal to a 7-year-old kid.
"""
    full_prompt = TMAI_prefix + "\n" + user_prompt
    print("Full prompt sent to Replicate:")
    print(full_prompt)

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
        print("Replicate returned output.")

        if num_outputs == 1:
            image_data = [output[0]] if output else None
        elif num_outputs == 4:
            image_data = output
        else:
            image_data = [output[0]] if output else None

        if image_data:
            print("Uploading image(s) to Slack...")
            upload_result = upload_images_to_slack(image_data, channel_id, user_prompt)
            print("Slack upload returned:", upload_result)

            if isinstance(upload_result, dict) and upload_result.get("error"):
                # We had an error from the Slack upload
                slack_message = {
                    "response_type": "ephemeral",
                    "text": f"Error uploading image(s) to Slack: {upload_result.get('error')}"
                }
            else:
                # Build final text with the Slack file permalinks
                link_text = "\n".join(upload_result)
                slack_message = {
                    "response_type": "in_channel",
                    "text": "Here are your generated TMAI images:\n" + link_text
                }
        else:
            print("No image data received from Replicate.")
            slack_message = {
                "response_type": "ephemeral",
                "text": "Failed to generate an image."
            }

    except Exception as e:
        print("Exception during image generation:", str(e))
        slack_message = {
            "response_type": "ephemeral",
            "text": f"An error occurred: {str(e)}"
        }

    print("Posting final message to Slack via response_url:", response_url)
    resp = requests.post(response_url, json=slack_message)
    print(f"Slack response update status: {resp.status_code}, body: {resp.text}")


@app.route('/slack/command', methods=['POST'])
def slack_command_endpoint():
    # Slack includes a channel_id in slash commands so you can post back to the same channel
    channel_id = request.form.get("channel_id")
    response_url = request.form.get("response_url")
    text = request.form.get("text", "")

    print(f"Received Slack channel_id: {channel_id}")
    print(f"Received Slack response_url: {response_url}")
    print("Received Slack text:", text)

    # Default parameter values
    aspect_ratio = "1:1"
    num_outputs = 1

    # Split input into prompt and parameter parts
    if "--" in text:
        parts = text.split("--", 1)
        user_prompt = parts[0].strip()
        params_text = "--" + parts[1]
    else:
        user_prompt = text.strip()
        params_text = ""

    print("User prompt:", user_prompt)
    print("Params text:", params_text)

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

    print(f"Parsed parameters: aspect_ratio={aspect_ratio}, num_outputs={num_outputs}")

    # Immediately respond to Slack to acknowledge receipt
    ack_response = {
        "response_type": "ephemeral",
        "text": "Processing your image... This might take a moment."
    }
    print("Sending immediate acknowledgement to Slack.")

    # Kick off background thread to generate and upload images
    thread = threading.Thread(
        target=process_image_generation,
        args=(user_prompt, aspect_ratio, num_outputs, response_url, channel_id)
    )
    thread.start()

    return jsonify(ack_response)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting app on port {port}")
    app.run(debug=True, host='0.0.0.0', port=port)
