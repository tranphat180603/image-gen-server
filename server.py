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

# Google Drive settings (example values)
CLIENT_ID = "87176438828-ppjnlepvhhvt7n3ctej6napr299r2e7n.apps.googleusercontent.com"
CLIENT_SECRET = "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"
REFRESH_TOKEN = "1//044dJM9CUUhLhCgYIARAAGAQSNwF-L9IrlW8RvPw9jKqqt9UI-zRi2hLvAn7QvjDtgzbi7Ohp6JA3xGysPrKVpuFzSH5p5vuBm_M"
folder_id = "1_B31euFkFoYAbzKGNyNu2bgc2uKb-qSE"  # Google Drive folder id

def save_and_get_public_url_from_image(image_data, user_prompt):
    # First, get a new access token from Google
    token_url = "https://oauth2.googleapis.com/token"
    payload = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "refresh_token": REFRESH_TOKEN,
        "grant_type": "refresh_token",
    }
    token_resp = requests.post(token_url, data=payload)
    if token_resp.status_code != 200:
        return {"error": f"Failed to get new GDrive access token. Status={token_resp.status_code}, Body={token_resp.text}"}
    token_json = token_resp.json()
    GDRIVE_ACCESS_TOKEN = token_json.get("access_token")
    if not GDRIVE_ACCESS_TOKEN:
        return {"error": f"Could not parse 'access_token' from Google response: {token_json}"}
    
    # Prepare the authorization header using the new token
    gdrive_headers = {"Authorization": f"Bearer {GDRIVE_ACCESS_TOKEN}"}
    
    # Read the content (bytes) of the image
    image_bytes = image_data.read()
    
    # Define the file name (make sure it’s a safe file name; here we simply use the user prompt)
    file_name = f"TMAI_{user_prompt[:20]}.png"  # limiting length for safety
    
    # Save image to Google Drive
    upload_url = "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart"
    metadata = {
        "name": file_name,
        "parents": [folder_id]
    }
    files_data = {
        "metadata": ("metadata.json", json.dumps(metadata), "application/json"),
        "file": (file_name, image_bytes, "image/png"),
    }
    create_resp = requests.post(upload_url, headers=gdrive_headers, files=files_data)
    if create_resp.status_code != 200:
        # Return an error message
        return {"error": f"Failed to upload file. Status={create_resp.status_code}, Body={create_resp.text}"}
    
    upload_json = create_resp.json()
    drive_file_id = upload_json.get("id")
    if not drive_file_id:
        return {"error": "Drive file id not found in upload response."}
    
    # Generate public URL by setting the file's permission to public
    permission_url = f"https://www.googleapis.com/drive/v3/files/{drive_file_id}/permissions"
    permission_body = {"role": "reader", "type": "anyone"}
    perm_resp = requests.post(permission_url, headers=gdrive_headers, json=permission_body)
    if perm_resp.status_code not in [200, 204]:
        return {"error": f"Failed to set file permissions. Status={perm_resp.status_code}, Body={perm_resp.text}"}
    
    # Retrieve detailed metadata, including public links
    file_metadata_url = f"https://www.googleapis.com/drive/v3/files/{drive_file_id}?fields=id,name,mimeType,size,webViewLink,webContentLink"
    meta_resp = requests.get(file_metadata_url, headers=gdrive_headers)
    if meta_resp.status_code != 200:
        return {"error": f"Failed to retrieve file metadata. Status={meta_resp.status_code}, Body={meta_resp.text}"}
    
    meta_data = meta_resp.json()
    public_link = meta_data.get("webViewLink") or meta_data.get("webContentLink")
    if not public_link:
        public_link = f"https://drive.google.com/file/d/{drive_file_id}/view"
    
    return public_link

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
        image_data = output[0] if output else None
        if image_data:
            image_public_url = save_and_get_public_url_from_image(image_data, user_prompt)
            if isinstance(image_public_url, dict) and image_public_url.get("error"):
                # Error occurred during Drive upload
                slack_message = {
                    "response_type": "ephemeral",
                    "text": f"Error uploading image: {image_public_url.get('error')}"
                }
            else:
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
                            "image_url": image_public_url,
                            "alt_text": "Generated Image"
                        }
                    ]
                }
        else:
            slack_message = {
                "response_type": "ephemeral",
                "text": "Failed to generate an image."
            }
    except Exception as e:
        slack_message = {
            "response_type": "ephemeral",
            "text": f"An error occurred: {str(e)}"
        }
    
    # Update Slack using the response_url
    requests.post(response_url, json=slack_message)

@app.route('/slack/command', methods=['POST'])
def slack_command_endpoint():
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
