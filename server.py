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

# Slack settings (make sure this is a modern bot token with proper scopes)
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")

def upload_file_to_slack_external(file_name, image_bytes, channel_id, title=None, filetype="png"):
    """
    Upload a file to Slack using the new external upload methods.
    Returns a dict with file_id and permalink on success, or an "error" key.
    """
    get_url = "https://slack.com/api/files.getUploadURLExternal"
    headers = {
        "Authorization": f"Bearer {SLACK_BOT_TOKEN}",
        "Content-Type": "application/json; charset=utf-8"
    }
    
    # Ensure the payload has the required fields.
    payload = {
        "filename": file_name,
        "title": title or file_name,
        "filetype": filetype,
        "length": len(image_bytes)
    }
    
    # Debug: print payload to verify values are set correctly.
    print("Payload for files.getUploadURLExternal:", payload)
    
    # Send as raw JSON string in the request body.
    resp = requests.post(get_url, headers=headers, data=json.dumps(payload))
    data = resp.json()
    if not data.get("ok"):
        return {"error": f"Error getting upload URL: {data}"}
    
    upload_url = data.get("upload_url")
    file_id = data.get("file_id")
    if not upload_url or not file_id:
        return {"error": "Missing upload_url or file_id in response from files.getUploadURLExternal"}
    
    # Step 2: Upload the file bytes to the pre-signed URL.
    put_resp = requests.put(upload_url, data=image_bytes, headers={"Content-Type": "application/octet-stream"})
    if put_resp.status_code != 200:
        return {"error": f"Error uploading file bytes: status {put_resp.status_code}, {put_resp.text}"}
    
    # Step 3: Finalize the upload.
    complete_url = "https://slack.com/api/files.completeUploadExternal"
    complete_payload = {
        "file_id": file_id,
        "channels": channel_id
    }
    complete_resp = requests.post(complete_url, headers=headers, data=json.dumps(complete_payload))
    complete_data = complete_resp.json()
    if not complete_data.get("ok"):
        return {"error": f"Error completing file upload: {complete_data}"}
    
    # Slack returns the file object; get the permalink.
    file_info = complete_data.get("file", {})
    permalink = file_info.get("permalink_public") or file_info.get("permalink")
    return {"file_id": file_id, "permalink": permalink}



def upload_images_to_slack(image_data_list, channel_id, user_prompt):
    """
    For each image (a file-like object), upload it using the new external upload flow.
    Returns a list of permalinks or a dict with an "error".
    """
    if not SLACK_BOT_TOKEN:
        return {"error": "SLACK_BOT_TOKEN not set or invalid."}
    
    file_links = []
    for idx, image_data in enumerate(image_data_list):
        # Read bytes from the image file-like object.
        image_bytes = image_data.read()
        file_name = f"TMAI_{int(time.time())}_{idx+1}.png"
        result = upload_file_to_slack_external(file_name, image_bytes, channel_id, title=f"Generated image {idx+1}")
        if result.get("error"):
            return {"error": result.get("error")}
        file_links.append(result.get("permalink", "No permalink found"))
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
        # Decide on the image data list based on the number of outputs
        if num_outputs == 1:
            image_data = [output[0]] if output else None
        elif num_outputs == 4:
            image_data = output
        else:
            image_data = [output[0]] if output else None
        
        if image_data:
            print("Uploading image(s) to Slack using the new external upload methods...")
            image_public_urls = upload_images_to_slack(image_data, channel_id, user_prompt)
            print("Slack upload returned:", image_public_urls)
            if isinstance(image_public_urls, dict) and image_public_urls.get("error"):
                slack_message = {
                    "response_type": "ephemeral",
                    "text": f"Error uploading image(s) to Slack: {image_public_urls.get('error')}"
                }
            else:
                # Build final text with the Slack file permalinks.
                link_text = "\n".join(image_public_urls)
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
    print("Final Slack payload:", json.dumps(slack_message, indent=2))
    resp = requests.post(response_url, json=slack_message)
    print(f"Slack response update status: {resp.status_code}, body: {resp.text}")

@app.route('/slack/command', methods=['POST'])
def slack_command_endpoint():
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
    
    # Immediately respond to Slack to acknowledge receipt.
    ack_response = {
        "response_type": "ephemeral",
        "text": "Processing your image... This might take a moment."
    }
    print("Sending immediate acknowledgement to Slack.")
    
    # Start a background thread to process image generation.
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
