from flask import Flask, request, jsonify
import requests
import os
import replicate
import threading
import json
import time
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

app = Flask(__name__)

# Get API keys from environment variables
REPLICATE_API_TOKEN = os.environ.get('REPLICATE_API_TOKEN')
SLACK_BOT_TOKEN = os.environ.get('SLACK_BOT_TOKEN')
TMAI_VERSION = os.environ.get('TMAI_VERSION', "a3409648730239101538d4cf79f2fdb0e068a5c7e6509ad86ab3fae09c4d6ef8")
LUCKY_VERSION = os.environ.get('LUCKY_VERSION', "499a35887d318d3e889af1a5968850fb8f2b508095c73d04b8734c5b018ec43e")

# Validate required environment variables
if not REPLICATE_API_TOKEN or not SLACK_BOT_TOKEN:
    raise ValueError("Missing required environment variables. Please set REPLICATE_API_TOKEN and SLACK_BOT_TOKEN")

def upload_file_to_slack_external(file_name_arr, image_bytes_arr, channel_id, title=None, filetype="png"):
    file_ids = []
    for file_name, image_bytes in zip(file_name_arr, image_bytes_arr):
        get_url = "https://slack.com/api/files.getUploadURLExternal"
        headers = {
            "Authorization": f"Bearer {SLACK_BOT_TOKEN}"
        }
        params = {
            "filename": file_name,
            "length": len(image_bytes)
        }
        # 1) GET the presigned URL
        resp = requests.get(get_url, headers=headers, params=params)
        data = resp.json()
        if not data.get("ok"):
            return {"error": f"Error getting upload URL: {data}"}
        
        upload_url = data.get("upload_url")
        file_id = data.get("file_id")
        file_ids.append(file_id)
        if not upload_url or not file_id:
            return {"error": "Missing upload_url or file_id"}
        
        # 2) POST file bytes (multipart/form-data)
        files = {
            "filename": (file_name, image_bytes, "application/octet-stream")
        }
        post_resp = requests.post(upload_url, files=files)
        if post_resp.status_code != 200:
            return {"error": f"Error uploading file bytes: {post_resp.status_code}, {post_resp.text}"}
        elif post_resp.status_code == 200:
            print("Successfully uploaded the file")

    # 3) Complete the upload
    complete_url = "https://slack.com/api/files.completeUploadExternal"
    files_arr = []
    for file_id, file_name in zip(file_ids, file_name_arr):
        file_info = {
            "id": file_id,
            "title": file_name,
        }
        files_arr.append(file_info)
    print("Finished appending all images")
    print(f"Files array: {files_arr}")
    complete_payload = {
        "files": files_arr,
        "channel_id": channel_id
    }
    complete_headers = {
        "Authorization": f"Bearer {SLACK_BOT_TOKEN}",
        "Content-Type": "application/json"
    }
    complete_resp = requests.post(complete_url, headers=complete_headers, json=complete_payload)
    complete_data = complete_resp.json()
    if not complete_data.get("ok"):
        return {"error": f"Error completing file upload: {complete_data}"}
    
    # 4) Return the permalink
    file_info = complete_data.get("file", {})
    permalink = file_info.get("permalink_public") or file_info.get("permalink")
    print(f"Permalinks: {permalink}")
    return {"file_id": file_id, "permalink": permalink}


def upload_images_to_slack(image_data_list, channel_id, user_prompt):
    """
    For each image (a file-like object), upload it using the new external upload flow.
    Returns a list of permalinks or a dict with an "error".
    """
    if not SLACK_BOT_TOKEN:
        return {"error": "SLACK_BOT_TOKEN not set or invalid."}
    
    image_bytes_arr = []
    file_name_arr = []
    for idx, image_data in enumerate(image_data_list):
        # Read bytes from the image file-like object.
        image_bytes = image_data.read()
        file_name = f"TMAI_{int(time.time())}_{idx+1}.png"
        image_bytes_arr.append(image_bytes)
        file_name_arr.append(file_name)
    #because we are sending multiple files, we need to send a list of file names and a list of image bytes
    result = upload_file_to_slack_external(file_name_arr, image_bytes_arr, channel_id, title=f"Generated image {idx+1}")
    if result.get("error"):
        return {"error": result.get("error")}
    return 

def process_image_generation(user_prompt, aspect_ratio, num_outputs, num_infer_steps, lora_scale, response_url, channel_id, character):
    print("Starting image generation with Replicate...")
    if character == "TMAI":
        TMAI_prefix = """TMAI, a yellow robot which has a rounded rectangular head with black eyes. TMAI's proportions are balanced, avoiding an overly exaggerated head-to-body ratio. TMAI's size is equal to a 7-year-old kid."""
        full_prompt = TMAI_prefix + "\n" + "TMAI " + user_prompt
        inputs = {
                "prompt": full_prompt,
                "model": "dev",
                "go_fast": False,
                "lora_scale": lora_scale,
                "num_outputs": num_outputs,
                "aspect_ratio": aspect_ratio,
                "output_format": "png",
                "guidance_scale": 3,
                "extra_lora_scale": 1,
                "num_inference_steps": num_infer_steps,
            }
        output = replicate.run(
            "token-metrics/tmai-imagegen-iter3:" + TMAI_VERSION,
            input=inputs
        )
    elif character == "LUCKY":
        LUCKY_prefix = """LUCKY, an orange French bulldog with upright ears, always wearing a collar with the word 'LUCKY' boldly written on it."""
        full_prompt = LUCKY_prefix + "\n" + "LUCKY " + user_prompt
        inputs = {
            "prompt": full_prompt,
            "model": "dev",
            "go_fast": False,
            "lora_scale": lora_scale,
            "num_outputs": num_outputs,
            "aspect_ratio": aspect_ratio,
            "output_format": "png",
            "guidance_scale": 3,
            "output_quality": 80,
            "prompt_strength": 0.8,
            "extra_lora_scale": 1,
            "num_inference_steps": num_infer_steps,
            "disable_safety_checker": True
        }
        output = replicate.run(
            "token-metrics/lucky-imagegen-iter1:" + LUCKY_VERSION,
            input= inputs
        )
    print("Full prompt sent to Replicate:")
    print(full_prompt)
    print("Params:")
    print(inputs)

    # Decide on the image data list based on the number of outputs
    if num_outputs == 1:
        image_data = [output[0]] if output else None
    elif num_outputs == 4:
        image_data = output
    else:
        image_data = [output[0]] if output else None
    try:
        if image_data:
            print("Uploading image(s) to Slack using the new external upload methods...")
            slack_message = {
                "response_type": "in_channel",
                "text": "Generated images:\n"
            }
            image_public_urls = upload_images_to_slack(image_data, channel_id, user_prompt)
            if isinstance(image_public_urls, dict) and image_public_urls.get("error"):
                slack_message = {
                    "response_type": "in_channel",
                    "text": f"Error uploading image(s) to Slack: {image_public_urls.get('error')}"
                }
        else:
            print("No image data received from Replicate.")
            slack_message = {
                "response_type": "in_channel",
                "text": "Failed to generate an image."
            }
    except Exception as e:
        print("Exception during image generation:", str(e))
        slack_message = {
            "response_type": "in_channel",
            "text": f"An error occurred: {str(e)}"
        }
    
    print("Posting final message to Slack via response_url:", response_url)
    print("Final Slack payload:", json.dumps(slack_message, indent=2))
    resp = requests.post(response_url, json=slack_message)
    print(f"Slack response update status: {resp.status_code}, body: {resp.text}")

def slack_command_endpoint(character):
    channel_id = request.form.get("channel_id")
    response_url = request.form.get("response_url")
    text = request.form.get("text", "")

    print(f"Received Slack channel_id: {channel_id}")
    print(f"Received Slack response_url: {response_url}")
    print("Received Slack text:", text)
    
    # Default parameter values
    aspect_ratio = "1:1"
    num_outputs = 4
    num_infer_steps = 50
    lora_scale = 0.96

    # Split input into prompt and parameter parts
    if "--" in text:
        parts = text.split("--", 1)
        user_prompt = parts[0].strip()
        params_text = "--" + parts[1]
    else:
        user_prompt = text.strip()
        params_text = ""

    tokens = params_text.split()
    idx = 0
    text_to_render = None
    
    while idx < len(tokens):
        token = tokens[idx]
        if token == "--words":
            if idx + 1 < len(tokens):
                # Collect all text until the next parameter or end
                text_parts = []
                idx += 1
                while idx < len(tokens) and not tokens[idx].startswith("--"):
                    text_parts.append(tokens[idx])
                    idx += 1
                text_to_render = " ".join(text_parts)
                # Add spacing to TMAI if it exists
                if "TMAI" in text_to_render:
                    text_to_render = text_to_render.replace("TMAI", "T M A I")
                continue
        elif token in ["--aspect_ratio", "--ar"]:
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
                        num_outputs = 4
                except ValueError:
                    num_outputs = 4
            idx += 2
        elif token == "--detailed_level":
            if idx + 1 < len(tokens):
                try:
                    lev = tokens[idx + 1]
                    if lev == "high":
                        num_infer_steps = 50
                    elif lev == "medium":
                        num_infer_steps = 40
                    elif lev == "low":
                        num_infer_steps = 28
                except ValueError:
                    num_infer_steps = 50
            idx += 2
        elif token == "--mascot_style":
            if idx + 1 < len(tokens):
                try:
                    sty = float(tokens[idx + 1])
                    if 0.8 <= sty <= 1:
                        lora_scale = sty
                    else:
                        lora_scale = 0.96
                except ValueError:
                    lora_scale = 0.96
            idx += 2
        else:
            idx += 1

    # Initialize enhanced_prompt with the user_prompt first
    enhanced_prompt = user_prompt

    # If text_to_render exists, enhance the prompt
    if text_to_render:
        text_enhancement = f'''Clear, legible text that reads exactly "{text_to_render}", 
        rendered in high contrast, sharp focus, centered composition,
        professional typography, crisp edges, no distortion,
        perfectly readable text, front-facing text, text stands out against the background,
        each letter clearly defined and spaced'''
        # Replace the entire "--words" parameter portion in the original prompt
        words_part = f"--words {text_to_render}"
        for token in tokens:
            if token == "--words":
                start_idx = user_prompt.find(words_part)
                if start_idx != -1:
                    enhanced_prompt = user_prompt[:start_idx] + text_enhancement + user_prompt[start_idx + len(words_part):]
                break

    print(f"Enhanced prompt: {enhanced_prompt}")

    # Create the character prefix based on the character
    if character == "TMAI":
        char_prefix = """TMAI, a yellow robot which has a rounded rectangular head with black eyes. TMAI's proportions are balanced, avoiding an overly exaggerated head-to-body ratio. TMAI's size is equal to a 7-year-old kid."""
    else:  # LUCKY
        char_prefix = """LUCKY, an orange French bulldog with upright ears, always wearing a collar with the word 'LUCKY' boldly written on it."""
    
    full_prompt = char_prefix + "\n" + character + " " + enhanced_prompt

    print(f"Full prompt with enhanced prompt: {full_prompt}")

    # Format the parameters
    params = []
    for idx, token in enumerate(tokens):
        if token in ["--aspect_ratio", "--ar"] and idx + 1 < len(tokens):
            params.append(f"`--ar {tokens[idx + 1]}`")
        elif token == "--num_outputs" and idx + 1 < len(tokens):
            params.append(f"`--num_outputs {tokens[idx + 1]}`")
        elif token == "--detailed_level" and idx + 1 < len(tokens):
            params.append(f"`--detailed_level {tokens[idx + 1]}`")
        elif token == "--mascot_style" and idx + 1 < len(tokens):
            params.append(f"`--mascot_style {tokens[idx + 1]}`")
        elif token == "--words" and idx + 1 < len(tokens):
            # Collect all text until the next parameter
            text_parts = []
            i = idx + 1
            while i < len(tokens) and not tokens[i].startswith("--"):
                text_parts.append(tokens[i])
                i += 1
            text = " ".join(text_parts)
            params.append(f"`--words \"{text}\"`")

    # Create formatted response with styled parameters
    formatted_response = f"Processing your image... This might take a moment.\n\n*Processed prompt:*\n{full_prompt}"
    if params:
        formatted_response += f"\n{' '.join(params)}"

    # Immediately respond to Slack to acknowledge receipt with the processed prompt
    ack_response = {
        "response_type": "in_channel",
        "text": formatted_response
    }    

    # Start a background thread to process image generation with enhanced prompt
    thread = threading.Thread(
        target=process_image_generation,
        args=(enhanced_prompt, aspect_ratio, num_outputs, num_infer_steps, lora_scale, response_url, channel_id, character)
    )
    thread.start()
    
    return jsonify(ack_response)

@app.route('/slack/TMAI', methods=['POST'])
def slack_TMAI_endpoint():
    return slack_command_endpoint(character = "TMAI")

@app.route('/slack/LUCKY', methods=['POST'])
def slack_LUCKY_endpoint():
    return slack_command_endpoint(character = "LUCKY")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting app on port {port}")
    app.run(debug=True, host='0.0.0.0', port=port)
