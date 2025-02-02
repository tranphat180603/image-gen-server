from flask import Flask, request, jsonify
import os
import replicate

app = Flask(__name__)

# Replace with your actual token (set as an environment variable)
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")

@app.route('/slack/command', methods=['POST'])
def slack_command():
    # Parse text from Slack command payload
    text = request.form.get('text', '')
    
    # Default parameter values
    aspect_ratio = "1:1"
    num_outputs = 1

    # Split the input into two parts:
    # 1. The main prompt (everything before the first parameter flag)
    # 2. The parameters (everything after)
    if "--" in text:
        parts = text.split("--", 1)
        user_prompt = parts[0].strip()
        params_text = "--" + parts[1]  # add back the --
    else:
        user_prompt = text.strip()
        params_text = ""

    # Parse parameters from the params_text.
    tokens = params_text.split()
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token == "--aspect_ratio":
            if idx + 1 < len(tokens):
                ar = tokens[idx + 1]
                # Only accept certain aspect ratios
                if ar in ["1:1", "16:9", "9:16", "21:9", "9:21"]:
                    aspect_ratio = ar
                    print(f"Returning TMAI image with aspect ratio {aspect_ratio}")
                else:
                    aspect_ratio = "1:1"
                    print(f"Invalid aspect_ratio provided; defaulting to {aspect_ratio}")
            idx += 2
        elif token == "--num_outputs":
            if idx + 1 < len(tokens):
                try:
                    no = int(tokens[idx + 1])
                    # Only allow 1 or 4 as valid options
                    if no == 1 or no == 4:
                        num_outputs = no
                        print(f"Returning {num_outputs} images of TMAI")
                    else:
                        num_outputs = 1
                        print(f"Invalid num_outputs provided; defaulting to {num_outputs}")
                except ValueError:
                    num_outputs = 1
                    print("num_outputs is not an integer; defaulting to 1")
            idx += 2
        else:
            idx += 1

    # Prefix for the prompt
    TMAI_prefix = """
TMAI, a yellow robot which has a rounded rectangular head with glossy black eyes. 
TMAI’s proportions are balanced, avoiding an overly exaggerated head-to-body ratio. TMAI’s size is equal to a 7-year-old kid.
"""

    # Build the full prompt by combining the prefix and the user's main prompt
    full_prompt = TMAI_prefix + "\n" + user_prompt

    # Build the Replicate API payload using the replicate Python package
    output = replicate.run(
        "token-metrics/tmai-imagegen-iter3:a3409648730239101538d4cf79f2fdb0e068a5c7e6509ad86ab3fae09c4d6ef8",
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
    
    # Check if output is valid
    if not output:
        return jsonify({
            "response_type": "ephemeral",
            "text": "Failed to get the generated image."
        })
    
    image_url = output[0]
    
    # Build a Slack message with an image block
    slack_response = {
        "response_type": "in_channel",  # visible to the channel
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
    # Bind to the port provided by the environment, defaulting to 5000 for local testing
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
