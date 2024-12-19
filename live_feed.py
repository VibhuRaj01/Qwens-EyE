import base64
import cv2
import numpy as np
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from qwen_vl_utils import process_vision_info
import torch

cap = cv2.VideoCapture(0)

# Create the config for loading the model in 8-bit precision
bnb_config = BitsAndBytesConfig(load_in_8bit=True)

# Load the model on the available device(s) with 8-bit precision
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  # Automatically choose GPU if available, else CPU
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype="auto",
    device_map="auto",  # Automatically place model on available devices (like GPU)
    quantization_config=bnb_config,  # Apply the 8-bit quantization config
)

# Load the processor
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# Read the system prompt from the file
with open("system_prompt.txt", "r") as file:
    sys_prompt = file.read()


def generate_in_batches(model, inputs, batch_size=1):
    generated_texts = []
    total_input_ids = len(inputs["input_ids"])

    # Process inputs in batches
    for start_idx in range(0, total_input_ids, batch_size):
        end_idx = min(start_idx + batch_size, total_input_ids)

        # Slice the inputs to create a batch
        batch_inputs = {key: value[start_idx:end_idx] for key, value in inputs.items()}

        # Generate the output for the batch
        generated_ids = model.generate(**batch_inputs, max_new_tokens=128)

        # Trim the generated IDs to remove the input length
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(batch_inputs["input_ids"], generated_ids)
        ]

        # Decode the output text
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        # Collect the generated text
        generated_texts.extend(output_text)

    return generated_texts


def encode_image_as_base64(frame):
    _, buffer = cv2.imencode(".jpg", frame)
    base64_image = base64.b64encode(buffer).decode("utf-8")
    return base64_image


while True:
    ret, frame = cap.read()
    if not ret:
        break

    base64_image = encode_image_as_base64(frame)

    messages = [
        {"role": "system", "content": sys_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": base64_image,
                },
                {
                    "type": "text",
                    "text": "This is an image of a live feed. Reply according to what you see in the image.",
                },
            ],
        },
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}
    output_texts = generate_in_batches(model, inputs, batch_size=1)
    print(output_texts)

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
