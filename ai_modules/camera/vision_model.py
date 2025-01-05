import logging
import os
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from qwen_vl_utils import process_vision_info
import torch
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(
                    filename="Logs",
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO
                    )


def read_sys_prompt(file_path):
    """Read the system prompt from a file."""
    try:
        with open(file_path, encoding="utf8") as file:
            return file.read()
    except FileNotFoundError:
        logging.error(f"File not found at: {file_path}")
        raise
    except Exception as e:
        logging.error(f"An error occurred while reading the system prompt: {e}")
        raise


def init_model():
    """Initialize the model and processor."""
    # Determine the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"The device in use is : {device}")

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    # Load the model with 8-bit precision
    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            torch_dtype="auto",
            device_map="auto",
            quantization_config=bnb_config,
        )
    # try:
    #     model = Qwen2VLForConditionalGeneration.from_pretrained(
    #         "Qwen/Qwen2-VL-2B-Instruct",
    #         device_map="auto"
    #     )
    except Exception as e:
        logging.error(f"An error occurred while initializing the model: {e}")
        raise

    # Set pixel constraints
    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28

    # Load the processor
    try:
        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct",
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    except Exception as e:
        logging.error(f"An error occurred while initializing the processor: {e}")
        raise

    return model, processor, device


def get_llm_out(model, processor, device, sys_prompt, image, text) -> str:
    """Generate LLM output based on the image and text."""
    if not os.path.exists(image):
        logging.error(f"Image file does not exist: {image}")
        return "Image file not found."
    
    messages = [
        {"role": "system", "content": sys_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": text},
            ],
        },
    ]

    try:
        # Process the text using the processor
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process the vision info
        image_inputs, video_inputs = process_vision_info(messages)

        # Prepare inputs for the model
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # Move inputs to the correct device
        inputs = inputs.to(device)

        # Generate predictions
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
    except Exception as e:
        logging.error(f"An error occurred with LLM: {e}")
        return f"An error has occurred with LLM: {e}"

    return output_text


# # Example Usage
# if __name__ == "__main__":
#     # Read the system prompt
#     sys_prompt = read_sys_prompt(
#         "D:\Project\Video-Description\system_prompt.txt"
#     )

#     # Initialize the model and processor
#     model, processor, device = init_model()

#     # Test the function
#     result = get_llm_out(
#         model,
#         processor,
#         device,
#         sys_prompt,
#         r"C:\Users\vibhu\Downloads\wallpapers\wp12796311-porsche-911-gt3-rs-4k-wallpapers.jpg",
#         "Do you see a car in this image?",
#     )
#     print(result)
