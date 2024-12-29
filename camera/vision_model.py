from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from qwen_vl_utils import process_vision_info
import torch
import warnings

warnings.filterwarnings("ignore")


def read_sys_prompt(file_path):
    """Read the system prompt from a file."""
    with open(file_path, "r") as file:
        return file.read()


def init_model():
    """Initialize the model and processor."""
    # Create the config for loading the model in 8-bit precision
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    # Determine the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model with 8-bit precision
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=bnb_config,
    )

    # Set pixel constraints
    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28

    # Load the processor
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    return model, processor, device


def get_llm_out(model, processor, device, sys_prompt, image, text):
    """Generate LLM output based on the image and text."""
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

    try:
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
        )
    except Exception as e:
        return f"An error has occurred with LLM: {e}"

    return output_text


# # Example Usage
# if __name__ == "__main__":
#     # Read the system prompt
#     sys_prompt = read_sys_prompt(
#         "/home/bibu/Projects/VideoDescription/Video-Description/system_prompt.txt"
#     )

#     # Initialize the model and processor
#     model, processor, device = init_model()

#     # Test the function
#     result = get_llm_out(
#         model,
#         processor,
#         device,
#         sys_prompt,
#         r"/home/bibu/Downloads/chill guy.jpg",
#         "Describe the guy only, not the surroundings!",
#     )
#     print(result)
