from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from qwen_vl_utils import process_vision_info
import torch
import warnings

warnings.filterwarnings("ignore")

# Create the config for loading the model in 8-bit precision
bnb_config = BitsAndBytesConfig(load_in_8bit=True)

# Load the model on the available device(s) with 8-bit precision
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  # Automatically choose GPU if available, else CPU
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=bnb_config,
)

min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28

# Load the processor
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels
)

# Read the system prompt from the file
with open(
    r"/home/bibu/Projects/VideoDescription/Video-Description/system_prompt.txt", "r"
) as file:
    sys_prompt = file.read()


def get_llm_out(image, text):
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

    inputs = inputs.to("cuda")
    try:
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
        return f"An error has occured with LLM {e}"

    return output_text


# Testing
# print(
#     get_llm_out(
#         r"/home/bibu/Downloads/chill guy.jpg",
#         "describe the guy only not the surroundings!",
#     )
# )
