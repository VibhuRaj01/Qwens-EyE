from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# Ensure that bitsandbytes is available (this will allow 8-bit loading)
from transformers import BitsAndBytesConfig

torch.cuda.empty_cache()

# Create the config for loading the model in 8-bit precision
bnb_config = BitsAndBytesConfig(load_in_4bit=True)

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

# Define the messages
messages = [
    {"role": "system", "content": sys_prompt},  # Corrected the syntax error
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": r"/home/bibu/Downloads/Juan Meme.jpg",
            },
            {"type": "text", "text": "Describe the image."},
        ],
    },
]

# Apply the chat template
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

# Move the input tensors to the same device as the model (CPU or GPU)
inputs = {key: value.to(device) for key, value in inputs.items()}

# Debugging: Print the shapes of the input tensors
print("Input IDs shape:", inputs["input_ids"].shape)
print("Attention Mask shape:", inputs["attention_mask"].shape)
if "image" in inputs:
    print("Image inputs shape:", inputs["image"].shape)


# Define a function to process in smaller batches to avoid out-of-memory errors
def generate_in_batches(model, inputs, batch_size=1):
    generated_texts = []
    total_input_ids = len(inputs["input_ids"])

    # Process inputs in batches
    for start_idx in range(0, total_input_ids, batch_size):
        end_idx = min(start_idx + batch_size, total_input_ids)

        # Slice the inputs to create a batch
        batch_inputs = {key: value[start_idx:end_idx] for key, value in inputs.items()}

        # Debugging: Print the shapes of the batch input tensors
        print("Batch Input IDs shape:", batch_inputs["input_ids"].shape)
        print("Batch Attention Mask shape:", batch_inputs["attention_mask"].shape)
        if "image" in batch_inputs:
            print("Batch Image inputs shape:", batch_inputs["image"].shape)

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


# Inference: Generate the output in batches to avoid OOM error
output_texts = generate_in_batches(model, inputs, batch_size=1)

# Print the output
print(output_texts)
