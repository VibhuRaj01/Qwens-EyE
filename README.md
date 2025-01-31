# Vision LLM with Microphone and Camera Integration

This project creates a local Vision Language Model (LLM) using Hugging Face's transformers, with integration for both microphone and camera functionalities. The model captures images when the user presses the `A` key and starts recording audio when the same key is pressed again. It combines both functionalities to process and generate multimodal results based on input from both the camera and microphone.

## Features
- **Camera Integration**: Captures an image when the user presses the `A` key.
- **Microphone Integration**: Records audio when the user presses the `A` key.
- **Hugging Face LLM**: Uses a Vision LLM model from Hugging Face to process the captured data.
- **Local Execution**: Runs completely offline, without needing an internet connection once set up.

## Requirements

- Python 3.10.x +
- Cuda 12.4 +

### Install Required Packages

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

### Setup Instructions
- Clone the Repository
```bash
git clone https://github.com/your-username/vision-llm.git
cd vision-llm
```
- Running the Program
```bash
python driver.py
```

