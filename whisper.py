import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import sounddevice as sd
import numpy as np

# Initialize Whisper Model and Processor
model_name = "openai/whisper-base"
model = WhisperForConditionalGeneration.from_pretrained(model_name)
processor = WhisperProcessor.from_pretrained(model_name)

# SoundDevice Settings
RATE = 16000  # Sampling rate
CHANNELS = 1  # Mono audio
RECORD_SECONDS = 5  # Segment duration before processing
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = model.to(DEVICE)


def record_audio(duration, rate):
    """Record audio using sounddevice."""
    print("Recording...")
    audio = sd.rec(
        int(duration * rate), samplerate=rate, channels=CHANNELS, dtype="int16"
    )
    sd.wait()  # Wait until recording is finished
    return audio.flatten()


def process_audio_and_transcribe(audio, rate):
    """Process audio and perform transcription."""
    inputs = [audio]  # Wrap in list for batch processing
    processed_inputs = processor(
        inputs,
        return_tensors="pt",
        truncation=False,
        padding="longest",
        return_attention_mask=True,
        sampling_rate=rate,
    )

    # Re-process short inputs
    if processed_inputs.input_features.shape[-1] < 3000:
        processed_inputs = processor(
            inputs,
            return_tensors="pt",
            sampling_rate=rate,
        )

    # Generate transcription (keep everything in float32)
    results = model.generate(
        **processed_inputs.to(DEVICE),  # Avoid float16 conversion
    )
    transcription = processor.batch_decode(results, skip_special_tokens=True)
    return transcription[0]


print("Live Speech-to-Text using Whisper. Speak now...")

while True:
    # Record in segments to process in real-time
    audio = record_audio(RECORD_SECONDS, RATE)
    transcription = process_audio_and_transcribe(audio, RATE)
    print(f"Transcription: {transcription}")

    # Continue listening or exit
    cont = input("Continue listening? (y/n): ")
    if cont.lower() != "y":
        break
