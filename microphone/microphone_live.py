import pyaudio
import numpy as np
import torch
from transformers import pipeline

# Initialize the Whisper ASR pipeline
whisper = pipeline(
    "automatic-speech-recognition",
    "openai/whisper-large-v3-turbo",
    torch_dtype=torch.float16,
    device="cuda:0" if torch.cuda.is_available() else "cpu",
)


def live_speech_to_text():
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024

    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
    )

    print("Recording... (Press Ctrl+C to stop)")

    try:
        while True:
            frames = []
            for _ in range(0, int(RATE / CHUNK * 5)):  # Capture 5 seconds of audio
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)

            # Convert audio to numpy array for processing
            audio_data = (
                np.frombuffer(b"".join(frames), np.int16).astype(np.float32) / 32768.0
            )

            # Pass the audio data to Whisper for transcription
            transcription = whisper(audio_data, return_timestamps=False)
            text = transcription.get("text", "").strip()

            if text:
                return text
    except KeyboardInterrupt:
        print("\nRecording stopped.")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()
