import logging
import pyaudio
import numpy as np
import torch
from transformers import pipeline
from pynput import keyboard

# Initialize the Whisper ASR pipeline
try:
    whisper = pipeline(
        "automatic-speech-recognition",
        "openai/whisper-large-v3-turbo",
        torch_dtype=torch.float16,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
except Exception as e:
    logging.error(f"An error occurred while initializing the Whisper pipeline: {e}")
    raise


def live_speech_to_text():
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024

    audio = pyaudio.PyAudio()
    stream = None

    try:
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )
    except Exception as e:
        logging.error(f"An error occurred while opening the audio stream: {e}")
        return "Could not understand the audio"

    logging.info("Press 's' to start/stop recording...")

    recording = False
    frames = []

    def on_press(key):
        nonlocal recording, frames
        try:
            if key.char == "q":
                return "Good day"

            if key.char == "s":
                if not recording:
                    logging.info("Recording started.")
                    recording = True
                    frames = []  # Clear previous frames

                else:
                    logging.info("Recording stopped.")
                    recording = False

                    # Convert audio to numpy array for processing
                    audio_data = (
                        np.frombuffer(b"".join(frames), np.int16).astype(np.float32)
                        / 32768.0
                    )

                    # Pass the audio data to Whisper for transcription
                    transcription = whisper(audio_data, return_timestamps=False)
                    text = transcription.get("text", "").strip()

                    if text:
                        logging.info(f"Transcription: {text}")
                        return text
        except AttributeError:
            pass  # Ignore special keys

    # Set up the keyboard listener
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    try:
        while True:
            if recording:
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
    except KeyboardInterrupt:
        logging.info("\nStopped by user.")
    finally:
        listener.stop()
        if stream:
            stream.stop_stream()
            stream.close()
        audio.terminate()

    return "Could not understand the audio"
