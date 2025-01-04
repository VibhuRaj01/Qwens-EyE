import logging
import pyaudio
import numpy as np
import torch
from transformers import pipeline
from pynput import keyboard

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

    logging.info("Press 'a' to start/stop recording...")
    logging.info("Press 'q' to quit...")

    recording = False
    frames = []
    transcription_result = ""

    def on_press(key):
        nonlocal recording, frames, transcription_result
        try:
            if key.char == "q":
                logging.info("Exiting...")
                listener.stop()
                return False  # Stop the listener

            if key.char == "a":
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
                        transcription_result = text
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
        listener.join()
        if stream:
            stream.stop_stream()
            stream.close()
        audio.terminate()

    return transcription_result if transcription_result else "Could not understand the audio"