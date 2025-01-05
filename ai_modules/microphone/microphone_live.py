import logging
import pyaudio
import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pynput import keyboard

# Configure logging
logging.basicConfig(
                    filename="Logs",
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO
                    )

device="cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = "openai/whisper-large-v3-turbo"

# Initialize the Whisper ASR pipeline
try:
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    whisper = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
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