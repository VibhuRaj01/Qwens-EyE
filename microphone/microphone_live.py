import pyaudio
import speech_recognition as sr


def live_speech_to_text():
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024

    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
    )
    recognizer = sr.Recognizer()

    print("Recording... (Press Ctrl+C to stop)")

    try:
        while True:
            frames = []
            for _ in range(0, int(RATE / CHUNK * 2)):
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)

            audio_data = b"".join(frames)
            audio_data = sr.AudioData(audio_data, RATE, 2)

            try:
                text = recognizer.recognize_google(audio_data)
                if text.strip():
                    return text
            except sr.UnknownValueError:
                return "Could not understand the audio"
            except sr.RequestError as e:
                return f"API Error: {e}"
    except KeyboardInterrupt:
        print("\nRecording stopped.")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()
