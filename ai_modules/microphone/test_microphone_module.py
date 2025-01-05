import logging
import microphone_live

# Configure logging
logging.basicConfig(
                    filename="Logs",
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO
                    )

def test_live_speech_to_text():
    try:
        logging.info("Starting live speech to text test...")
        transcription = microphone_live.live_speech_to_text()
        logging.info(f"Transcription Result: {transcription}")
    except Exception as e:
        logging.error(f"An error occurred during the test: {e}")

if __name__ == "__main__":
    test_live_speech_to_text()