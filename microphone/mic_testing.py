from is_a_question import is_a_question
from microphone_live import live_speech_to_text


def main():
    print("Listening for speech...")
    while True:
        text = live_speech_to_text()
        if text:
            if is_a_question(text):
                print(f"Question detected!\nThe question is: {text}")
                break


if __name__ == "__main__":
    main()
