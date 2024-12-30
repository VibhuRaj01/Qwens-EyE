import cv2
import os
import sys
from camera.vision_model import read_sys_prompt, init_model, get_llm_out
from microphone.is_a_question import is_a_question
from microphone.microphone_live import live_speech_to_text


sys_prompt_path = (
    "/home/bibu/Projects/VideoDescription/Video-Description/system_prompt.txt"
)
image_folder = "/home/bibu/Projects/VideoDescription/Video-Description/images"


def start_live_feed_with_speech(
    image_folder_path=image_folder, system_prompt_path=sys_prompt_path
):
    # Check directories
    if not os.path.exists(image_folder_path):
        raise FileNotFoundError(f"Folder not found at: {image_folder_path}")
    if not os.path.exists(system_prompt_path):
        raise FileNotFoundError(f"File not found at: {system_prompt_path}")

    # Initialize model and read system prompt
    model, processor, device = init_model()
    sys_prompt = read_sys_prompt(system_prompt_path)

    os.makedirs(image_folder_path, exist_ok=True)

    # Open the video capture
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        sys.exit("Error: Could not open video capture.")

    frame_count = 0

    print("Listening for speech...")

    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            if not ret:
                print("Error: Failed to capture frame.")
                break

            # Display the resulting frame
            cv2.imshow("Live Feed", frame)

            # Check for speech
            text = live_speech_to_text()
            if is_a_question(text) and text != "Could not understand the audio":
                print(f"Question detected!\nThe question is: {text}")

                # Save the current frame
                frame_count += 1
                image_path = os.path.join(image_folder_path, f"frame_{frame_count}.png")
                cv2.imwrite(image_path, frame)
                print(f"Saved frame to {image_path}")

                # Send the image path and the question to get_llm_out
                response = get_llm_out(
                    model, processor, device, sys_prompt, image_path, text
                )
                print(f"Response from get_llm_out: {response}")

            # Check if the 'Q' key is pressed to quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    start_live_feed_with_speech()
