import cv2
import os
import sys
import logging
from ai_modules.camera.vision_model import read_sys_prompt, init_model, get_llm_out
from ai_modules.microphone.microphone_live import live_speech_to_text

logging.basicConfig(level=logging.INFO)

sys_prompt_path = "D:\Project\Video-Description\ai_modules\system_prompt.txt"
image_folder = "D:\Project\Video-Description\ai_modules\images"


def check_directories(image_folder_path: str, system_prompt_path: str):
    if not os.path.exists(image_folder_path):
        logging.error(f"Folder not found at: {image_folder_path}")
        raise FileNotFoundError(f"Folder not found at: {image_folder_path}")
    if not os.path.exists(system_prompt_path):
        logging.error(f"File not found at: {system_prompt_path}")
        raise FileNotFoundError(f"File not found at: {system_prompt_path}")


def initialize_model_and_prompt(system_prompt_path: str):
    model, processor, device = init_model()
    sys_prompt = read_sys_prompt(system_prompt_path)
    return model, processor, device, sys_prompt


def capture_and_process_frame(
    cap: cv2.VideoCapture,
    image_folder_path: str,
    frame_count: int,
    model,
    processor,
    device,
    sys_prompt,
):
    ret, frame = cap.read()
    if not ret:
        logging.error("Error: Failed to capture frame.")
        return frame_count, None

    cv2.imshow("Live Feed", frame)

    text = live_speech_to_text()
    logging.info(f"Question detected!\nThe question is: {text}")

    frame_count += 1
    image_path = os.path.join(image_folder_path, f"frame_{frame_count}.png")
    cv2.imwrite(image_path, frame)
    logging.info(f"Saved frame to {image_path}")

    response = get_llm_out(model, processor, device, sys_prompt, image_path, text)
    logging.info(f"Response from get_llm_out: {response}")

    return frame_count, frame


def start_live_feed_with_speech(
    image_folder_path: str = image_folder, system_prompt_path: str = sys_prompt_path
):
    check_directories(image_folder_path, system_prompt_path)
    model, processor, device, sys_prompt = initialize_model_and_prompt(
        system_prompt_path
    )
    os.makedirs(image_folder_path, exist_ok=True)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        logging.error("Error: Could not open video capture.")
        sys.exit("Error: Could not open video capture.")

    frame_count = 0

    logging.info("Listening for speech...")

    try:
        while True:
            frame_count, frame = capture_and_process_frame(
                cap,
                image_folder_path,
                frame_count,
                model,
                processor,
                device,
                sys_prompt,
            )

            # Check if the 'Q' key is pressed to quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    start_live_feed_with_speech()
