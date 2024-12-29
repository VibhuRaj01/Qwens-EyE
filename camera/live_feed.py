import cv2
import os
import sys
from vision_model import read_sys_prompt, init_model, get_llm_out

# Initialize model and read system prompt
model, processor, device = init_model()
sys_prompt = read_sys_prompt(
    "/home/bibu/Projects/VideoDescription/Video-Description/system_prompt.txt"
)

# Create a directory to save images if it doesn't exist
image_folder = "/home/bibu/Projects/VideoDescription/Video-Description/images"
os.makedirs(image_folder, exist_ok=True)

# Open the video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    sys.exit("Error: Could not open video capture.")

frame_count = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Display the resulting frame
    cv2.imshow("Live Feed", frame)

    # Check if the 'A' key is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("a"):
        # Save the frame to the images folder
        frame_count += 1
        image_path = os.path.join(image_folder, f"frame_{frame_count}.png")
        cv2.imwrite(image_path, frame)
        print(f"Saved frame to {image_path}")

        # Send the image path and the text to get_llm_out
        text = "describe the image"
        response = get_llm_out(model, processor, device, sys_prompt, image_path, text)
        print(f"Response from get_llm_out: {response}")

    # Check if the 'Q' key is pressed to quit
    if key == ord("q"):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
