from RealtimeSTT import AudioToTextRecorder
from ai_modules.camera.vision_model import read_sys_prompt, init_model, get_llm_out
import cv2
import logging
import os
import threading

# Configure logging
logging.basicConfig(
    filename="Logs",
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)


class VisionLanguageApp:
    """Class to handle vision-language processing with real-time camera and microphone input"""

    def __init__(self):
        """Initialize the application and its components"""
        self.logger = logging.getLogger(__name__)
        self.camera_running = False
        self.model = None
        self.processor = None
        self.device = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the vision-language model and its components"""
        try:
            # Read the system prompt
            sys_prompt = read_sys_prompt(
                "D:\Project\Video-Description\system_prompt.txt"
            )

            # Initialize the model and processor
            self.model, self.processor, self.device = init_model()
            self.sys_prompt = sys_prompt
            self.logger.info("Model and processor initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing model: {e}")
            raise

    def _capture_camera_feed(self):
        """
        Capture live camera feed and save the current frame periodically.
        Runs in a separate thread.
        """
        cap = cv2.VideoCapture(0)
        self.camera_running = True

        while self.camera_running:
            ret, frame = cap.read()
            if not ret:
                self.logger.error("Error accessing camera")
                break

            # Save the frame as an image
            cv2.imwrite("captured_image.jpg", frame)

            # Add a small delay
            cv2.waitKey(1000)  # Save an image every second

        cap.release()
        self.logger.info("Camera feed capture stopped")

    def start_camera_capture(self):
        """Start the camera feed capture in a separate thread"""
        try:
            self.camera_running = True
            camera_thread = threading.Thread(
                target=self._capture_camera_feed, daemon=True
            )
            camera_thread.start()
            self.logger.info("Camera feed capture started in background thread")
        except Exception as e:
            self.logger.error(f"Error starting camera capture: {e}")
            raise

    def stop_camera_capture(self):
        """Stop the camera feed capture"""
        self.camera_running = False
        if os.path.exists("captured_image.jpg"):
            os.remove("captured_image.jpg")

    def get_transcribed_text(self):
        """
        Returns the transcribed text from the AudioToTextRecorder.
        """
        try:
            with AudioToTextRecorder() as recorder:
                print("Please start speaking...")
                text = str(recorder.text())
                recorder.shutdown()
                print(f"Transcription completed: {text}")
                return text
        except Exception as e:
            self.logger.error(f"Error during transcription: {e}")
            raise

    def main_loop(self):
        """Main processing loop to handle real-time inputs and generate responses"""
        try:
            while True:
                # Get transcribed text from microphone
                transcribed_text = self.get_transcribed_text()

                # Generate response using vision-language model
                if os.path.exists("captured_image.jpg"):
                    response = get_llm_out(
                        self.model,
                        self.processor,
                        self.device,
                        self.sys_prompt,
                        "captured_image.jpg",
                        transcribed_text,
                    )
                    print("LLM Response:", response)
                    self.logger.info(f"LLM Response: {response}")
                else:
                    print("No image available")
                    self.logger.warning("No image available")

        except Exception as e:
            self.logger.error(f"Main loop error: {e}")
            print(f"An error occurred: {e}")
        finally:
            # Clean up
            self.stop_camera_capture()
            print("Application closed")

    def __del__(self):
        """Clean up resources when object is destroyed"""
        self.stop_camera_capture()
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "processor"):
            del self.processor
        if hasattr(self, "device"):
            del self.device
        self.logger.info("Application resources cleaned up")


if __name__ == "__main__":
    app = VisionLanguageApp()
    app.start_camera_capture()
    try:
        app.main_loop()
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
        app.stop_camera_capture()
