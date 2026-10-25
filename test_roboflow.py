import os
# pyrefly: ignore[missing-import]
import cv2
# pyrefly: ignore[missing-import]
import numpy as np
# pyrefly: ignore[missing-import]
from dotenv import load_dotenv
# pyrefly: ignore[missing-import]
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

# Load .env variables
load_dotenv()

api_key = os.getenv("ROBOFLOW_API_KEY")
model_id = os.getenv("ROBOFLOW_MODEL_ID", "cricket-drs-system/4")

if not api_key:
    raise ValueError("ROBOFLOW_API_KEY not found in .env file!")

print(f"Connecting to Roboflow using model: {model_id}...")

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=api_key
).configure(InferenceConfiguration(
    api_key_transport="header"
))

# Generate a synthetic test frame (green pitch with red circle)
test_frame = np.full((720, 1280, 3), (40, 100, 30), dtype=np.uint8)
cv2.circle(test_frame, (640, 360), 20, (30, 30, 200), -1) # Red ball
test_img_path = "outputs/sample_test.jpg"
os.makedirs("outputs", exist_ok=True)
cv2.imwrite(test_img_path, test_frame)

print(f"Running inference on {test_img_path}...")
result = client.infer(test_img_path, model_id=model_id)

print("\n--- API Connection Successful! ---")
print("Response keys:", list(result.keys()) if isinstance(result, dict) else type(result))
print("Predictions count:", len(result.get("predictions", [])) if isinstance(result, dict) else "N/A")
print("Sample prediction response:", result)
