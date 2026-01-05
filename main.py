from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import easyocr
import cv2
import numpy as np
import io

app = FastAPI()

# Enable CORS so the frontend can communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Models
# REPLACE 'best.pt' with the path to your trained model from Phase 1
print("Loading models...")
model = YOLO("best.pt") 
reader = easyocr.Reader(['en'], gpu=True)
print("Models loaded.")

@app.post("/detect")
async def detect_plate(file: UploadFile = File(...)):
    # 1. Read Image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 2. Detect Plate using YOLO
    results = model(img)
    
    response_data = {"detected": False, "text": "", "confidence": 0.0}

    for result in results:
        for box in result.boxes:
            # Get coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            
            # Crop the license plate
            plate_crop = img[y1:y2, x1:x2]
            
            # 3. OCR on the crop
            # Convert to grayscale for better OCR
            gray_plate = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            ocr_result = reader.readtext(gray_plate)
            
            # Extract text if OCR found anything
            if ocr_result:
                # ocr_result format: ([[coords], text, confidence])
                detected_text = ocr_result[0][1] 
                response_data = {
                    "detected": True,
                    "text": detected_text.upper(),
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2]
                }
                # Break after first plate found (assuming 1 plate per image)
                break
        if response_data["detected"]:
            break

    return response_data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)