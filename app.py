import os
import uuid
import cv2

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "yolo11n.pt")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

CONFIDENCE_THRESHOLD = 0.50


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD YOLO MODEL
# ============================================================

print("=" * 60)
print("YOLO CAR DETECTION SERVER")
print("=" * 60)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"YOLO model not found: {MODEL_PATH}\n"
        "Make sure yolo11n.pt is inside the project folder."
    )

print(f"Loading model: {MODEL_PATH}")

model = YOLO(MODEL_PATH)

print("YOLO model loaded successfully.")
print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
print("=" * 60)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="YOLO Car Detection",
    description="Car detection from uploaded videos using YOLO",
    version="1.0"
)


# ============================================================
# STATIC FILES
# ============================================================

app.mount(
    "/uploads",
    StaticFiles(directory=UPLOAD_DIR),
    name="uploads"
)

app.mount(
    "/outputs",
    StaticFiles(directory=OUTPUT_DIR),
    name="outputs"
)

app.mount(
    "/frontend",
    StaticFiles(directory=FRONTEND_DIR),
    name="frontend"
)


# ============================================================
# HOME PAGE
# ============================================================

@app.get("/")
def home():
    index_file = os.path.join(FRONTEND_DIR, "index.html")

    if not os.path.exists(index_file):
        return JSONResponse({
            "message": "YOLO Car Detection API is running.",
            "error": "frontend/index.html was not found."
        })

    return FileResponse(index_file)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "running",
        "model": "YOLO11n",
        "confidence_threshold": CONFIDENCE_THRESHOLD
    }


# ============================================================
# VIDEO DETECTION
# ============================================================

@app.post("/predict-video")
async def predict_video(file: UploadFile = File(...)):

    # --------------------------------------------------------
    # Check file
    # --------------------------------------------------------

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No video file was provided."
        )

    allowed_extensions = {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm"
    }

    original_extension = os.path.splitext(file.filename)[1].lower()

    if original_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported video format. "
                "Use MP4, AVI, MOV, MKV or WEBM."
            )
        )

    # --------------------------------------------------------
    # Generate unique filenames
    # --------------------------------------------------------

    unique_id = uuid.uuid4().hex

    input_filename = f"{unique_id}{original_extension}"
    output_filename = f"{unique_id}_detected.mp4"

    input_path = os.path.join(
        UPLOAD_DIR,
        input_filename
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        output_filename
    )

    # --------------------------------------------------------
    # Save uploaded video
    # --------------------------------------------------------

    try:

        with open(input_path, "wb") as buffer:

            while True:

                chunk = await file.read(1024 * 1024)

                if not chunk:
                    break

                buffer.write(chunk)

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Failed to save uploaded video: {str(e)}"
        )

    # --------------------------------------------------------
    # Open input video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():

        if os.path.exists(input_path):
            os.remove(input_path)

        raise HTTPException(
            status_code=400,
            detail="Could not open the uploaded video."
        )

    # --------------------------------------------------------
    # Get video properties
    # --------------------------------------------------------

    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    # Some videos report an invalid FPS.
    if fps <= 0:
        fps = 25.0

    print()
    print("=" * 60)
    print("PROCESSING VIDEO")
    print("=" * 60)
    print(f"Input:       {file.filename}")
    print(f"Resolution:  {width} x {height}")
    print(f"FPS:         {fps:.2f}")
    print(f"Frames:      {total_frames}")
    print("=" * 60)

    # --------------------------------------------------------
    # Create output video
    # --------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():

        cap.release()

        if os.path.exists(input_path):
            os.remove(input_path)

        raise HTTPException(
            status_code=500,
            detail="Could not create output video."
        )

    # --------------------------------------------------------
    # Process frames
    # --------------------------------------------------------

    frame_number = 0
    total_detections = 0
    frames_with_detections = 0

    try:

        while True:

            success, frame = cap.read()

            if not success:
                break

            frame_number += 1

            # ------------------------------------------------
            # YOLO inference
            # ------------------------------------------------

            results = model(
                frame,
                conf=CONFIDENCE_THRESHOLD,
                verbose=False
            )

            result = results[0]

            # ------------------------------------------------
            # Count detections
            # ------------------------------------------------

            if result.boxes is not None:

                detection_count = len(result.boxes)

                if detection_count > 0:
                    frames_with_detections += 1
                    total_detections += detection_count

            # ------------------------------------------------
            # Draw bounding boxes
            # ------------------------------------------------

            annotated_frame = result.plot()

            # ------------------------------------------------
            # Write frame to output video
            # ------------------------------------------------

            writer.write(annotated_frame)

            # ------------------------------------------------
            # Console progress
            # ------------------------------------------------

            if frame_number % 30 == 0:

                if total_frames > 0:

                    progress = (
                        frame_number / total_frames
                    ) * 100

                    print(
                        f"Progress: {progress:6.2f}% | "
                        f"Frame: {frame_number}/{total_frames} | "
                        f"Detections: {total_detections}"
                    )

                else:

                    print(
                        f"Frame: {frame_number} | "
                        f"Detections: {total_detections}"
                    )

    except Exception as e:

        cap.release()
        writer.release()

        if os.path.exists(input_path):
            os.remove(input_path)

        if os.path.exists(output_path):
            os.remove(output_path)

        raise HTTPException(
            status_code=500,
            detail=f"Error during YOLO processing: {str(e)}"
        )

    finally:

        cap.release()
        writer.release()

    # --------------------------------------------------------
    # Verify output
    # --------------------------------------------------------

    if not os.path.exists(output_path):

        if os.path.exists(input_path):
            os.remove(input_path)

        raise HTTPException(
            status_code=500,
            detail="Output video was not created."
        )

    output_size = os.path.getsize(output_path)

    print()
    print("=" * 60)
    print("VIDEO PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Frames processed:       {frame_number}")
    print(f"Frames with detections: {frames_with_detections}")
    print(f"Total detections:       {total_detections}")
    print(f"Output:                 {output_path}")
    print(f"Output size:            {output_size / (1024 * 1024):.2f} MB")
    print("=" * 60)

    # --------------------------------------------------------
    # Return result to frontend
    # --------------------------------------------------------

    return {
        "success": True,
        "message": "Video processed successfully.",
        "original_filename": file.filename,
        "frames_processed": frame_number,
        "frames_with_detections": frames_with_detections,
        "total_detections": total_detections,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "video_url": f"/outputs/{output_filename}"
    }


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=False
    )