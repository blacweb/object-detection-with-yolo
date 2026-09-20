import os
import time
import uuid
import cv2

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "yolo11n.pt")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

CONFIDENCE_THRESHOLD = 0.50

# Smaller image = faster CPU inference
YOLO_IMAGE_SIZE = 320


# ============================================================
# DIRECTORIES
# ============================================================

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FRONTEND_DIR, exist_ok=True)


# ============================================================
# LOAD YOLO
# ============================================================

print("=" * 60)
print("YOLO LIVE DETECTION SERVER")
print("=" * 60)

print("Loading:", MODEL_PATH)

model = YOLO(MODEL_PATH)

print("YOLO model loaded.")
print("Device: CPU")
print("Image size:", YOLO_IMAGE_SIZE)
print("Confidence:", CONFIDENCE_THRESHOLD)

print("=" * 60)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="YOLO Live Car Detection"
)


# ============================================================
# FRONTEND
# ============================================================

@app.get("/")
async def home():

    return FileResponse(
        os.path.join(
            FRONTEND_DIR,
            "index.html"
        )
    )


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
async def health():

    return {
        "status": "ok",
        "model": "yolo11n.pt",
        "device": "cpu",
        "image_size": YOLO_IMAGE_SIZE,
        "confidence": CONFIDENCE_THRESHOLD
    }


# ============================================================
# UPLOAD VIDEO
# ============================================================

@app.post("/upload-video")
async def upload_video(
    file: UploadFile = File(...)
):

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No video selected."
        )


    extension = os.path.splitext(
        file.filename
    )[1].lower()


    allowed_extensions = {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm"
    }


    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail="Unsupported video format."
        )


    # Create unique filename
    filename = (
        str(uuid.uuid4())
        + extension
    )


    video_path = os.path.join(
        UPLOAD_DIR,
        filename
    )


    print()
    print("=" * 60)
    print("VIDEO UPLOAD")
    print("=" * 60)

    print(
        "Original:",
        file.filename
    )

    print(
        "Saved:",
        filename
    )


    # Save video
    with open(
        video_path,
        "wb"
    ) as buffer:

        while True:

            chunk = await file.read(
                1024 * 1024
            )

            if not chunk:
                break

            buffer.write(chunk)


    print("Upload complete.")

    print("=" * 60)


    return {
        "success": True,
        "filename": filename,
        "stream_url":
            f"/video-stream/{filename}"
    }


# ============================================================
# YOLO FRAME GENERATOR
# ============================================================

def generate_frames(filename):

    video_path = os.path.join(
        UPLOAD_DIR,
        filename
    )


    if not os.path.exists(video_path):

        print(
            "Video does not exist:",
            video_path
        )

        return


    cap = cv2.VideoCapture(
        video_path
    )


    if not cap.isOpened():

        print(
            "Could not open video."
        )

        return


    fps = cap.get(
        cv2.CAP_PROP_FPS
    )


    if fps <= 0:
        fps = 25


    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )


    print()
    print("=" * 60)
    print("STARTING LIVE YOLO DETECTION")
    print("=" * 60)

    print(
        "FPS:",
        fps
    )

    print(
        "Total frames:",
        total_frames
    )


    frame_number = 0

    start_time = time.time()


    try:

        while True:

            # --------------------------------------------
            # READ FRAME
            # --------------------------------------------

            success, frame = cap.read()


            if not success:
                break


            frame_number += 1


            # --------------------------------------------
            # YOLO
            # --------------------------------------------

            inference_start = time.time()


            results = model.predict(

                source=frame,

                imgsz=YOLO_IMAGE_SIZE,

                conf=CONFIDENCE_THRESHOLD,

                device="cpu",

                verbose=False

            )


            result = results[0]


            # --------------------------------------------
            # DETECTIONS
            # --------------------------------------------

            if result.boxes is not None:

                detection_count = len(
                    result.boxes
                )

            else:

                detection_count = 0


            # --------------------------------------------
            # DRAW BOXES
            # --------------------------------------------

            annotated = result.plot()


            # --------------------------------------------
            # FPS
            # --------------------------------------------

            inference_time = (
                time.time()
                - inference_start
            )


            if inference_time > 0:

                inference_fps = (
                    1 / inference_time
                )

            else:

                inference_fps = 0


            # --------------------------------------------
            # TEXT ON VIDEO
            # --------------------------------------------

            cv2.putText(

                annotated,

                f"YOLO LIVE | Frame: {frame_number}",

                (20, 35),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                (0, 255, 0),

                2

            )


            cv2.putText(

                annotated,

                f"Cars: {detection_count}",

                (20, 70),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.75,

                (0, 255, 0),

                2

            )


            cv2.putText(

                annotated,

                f"FPS: {inference_fps:.1f}",

                (20, 105),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (255, 255, 0),

                2

            )


            # --------------------------------------------
            # JPEG
            # --------------------------------------------

            success, encoded = cv2.imencode(

                ".jpg",

                annotated,

                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    80
                ]

            )


            if not success:
                continue


            frame_bytes = encoded.tobytes()


            # --------------------------------------------
            # SEND FRAME
            # --------------------------------------------

            yield (

                b"--frame\r\n"

                b"Content-Type: image/jpeg\r\n"

                b"Content-Length: "

                + str(
                    len(frame_bytes)
                ).encode()

                + b"\r\n\r\n"

                + frame_bytes

                + b"\r\n"

            )


            # --------------------------------------------
            # SERVER PROGRESS
            # --------------------------------------------

            if frame_number % 30 == 0:

                elapsed = (
                    time.time()
                    - start_time
                )


                processing_fps = (

                    frame_number
                    / elapsed

                    if elapsed > 0
                    else 0

                )


                print(

                    f"Frame: "
                    f"{frame_number}/{total_frames} | "

                    f"Processing FPS: "
                    f"{processing_fps:.2f}"

                )


    finally:

        cap.release()


        elapsed = (
            time.time()
            - start_time
        )


        print()
        print("=" * 60)
        print("LIVE DETECTION FINISHED")
        print("=" * 60)

        print(
            "Frames:",
            frame_number
        )

        print(
            "Time:",
            round(
                elapsed,
                2
            ),
            "seconds"
        )

        print("=" * 60)


# ============================================================
# VIDEO STREAM
# ============================================================

@app.get(
    "/video-stream/{filename}"
)
async def video_stream(
    filename: str
):

    # Security: don't allow paths
    filename = os.path.basename(
        filename
    )


    video_path = os.path.join(
        UPLOAD_DIR,
        filename
    )


    if not os.path.exists(
        video_path
    ):

        raise HTTPException(
            status_code=404,
            detail="Video not found."
        )


    return StreamingResponse(

        generate_frames(filename),

        media_type=
        "multipart/x-mixed-replace; boundary=frame"

    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    import uvicorn


    uvicorn.run(

        app,

        host="127.0.0.1",

        port=8000,

        reload=False

    )