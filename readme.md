# Real-Time YOLO Car Detection System

A real-time computer vision application that uses a custom-trained YOLO model to detect cars frame-by-frame from video.

The project covers the complete machine learning pipeline — from dataset preparation and model training to evaluation and deployment through a FastAPI backend with a web-based frontend.

---

## Overview

This project was developed as a Computer Vision system for detecting cars in images and video streams.

A custom YOLO model was trained on a car detection dataset and then integrated into a FastAPI application.

Instead of processing an entire video and returning the result afterward, the system processes video frames continuously and performs object detection in real time.

### Core Pipeline

```text
Video / Camera
      │
      ▼
   Video Frame
      │
      ▼
 YOLO Detection
      │
      ▼
Bounding Boxes
      │
      ▼
Detected Cars
      │
      ▼
FastAPI Backend
      │
      ▼
Web Frontend