# ✋ Hand Gesture Recognition API - Production Backend

This repository contains the **production version** of our Hand Gesture Recognition system. It serves a trained SVM model through a REST API using FastAPI, enabling real-time hand gesture classification to control a maze game.

---

## 📌 Project Summary

After developing and training a hand gesture classifier using MediaPipe and an SVM model, this production repo focuses on:

- Serving the model via a FastAPI backend
- Containerizing the app with Docker
- Monitoring performance metrics with Prometheus & Grafana
- Deploying the app using clawcloud 
- Integrating the backend with the maze navigation frontend




## Monitoring Metrics
I instrumented the FastAPI app using prometheus_fastapi_instrumentator and set up Prometheus + Grafana via Docker Compose.

| Type       | Metric Name                         | Why?                                                                                |
| ---------- | ----------------------------------- | ----------------------------------------------------------------------------------- |
| **Model**  | `python_gc_objects_collected_total` | Measures memory pressure and garbage collection which can affect inference latency. |
| **Data**   | `http_requests_total`               | Tracks how many requests are received at the prediction endpoint.                   |
| **Server** | `process_resident_memory_bytes`     | Indicates RAM usage of the API process — useful for scaling and debugging.          |

## 🔗 API Reference
POST /predict

json
Copy
Edit
{
  "landmarks": [0.12, 0.33, ..., 0.54]  // length = 42
}
Response:

json
Copy
Edit
{
  "class": "left",
}

GET /metrics – Exposes Prometheus metrics

## 🛠️ Tech Stack
Python + FastAPI

scikit-learn (SVM)

MediaPipe (landmarks extraction)

Docker, Docker Compose

Prometheus & Grafana

pytest (unit testing)

Deployed on AWS/ClawCloud


