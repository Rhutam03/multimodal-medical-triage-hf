---
title: Multimodal Medical Triage
emoji: 🩺
colorFrom: blue
colorTo: pink
sdk: gradio
sdk_version: 3.50.2
app_file: app/app.py
pinned: false
---

# Multimodal Medical Triage System

## Overview
This project demonstrates a **multimodal medical triage system** that can analyze
**medical images**, **text descriptions**, or **both together** to produce a
triage-level prediction.

The system is designed as a **robust deployment-ready AI pipeline** rather than a
clinical diagnostic model.

---

## Features
- Accepts **medical images only**
- Accepts **medical text only**
- Supports **image + text multimodal inference**
- Robust to missing inputs (image or text can be omitted)
- Deployed using **Gradio on Hugging Face Spaces**
- CPU-only (no CUDA dependency)

---

## Architecture
- **Image Encoder**: Lightweight CNN
- **Text Encoder**: Token-based embedding encoder
- **Fusion Head**: Feature concatenation + classifier
- **Inference Layer**: Safe, lazy-loaded model execution

---

## Tech Stack
- Python 3.10
- PyTorch
- Gradio
- Hugging Face Spaces

---

## How to Use
1. Upload a medical image *(optional)*
2. Enter a medical description *(optional)*
3. Click **Submit**
4. View the predicted triage level

You may provide **either input or both**.

---

## Output Classes
- **Low Risk**
- **Medium Risk**
- **High Risk**

---

## Deployment
This application is deployed on **Hugging Face Spaces** and automatically exposes:
- A web-based UI
- A REST-style `/run/predict` API endpoint

---

## Disclaimer ⚠️
This project is intended **for educational and research purposes only**.
It is **not a medical device** and **must not be used for clinical diagnosis,
treatment, or decision-making**.

---

## Author
**Rhutam Mahajan**
