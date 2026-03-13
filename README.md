# Multimodal Medical Triage

Online app for lesion image + clinical note triage prediction.

## Stack

- Frontend: React + TypeScript + Vite
- Backend: FastAPI + PyTorch
- Model Serving: Docker + AWS App Runner
- Website Hosting: Amazon S3 + CloudFront
- Storage: Amazon S3
- Prediction History: Amazon DynamoDB
- CI/CD: GitHub Actions + AWS OIDC

## Architecture

```text
Users
  ↓
CloudFront
  ↓
S3 static frontend (React)
  ↓
FastAPI backend on AWS App Runner
  ↓
PyTorch multimodal model
  ↓
S3 uploads + DynamoDB prediction history