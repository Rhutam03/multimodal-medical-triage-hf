# Multimodal Medical Triage

A multimodal machine learning project for lesion image and clinical note triage prediction.

This project combines computer vision, text processing, model evaluation, cloud deployment, and modern frontend development into one end to end system. The application accepts a lesion image, a short clinical note, or both together, and predicts one of three triage categories: **Low Risk**, **Medium Risk**, or **High Risk**.

The goal of this project was not just to train a model, but to build a complete and deployment ready pipeline that reflects the kind of work done in real machine learning and software engineering environments.


## Project Overview

Healthcare data often comes from more than one source. In many real settings, decisions are made using both visual information and written clinical context rather than either one alone. This project explores that idea through a multimodal triage system that brings image and text signals together in a single prediction workflow.

 Skills used in this project:

- Multimodal machine learning
- PyTorch model development
- Image and text preprocessing
- Evaluation of a multiclass classification system
- FastAPI backend development
- React and TypeScript frontend development
- AWS deployment
- CI/CD with GitHub Actions and AWS OIDC

## What the System Does

- Accepts a lesion image as input
- Accepts a clinical note as input
- Supports image only, text only, and combined multimodal inference
- Predicts one of three triage levels:
  - Low Risk
  - Medium Risk
  - High Risk
- Stores uploaded assets in Amazon S3
- Tracks prediction history in DynamoDB
- Serves the backend on AWS App Runner
- Hosts the frontend with Amazon S3 and CloudFront
- Uses GitHub Actions for deployment automation


## Tech Stack

### Backend
- Python
- PyTorch
- FastAPI
- scikit-learn
- NumPy
- pandas

### Frontend
- React
- TypeScript
- Vite

### Cloud and Deployment
- Docker
- AWS App Runner
- Amazon S3
- Amazon CloudFront
- Amazon DynamoDB
- GitHub Actions
- AWS OIDC



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
