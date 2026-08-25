# Multimodal Medical Triage

A multimodal learning project using **lesion images** and **supporting clinical text**.

This repository combines model development, data preprocessing, training, inference, backend API engineering, frontend application development, containerization, and cloud deployment workflows into a single end to end project.

## Overview

Clinical triage decisions are rarely based on a single source of information. In dermatology and related screening workflows, clinicians often evaluate both **visual evidence** and **contextual notes** such as lesion location, symptom progression, patient history, and other structured observations. This project is designed around that practical reality.

The system accepts:

- a **lesion image**
- **clinical notes or structured metadata**
- a multimodal inference pipeline that produces one of three triage outcomes:
  - **Low Risk**
  - **Medium Risk**
  - **High Risk**

The project is structured as a deployable application rather than a research notebook alone. It includes a trained model, inference service, user interface, local prediction history, Dockerized backend packaging, and CI/CD workflows for cloud deployment.

## Motivation

The motivation for this project is both **clinical** and **engineering driven**.

From a clinical perspective, visual assessment alone is often insufficient for risk prioritization. A lesion that appears similar across two cases may represent different urgency levels when paired with supporting context such as recent change in size, anatomical site, or patient notes. A multimodal system is therefore a more realistic approximation of how triage decisions are made in practice.

From an engineering perspective, many machine learning projects stop at model training and do not demonstrate how a model is operationalized for real use. This project was built to address that gap by showing the complete lifecycle of an applied AI product:

- data preparation
- multimodal representation learning
- model evaluation
- runtime inference
- API design
- frontend integration
- deployment automation

The objective is not only to build a classifier, but to demonstrate how machine learning models can be packaged into a structured, usable, and production oriented application.

## Objectives

This project was developed with the following goals:

- Build a **multimodal triage model** that combines image and text features
- Design a **repeatable training pipeline** for supervised learning
- Expose model inference through a **FastAPI backend**
- Create a **React frontend** for case submission and result interpretation
- Support **recent prediction history** for application usability
- Package the backend with **Docker**
- Automate backend and frontend deployment through **GitHub Actions**
- Organize the codebase in a way that reflects industry style ML application architecture

## Key Features

- Multimodal lesion risk prediction using image and text
- Three level triage classification: Low, Medium, High Risk
- FastAPI based inference API
- React and TypeScript frontend for interactive case analysis
- Local persistence of recent prediction history
- Dockerized backend
- CI/CD workflows for cloud deployment
- AWS oriented deployment structure using ECR, S3, CloudFront, and App Runner compatible configuration

## System Workflow

### 1. Case Submission

The user interacts with the frontend and submits:

- a lesion image
- optional clinical notes
- optional contextual fields such as age, sex, and lesion site

### 2. Backend Request Handling

The frontend sends a `multipart/form-data` request to the backend API. The backend accepts:

- `file` or `image`
- `note_text` or `notes`
- `age`
- `sex`
- `site`

### 3. Input Preprocessing

The backend performs:

- image loading and RGB conversion
- image resizing and tensor transformation
- text cleaning and normalization
- tokenization and vocabulary based encoding

### 4. Multimodal Inference

The model processes both modalities:

- image features are extracted through a pretrained CNN backbone
- text features are extracted through a lightweight embedding based encoder
- both feature vectors are fused and passed to a classifier head

### 5. Prediction Output

The API returns:

- predicted class index
- triage label
- model confidence
- class probabilities
- normalized text used for inference
- model metadata

### 6. Prediction History

Recent predictions are stored in a local JSON file and made available to the frontend for display in the history panel.


## Results

The shipped model was evaluated on 5,067 held out cases across the three triage levels. The
figures below are the run recorded in `backend/models/best_metrics.json`, which is the checkpoint
the deployed API serves.

| Metric | Score |
| --- | --- |
| Accuracy | 79.79% |
| Macro F1 | 76.57% |
| Balanced accuracy | 76.80% |

Per class:

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| Low Risk | 87.02% | 88.00% | 87.51% | 2,576 |
| Medium Risk | 66.23% | 70.82% | 68.45% | 922 |
| High Risk | 76.08% | 71.57% | 73.76% | 1,569 |

Medium Risk is the hardest class, which is what you would expect: it sits between the other two
and shares symptoms with both. The confusion matrix in the metrics file shows most of its errors
land in the adjacent classes rather than jumping from Low to High.

Predictions below the 0.70 confidence threshold are not forced into a class. They are flagged for
manual review instead, because in a triage setting a wrong confident answer costs more than a
deferred one.

## Tech Stack

PyTorch for the fusion model, with a ResNet-18 image branch and a clinical text branch. FastAPI
serves inference, the frontend is React with TypeScript, and both are Dockerized and deployed to
AWS through GitHub Actions: the backend image to ECR and App Runner, the frontend build to S3.
