# Google Cloud Setup Guide

This guide walks you through the setup steps needed on Google Cloud Platform (GCP) to support the `vidyaNav-ai-backend` project, especially when using Vertex AI and Google Cloud Storage.

---

## 1. Create a Google Cloud Project

- Visit the [Google Cloud Console](https://console.cloud.google.com/).
- Create a new project and note the **Project ID** (e.g: `my-root-vidyaNav-ai-project`).
- Enable billing and enable required services:

```bash
gcloud config set project my-vidyaNav-ai-project
gcloud services enable aiplatform.googleapis.com
gcloud services enable storage.googleapis.com
```

---

## 2. Create and Download a Service Account Key

1. Navigate to **IAM & Admin → Service Accounts**.
2. Create a service account
3. Assign the following roles:
   - `Vertex AI User`
   - `Storage Object Admin`
4. Add a key and download the secrects as `creds.json`. Save it in the root directory and add it to `.gitignore`.

---

## 3. Set Up Application Default Credentials

Setup `.env` and `.env.docker` files by seeing examples. 

---

## 4. Google Cloud Storage

Files and images are stored in Google Cloud Storage. So, need to create a bucket.

```bash
Go to Google Cloud Console → Cloud Storage → Create a bucket.
```
The bucket name is used in the `.env` file.

---