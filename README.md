# End-to-End Chest Cancer Classification 

An **End-to-End Deep Learning system** for **Chest CT Scan Cancer Classification**, built with **PyTorch**, **FastAPI**, **Streamlit**, **Docker**, **DVC**, and **MLflow**.

This repository follows **MLOps practices** using modular pipelines, reproducibility, experiment tracking, containerized deployment, and cloud-native CI/CD for efficient implementation.

---

## Problem Statement

Chest cancer is one of the leading causes of cancer-related deaths.  
Early detection using **CT scan imaging** enables faster diagnosis and better patient outcomes.

This system classifies chest CT scans into:
- **Normal**
- **Cancer (Adenocarcinoma)**

---

---

## Why ResNet-18?

- Lightweight
- Fast inference
- Strong transfer learning
- Ideal for medical datasets

---

## Model Overview

| Component | Description |
|---------|-------------|
| Architecture | ResNet-18 (Transfer Learning) |
| Framework | PyTorch |
| Task | Binary Image Classification |
| Loss | BCEWithLogitsLoss |
| Optimizer | AdamW |
| Metrics | Accuracy, Precision, Recall, F1 |
| Tracking | MLflow (DAGsHub) |

---

## Project Directory Structure (Authoritative)

```
CHEST-CANCER-NET/
├── .github/workflows/
│   └── deploy.yml
├── artifacts/
│   ├── data_ingestion/
│   ├── hyperparameter_tuning/
│   ├── prepare_base_model/
│   └── training/
│       └── model_best_hparams.pt
├── models/
│   └── model_best_hparams.pt
├── config/
│   └── config.yaml
├── logs/
├── research/
├── src/cancer_clf/
│   ├── components/
│   ├── pipelines/
│   ├── config/
│   ├── entity/
│   ├── logger/
│   └── utils/
├── app.py
├── streamlit_app.py
├── Dockerfile.api
├── Dockerfile.streamlit
├── docker-compose.yml
├── dvc.yaml
├── params.yaml
├── requirements.txt
├── setup.py
└── README.md
```

---

## End-to-End Pipeline

1. Data Ingestion  
2. Prepare Base Model  
3. Hyperparameter Tuning (Optuna)  
4. Model Training  
5. Model Evaluation (MLflow)  
6. Inference (FastAPI + Streamlit)  

---

## Local Development

```bash
git clone https://github.com/<your-username>/chest-cancer-net.git
cd chest-cancer-net

conda create -n chest-cancer python=3.10 -y
conda activate chest-cancer
pip install -r requirements.txt
```

Run full pipeline:
```bash
dvc repro
```

---

## Docker (Local)

```bash
docker-compose up --build
```

---

# AWS CONFIGURATION 

This setup supports **CI/CD using GitHub Actions → Amazon ECR → EC2**,  
✅ No secrets stored on EC2  
✅ No retraining during deployment  

---

## 🧱 Architecture

```
GitHub Actions (CI)
 ├── Build Docker images
 ├── Push images to Amazon ECR
 ↓
EC2 (CD)
 ├── Pull images from ECR (IAM Role)
 ├── Run containers using Docker Compose
```

---

## PART 1️⃣ — AWS Prerequisites

- AWS account
- Billing enabled
- Region selected (example: ap-southeast-1)

---

## PART 2️⃣ — Create ECR Repositories

Create **two private repositories**:

### Repository 1 (API)
- Name: `chest-cancer-api`

### Repository 2 (UI)
- Name: `chest-cancer-ui`

Note:
- AWS Account ID
- AWS Region

---

## PART 3️⃣ — IAM User (GitHub Actions CI)

### Create User
- Name: `github-actions-ci`
- Access type: Programmatic

### Attach Policy
- `AmazonEC2ContainerRegistryFullAccess`

### Save Credentials
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`

---

## PART 4️⃣ — IAM Role (EC2 CD)

### Create Role
- Trusted entity: EC2

### Attach Policy
- `AmazonEC2ContainerRegistryReadOnly`

Role name:
```
EC2-ECR-READONLY
```

---

## PART 5️⃣ — Create EC2 Instance

- AMI: Ubuntu 22.04
- Instance type: t2.micro / t3.micro
- Key pair: `infer.pem`
- IAM Role: `EC2-ECR-READONLY`

### Security Group (IMPORTANT)

| Type | Port | Source |
|----|----|----|
| SSH | 22 | 0.0.0.0/0 |
| Custom TCP | 8000 | 0.0.0.0/0 |
| Custom TCP | 8501 | 0.0.0.0/0 |

---

## PART 6️⃣ — EC2 Setup (One-Time)

```bash
ssh -i infer.pem ubuntu@<EC2_PUBLIC_IP>

sudo apt update -y
sudo apt upgrade -y

curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

sudo apt install -y docker-compose-plugin awscli
sudo usermod -aG docker ubuntu
newgrp docker
```

Verify:
```bash
docker --version
docker compose version
aws --version
aws sts get-caller-identity
```

⚠️ Do NOT run `aws configure`  
⚠️ Do NOT add AWS keys on EC2  

---

## PART 7️⃣ — GitHub Secrets

Add in **GitHub → Settings → Secrets → Actions**

### AWS (CI)
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`
- `AWS_ACCOUNT_ID`

### EC2 (CD)
- `EC2_HOST`
- `EC2_USERNAME` → ubuntu
- `EC2_SSH_KEY` → contents of `infer.pem`

---

## PART 8️⃣ — CI/CD Flow

On push to `main`:

**CI**
- Build API & UI Docker images
- Push images to ECR

**CD**
- SSH into EC2
- Pull latest images
- Generate docker-compose.yml
- Restart containers

---

## PART 9️⃣ — Verify Deployment

- API: `http://<EC2_PUBLIC_IP>:8000/docs`
- Streamlit: `http://<EC2_PUBLIC_IP>:8501`




---

## PART 🔟 — Challenges Faced & Learnings

Building an end-to-end, production-ready MLOps system involves addressing real-world engineering challenges. Below are the key challenges encountered during development and deployment, along with the solutions and learnings.

---

### 1️⃣ Docker Build Context & `.dockerignore` Issues

**Challenge:**  
Docker builds failed due to ignored model files or missing build context.

**Solution:**  
- Carefully configured `.dockerignore`
- Explicitly allowed inference models while excluding large training artifacts
- Added fail-fast validation inside Dockerfiles

**Learning:**  
> Docker context management is critical for ML workloads due to large artifacts.

---

### 2️⃣ Secure AWS Authentication Without Secrets on EC2

**Challenge:**  
AWS CLI failed on EC2 due to missing credentials during deployment.

**Solution:**  
- Used **IAM User** for GitHub Actions (CI)
- Used **IAM Role** for EC2 (CD)
- Eliminated AWS access keys on EC2 entirely

**Learning:**  
> IAM roles are the safest and recommended way to authenticate AWS services running on EC2.

---

### 3️⃣ Docker Compose Deployment Failures

**Challenge:**  
`docker compose` failed because `docker-compose.yml` was not present on EC2.

**Solution:**  
- CI/CD pipeline dynamically generates `docker-compose.yml` during deployment
- Removed dependency on manual EC2 setup or repository cloning

**Learning:**  
> Deployment pipelines should be **idempotent** and should not rely on server state.

---


## PART 1️⃣1️⃣ — Future Improvements

While this system is production-ready, the following improvements can further enhance scalability, security, and observability.

---

### 🚀 Infrastructure Improvements
- Migrate deployment from EC2 to **Amazon ECS or EKS**
- Add **Auto Scaling Groups** for high availability
- Introduce **Application Load Balancer (ALB)**

---

### 🔐 Security Enhancements
- Enable **HTTPS** using Nginx + Let’s Encrypt
- Restrict SSH access using IP whitelisting
- Apply stricter IAM least-privilege policies

---

### 📈 Monitoring & Observability
- Integrate **AWS CloudWatch** for logs and metrics
- Add application health checks
- Track inference latency and throughput


### 🧠 ML Improvements
- Extend to multi-class classification (additional cancer types)
- Add MLflow Model Registry integration
- Implement shadow deployment for new models
- Add data drift detection


⭐ Star this repository if it helped you!
