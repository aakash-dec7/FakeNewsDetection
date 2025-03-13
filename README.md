# Fake News Detection

This repository contains an end-to-end pipeline for **Fake News Detection** using an `XGBoost` model, implemented with scikit-learn. The project follows a modular structure, ensuring enhanced readability, maintainability, and scalability.

## Key Features

- **XGBoost Model** for high-performance classification.

- **Data Version Control (DVC)** for managing and tracking data, model training, and evaluation pipelines.

- **Word2Vec Vectorizer** for efficient text embedding and feature extraction.

- **MLflow & DagsHub Integration** for experiment tracking, model registry, and version control.

- **Amazon Elastic Container Registry (ECR)** for storing and managing Docker container images.

- **Amazon Elastic Kubernetes Service (EKS)** for seamless deployment and scalability.

- **CI/CD Implementation** using GitHub Actions for automated testing, model building, and deployment.

## Prerequisites

Ensure the following dependencies and services are installed and configured:

- Python 3.10
- AWS Account
- AWS CLI
- Docker Desktop (for local image testing)
- DagsHub Account (for experiment tracking)
- Git & GitHub (for version control)

## Dataset

### Source

[LIAR Dataset](https://www.kaggle.com/datasets/doanquanvietnamca/liar-dataset/data)

### Description

The dataset consists of:

- **Statement**
- **Label**

## Model Architecture

The **Fake News Detection** model consists of the following key components:

1. **Text Preprocessing & Feature Extraction**
   - Tokenization, `stopword` removal, and text normalization.
   - `Word2Vec` embeddings for feature extraction.

2. **Model Selection & Training**
   - **Base Model**: `XGBoost` (XGBClassifier) is used for classification.
   - **Hyperparameter Optimization**: `GridSearchCV` is employed to identify the optimal hyperparameters.
   - **Cross-Validation**: `StratifiedKFold` ensures balanced class distribution during training.

3. **Performance Evaluation & Model Selection**
   - Evaluation metrics: `Accuracy`, `Precision`, `Recall`, `F1-score`, and `AUC-ROC`.
   - The **best-performing model** is selected based on `cross-validation` results.

4. **Model Serialization & Deployment**
   - The trained model is saved as `model.pth` for deployment.
   - **CI/CD** automates model deployment using `Docker` and `Kubernetes`.

## Installation & Setup

### Clone the Repository

```sh
git clone https://github.com/aakash-dec7/FakeNewsDetection.git
cd FakeNewsDetection
```

### Install Dependencies

```sh
pip install -r requirements.txt
```

### Initialize DVC for Pipeline Management

```sh
dvc init
```

## DVC Pipeline Stages

1. **Data Ingestion** - Fetches and stores the raw dataset.
2. **Data Validation** - Ensures data quality and integrity.
3. **Data Preprocessing** - Cleans, tokenizes, and prepares text data.
4. **Data Transformation** - Transforms text into numerical embeddings using Word2Vec.
5. **Model Training** - Trains the model with hyperparameter tuning and cross-validation.
6. **Model Evaluation** - Assesses model performance and selects the best version.

To execute the pipeline:

```sh
dvc repro
```

The trained model will be saved in:

```sh
artifacts/model/model.pkl
```

## Deployment

### Create an Amazon ECR Repository

Ensure that the Amazon ECR repository exists with the appropriate name as specified in `setup.py`:

```python
setup(
    name="fnd",
    version="1.0.0",
    author="Aakash Singh",
    author_email="aakash.dec7@gmail.com",
    packages=find_packages(),
)
```

### Create an Amazon EKS Cluster

Execute the following command to create an Amazon EKS cluster:

```sh
eksctl create cluster --name <cluster-name> \
    --region <region> \
    --nodegroup-name <nodegroup-name> \
    --nodes <number-of-nodes> \
    --nodes-min <nodes-min> \
    --nodes-max <nodes-max> \
    --node-type <node-type> \
    --managed
```

### Push Code to GitHub & Configure CI/CD

Before pushing the code, ensure that the necessary GitHub Actions secrets are added under **Settings > Secrets and Variables > Actions**:

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_DEFAULT_REGION`
- `ECR_REGISTRY_URI`

Push the code to GitHub:

```sh
git add .
git commit -m "Initial commit"
git push origin main
```

### CI/CD Automation

GitHub Actions will automate the CI/CD process, ensuring that the model is built, tested, and deployed to **Amazon EKS**.

## Accessing the Deployed Application

Once the deployment is successful:

1. Navigate to **EC2 Instances** in the AWS Console.
2. Under **Security Groups**, update inbound rules to allow public traffic.

Retrieve the external IP of the deployed Kubernetes service:

```sh
kubectl get svc
```

Copy the `EXTERNAL-IP` and access the application:

```text
http://<EXTERNAL-IP>:5000
```

The Fake News Detection application is now successfully deployed and accessible online.

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.
