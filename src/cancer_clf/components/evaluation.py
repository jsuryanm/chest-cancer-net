from pathlib import Path
from urllib.parse import urlparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix

import mlflow
import mlflow.pytorch
import dagshub

from src.cancer_clf.entity.config_entity import EvaluationConfig
from src.cancer_clf.utils.common import save_json
from src.cancer_clf.logger.logger import logger
from src.cancer_clf.utils.seeds import set_seed


# DAGsHub handles MLflow tracking
dagshub.init(
    repo_owner="jsm.dgme",
    repo_name="chest-cancer-net",
    mlflow=True,
)


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        set_seed(self.config.params_seed)

        self.accuracy = Accuracy(task="binary").to(self.device)
        self.precision = Precision(task="binary").to(self.device)
        self.recall = Recall(task="binary").to(self.device)
        self.f1 = F1Score(task="binary").to(self.device)
        self.confmat = ConfusionMatrix(task="binary").to(self.device)

    def load_model(self):
        self.model = models.resnet18(weights=None)

        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.Linear(128, 1),
        )

        self.model.load_state_dict(
            torch.load(
                self.config.path_of_model,
                map_location=self.device,
                weights_only=True,
            )
        )

        self.model = self.model.to(self.device)
        self.model.eval()

    def _create_test_loader(self):
        image_size = tuple(self.config.params_image_size)

        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        dataset = datasets.ImageFolder(
            root=self.config.test_data,
            transform=transform,
        )

        self.test_loader = DataLoader(
            dataset,
            batch_size=self.config.params_batch_size,
            shuffle=False,
            pin_memory=True,
        )

        logger.info(f"Testing samples: {len(dataset)}")
        logger.info(f"Test batches: {len(self.test_loader)}")
    

    def evaluation(self):
        # ---- MLflow setup ----
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_experiment("chest-cancer-net")

        with mlflow.start_run(run_name="evaluation"):
            # ---- Load + eval ----
            self.load_model()
            self._create_test_loader()

            loss_fn = nn.BCEWithLogitsLoss()
            total_loss = 0.0

            self.accuracy.reset()
            self.precision.reset()
            self.recall.reset()
            self.f1.reset()
            self.confmat.reset()

            with torch.no_grad():
                for images, labels in self.test_loader:
                    images = images.to(self.device)
                    labels = labels.float().to(self.device)

                    logits = self.model(images).squeeze(1)
                    loss = loss_fn(logits, labels)
                    total_loss += loss.item()

                    preds = (torch.sigmoid(logits) > 0.5).int()

                    self.accuracy(preds, labels.int())
                    self.precision(preds, labels.int())
                    self.recall(preds, labels.int())
                    self.f1(preds, labels.int())
                    self.confmat(preds, labels.int())

            avg_loss = total_loss / len(self.test_loader)

            self.test_scores = {
                "loss": avg_loss,
                "accuracy": self.accuracy.compute().item(),
                "precision": self.precision.compute().item(),
                "recall": self.recall.compute().item(),
                "f1_score": self.f1.compute().item(),
            }

            logger.info(f"Final test metrics: {self.test_scores}")

            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(self.test_scores)

            mlflow.pytorch.log_model(
                pytorch_model=self.model,
                name="model",
                registered_model_name="ChestCancerNet"
            )

            save_json(Path("test_scores.json"), self.test_scores)
