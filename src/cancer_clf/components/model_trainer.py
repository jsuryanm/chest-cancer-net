import os 
import urllib.request as request 
from zipfile import ZipFile 
import time 
from pathlib import Path
from collections import Counter
import json

import matplotlib.pyplot as plt 
import seaborn as sns

import torch
from torch import nn 
from torch.utils.data import DataLoader,WeightedRandomSampler
from torchvision import datasets,transforms,models

from torchmetrics import (Accuracy,
                          Precision,
                          Recall,
                          F1Score,
                          ConfusionMatrix)

from src.cancer_clf.entity.config_entity import TrainingConfig
from src.cancer_clf.utils.seeds import set_seed
from src.cancer_clf.logger.logger import logger

class Training:
    def __init__(self,config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        set_seed(self.config.params_seed)
        

        
        self.train_accuracy = Accuracy(task="binary").to(self.device)

        self.val_accuracy = Accuracy(task="binary").to(self.device)
        self.val_precision = Precision(task="binary").to(self.device)
        self.val_recall = Recall(task="binary").to(self.device)
        self.val_f1 = F1Score(task="binary").to(self.device)
        self.confmat = ConfusionMatrix(task='binary').to(self.device)

    def _load_best_hyperparameters(self):
        path = Path("artifacts/hyperparameter_tuning/best_params.json")

        if not path.exists():
            logger.info("No best_params.json found. Using default hyperparameters.")
            return None 
        
        logger.info(f"Loading best hyperparameters from {path}")
        with open(path,"r") as f:
            return json.load(f)


    def load_model(self,dropout: float):
        self.model = models.resnet18(weights="IMAGENET1K_V1")
        for param in self.model.parameters():
            param.requires_grad = False


        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features,128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128,1)
        )

        self.model.load_state_dict(
            torch.load(
                self.config.updated_base_model_path,
                map_location=self.device,
                weights_only=True
            ),
            strict=False
        )

        self.model = self.model.to(self.device)

    def get_dataloaders(self,batch_size: int):
        image_size = tuple(self.config.params_image_size)

        valid_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])
        ])

        if self.config.params_is_augmentation:

            train_transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomRotation(10),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406],
                                     std=[0.229,0.224,0.225])
            ])
        
        else:
            train_transform = valid_transform

        generator = torch.Generator().manual_seed(self.config.params_seed)

        train_ds = datasets.ImageFolder(root=self.config.train_data,
                                        transform=train_transform)
        val_ds = datasets.ImageFolder(root=self.config.val_data,
                                      transform=valid_transform)
        

        targets = [labels for _,labels in train_ds.samples]
        class_counts = Counter(targets)

        class_weights = {cls:1.0/count for cls,count in class_counts.items()}

        sample_weights = [class_weights[label] for label in targets]

        sampler = WeightedRandomSampler(weights=sample_weights,
                                        num_samples=len(sample_weights),
                                        generator=generator,
                                        replacement=True)

        self.train_loader = DataLoader(train_ds,
                                       batch_size=batch_size,
                                       sampler=sampler,
                                       pin_memory=True)
        
        self.val_loader = DataLoader(val_ds,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     pin_memory=True)
        
        logger.info(f"Train samples: {len(train_ds)}")
        logger.info(f"Val samples: {len(val_ds)}")
        logger.info(f"Train batches:{len(self.train_loader)}")
        logger.info(f"Val batches:{len(self.val_loader)}")

        
    def train(self):
        self.history = {"train_loss": [],
                        "val_loss": [],
                        "train_acc": [],
                        "val_acc": []
                        }
        
        best_params = self._load_best_hyperparameters()

        if best_params:
            lr = best_params["learning_rate"]
            batch_size = best_params["batch_size"]
            dropout = best_params["dropout"]
            weight_decay = best_params["weight_decay"]
            suffix = "best"
            logger.info("Starting model training with the best combination hyperparameters")
        
        else:
            lr = self.config.params_learning_rate
            batch_size = self.config.params_batch_size
            dropout = 0.6 
            weight_decay = 5e-4
            suffix = "default"
            logger.info("Training with default hyperparameters")

        best_val_loss = float("inf")
        patience = 10
        counter = 0

        self.load_model(dropout=dropout)
        self.get_dataloaders(batch_size=batch_size)


        loss_fn = nn.BCEWithLogitsLoss()

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                               mode="min",
                                                               factor=0.3,
                                                               patience=2)

        for epoch in range(self.config.params_epochs):

            self.model.train()
            train_loss = 0.0

            self.train_accuracy.reset()

            for images, labels in self.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                logits = self.model(images).squeeze(1)
                labels = labels.float()
                loss = loss_fn(logits, labels)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).int()
                
                self.train_accuracy(preds, labels.int())

            train_loss /= len(self.train_loader)
            train_acc = self.train_accuracy.compute().item()

            self.model.eval()
            val_loss = 0.0

            self.val_accuracy.reset()
            self.val_precision.reset()
            self.val_recall.reset()
            self.val_f1.reset()
            self.confmat.reset()

            with torch.no_grad():
                for images, labels in self.val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    logits = self.model(images).squeeze(1)
                    labels = labels.float()

                    loss = loss_fn(logits, labels)

                    val_loss += loss.item()
                
                    probs = torch.sigmoid(logits)
                    preds = (probs > 0.5).int()

                    self.val_accuracy(preds, labels.int())
                    self.val_precision(preds, labels.int())
                    self.val_recall(preds, labels.int())
                    self.val_f1(preds, labels.int())
                    self.confmat(preds,labels.int())


            val_loss /= len(self.val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0 
                model_path = self.config.trained_model_path.with_name(f"model_{suffix}.pt")
                self.save_model(path=model_path,
                                model=self.model)
                logger.info(f"Saving model at {epoch + 1} with train_loss:{train_loss:.4f} and val_loss:{val_loss:.4f}")
            else:
                counter += 1 
                if counter >= patience:
                    logger.info("Early stopping triggered")
                    break 

            val_acc = self.val_accuracy.compute().item()
            precision = self.val_precision.compute().item()
            recall = self.val_recall.compute().item()
            f1 = self.val_f1.compute().item()

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)

            scheduler.step(val_loss)

            if (epoch + 1) % 5 == 0:
                logger.info(
                    f"Epoch [{epoch+1}/{self.config.params_epochs}] | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Train Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_acc:.4f} | "
                    f"Val Precision: {precision:.4f} | "
                    f"Val Recall: {recall:.4f} | "
                    f"Val F1: {f1:.4f}"
                )

        # model_path = self.config.trained_model_path.with_name(f"model_{suffix}.pt")
        # self.save_model(path=model_path, model=self.model)


        self._plot_confusion_matrix(suffix)
        self._plot_loss_curves(suffix)


    def _plot_confusion_matrix(self,suffix: str):
        cm = self.confmat.compute().cpu().numpy()

        plt.figure(figsize=(6, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Normal", "Cancer"],
            yticklabels=["Normal", "Cancer"],
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix ({suffix})")

        save_path = self.config.root_dir / f"confusion_matrix_{suffix}.png"
        plt.savefig(save_path)
        plt.close()

        logger.info(f"Confusion matrix saved at: {save_path}")

    def _plot_loss_curves(self,suffix: str):
        epochs = range(1, len(self.history["train_loss"]) + 1)

        plt.figure(figsize=(8, 6))
        plt.plot(epochs, self.history["train_loss"], label="Train Loss")
        plt.plot(epochs, self.history["val_loss"], label="Validation Loss")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Training vs Validation Loss ({suffix})")
        plt.legend()
        plt.grid(True)

        save_path = self.config.root_dir / f"loss_curve_{suffix}.png"
        plt.savefig(save_path)
        plt.close()

        logger.info(f"Loss curve saved at: {save_path}")


    @staticmethod
    def save_model(path: Path,
                   model: nn.Module):
        path.parent.mkdir(parents=True,exist_ok=True)
        torch.save(model.state_dict(),path)