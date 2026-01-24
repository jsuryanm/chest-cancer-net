import os 
import urllib.request as request 
from zipfile import ZipFile 
import time 
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt 
import seaborn as sns

import torch
from torch import nn 
from torch.utils.data import DataLoader,random_split,WeightedRandomSampler
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
        self.history = {"train_loss": [],
                        "val_loss": [],
                        "train_acc": [],
                        "val_acc": []
                        }

        
        self.train_accuracy = Accuracy(task="binary").to(self.device)

        self.val_accuracy = Accuracy(task="binary").to(self.device)
        self.val_precision = Precision(task="binary").to(self.device)
        self.val_recall = Recall(task="binary").to(self.device)
        self.val_f1 = F1Score(task="binary").to(self.device)
        self.confmat = ConfusionMatrix(task='binary').to(self.device)

    def load_model(self):
        self.model = models.resnet18(weights="IMAGENET1K_V1")

        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features,128),
            nn.ReLU(),
            nn.Dropout(p=0.6),
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

    def get_dataloaders(self):
        image_size = tuple(self.config.params_image_size)

        full_dataset = datasets.ImageFolder(self.config.training_data)

        val_size = int(0.4*len(full_dataset))
        train_size = len(full_dataset) - val_size

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

        train_ds,val_ds = random_split(full_dataset,
                                       [train_size,val_size],
                                       generator=generator)
        
        train_ds.dataset.transform = train_transform
        val_ds.dataset.transform = valid_transform

        train_indices = train_ds.indices
        targets = [full_dataset.samples[i][1] for i in train_indices]
        class_counts = Counter(targets)

        class_weights = {cls:1.0/count for cls,count in class_counts.items()}

        sample_weights = [class_weights[full_dataset.samples[i][1]] for i in train_indices]

        sampler = WeightedRandomSampler(weights=sample_weights,
                                        num_samples=len(train_indices),
                                        generator=generator,
                                        replacement=True)

        self.train_loader = DataLoader(train_ds,
                                       batch_size=self.config.params_batch_size,
                                       sampler=sampler,
                                       pin_memory=True)
        
        self.val_loader = DataLoader(val_ds,
                                     batch_size=self.config.params_batch_size,
                                     shuffle=False,
                                     pin_memory=True)
        
        logger.info(f"Total samples: {len(full_dataset)}")
        logger.info(f"Train samples: {len(train_ds)}")
        logger.info(f"Val samples: {len(val_ds)}")
        logger.info(f"Train batches:{len(self.train_loader)}")
        logger.info(f"Val batches:{len(self.val_loader)}")

        
    def train(self):
        best_val_loss = float("inf")
        patience = 10
        counter = 0


        loss_fn = nn.BCEWithLogitsLoss()

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.params_learning_rate,
            weight_decay=1e-4
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
                self.save_model(path=self.config.trained_model_path,model=self.model)
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

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )

        self._plot_confusion_matrix()
        self._plot_loss_curves()


    def _plot_confusion_matrix(self):
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
        plt.title("Confusion Matrix")

        save_path = self.config.root_dir / "confusion_matrix.png"
        plt.savefig(save_path)
        plt.close()

        logger.info(f"Confusion matrix saved at: {save_path}")

    def _plot_loss_curves(self):
        epochs = range(1, len(self.history["train_loss"]) + 1)

        plt.figure(figsize=(8, 6))
        plt.plot(epochs, self.history["train_loss"], label="Train Loss")
        plt.plot(epochs, self.history["val_loss"], label="Validation Loss")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.grid(True)

        save_path = self.config.root_dir / "loss_curve.png"
        plt.savefig(save_path)
        plt.close()

        logger.info(f"Loss curve saved at: {save_path}")


    @staticmethod
    def save_model(path: Path,
                   model: nn.Module):
        path.parent.mkdir(parents=True,exist_ok=True)
        torch.save(model.state_dict(),path)