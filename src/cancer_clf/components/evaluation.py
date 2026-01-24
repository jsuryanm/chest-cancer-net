from pathlib import Path 
from urllib.parse import urlparse

import matplotlib.pyplot as plt 
import seaborn as sns  

from src.cancer_clf.entity.config_entity import EvaluationConfig
from src.cancer_clf.utils.common import save_json 

import torch
from torch import nn 
from torch.utils.data import DataLoader 
from torchvision import datasets,models,transforms
from torchmetrics import (Accuracy,
                          Precision,
                          Recall,
                          F1Score,
                          ConfusionMatrix)

from src.cancer_clf.entity.config_entity import EvaluationConfig
from src.cancer_clf.utils.common import save_json
from src.cancer_clf.logger.logger import logger 
from src.cancer_clf.utils.seeds import set_seed

import mlflow 
import mlflow.pytorch
import dagshub

dagshub.init(repo_owner='jsm.dgme', repo_name='chest-cancer-net', mlflow=True)

class Evaluation:
    def __init__(self,config: EvaluationConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        set_seed()

        self.accuracy = Accuracy(task="binary").to(self.device)
        self.precision = Precision(task="binary").to(self.device)
        self.recall = Recall(task="binary").to(self.device)
        self.f1 = F1Score(task='binary').to(self.device)
        self.confmat = ConfusionMatrix(task='binary').to(self.device)

    def load_model(self):
        # Loads the trained model
        self.model = models.resnet(weights=None)

        in_features = self.model.fc.in_features 

        self.model.fc = nn.Sequential(nn.Linear(in_features,128),
                                      nn.ReLU(),
                                      nn.Dropout(p=0.6),
                                      nn.Linear(128,1))
        
        self.model.load_state_dict(torch.load(self.config.path_of_model,
                                              map_location=self.device,
                                              weights_only=True))
        
        self.model = self.model.to(self.device)
        self.model.eval()

    def _create_valid_loader(self):
        # Create a validation dataloader 
        
        image_size = tuple(self.config.params_image_size)

        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            ])
        

        dataset = datasets.ImageFolder(root=self.config.training_data,
                                       transform=transform)
        
        generator = torch.Generator().manual_seed(self.config.params_seed)
        
        self.valid_loader = DataLoader(dataset=dataset,
                                       batch_size=self.config.params_batch_size,
                                       shuffle=False,
                                       pin_memory=True,
                                       generator=generator)
        
        logger.info(f"Evaluation samples:{len(dataset)}")
        logger.info(f"No of batches in val_loader:{len(self.valid_loader)}")

    def evaluation(self):
        self.load_model()
        self._create_valid_loader()

        loss_fn = nn.BCEWithLogitsLoss()
        total_loss = 0.0 

        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()
        self.confmat.reset()

        with torch.no_grad():
            for epoch in range(self.config.params_epochs):
                for images,labels in self.valid_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    logits = self.model(images).squeeze(1)
                    loss = loss_fn(logits,labels)
                    total_loss += loss.item()

                    probs = torch.sigmoid(logits)
                    preds = (probs > 0.5).int()

                    self.accuracy(preds,labels.int())
                    self.precision(preds,labels.int())
                    self.recall(preds,labels.int())
                    self.f1(preds,labels.int())
                    self.confmat(preds,labels.int())
                
            avg_loss = total_loss / len(self.valid_loader)

            self.scores = {"loss":avg_loss,
                        "accuracy":self.accuracy.compute().item(),
                        "precision":self.precision.compute().item(),
                        "recall":self.recall.compute().item(),
                        "f1_score":self.f1.compute().item()}
            
            logger.info(f"Evaluation metrics: {self.scores}")
    
    def save_score(self):
        save_json(Path("scores.json"),self.scores)
    
    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_type = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)