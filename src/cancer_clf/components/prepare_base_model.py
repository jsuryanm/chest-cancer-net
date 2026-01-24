import os 
import urllib.request as request 
from pathlib import Path

import torch 
from torch import nn
from torchvision.models import resnet18,ResNet18_Weights 

from src.cancer_clf.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self,config: PrepareBaseModelConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_base_model(self):
        
        weights = (ResNet18_Weights.IMAGENET1K_V1
                   if self.config.params_weights == "imagenet"
                   else None)
        
        model = resnet18(weights=weights)

        self.model = model.to(self.device)

        self.save_model(path=self.config.base_model_path,
                        model=self.model)
    
    @staticmethod
    def _prepare_full_model(
        model: nn.Module,
        classes: int,
        freeze_all: bool,
    ):
        # Freeze entire backbone
        if freeze_all:
            for param in model.parameters():
                param.requires_grad = False

        # Replace classifier head (ResNet uses `fc`)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.Linear(128, classes),
        )

        return model

    
    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,   # freeze backbone for small dataset
        )

        self.full_model = self.full_model.to(self.device)

        self.save_model(
            path=self.config.updated_base_model_path,
            model=self.full_model
        )


    @staticmethod
    def save_model(path: Path,model: nn.Module):
        path.parent.mkdir(parents=True,exist_ok=True)
        torch.save(model.state_dict(),path)