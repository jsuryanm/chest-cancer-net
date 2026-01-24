import os 
import urllib.request as request 
from pathlib import Path

import torch 
from torch import nn
from torchvision.models import vgg16,VGG16_Weights 

from src.cancer_clf.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self,config: PrepareBaseModelConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_base_model(self):
        
        weights = (VGG16_Weights.IMAGENET1K_V1
                   if self.config.params_weights == "imagenet"
                   else None)
        
        model = vgg16(weights=weights)

        self.model = model.to(self.device)

        self.save_model(path=self.config.base_model_path,
                        model=self.model)
    
    @staticmethod
    def _prepare_full_model(model,
                            classes,
                            freeze_all,
                            freeze_till):
        if freeze_all:
            for param in model.parameters():
                param.requires_grad = False 
            
        elif freeze_till is not None and freeze_till > 0:
            for param in model.features[:-freeze_till].parameters():
                param.requires_grad = False

        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features=in_features,
                                        out_features=classes)
        return model 
    
    def update_base_model(self):
        self.full_model = self._prepare_full_model(model=self.model,
                                                   classes=self.config.params_classes,
                                                   freeze_all=True,
                                                   freeze_till=None)
        
        self.full_model = self.full_model.to(self.device)
        self.save_model(
            path=self.config.updated_base_model_path,
            model=self.full_model
        )

    @staticmethod
    def save_model(path: Path,model: nn.Module):
        path.parent.mkdir(parents=True,exist_ok=True)
        torch.save(model.state_dict(),path)