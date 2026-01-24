import json 
from pathlib import Path 
from collections import Counter 

import optuna 
import torch 
from torch import nn 
from torch.utils.data import DataLoader,WeightedRandomSampler
from torchvision import datasets,models,transforms

from src.cancer_clf.entity.config_entity import HyperparameterTuningConfig
from src.cancer_clf.utils.seeds import set_seed 
from src.cancer_clf.logger.logger import logger

class HyperparameterTuning:
    def __init__(self,config: HyperparameterTuningConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        set_seed(self.config.params_seed)

    def _build_dataloaders(self,batch_size):
        image_size = tuple(self.config.params_image_size)

        train_tf = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        val_tf = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        train_ds = datasets.ImageFolder(self.config.train_data,
                                        transform=train_tf)
        
        val_ds = datasets.ImageFolder(self.config.val_data,
                                      transform=val_tf)

        generator = torch.Generator().manual_seed(self.config.params_seed)

        targets = [label for _, label in train_ds.samples]
        class_counts = Counter(targets)
        sample_weights = [1.0 / class_counts[label] for label in targets]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True
        )

        return train_loader, val_loader
    
    def _objective(self,trial: optuna.Trial):
        lr = trial.suggest_float("learning_rate",1e-5,5e-4,log=True)
        weight_decay = trial.suggest_float("weight_decay",1e-6,1e-3,log=True)
        dropout = trial.suggest_float("dropout",0.2,0.6)
        batch_size = trial.suggest_categorical("batch_size",self.config.batch_size_choices)

        model = models.resnet18(weights="IMAGENET1K_V1")

        for param in model.parameters():
            param.requires_grad = False 
        
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(in_features,128),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(128,1))
        
        model = model.to(self.device)

        train_loader,val_loader = self._build_dataloaders(batch_size=batch_size)
        
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(params=model.fc.parameters(),
                                      lr=lr,
                                      weight_decay=weight_decay)

        best_val_loss = float("inf")

        for epoch in range(self.config.max_epochs):
            model.train()

            for images,labels in train_loader:
                images = images.to(self.device) 
                labels = labels.float().to(self.device)

                optimizer.zero_grad()
                logits = model(images).squeeze(1)
                
                loss = loss_fn(logits,labels)
                loss.backward()
                optimizer.step()
            
            model.eval()
            val_loss = 0.0 

            with torch.no_grad():
                for images,labels in val_loader:
                    images = images.to(self.device)
                    labels = labels.float().to(self.device)

                    logits = model(images).squeeze(1)
                    loss = loss_fn(logits,labels)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            best_val_loss = min(best_val_loss,val_loss)

            trial.report(val_loss,step=epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return best_val_loss
    
    def run(self):
        output_path = Path("artifacts/hyperparameter_tuning/best_params.json")
        

        if output_path.exists():
            logger.info(
                f"Best hyperparameters already exist at {output_path}. "
                "Skipping hyperparameter tuning."
            )
            return
        
        logger.info("Starting hyperparameter tuning with Optuna")

        study = optuna.create_study(study_name=self.config.study_name,
                                    direction=self.config.direction)
        
        study.optimize(self._objective,n_trials=self.config.n_trials)

        logger.info("Hyperparameter tuning completed")
        logger.info(f"Best trial value: {study.best_value}")
        logger.info(f"Best model parameters:{study.best_params}")

        output_path = Path("artifacts/hyperparameter_tuning/best_params.json")
        output_path.parent.mkdir(parents=True,exist_ok=True)

        with open(output_path,"w") as f:
            json.dump(study.best_params,f,indent=4)
        
        logger.info(f"Best hyperparameters saved to:{output_path}")