import torch 
from torch import nn
from torchvision import models,transforms
from PIL import Image 
import os 

class PredictionPipeline:
    def __init__(self,filename: str | None):
        self.filename = filename 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456, 0.406],
                                 std=[0.229,0.224,0.225])
        ])

        self.model = self._load_model()

    def _load_model(self) -> nn.Module:
        model = models.resnet18(weights=None)

        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(in_features,128),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.6),
                                 nn.Linear(128,1))

        model_path = os.path.join("artifacts","training","model_best_hparams.pt")

        model.load_state_dict(torch.load(model_path,
                                         map_location=self.device,
                                         weights_only=True))
        
        model = model.to(self.device)
        model.eval()

        return model 
    
    def predict(self):
        image = Image.open(self.filename).convert("L")
        image = self.transform(image)
        image = image.unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(image).squeeze(1)
            prob_normal = torch.sigmoid(logits).item()
            pred = (prob_normal > 0.5)

            if pred:
                label = "Normal"
                confidence = prob_normal

            else:
                label = "Cancer (adenocarcinoma)"
                confidence = 1.0 - prob_normal 

        return [{"image":label,
                 "confidence":round(confidence,4)}]