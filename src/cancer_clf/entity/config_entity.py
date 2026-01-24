from dataclasses import dataclass 
from pathlib import Path 

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path 

    dataset_dir: Path 
    train_dir: Path
    val_dir: Path
    test_dir: Path

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_learning_rate: float 
    params_include_top: bool
    params_weights: str
    params_classes: int

@dataclass(frozen=True)
class HyperparameterTuningConfig:
    train_data: Path 
    val_data: Path
    params_image_size: list
    params_image_size: int 
    params_seed: int 
    n_trials: int
    max_epochs: int
    study_name: str
    direction: str
    batch_size_choices: list[int]

@dataclass(frozen=True)
class TrainingConfig: 
    root_dir: Path
    best_model_path: Path
    updated_base_model_path: Path
    train_data: Path
    val_data: Path
    params_epochs: int 
    params_batch_size: int 
    params_learning_rate: float
    params_is_augmentation: bool 
    params_image_size: list
    params_seed: int 
    params_classes: int

@dataclass 
class EvaluationConfig:
    path_of_model: Path
    test_data: Path 
    all_params: dict 
    mlflow_uri: str
    params_image_size: list
    params_epochs: int 
    params_batch_size: int
    params_seed: int
