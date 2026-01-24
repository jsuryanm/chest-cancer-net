import os 

from src.cancer_clf.constants.constant import * 
from src.cancer_clf.utils.common import read_yaml,create_directories
from src.cancer_clf.entity.config_entity import (DataIngestionConfig,
                                                 PrepareBaseModelConfig,
                                                 HyperparameterTuningConfig,
                                                 TrainingConfig,
                                                 EvaluationConfig)


class ConfigurationManager:
    def __init__(self,
                 config_filepath=CONFIG_FILE_PATH,
                 params_filepath=PARAMS_FILE_PATH):
        
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        
        create_directories([self.config.artifacts_root])

    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        
        data_ingestion_config = DataIngestionConfig(root_dir=Path(config.root_dir),
                                                    source_URL=config.source_URL,
                                                    local_data_file=Path(config.local_data_file),
                                                    unzip_dir=Path(config.unzip_dir),
                                                    dataset_dir=Path(config.dataset_dir),
                                                    train_dir=Path(config.train_dir),
                                                    val_dir=Path(config.val_dir),
                                                    test_dir=Path(config.test_dir))
        return data_ingestion_config 
    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        
        config = self.config.prepare_base_model
        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(root_dir=Path(config.root_dir),
                                                           base_model_path=Path(config.base_model_path),
                                                           updated_base_model_path=Path(config.updated_base_model_path),
                                                           params_image_size=self.params.IMAGE_SIZE,
                                                           params_learning_rate=self.params.LEARNING_RATE,
                                                           params_include_top=self.params.INCLUDE_TOP,
                                                           params_weights=self.params.WEIGHTS,
                                                           params_classes=self.params.CLASSES)
        return prepare_base_model_config
    
    def get_hyperparameter_tuning_config(self) -> HyperparameterTuningConfig:
        config = self.config.hyperparameter_tuning
        ingestion = self.config.data_ingestion

        tuning_config = HyperparameterTuningConfig(train_data=Path(ingestion.train_dir),
                                                   val_data=Path(ingestion.val_dir),
                                                   params_image_size=self.params.IMAGE_SIZE,
                                                   params_seed=self.params.SEED,
                                                   n_trials = config.n_trials,
                                                   max_epochs=config.n_trials,
                                                   study_name=config.study_name,
                                                   direction=config.direction,
                                                   batch_size_choices=config.batch_size_choices)
        return tuning_config
    
    def get_training_config(self) -> TrainingConfig:
        training = self.config.training 
        prepare_base_model = self.config.prepare_base_model 
        params = self.params 
        ingestion = self.config.data_ingestion
        
        create_directories([Path(training.root_dir)])

        training_config = TrainingConfig(root_dir=Path(training.root_dir),
                                         updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
                                         best_model_path=Path(training.best_model_path),
                                         train_data=Path(ingestion.train_dir),
                                         val_data=Path(ingestion.val_dir),
                                         params_epochs=params.EPOCHS,
                                         params_batch_size=params.BATCH_SIZE,
                                         params_is_augmentation=params.AUGMENTATION,
                                         params_image_size=params.IMAGE_SIZE,
                                         params_seed=params.SEED,
                                         params_learning_rate=params.LEARNING_RATE,
                                         params_classes=params.CLASSES)
        return training_config
    
    def get_evaluation_config(self) -> EvaluationConfig:
        eval_config = EvaluationConfig(path_of_model="artifacts/training/model_best_hparams.pt",
                                       test_data="artifacts/data_ingestion/Chest-CT-Scan-data/test",
                                       mlflow_uri="https://dagshub.com/jsm.dgme/chest-cancer-net.mlflow",
                                       all_params=self.params,
                                       params_image_size=self.params.IMAGE_SIZE,
                                       params_batch_size=self.params.BATCH_SIZE,
                                       params_seed=self.params.SEED,
                                       params_epochs=self.params.EPOCHS)
        
        return eval_config