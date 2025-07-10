# src/train.py
import os
import torch
from ultralytics import YOLO
from roboflow import Roboflow
from config import Config
from utils import setup_logging, apply_nms_gpu_workaround

# Setup logging
logger = setup_logging()

def download_dataset():
    """Downloads the dataset from Roboflow if it doesn't exist."""
    if not os.path.exists(Config.DATASET_DIR):
        logger.info("Downloading dataset from Roboflow...")
        if not Config.ROBOFLOW_API_KEY:
            logger.error("Roboflow API key not found! Set it in the .env file.")
            raise ValueError("Missing Roboflow API key")
        
        rf = Roboflow(api_key=Config.ROBOFLOW_API_KEY)
        project = rf.workspace("cucumber-bdhva").project("my-first-project-q8blq")
        version = project.version(1)
        dataset = version.download("yolov8")
        # Ensure the downloaded folder has the expected name
        os.rename(dataset.location, Config.DATASET_DIR)
        logger.info(f"Dataset downloaded to {Config.DATASET_DIR}")
    else:
        logger.info("Dataset already exists. Skipping download.")


class YOLOTrainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self.output_dir = Config.OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        self.model = None

    def train(self):
        try:
            logger.info("Starting model training...")
            self.model = YOLO('yolov8n.pt')
            self.model.train(
                data=Config.DATASET_YAML,
                epochs=Config.EPOCHS,
                imgsz=Config.IMG_SIZE,
                batch=Config.BATCH_SIZE,
                name='yolov8_cucumber_disease',
                project=self.output_dir,
                device=self.device,
                patience=Config.PATIENCE,
                workers=Config.WORKERS,
                optimizer=Config.OPTIMIZER,
                verbose=True
            )
            logger.info("Training completed successfully.")
            return self.model
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            raise

    # You can add evaluate, predict, and export methods here as well...

if __name__ == "__main__":
    try:
        # Apply workarounds if needed
        apply_nms_gpu_workaround()
        
        # Download data
        download_dataset()

        # Start training
        trainer = YOLOTrainer()
        trainer.train()
        
        logger.info("Main script finished successfully.")
    except Exception as e:
        logger.error(f"An error occurred in the main script: {e}", exc_info=True)