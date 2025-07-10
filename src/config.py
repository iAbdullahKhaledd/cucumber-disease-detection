# src/config.py
import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

class Config:
    # Safely get the API key from the environment
    ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

    # Paths (make them relative so they work on any machine)
    DATASET_DIR = "My-First-Project-1"
    DATASET_YAML = os.path.join(DATASET_DIR, "data.yaml")
    OUTPUT_DIR = "runs/detect"
    
    # Training Parameters
    EPOCHS = 60
    BATCH_SIZE = 16
    IMG_SIZE = 640
    PATIENCE = 20
    WORKERS = 8 # Adjusted for general use
    OPTIMIZER = 'SGD'