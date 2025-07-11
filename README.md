# YOLOv8 Cucumber Disease Detection

This project is a collaborative effort to build an end-to-end computer vision pipeline for detecting diseases in cucumbers using YOLOv8.

## Team & Contributions

This project was brought to life by:

*   **Abdullah Khaled** ([@iAbdullahKhaledd](https://github.com/iAbdullahKhaledd))

*   **Yousef Kamel** ([@YousefKamell](https://github.com/YousefKamell))

*   **Moaz Nasser** ([@Moaznasser](https://github.com/Moaznasser))

## Features

-   **Model:** YOLOv8n
-   **Dataset:** A custom-built dataset sourced and annotated on Roboflow, containing 8 classes of cucumber health status (e.g., Anthracnose, Powdery Mildew, Healthy).[https://universe.roboflow.com/cucumber-bdhva/cucmber-healthy-diseases]
-   **End-to-End Pipeline:** Includes scripts for data downloading, training, and evaluation.
-   **Structured Code:** The project is organized into a `src` directory with classes and utility functions for clean, reusable code.
-   **Dependency Management:** Uses a `requirements.txt` file for easy environment setup.
-   **Secure API Key Handling:** Uses a `.env` file to manage the Roboflow API key securely.

## Project Structure

```
cucumber-disease-detection/
├── .gitignore
├── README.md
├── requirements.txt
├── notebooks/
│   └── YOLOv8AABB_Cucumber.ipynb
├── src/
│   ├── config.py
│   ├── train.py
│   └── utils.py
└── .env.example
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/iAbdullahKhaledd/cucumber-disease-detection.git
    cd cucumber-disease-detection
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your environment variables:**
    -   Copy the example file: `cp .env.example .env`
    -   Open the `.env` file and add your Roboflow API key.

## Usage

To train the model, run the main training script:

```bash
python src/train.py
```

The training logs will be saved to a `.log` file, and the model weights and results will be saved in the `runs/detect/` directory.

## Results

The model was trained for 60 epochs and achieved the following performance on the validation set:

-   **mAP@0.5-0.95:** 0.723
-   **mAP@0.5:** 0.939
-   **Precision:** 0.937
-   **Recall:** 0.904

### Training Performance Graphs

The following graphs show the model's performance improving over the training epochs. The loss (box_loss, cls_loss, dfl_loss) decreases while the performance metrics (Precision, Recall, mAP) increase, indicating successful learning.

![Training Results](https://github.com/user-attachments/assets/c7d9ffa0-0e41-4d76-9531-6d6022b24e70)

### Confusion Matrix

The normalized confusion matrix below shows the model's performance for each class. It helps visualize where the model excels and where it might be confused. The strong diagonal line indicates high accuracy across all classes.

![Normalized Confusion Matrix](https://github.com/user-attachments/assets/30cdbcdd-7f77-416a-83cd-d7acbd61b580)

### Precision-Recall Curve

The PR curve shows the tradeoff between precision and recall for different thresholds. A high area under the curve (AUC) represents both high recall and high precision, which is a sign of a high-accuracy model.

![PR Curve](https://github.com/user-attachments/assets/f0f59eeb-47ce-4649-9fe7-ae5606a95f3d)

### Sample Model Predictions

Here is an example of the model's predictions on a batch of validation images, showing its ability to detect different diseases in real-world scenarios.

![Sample Predictions](https://github.com/user-attachments/assets/12de729d-99a8-45dc-b427-d2f73234f67c)
