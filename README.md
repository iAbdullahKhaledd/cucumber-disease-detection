# YOLOv8 Cucumber Disease Detection

This project is a collaborative effort to build an end-to-end computer vision pipeline for detecting diseases in cucumbers using YOLOv8.

## Team & Contributions

This project was brought to life by:

*   **[Abdullah Khaled]** ([@iAbdullahKhaledd](https://github.com/iAbdullahKhaledd))

*   **Yousef Kamel** ([@YousefKamell](https://github.com/YousefKamell))

## Features

-   **Model:** YOLOv8n
-   **Dataset:** A custom-built dataset sourced and annotated on Roboflow, containing 8 classes of cucumber health status (e.g., Anthracnose, Powdery Mildew, Healthy).
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