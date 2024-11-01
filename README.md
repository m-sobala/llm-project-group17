# Large Language Models Project Group 17

## Contents
- [Overview](#overview)
- [Installation](#installation)
- [Model Evaluation](#model-evaluation)
- [Code](#code)

## Overview
This repository hosts the code for the Large Language Models Project of Group 17. It includes code for data preprocessing, model fine-tuning, context-enhanced prompting, and model evaluation.

## Installation

1. **Open terminal**.

2. **Create and activate a Python virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
    ```

3. **Install required packages**
    ```bash
    pip3 install -r requirements.txt
    ```

## Model Evaluation

1. **Execute the evaluation script**
    ```bash
    python3 evaluate_models.py
    ```

2. **Evaluation metrics** are available in the terminal once the script executed. A plot is generated, and can be found in the root directory under the name ```'bleu_scores_plot.png'```.

## Code
All code is implemeneted in the ```src``` directory. The directory contains 5 files:
- config - constants and configurations shared across the entire codebase
- context - code for context generation
- evaluation - code used for evaluation of model performance
- fine_tuning - code used for fine-tuning the models using Full Parameter Fine-Tuning and LoRA
- utils - helper methods used across the entire codebase
