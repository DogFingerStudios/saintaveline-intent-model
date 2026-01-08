# Saint Aveline Intent Model

This repository contains the code for training and evaluating an intent classification model for the Saint Aveline project. This model is designed to take user commands and classify them into predefined intents to facilitate appropriate responses.

The model is trained in Python, but is consumed by C# and Unity.

## Requirements

- Python 3.11.x is being used for development

## Setup
#### 1. Create and activate a virtual environment:
```bash
$> python -m venv venv
$> source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

#### 2. Install the required packages:
```bash
$> pip install -r requirements.txt
```

## Train the Model
```bash
$> python train.py
```

## Evaluate the Model
```bash
$> python test.py
```

## Build for C#
```bash
$> python python train.py   # outputs intent_model.pt
$> python export_model.py   # converts intent_model.pt to intent_model.onnx
$> python export_vcab_json.py   # exports the vocabulary to a JSON file
```