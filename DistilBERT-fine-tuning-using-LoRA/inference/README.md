# SMS Finance - Text Classification with DistilBERT and LoRA

This project implements SMS text classification using DistilBERT fine-tuned with LoRA (Low-Rank Adaptation) for efficient training and inference.

## Project Overview

The system classifies SMS messages into 15 different categories:
- **Transportation**: Bus, Cab, Flight, Train
- **Services**: Appointment, Delivery, PickUp, Reservation
- **Financial**: Payment, Expiry
- **Entertainment**: Movie, Hotel
- **Communication**: ham (legitimate), info, spam

## Features

- **Efficient Fine-tuning**: Uses LoRA to reduce trainable parameters by ~99%
- **Multi-class Classification**: Handles 15 different SMS categories
- **Optimized Model**: Uses DistilBERT for faster inference
- **Comprehensive Evaluation**: Includes accuracy metrics and detailed classification reports

## Project Structure

```
final_v1/
├── DistilBERT-fine-tuning-using-LoRA/
│   ├── category_distilbert_LoRA.ipynb    # Main training notebook
│   └── README.md                         # Implementation details
├── data/                                 # Data directory (to be created)
├── models/                               # Saved models (to be created)
├── requirements.txt                      # Dependencies
└── README.md                            # This file
```

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Or install individually:
```bash
pip install datasets transformers accelerate evaluate peft torch pandas scikit-learn
```

## Usage

1. **Data Preparation**: 
   - Place your SMS dataset in the `data/` directory
   - Ensure it has 'Message' and 'Label' columns

2. **Training**:
   - Open `category_distilbert_LoRA.ipynb`
   - Update the data path to point to your dataset
   - Run the notebook cells sequentially

3. **Inference**:
   - Load the trained model using the provided code
   - Use the model for predictions on new SMS messages

## Model Configuration

- **Base Model**: `distilbert-base-uncased`
- **LoRA Configuration**:
  - Rank (r): 4
  - Alpha: 32
  - Dropout: 0.01
  - Target modules: Query layer only
- **Training Parameters**:
  - Learning rate: 2e-5
  - Batch size: 16
  - Epochs: 5
  - Max sequence length: 128

## Performance

- **Trainable Parameters**: ~639K (0.95% of total parameters)
- **Total Parameters**: ~67.6M
- **Memory Efficiency**: Significant reduction in memory usage
- **Training Speed**: Faster training due to fewer parameters

## Results

The model shows improved classification accuracy after fine-tuning:
- Before training: Random predictions
- After training: Meaningful category assignments
- Example: "OTP is 667778..." → classified as "spam"