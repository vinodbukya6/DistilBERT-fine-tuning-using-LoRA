#!/usr/bin/env python3
"""
Simple inference script for SMS classification
Usage: python inference.py --model_path ./models/sms_classifier --text "Your SMS text here"
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig

# Label mappings
ID_TO_LABEL = {
    0: 'Appointment', 1: 'Bus', 2: 'Cab', 3: 'Delivery', 4: 'Expiry',
    5: 'Flight', 6: 'Hotel', 7: 'Movie', 8: 'Payment', 9: 'PickUp',
    10: 'Reservation', 11: 'Train', 12: 'ham', 13: 'info', 14: 'spam'
}

def load_model(model_path):
    """Load the trained PEFT model."""
    print(f"Loading model from {model_path}")
    
    # Load PEFT config
    config = PeftConfig.from_pretrained(model_path)
    
    # Load base model
    base_model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model_name_or_path,
        num_labels=15,
        id2label=ID_TO_LABEL
    )
    
    # Load PEFT model
    model = PeftModel.from_pretrained(base_model, model_path)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    
    return model, tokenizer

def predict_sms(model, tokenizer, text, max_length=128):
    """Predict the category of an SMS message."""
    model.eval()
    
    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1)
        confidence = torch.softmax(outputs.logits, dim=1).max().item()
    
    predicted_label = ID_TO_LABEL[predictions.item()]
    
    return predicted_label, confidence

def main():
    parser = argparse.ArgumentParser(description="SMS Classification Inference")
    parser.add_argument("--model_path", required=True, help="Path to the trained model")
    parser.add_argument("--text", required=True, help="SMS text to classify")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(args.model_path)
    
    if args.interactive:
        print("Interactive SMS Classifier")
        print("Type 'quit' to exit")
        print("-" * 50)
        
        while True:
            text = input("Enter SMS text: ").strip()
            if text.lower() == 'quit':
                break
            
            if text:
                label, confidence = predict_sms(model, tokenizer, text)
                print(f"Category: {label}")
                print(f"Confidence: {confidence:.2%}")
                print("-" * 50)
    else:
        # Single prediction
        label, confidence = predict_sms(model, tokenizer, args.text)
        print(f"Text: {args.text}")
        print(f"Category: {label}")
        print(f"Confidence: {confidence:.2%}")

if __name__ == "__main__":
    main() 