#!/usr/bin/env python3
"""
SMS Text Classification using DistilBERT with LoRA
This script provides a command-line interface for training and evaluating the SMS classifier.
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    DataCollatorWithPadding,
    TrainingArguments, 
    Trainer
)
from peft import LoraConfig, get_peft_model
import evaluate

# Label mappings
LABEL_MAPPING = {
    'Appointment': 0, 'Bus': 1, 'Cab': 2, 'Delivery': 3, 'Expiry': 4,
    'Flight': 5, 'Hotel': 6, 'Movie': 7, 'Payment': 8, 'PickUp': 9,
    'Reservation': 10, 'Train': 11, 'ham': 12, 'info': 13, 'spam': 14
}

ID_TO_LABEL = {v: k for k, v in LABEL_MAPPING.items()}

class SMSClassifier:
    def __init__(self, model_name="distilbert-base-uncased", max_length=128):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.label_encoder = LabelEncoder()
        
    def load_data(self, data_path):
        """Load and preprocess the SMS dataset."""
        print(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        
        # Ensure required columns exist
        if 'Message' not in df.columns or 'Label' not in df.columns:
            raise ValueError("Dataset must contain 'Message' and 'Label' columns")
        
        # Convert messages to string
        df['Message'] = df['Message'].astype(str)
        
        # Encode labels
        df['label_encoding'] = self.label_encoder.fit_transform(df['Label'])
        
        # Remove duplicates and shuffle
        df = df.drop_duplicates(keep='first').sample(frac=1).reset_index(drop=True)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Label distribution:\n{df['Label'].value_counts()}")
        
        return df
    
    def prepare_datasets(self, df, test_size=0.2, random_state=42):
        """Split data into train and test sets."""
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state
        )
        
        # Convert to HuggingFace datasets
        train_dataset = Dataset.from_pandas(train_df)
        test_dataset = Dataset.from_pandas(test_df)
        
        return train_dataset, test_dataset
    
    def setup_tokenizer(self):
        """Initialize the tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            model_max_length=self.max_length,
            return_tensors="pt",
            add_prefix_space=True
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def preprocess_function(self, examples):
        """Tokenize and prepare data for training."""
        tokens = self.tokenizer(
            examples["Message"], 
            padding="max_length", 
            truncation=True
        )
        tokens["labels"] = examples["label_encoding"]
        return tokens
    
    def setup_model(self, num_labels):
        """Initialize the model with LoRA configuration."""
        # Load base model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
            id2label=ID_TO_LABEL,
            label2id=LABEL_MAPPING
        )
        
        # Configure LoRA
        peft_config = LoraConfig(
            task_type="SEQ_CLS",
            r=4,
            lora_alpha=32,
            lora_dropout=0.01,
            target_modules=['q_lin']
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics."""
        metric = evaluate.load("accuracy")
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    
    def train(self, train_dataset, eval_dataset, output_dir, **training_args):
        """Train the model."""
        # Tokenize datasets
        tokenized_train = train_dataset.map(self.preprocess_function, batched=True)
        tokenized_eval = eval_dataset.map(self.preprocess_function, batched=True)
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=5,
            weight_decay=0.01,
            eval_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            **training_args
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )
        
        # Train the model
        print("Starting training...")
        trainer.train()
        
        # Evaluate on test set
        print("Evaluating model...")
        results = trainer.evaluate()
        print(f"Test accuracy: {results['eval_accuracy']:.4f}")
        
        return trainer, results
    
    def predict(self, texts, model_path=None):
        """Make predictions on new texts."""
        if model_path:
            # Load trained model
            from peft import PeftModel, PeftConfig
            config = PeftConfig.from_pretrained(model_path)
            inference_model = AutoModelForSequenceClassification.from_pretrained(
                config.base_model_name_or_path,
                num_labels=len(LABEL_MAPPING),
                id2label=ID_TO_LABEL,
                label2id=LABEL_MAPPING
            )
            self.model = PeftModel.from_pretrained(inference_model, model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(
                    text, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=self.max_length
                )
                outputs = self.model(**inputs)
                pred = torch.argmax(outputs.logits, dim=1)
                predictions.append(ID_TO_LABEL[pred.item()])
        
        return predictions

def main():
    parser = argparse.ArgumentParser(description="Train SMS classifier with DistilBERT and LoRA")
    parser.add_argument("--data_path", required=True, help="Path to the CSV dataset")
    parser.add_argument("--output_dir", default="./models/sms_classifier", help="Output directory for model")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--predict", nargs='+', help="Texts to predict (for inference mode)")
    parser.add_argument("--model_path", help="Path to trained model (for inference mode)")
    
    args = parser.parse_args()
    
    # Initialize classifier
    classifier = SMSClassifier()
    
    if args.predict:
        # Inference mode
        if not args.model_path:
            print("Error: --model_path is required for inference mode")
            return
        
        predictions = classifier.predict(args.predict, args.model_path)
        for text, pred in zip(args.predict, predictions):
            print(f"Text: {text}")
            print(f"Prediction: {pred}")
            print("-" * 50)
    else:
        # Training mode
        # Load and prepare data
        df = classifier.load_data(args.data_path)
        train_dataset, test_dataset = classifier.prepare_datasets(df, args.test_size)
        
        # Setup tokenizer and model
        classifier.setup_tokenizer()
        classifier.setup_model(len(LABEL_MAPPING))
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Train the model
        trainer, results = classifier.train(
            train_dataset, 
            test_dataset, 
            args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        
        # Save the model
        trainer.save_model(args.output_dir)
        print(f"Model saved to {args.output_dir}")
        
        # Test predictions
        test_texts = [
            "OTP is 667778 for the txn of INR 3721.00 at True Value on your AXIS bank CREDIT card",
            "Your Tkt Cancelled. PNR, 12136726454, Amt 521 will be refunded in your account.",
            "Next Thursday at 9 pm"
        ]
        
        predictions = classifier.predict(test_texts)
        print("\nSample predictions:")
        for text, pred in zip(test_texts, predictions):
            print(f"Text: {text[:50]}...")
            print(f"Prediction: {pred}")
            print("-" * 50)

if __name__ == "__main__":
    main() 