# Quick Start Guide - SMS Classifier

This guide will help you get started with the SMS classification system in minutes.

## Prerequisites

1. **Python 3.8+** installed
2. **pip** package manager
3. **Git** (optional, for cloning)

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Verify installation:**
```bash
python -c "import torch, transformers, peft; print('All packages installed successfully!')"
```

## Quick Test with Sample Data

1. **Train the model with sample data:**
```bash
python train_sms_classifier.py --data_path data/sample_sms_data.csv --output_dir models/test_model
```

2. **Test inference:**
```bash
python inference.py --model_path models/test_model --text "Your OTP is 123456 for transaction"
```

3. **Interactive mode:**
```bash
python inference.py --model_path models/test_model --interactive
```

## Using Your Own Data

1. **Prepare your dataset:**
   - Create a CSV file with columns: `Message`, `Label`
   - Ensure your labels match the predefined categories (see README.md)

2. **Train the model:**
```bash
python train_sms_classifier.py --data_path your_data.csv --output_dir models/your_model
```

3. **Customize training parameters:**
```bash
python train_sms_classifier.py \
    --data_path your_data.csv \
    --output_dir models/your_model \
    --epochs 10 \
    --batch_size 32 \
    --learning_rate 1e-5
```

## Using the Jupyter Notebook

1. **Start Jupyter:**
```bash
jupyter notebook
```

2. **Open the notebook:**
   - Navigate to `DistilBERT-fine-tuning-using-LoRA/category_distilbert_LoRA.ipynb`
   - Run cells sequentially

## Model Performance

- **Memory efficient:** Uses LoRA to reduce trainable parameters by ~99%
- **Fast training:** Optimized for quick fine-tuning
- **Accurate:** Multi-class classification for 15 SMS categories

## Troubleshooting

### Common Issues:

1. **CUDA out of memory:**
   - Reduce batch size: `--batch_size 8`
   - Use CPU: Add `--device cpu` (if supported)

2. **Import errors:**
   - Ensure all packages are installed: `pip install -r requirements.txt`
   - Check Python version: `python --version`

3. **Model loading errors:**
   - Verify model path exists
   - Check if model was saved correctly

### Getting Help:

- Check the main README.md for detailed documentation
- Review the Jupyter notebook for implementation details
- Ensure your data format matches the expected structure

## Next Steps

1. **Experiment with different models:**
   - Try different base models in the configuration
   - Adjust LoRA parameters for your use case

2. **Improve performance:**
   - Add more training data
   - Experiment with hyperparameters
   - Use data augmentation techniques

3. **Deploy the model:**
   - Create a web API using Flask/FastAPI
   - Integrate with your SMS processing pipeline
   - Set up automated retraining

## Example Workflow

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train model
python train_sms_classifier.py --data_path data/sample_sms_data.csv --output_dir models/my_model

# 3. Test predictions
python inference.py --model_path models/my_model --text "Meeting at 3 PM tomorrow"

# 4. Interactive testing
python inference.py --model_path models/my_model --interactive
```

That's it! You're now ready to classify SMS messages with your custom-trained model. 