# DistilBERT-fine-tuning-using-LoRA

Below is a high-level procedure for supervised model fine-tuning

1. Choose fine-tuning task (text classification, summarization, question answering)
2. Prepare training dataset create (100–10k) input-output pairs and preprocess data (tokenize, truncate, and pad text, etc).
3. Choose a base model (experiment with different models and choose one that performs best on the desired task).
4. Fine-tune model using PEFT (PEFT involves augmenting a base model with a relatively small number of trainable parameters.)
5. Evaluate model performance on unseen data
