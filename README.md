# LLMs-fine-tuning-LoRA-and-QLoRA

# 1. DistilBERT-fine-tuning-using-LoRA
Below is a high-level procedure for supervised model fine-tuning

Choose fine-tuning task (text classification, summarization, question answering)
Prepare training dataset create (100–10k) input-output pairs and preprocess data (tokenize, truncate, and pad text, etc).
Choose a base model (experiment with different models and choose one that performs best on the desired task).
Fine-tune model using PEFT (PEFT involves augmenting a base model with a relatively small number of trainable parameters.)
Evaluate model performance on unseen data

# 2. Fine-tuning Mistral-7b-Instruct using QLoRA to Respond to YouTube Comments
Quantized Low-Rank Adaptation Steps:

4-bit NormalFloat - encodes numbers with just 4 bits
Double Quantization - quantizing the quantization constants, the constants generated from block-wise quantization aproach from input tensors
Paged optimizers - It transfers pages of memory from the GPU to the CPU when the GPU hits its limits.
LoRA (Parameter Efficient Fine-tuning method) - adds a relatively small number of trainable parameters while keeping the original parameters fixed.

# 3. Improving Fine-tuned Model using RAG Pipeline & LLama Index
Stage 1 Document Preparation & Indexing: Load Docs -> Chunk Docs(LLMs have limited context window) -> Embed Chunks(word-embeddings) -> Load into VectorDB(VectorStoreIndex).
Stage 2 Query Processing & Docs Retrieval: Input Query -> Retrieve Relevant Docs from VectorDB -> Prompt Template with Context.
Stage 3 Response Generation: Prompt + Query -> Tokenizer -> LLM Inference -> Generate Response
