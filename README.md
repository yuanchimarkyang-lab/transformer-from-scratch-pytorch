# Mini-project: transformer implementation from scratch in pytorch
## Project Overview
This mini-project implements an encoder–decoder Transformer from scratch and trains the model on the [Multi30K German-to-English translation task](https://huggingface.co/datasets/bentrevett/multi30k), achieving a corpus BLEU score above 30 on the held-out test set after 50 epochs.

The goal of this mini-project is to build hands-on experience with the key components of the Transformer architecture. 

## Architecture Implemented
- scaled dot-product attention
- multi-head attention
- encoder/decoder
- positional encoding
- causal and padding masks
- autoregressive decoding

Note that these are implemented without nn.Transformer / nn.MultiheadAttention.

## Dataset and Training
- Dataset: [Multi30K German-to-English translation task](https://huggingface.co/datasets/bentrevett/multi30k)
    - Train: 29,000 German-English sentence pairs
    - Validation: 1,014
    - Test: 1,000
- Architecture: Encoder-Decoder Transformer
    - Encoder Blocks: 3
    - Decoder Blocks: 3
    - Embedding Size: 256
    - Number of Attention Head: 8
    - Inner Dimentions of Feed-Forward Networks: 512
    - Maximum Sequence Length: 100
- Tokenizer: Google's SentencePiece tokenizer, trained on the training set.
- Dictionary Size: 8,000 for both English and German
- Optimizer: Adam
    - Learning rate: 1e-1 initially, scheduled to reduce on plateau with factor 0.5, patience 5, and minimum learning rate 1e-4
- Batch size: 64
- Training duration: 100 epochs
- Loss function: CrossEntropyLoss
- Evaluation metric: BLEU score


## Results



## Notes


## Running the project

This project uses [uv] for dependency management.

```bash
git clone <repository-url>
cd transformer-from-scratch-pytorch
uv sync
```

Prepare the training data
```bash
uv run python preapare_data.py
```

Train the model
```bash
uv run python train.py
```

Evaluate the saved checkpoint
```bash
uv run python evaluate.py
```

Plot the training trajectory
```bash
uv run python plot_training_trajectory.py
```



## Repository structure
```text
transformer-from-scratch-pytorch/
├── transformer/
│   ├── config.py
│   ├── constants.py
│   ├── data.py
│   ├── model.py
├── result/
│   └── baseline/
├── prepare_data.py
├── train.py
├── evaluate.py
├── pyproject.toml
└── README.md
```
