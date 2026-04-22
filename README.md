# MASC 515 Assignment 3: AI and MicroGPT

This repository contains the pure, dependency-free implementation of a GPT model, enhanced with four advanced machine learning algorithms.

## Algorithms Implemented and Underlying Ideas

### 1. GELUs (Gaussian Error Linear Units)
* GELU is a smoother, probabilistic alternative to the standard ReLU activation function. Instead of abruptly cutting off negative values, it weights inputs by their cumulative probability under a Gaussian distribution. This prevents the "dead neuron" problem and allows for better gradient flow in deep networks.

### 2. LoRA (Low Rank Adaptation)
* Training all parameters of a large language model is computationally expensive. LoRA freezes the pre-trained model weights and injects trainable rank decomposition matrices (A and B) into the architecture. It drastically reduces the number of trainable parameters while maintaining performance for downstream tasks.

### 3. RoPE (Rotary Position Embedding)
* Instead of adding absolute position embeddings to tokens, RoPE encodes positional information by rotating the Query and Key representations in a 2D complex plane. The rotation angle depends on the token's position, allowing the attention mechanism to gracefully capture relative token distances.

### 4. Mixture of Experts (MoE)
*  MoE scales model capacity without significantly increasing compute cost. It replaces a dense feed-forward network with a "Router" (gating network) and multiple "Experts". For each token, the router predicts which expert is best suited to process it, performing a weighted sum of their outputs.
