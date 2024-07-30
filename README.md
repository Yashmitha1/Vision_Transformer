# Vision Transformer from Scratch

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset Preparation](#dataset-preparation)
- [Model Implementation](#model-implementation)
- [Training and Evaluation](#training-and-evaluation)
- [Transfer Learning](#transfer-learning)
- [Results](#results)
- [Insights](#insights)
- [References](#references)

## Overview

This project involves building a Vision Transformer (ViT) from scratch, training it on MNIST and CIFAR-10 datasets, and evaluating its performance using different hyperparameters. The project also explores transfer learning by fine-tuning a pre-trained ViT model for a scene recognition task.

## Project Structure


## Dataset Preparation

1. **MNIST**:
   - Download the MNIST dataset using the datasets library:
     ```python
     from datasets import load_dataset

     mnist = load_dataset('mnist')
     ```

2. **CIFAR-10**:
   - Download the CIFAR-10 dataset using the datasets library:
     ```python
     from datasets import load_dataset

     cifar10 = load_dataset('cifar10')
     ```

3. **Scene Recognition**:
   - Prepare the dataset by augmenting and separating the class labels using `ImageFolder`.

## Model Implementation

The Vision Transformer (ViT) implementation includes:
- Manual patching and patching using CNN.
- Sinusoidal positional embeddings and learnable positional embeddings.
- Experimentation with different patch sizes, number of heads, and number of encoder blocks.

## Training and Evaluation

- Trained the ViT model on MNIST and CIFAR-10 datasets.
- Observed a performance of around 88% accuracy on MNIST and 44% accuracy on CIFAR-10.
- Performance on CIFAR-10 improved when using CNN-based patching instead of manual patching, while it decreased in the case of MNIST.

## Transfer Learning

- Prepared the scene recognition dataset using `ImageFolder` with augmentation and class separation.
- Used the `vit_base_patch16_224` model and modified the last classification head to match the dataset.
- Fine-tuned the pre-trained model on the scene recognition dataset.

## Results

- **MNIST**:
  - Achieved accuracy: 88%

- **CIFAR-10**:
  - Achieved accuracy: 44% with manual patching, improved with CNN-based patching.

- **Scene Recognition**:
  - Achieved accuracy: 95%

## Insights

- **Architecture Insights**:
  - Vision Transformers effectively handle image data by converting it into sequences and applying Transformer models.
  - Positional embeddings (both sinusoidal and learnable) are crucial for retaining spatial information in image patches.
  - CNN-based patching can improve performance on more complex datasets like CIFAR-10.

- **Hyperparameters**:
  - The performance of ViT models is sensitive to hyperparameters like learning rate, batch size, and the architecture's depth and width.
  - Experimenting with patch size, number of heads, and number of encoder blocks helps in understanding the model's behavior.

- **Transfer Learning**:
  - Pre-trained models significantly reduce training time and improve performance on related tasks.
  - Fine-tuning with a smaller learning rate helps in adapting the model to the new dataset while preserving learned features.

## References

- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [Hugging Face Transformers](https://huggingface.co/transformers/)




