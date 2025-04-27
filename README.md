# Pix2Pix Night-to-Day Image Translation

This repository contains an implementation of the Pix2Pix conditional generative adversarial network (cGAN) for translating nighttime images to daytime images using TensorFlow.

## Overview

This project implements the Pix2Pix architecture as described in the paper [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) by Isola et al. The model is trained on the night2day dataset, which contains paired images of the same scenes during nighttime and daytime.

## Features

- Full implementation of Pix2Pix GAN architecture in TensorFlow
- Data preprocessing pipeline with augmentation (random jitter, mirroring)
- U-Net generator with skip connections
- PatchGAN discriminator
- Generator loss combining GAN loss and L1 loss
- Training checkpointing and TensorBoard integration

## Dataset

The night2day dataset is used for training and testing. It contains 8,385 paired images of nighttime and daytime scenes. The dataset can be downloaded from the [Berkeley AI Research (BAIR) Laboratory](http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/).

## Model Architecture

### Generator

The generator uses a U-Net architecture with skip connections:

- Encoder: 8 downsampling blocks
- Decoder: 7 upsampling blocks with skip connections
- All convolutional layers use 4×4 kernels

### Discriminator

The discriminator follows a PatchGAN architecture:
- Takes both input and target images concatenated
- Classifies whether overlapping image patches are real or fake
- 70×70 effective receptive field

## Training Details

- LAMBDA = 100 (weight for L1 loss)
- Adam optimizer with learning rate 2e-4 and beta_1=0.5
- Batch size of 1
- Random cropping and mirroring for data augmentation
- Checkpoints saved every 5,000 steps
- Training progress tracked with TensorBoard

## Requirements

- TensorFlow 2.x
- Matplotlib
- NumPy
- Python 3.6+

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pix2pix-night2day.git
cd pix2pix-night2day

# Install dependencies
pip install tensorflow matplotlib numpy
```

## Usage

### Download Dataset

```python
!wget http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/night2day.tar.gz
!tar xvf night2day.tar.gz
```

### Training

```python
# Run the training script
python train.py
```

### Testing

```python
# Generate images using a trained model
python test.py
```

### Visualizing Results with TensorBoard

```python
%load_ext tensorboard
%tensorboard --logdir logs/
```

## Results

The model learns to translate nighttime scenes to corresponding daytime scenes. Example outputs:

- Input: Nighttime image
- Ground Truth: Actual daytime image
- Predicted: Generated daytime image

## Acknowledgments

- [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) paper by Isola et al.
- [TensorFlow Pix2Pix Tutorial](https://www.tensorflow.org/tutorials/generative/pix2pix)
- UC Berkeley BAIR Lab for the dataset

## License

MIT License

## Citation

If you use this code, please cite:
```
@inproceedings{isola2017image,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1125--1134},
  year={2017}
}
```
