# 🧠 AI-Based Medical Image Enhancement for Diagnostics Using GANs

This project implements a Generative Adversarial Network (GAN) to enhance diagnostic medical images. It uses the MNIST dataset as a proxy for medical images, demonstrating how GANs can improve image clarity and support diagnostic accuracy.

## 📝 Overview

Generative Adversarial Networks (GANs) consist of two neural networks — a *Generator* and a *Discriminator* — trained in opposition. The Generator creates synthetic images resembling real ones, while the Discriminator attempts to distinguish between real and synthetic data. Over time, the Generator learns to produce increasingly realistic images.

## 📂 Project Structure


main.py       # Core implementation of GAN model and training loop


## ⚙ Requirements

Make sure you have the following Python libraries installed:

bash
pip install tensorflow numpy


## ▶ How to Run

To start training the GAN, run the following command:

bash
python main.py


Training progress will be printed to the console every 100 epochs, showing discriminator loss/accuracy and generator loss.

## 🧠 Model Architecture

### Generator
- Dense layers with Batch Normalization and LeakyReLU activations
- Final layer reshaped into a 28x28 grayscale image using tanh activation

### Discriminator
- Dense layers with LeakyReLU activations
- Outputs a single value using sigmoid to classify real vs generated images

## 📊 Training Logs

Training logs are printed at intervals like:


Epoch 100/500 [D loss: 0.6543, acc.: 62.50%] [G loss: 0.8723]


You can adjust sample_interval in the code to change how frequently logs are printed.

## 🏥 Medical Use Case Potential

Although the MNIST dataset is used here, this GAN structure is applicable to real medical images for tasks like:

- Enhancing low-resolution MRI or CT scans
- Removing noise/artifacts from X-rays
- Generating synthetic images for training data augmentation

## 🔮 Future Improvements

- Use real medical datasets (e.g., ChestX-ray14, BraTS)
- Save and visualize generated images after training
- Upgrade to Deep Convolutional GAN (DCGAN) architecture
- Integrate into a user-friendly web interface using Flask or Streamlit
