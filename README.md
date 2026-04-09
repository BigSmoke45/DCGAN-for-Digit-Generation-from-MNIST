# DCGAN for Digit Generation from MNIST

A Deep Convolutional Generative Adversarial Network (DCGAN) implemented from scratch in TensorFlow, trained on the MNIST dataset to generate realistic handwritten digits.

---

## What it does

Trains a GAN where:
- The **generator** takes random noise and learns to produce fake digit images
- The **discriminator** learns to distinguish real MNIST digits from generated ones
- Both networks compete until the generator produces convincing results

After training, the model generates a 10×10 grid of 100 sample digits and saves them as PNG.

---

## Implementation highlights

- Custom training loop via `tf.GradientTape` — no `model.fit()`, full manual control over gradient updates for both networks
- Least-squares loss (LSGAN variant) instead of standard binary cross-entropy — more stable training
- Generator: `Dense → Reshape → Conv2DTranspose × 3` with `BatchNormalization` and `LeakyReLU`
- Discriminator: `Conv2D × 2 → Flatten → Dense` with `Dropout(0.3)`
- `@tf.function` decorator on the train step for graph-mode optimization
- Training loss curves saved as `training_loss.png`

---

## Tech Stack

`Python` · `TensorFlow 2.x` · `Keras` · `NumPy` · `Matplotlib` · `Pillow`

---

## Output

| File | Description |
|------|-------------|
| `training_loss.png` | Generator and discriminator loss curves over 50 epochs |
| `generated_digits.png` | Sample row of 8 generated digits |
| `example_100_digits.png` | 10×10 grid of 100 generated samples |

---

## Usage

```bash
pip install tensorflow numpy matplotlib pillow
python dcgan_mnist.py
```

Training runs for 50 epochs on the full MNIST dataset (60,000 images, batch size 256).

---

## Notes

University coursework project. The model uses a fixed noise dimension of 100 and Adam optimizer (lr=1e-4) for both networks.
