import tensorflow as tf
from tensorflow.keras import layers, models, Input
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Завантаження та попередня обробка даних MNIST
(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = (train_images.astype(np.float32) - 127.5) / 127.5  # Масштабування до [-1, 1]
train_images = np.expand_dims(train_images, axis=-1)
BUFFER_SIZE = 60000
BATCH_SIZE = 256

# Створення датасету
dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Визначення параметрів моделі
noise_dim = 100
num_epochs = 50

# Побудова генератора
def make_generator_model():
    model = models.Sequential()
    model.add(Input(shape=(noise_dim,)))
    model.add(layers.Dense(7 * 7 * 256, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

    return model

generator = make_generator_model()

# Побудова дискримінатора
def make_discriminator_model():
    model = models.Sequential()
    model.add(Input(shape=(28, 28, 1)))
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

discriminator = make_discriminator_model()

# Визначення функцій втрат
def generator_loss(fake_output):
    return tf.reduce_mean(tf.square(fake_output - 1))

def discriminator_loss(real_output, fake_output):
    real_loss = tf.reduce_mean(tf.square(real_output - 1))
    fake_loss = tf.reduce_mean(tf.square(fake_output))
    total_loss = real_loss + fake_loss
    return total_loss

# Оптимізатори
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Тренувальний цикл
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

def train(dataset, epochs):
    gen_losses = []
    disc_losses = []

    for epoch in range(epochs):
        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch)

        gen_losses.append(gen_loss)
        disc_losses.append(disc_loss)
        print(f'Epoch {epoch+1}, Gen Loss: {gen_loss}, Disc Loss: {disc_loss}')

    return gen_losses, disc_losses

# Тренування моделей
gen_losses, disc_losses = train(dataset, num_epochs)

# Збереження графіків кривих навчання
plt.figure(figsize=(10, 5))
plt.plot(gen_losses, label='Generator Loss')
plt.plot(disc_losses, label='Discriminator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_loss.png')  # Збереження графіку навчання
plt.close()

# Генерація зображень цифр
def generate_digits(generator, noise_dim, num_images):
    noise = tf.random.normal([num_images, noise_dim])
    generated_images = generator(noise, training=False)
    generated_images = (generated_images + 1) / 2.0  # Масштабування до [0, 1]
    return generated_images

# Вибір 8 зображень для створення 23112001
generated_images = generate_digits(generator, noise_dim, 8)

# Перетворення на зображення та створення одного зображення
fig, axes = plt.subplots(1, 8, figsize=(20, 2))
digits = [2, 3, 1, 1, 2, 0, 0, 1]
output_image = np.zeros((28, 28 * len(digits)))

for i, ax in enumerate(axes):
    img = generated_images[i, :, :, 0] * 255
    ax.imshow(img, cmap='gray')
    ax.axis('off')
    output_image[:, i*28:(i+1)*28] = img

plt.show()
fig.savefig('generated_digits.png')  # Збереження згенерованих цифр
plt.close()

# Збереження результату
output_image = output_image.astype(np.uint8)
Image.fromarray(output_image).save('23112001.png')

# Генерація 100 зображень для прикладу
example_images = generate_digits(generator, noise_dim, 100)

# Перетворення на одне велике зображення
example_image_grid = np.zeros((28 * 10, 28 * 10))  # 10x10 зображення
for i in range(10):
    for j in range(10):
        example_image_grid[i*28:(i+1)*28, j*28:(j+1)*28] = example_images[i*10 + j, :, :, 0] * 255

# Збереження прикладу 100 зображень
example_image_grid = example_image_grid.astype(np.uint8)
Image.fromarray(example_image_grid).save('example_100_digits.png')

# Виведення результату для перегляду
plt.figure(figsize=(10, 10))
plt.imshow(example_image_grid, cmap='gray')
plt.axis('off')
plt.show()