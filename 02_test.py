import os
os.system("clear")
pause89 = "Appuyez sur la touche Retour pour continuer..."

print("Configuration de TensorFlow"); input(pause89)
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
print(f"TensorFlow version: {tf.__version__}")

print("Chargement d'un jeu de données Fashion MNIST"); input(pause89)
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print("Exploration des données"); input(pause89)
train_images.shape
len(train_labels)
train_labels
len(test_labels)

print("Prétraitement des données"); input(pause89)
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

print("Affichage des 25 premières images"); input(pause89)
train_images = train_images / 255.0
test_images = test_images / 255.0
plt.figure(figsize=(10,10))
for i in range(25):
  plt.subplot(5,5,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(train_images[i], cmap=plt.cm.binary)
  plt.xlabel(class_names[train_labels[i]])
plt.show()

