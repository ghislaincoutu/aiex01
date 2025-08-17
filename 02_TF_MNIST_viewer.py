# TensorFlow -- Visionneuse de données MNIST de base

import os
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
os.system("clear")

def pause():
    programPause = input("Appuyez sur la touche Retour pour continuer...")

print("TensorFlow -- Visionneuse de données MNIST de base")
print(f"Version TensorFlow : {tf.__version__}")
pause()

print("\nChargement de la base de données MNIST")
pause()
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("\nDétermination du nombre d’images d’entraînement et d’images de test")
pause()
print(f"Total des images d’entraînement: {x_train.shape}, Étiquettes: {y_train.shape}")
print(f"Total des images de test: {x_test.shape}, Étiquettes: {y_test.shape}")

print("\nAffichage d'une image d’entraînement et de son étiquette")
pause()
sample_index = 58
plt.imshow(x_train[sample_index], cmap='gray')
plt.title(f"Label: {y_train[sample_index]}")
plt.axis('off')
plt.show()

print("\nAffichage d’une image de test et de son étiquette")
pause()
sample_index = 58
plt.imshow(x_test[sample_index], cmap='gray')
plt.title(f"Label: {y_test[sample_index]}")
plt.axis('off')
plt.show()
