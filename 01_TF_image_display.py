# TensorFlow -- Téléchargement et affichage d'une image

import os
import tensorflow as tf
import matplotlib.pyplot as plt
os.system("clear")

def pause():
    programPause = input("Appuyez sur la touche Retour pour continuer...")

print("TensorFlow -- Téléchargement et affichage d’une image")
print(f"Version TensorFlow : {tf.__version__}")
pause()

print("\nTéléchargement de l’image")
pause()
url = "https://farm6.staticflickr.com/5314/5887463535_a88f862a81_o.jpg"
image_path = tf.keras.utils.get_file(origin=url)

print("\nLecture et décodage de l’image")
pause()
image = tf.io.read_file(image_path)
image = tf.image.decode_image(image, channels=3)

print("\nAffichage de l'image")
pause()
plt.imshow(image)
plt.axis("off")
plt.show()
