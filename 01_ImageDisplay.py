# TensorFlow -- Affichage d'une image
import tensorflow as tf
import matplotlib.pyplot as plt

url = "https://farm6.staticflickr.com/5314/5887463535_a88f862a81_o.jpg"

# Download the image to a local cache directory
image_path = tf.keras.utils.get_file(origin=url)

# Load and decode the image
image = tf.io.read_file(image_path)
image = tf.image.decode_image(image, channels=3)  # Ensure RGB

# Display the image
plt.imshow(image)
plt.axis("off")
plt.show()
