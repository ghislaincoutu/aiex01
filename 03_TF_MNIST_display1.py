# TensorFlow -- Traitement d'un jeu de données MNIST

import os
import datetime
import tensorflow as tf
os.system("clear")

def pause():
    programPause = input("Appuyez sur la touche Retour pour continuer...")

print("TensorFlow -- Traitement d'un jeu de données MNIST")
print(f"Version TensorFlow : {tf.__version__}")
pause()

print("\nChargement d’un jeu de données MNIST")
pause()
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

print("\nCréation d’un modèle d’apprentissage automatique")
pause()
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10),
    ]
)
predictions = model(x_train[:1]).numpy()
predictions
tf.nn.softmax(predictions).numpy()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn(y_train[:1], predictions).numpy()
model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

print("\nEntraînement et évaluation du modèle")
pause()
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test, verbose=2)
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
probability_model(x_test[:5])

# Test d’utilisation du module TensorBoard
def tensorboard_function():
    print("\nDéfinition du répertoire où les journaux de TensorBoard seront enregistrés")
    pause()
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    print("\nCréation du rappel de TensorBoard")
    pause()
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    print("\nEntraînement et évaluation du modèle")
    pause()
    history = model.fit(
        x_train,
        y_train,
        epochs=5,
        validation_data=(x_test, y_test),
        callbacks=[tensorboard_callback],
    )

tensorboard_function()

print("\nExécuter la commande suivante pour activer TensorBoard :")
print ("tensorboard --logdir=logs/fit")

