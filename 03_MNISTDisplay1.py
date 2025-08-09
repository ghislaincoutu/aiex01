# TensorFlow -- Traitement d'un jeu de données MNIST


def pause():
    programPause = input("Appuyez sur la touche Retour pour continuer...")


import os
import datetime

os.system("clear")

print("Configuration de TensorFlow")
pause()
import tensorflow as tf

print(f"TensorFlow version: {tf.__version__}")

print("Chargement d’un jeu de données MNIST")
pause()
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

print("Création d’un modèle d’apprentissage automatique")
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

print("Entraînement et évaluation du modèle")
pause()
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test, verbose=2)
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
probability_model(x_test[:5])


# Test d’utilisation du module TensorBoard
def tensorboard_function():
    print("Définition du répertoire où les journaux de TensorBoard seront enregistrés")
    pause()
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print("Création du rappel de TensorBoard")
    pause()
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    print("Entraînement et évaluation du modèle")
    pause()
    history = model.fit(
        x_train,
        y_train,
        epochs=5,
        validation_data=(x_test, y_test),
        callbacks=[tensorboard_callback],
    )


tensorboard_function()

# À la fin du script, exécuter la commande suivante :
# tensorboard --logdir=logs/fit
