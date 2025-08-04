# aiex01
aiex01 — Expérimentations d’applications d’intelligence artificielle

## Installation de TensorFlow
Procédure d’installation de l’application, selon la documentation officielle de TensorFlow.

### Installation des modules Python3
```sh
sudo apt-get update
sudo apt-get install python3
sudo apt-get install python3-pip
sudo apt-get install python3-venv
sudo apt-get install python3-tk
```

### Création d’un environnement virtuel Python3
```sh
cd /var
mkdir aiex01
cd /var/aiex01
python3 -m venv ai01
source /var/aiex01/ai01/bin/activate
```
Saisir la commande `deactivate` pour sortir de l’environnement virtuel Python3.

### Installation de TensorFlow et de ses dépendances
```sh
source /var/aiex01/ai01/bin/activate
pip install --upgrade pip
pip install tensorflow
pip install matplotlib
```

### Vérification de l’installation
```sh
python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```

# Installation et utilisation du module TensorBoard
## Installation de TensorBoard
```sh
source /var/aiex01/ai01/bin/activate
pip install --upgrade pip
pip install tensorboard
```

## Activation de TensorBoard
Il faut au préalable générer les journaux dans le sous-répertoire `/var/aiex01/logs/fit` pour pouvoir obtenir un visuel.
```sh
tensorboard --logdir=logs/fit
```
