# aiex01
aiex01 — Expérimentations d’applications d’intelligence artificielle

## Dépôt Git
https://github.com/ghislaincoutu/aiex01

## Installation des modules Python3
```sh
sudo apt-get update
sudo apt-get install python3
sudo apt-get install python3-pip
sudo apt-get install python3-venv
sudo apt-get install python3-tk
```

## Installation de FFmpeg
```sh
sudo apt-get install ffmpeg
```

## Création d’un environnement virtuel Python3
```sh
cd /media/disk01
sudo mkdir aiex01
sudo chown $USER:$USER aiex01
cd /media/disk01/aiex01
python3 -m venv ai01
source /media/disk01/aiex01/ai01/bin/activate
```
Saisir la commande `deactivate` pour sortir de l’environnement virtuel Python3.

Commande pour activer l’environnemnet virtuel Python3 à partir du script `activate_venv.sh`.
```sh
cd /media/disk01/aiex01
source ./sh_activate_venv.sh
```

## Installation de TensorFlow et de ses dépendances
```sh
source /media/disk01/aiex01/ai01/bin/activate
pip install --upgrade pip
pip install tensorflow
pip install tensorflow-hub
pip install tensorflow-datasets
pip install matplotlib
pip install opencv-python
pip install opencv-python-headless
```
Dans le cas d’une installation de TensorFlow sur une machine virtuelle VirtualBox pour Windows, il faut installer le paquet `tensorflow-cpu` plutôt que `tensorflow`. Cependant, ça ne fonctionne pas si on installe `tensorflow-cpu` avec `tensorflow-hub`.
```sh
pip uninstall tensorflow
pip install tensorflow-cpu
```

### Vérification de l’installation
```sh
python3 -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```

## Installation et utilisation du module TensorBoard
### Installation de TensorBoard
```sh
source /media/disk01/aiex01/ai01/bin/activate
pip install --upgrade pip
pip install tensorboard
```

### Activation de TensorBoard
Il faut au préalable générer les journaux dans le sous-répertoire `/media/disk01/aiex01/logs/fit` pour pouvoir obtenir un visuel.
```sh
tensorboard --logdir=logs/fit
```
