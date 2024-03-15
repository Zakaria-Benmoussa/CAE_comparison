import train 
import sys

train.execute(model_name='CAE', depth=3, coeff_NB=4, database="MNIST", nb_epoch=10, pth_name="model.pth")

# Paramètres:   -model_name : type d'architecture ("CAE" ou "WDAED")
#               -depth : profondeur du réseau (CAE seulement)
#               -coeff_NB : coefficient de réduction de l'espace latent par rapport a l'entrée FC (CAE seulement)
#               -database : donnée d'entrainement ("MNIST" ou "CIFAR10")
#               -nb_epoch : nombre d'epoch pour l'entrainement
#               -pth_name : nom du fichier de sorti de l'entrainement