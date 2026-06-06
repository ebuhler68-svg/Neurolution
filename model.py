# =============================================================================
# model.py — Définition du réseau de neurones
# =============================================================================
# Un réseau de neurones est une série de "couches" qui transforment
# progressivement les données brutes (pixels) en une décision (chiffre 0-9).
# Chaque couche apprend à détecter des motifs de plus en plus abstraits.
# =============================================================================

import torch
import torch.nn as nn


class ReseauMNIST(nn.Module):
    """
    Perceptron multicouche (MLP) pour reconnaître les chiffres manuscrits.

    Architecture :
        Entrée (784)  →  Couche 1 (256)  →  Couche 2 (128)  →  Sortie (10)

    Pourquoi 784 en entrée ?
        Une image MNIST fait 28 x 28 pixels = 784 valeurs numériques.
        On "aplatit" l'image en une longue liste de 784 nombres,
        chacun représentant la luminosité d'un pixel (0 = noir, 1 = blanc).

    Pourquoi 10 en sortie ?
        Il y a 10 chiffres possibles (0 à 9). Chaque neurone de sortie
        représente la confiance du réseau pour un chiffre donné.
        Le chiffre prédit est celui dont le neurone a la valeur la plus haute.
    """

    def __init__(self):
        # On appelle le constructeur de nn.Module (obligatoire)
        super(ReseauMNIST, self).__init__()

        # --- COUCHE 1 : 784 entrées → 256 neurones cachés ---
        # nn.Linear est une couche "linéaire" (aussi appelée couche "dense" ou "fully connected").
        # Elle calcule : sortie = entrée × poids + biais
        # Les "poids" sont les paramètres que le réseau va apprendre.
        # 256 est un choix arbitraire : assez grand pour capter des motifs,
        # assez petit pour rester rapide.
        self.couche1 = nn.Linear(784, 256)

        # --- COUCHE 2 : 256 → 128 neurones cachés ---
        # Une deuxième couche permet au réseau d'apprendre des combinaisons
        # plus complexes des motifs détectés en couche 1.
        self.couche2 = nn.Linear(256, 128)

        # --- COUCHE DE SORTIE : 128 → 10 neurones ---
        # Un neurone par chiffre possible (0, 1, 2, ..., 9).
        self.sortie = nn.Linear(128, 10)

        # --- ACTIVATION ReLU ---
        # ReLU (Rectified Linear Unit) est une fonction d'activation.
        # Elle applique simplement : f(x) = max(0, x)
        # → si x est positif, elle le laisse passer
        # → si x est négatif, elle le met à zéro
        #
        # POURQUOI en a-t-on besoin ?
        # Sans activation, empiler plusieurs couches linéaires revient
        # à n'avoir qu'une seule couche linéaire (les maths se simplifient).
        # ReLU introduit de la "non-linéarité" : le réseau peut alors
        # apprendre des relations complexes et non-linéaires dans les données.
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Le "forward pass" : comment les données traversent le réseau.

        C'est le chemin que prend une image depuis l'entrée jusqu'à
        la prédiction finale. PyTorch appelle cette méthode automatiquement
        quand on fait : prediction = modele(image)

        Argument :
            x : un lot d'images, de forme (batch_size, 784)
                batch_size = nombre d'images traitées en même temps

        Retour :
            logits : tenseur de forme (batch_size, 10)
                     valeurs brutes de confiance pour chaque chiffre
                     (pas encore des probabilités — CrossEntropyLoss s'en charge)
        """

        # Étape 1 : passe par la couche 1, puis ReLU
        # La couche linéaire fait une transformation mathématique,
        # ReLU éteint les signaux négatifs (garde seulement le positif).
        x = self.relu(self.couche1(x))

        # Étape 2 : passe par la couche 2, puis ReLU
        # Le réseau raffine sa compréhension à partir des motifs de couche 1.
        x = self.relu(self.couche2(x))

        # Étape 3 : couche de sortie (PAS de ReLU ici)
        # On veut des valeurs brutes (logits), positives ou négatives.
        # La fonction de perte (CrossEntropyLoss) les transformera
        # en probabilités via une fonction Softmax en interne.
        x = self.sortie(x)

        return x
