# =============================================================================
# train.py — Entraînement du réseau de neurones sur MNIST
# =============================================================================
# L'entraînement suit toujours le même cycle, répété des milliers de fois :
#
#   1. DEVINE  : le réseau regarde une image et fait une prédiction
#   2. MESURE  : on calcule à quel point la prédiction est mauvaise (la "loss")
#   3. CORRIGE : on ajuste les poids pour que la prochaine prédiction soit meilleure
#   4. RECOMMENCE avec la prochaine image
#
# Ce cycle "devine → mesure l'erreur → ajuste" est l'essence même
# de l'apprentissage automatique.
# =============================================================================

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import ReseauMNIST

# =============================================================================
# HYPERPARAMÈTRES
# Ce sont les réglages globaux de l'entraînement.
# On les regroupe ici pour les trouver et modifier facilement.
# =============================================================================
EPOCHS       = 5      # Nombre de fois qu'on passe sur tout le dataset
BATCH_SIZE   = 64     # Nombre d'images traitées en même temps
LEARNING_RATE = 0.001  # "Vitesse d'apprentissage" : à quel point on corrige à chaque étape


# =============================================================================
# 1. DEVICE : GPU ou CPU ?
# =============================================================================
# On demande à PyTorch d'utiliser le GPU (cuda) s'il est disponible,
# sinon on se rabat sur le CPU. Le GPU traite des milliers d'opérations
# en parallèle, ce qui accélère énormément l'entraînement.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Appareil utilisé : {device}")
if device.type == "cuda":
    print(f"GPU : {torch.cuda.get_device_name(0)}")
print()


# =============================================================================
# 2. DONNÉES MNIST
# =============================================================================
# torchvision peut télécharger MNIST automatiquement dans ./data.
# MNIST contient 70 000 images de chiffres manuscrits (28x28 pixels, niveaux de gris).
#   → 60 000 images pour l'entraînement
#   → 10 000 images pour le test (jamais vues pendant l'entraînement)

# La "transformation" normalise les pixels : valeurs entre 0 et 1, puis centrées
# autour de la moyenne 0.1307 avec un écart-type de 0.3081 (valeurs standards pour MNIST).
# Cela aide le réseau à apprendre plus vite et plus stablement.
transform = transforms.Compose([
    transforms.ToTensor(),                        # Convertit l'image PIL en tenseur PyTorch
    transforms.Normalize((0.1307,), (0.3081,))   # Normalise les pixels
])

print("Téléchargement / chargement de MNIST...")
dataset_train = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
dataset_test  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# Le DataLoader découpe le dataset en "batchs" (lots) et les mélange aléatoirement.
# Traiter plusieurs images à la fois (batch) est bien plus rapide
# que de les traiter une par une.
loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
loader_test  = DataLoader(dataset_test,  batch_size=BATCH_SIZE, shuffle=False)

print(f"  Images d'entraînement : {len(dataset_train)}")
print(f"  Images de test        : {len(dataset_test)}")
print()


# =============================================================================
# 3. MODÈLE, PERTE ET OPTIMISEUR
# =============================================================================

# On instancie le réseau et on l'envoie sur le GPU
modele = ReseauMNIST().to(device)

# FONCTION DE PERTE : CrossEntropyLoss
# Elle mesure à quel point la prédiction du réseau est éloignée de la bonne réponse.
# Plus la prédiction est fausse, plus la loss est grande.
# L'objectif de l'entraînement : minimiser cette valeur.
critere = nn.CrossEntropyLoss()

# OPTIMISEUR : Adam
# C'est l'algorithme qui ajuste les poids du réseau après chaque batch.
# Il calcule dans quelle direction modifier chaque poids pour réduire la loss.
# Adam est une version intelligente de la descente de gradient :
# il adapte le learning rate automatiquement pour chaque poids.
optimiseur = torch.optim.Adam(modele.parameters(), lr=LEARNING_RATE)


# =============================================================================
# 4. FONCTIONS UTILITAIRES
# =============================================================================

def entrainer_une_epoch(modele, loader, critere, optimiseur, device):
    """Entraîne le modèle sur tout le dataset d'entraînement une fois."""
    modele.train()  # Mode entraînement : active le calcul des gradients
    loss_totale = 0.0

    for images, etiquettes in loader:
        # --- Déplacer les données vers le GPU ---
        images    = images.view(-1, 784).to(device)  # Aplatir 28x28 → 784
        etiquettes = etiquettes.to(device)

        # ÉTAPE "DEVINE" : forward pass
        # Le réseau regarde les images et produit ses prédictions (logits)
        predictions = modele(images)

        # ÉTAPE "MESURE L'ERREUR" : calcul de la loss
        # On compare les prédictions avec les vraies étiquettes
        loss = critere(predictions, etiquettes)

        # ÉTAPE "CORRIGE" : backward pass + mise à jour des poids
        optimiseur.zero_grad()   # Remet les gradients à zéro (sinon ils s'accumulent)
        loss.backward()          # Calcule comment modifier chaque poids pour réduire la loss
        optimiseur.step()        # Applique les corrections aux poids

        loss_totale += loss.item()

    return loss_totale / len(loader)  # Loss moyenne sur tous les batchs


def evaluer(modele, loader, device):
    """Évalue la précision du modèle sur un dataset (sans modifier les poids)."""
    modele.eval()  # Mode évaluation : désactive les gradients (plus rapide)
    corrects = 0
    total    = 0

    with torch.no_grad():  # On ne calcule pas les gradients pendant l'évaluation
        for images, etiquettes in loader:
            images     = images.view(-1, 784).to(device)
            etiquettes = etiquettes.to(device)

            predictions = modele(images)
            # torch.max retourne la valeur et l'indice du neurone le plus activé
            # L'indice correspond au chiffre prédit (0-9)
            _, chiffres_predits = torch.max(predictions, dim=1)

            corrects += (chiffres_predits == etiquettes).sum().item()
            total    += etiquettes.size(0)

    return 100.0 * corrects / total  # Précision en pourcentage


# =============================================================================
# 5. BOUCLE D'ENTRAÎNEMENT
# =============================================================================
print("=" * 55)
print("  Début de l'entraînement")
print("=" * 55)

meilleure_precision = 0.0

for epoch in range(1, EPOCHS + 1):

    # --- Entraînement : le réseau apprend sur les 60 000 images ---
    loss_moyenne = entrainer_une_epoch(modele, loader_train, critere, optimiseur, device)

    # --- Évaluation : on teste sur les 10 000 images jamais vues ---
    precision = evaluer(modele, loader_test, device)

    print(f"  Epoch {epoch}/{EPOCHS}  |  Loss : {loss_moyenne:.4f}  |  Précision test : {precision:.2f}%")

    # --- Sauvegarde du meilleur modèle ---
    # On ne garde que le modèle qui a obtenu la meilleure précision jusqu'ici.
    # Cela évite de sauvegarder un modèle qui aurait "sur-appris" en fin d'entraînement.
    if precision > meilleure_precision:
        meilleure_precision = precision
        torch.save(modele.state_dict(), "model.pt")

print("=" * 55)
print(f"  Entraînement terminé !")
print(f"  Meilleure précision : {meilleure_precision:.2f}%")
print(f"  Poids sauvegardés dans : model.pt")
print("=" * 55)
