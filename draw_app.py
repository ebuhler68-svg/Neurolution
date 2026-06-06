# =============================================================================
# draw_app.py — Interface de dessin avec prédiction en temps réel
# =============================================================================
# Ce programme ouvre une fenêtre Pygame avec deux zones :
#
#   GAUCHE : canvas de dessin (fond noir, trait blanc)
#   DROITE : panneau de résultats avec 10 barres de probabilité
#
# À chaque modification du dessin, l'image est :
#   1. Réduite en 28×28 pixels (taille standard MNIST)
#   2. Normalisée avec les mêmes valeurs que pendant l'entraînement
#   3. Passée au réseau de neurones → 10 probabilités (une par chiffre)
#
# Contrôles :
#   Clic-glissé : dessiner
#   C           : effacer
#   Échap       : quitter
# =============================================================================

import sys
import torch
import numpy as np
import pygame

from model import ReseauMNIST

# =============================================================================
# MISE EN PAGE
# =============================================================================
LARGEUR_CANVAS  = 280    # Zone de dessin : 280×280 pixels
HAUTEUR_CANVAS  = 280    # (multiple de 28, pour réduire proprement à 28×28)
LARGEUR_PANEL   = 420    # Largeur du panneau résultats (à droite)
LARGEUR_FENETRE = LARGEUR_CANVAS + LARGEUR_PANEL   # 700 px au total
HAUTEUR_FENETRE = 380

RAYON_PINCEAU = 12       # Demi-épaisseur du trait en pixels (→ trait ~24px)

# Palette de couleurs (format RGB)
NOIR       = (0,   0,   0  )
BLANC      = (255, 255, 255)
GRIS_FOND  = (35,  35,  45 )   # Fond du panneau droit
GRIS_BARRE = (60,  60,  80 )   # Barres "éteintes" (probabilité faible)
BLEU       = (80,  130, 200)   # Barres normales
VERT       = (80,  200, 120)   # Barre et chiffre du gagnant
GRIS_TEXTE = (140, 140, 160)   # Texte secondaire / aide


# =============================================================================
# CHARGEMENT DU MODÈLE
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

modele = ReseauMNIST()
try:
    # map_location permet de charger le modèle sur le bon device
    # même s'il a été sauvegardé sur un autre (ex: GPU → CPU)
    modele.load_state_dict(torch.load("model.pt", map_location=device))
except FileNotFoundError:
    print("ERREUR : model.pt introuvable. Lance d'abord train.py.")
    sys.exit(1)

modele.to(device)
# eval() : passe le modèle en mode "inférence"
# → les poids sont figés, on ne fait que prédire (pas d'apprentissage)
modele.eval()

print(f"Modèle chargé sur : {device}")
if device.type == "cuda":
    print(f"GPU : {torch.cuda.get_device_name(0)}")


# =============================================================================
# PRÉDICTION : canvas Pygame → probabilités softmax
# =============================================================================
def predire(surface_canvas):
    """
    Transforme le contenu visuel du canvas en une prédiction du réseau.

    Étapes :
      1. Vérifie que la zone n'est pas vide
      2. Réduit à 28×28 (taille MNIST) avec lissage antialiasing
      3. Convertit en niveaux de gris, normalise
      4. Passe au modèle → 10 probabilités (une par chiffre)

    Retourne : tableau numpy de 10 floats (somme = 1.0), ou None si vide.
    """
    # surfarray.array3d retourne un tableau numpy de forme (largeur, hauteur, 3)
    # chaque pixel est [R, G, B] avec des valeurs entre 0 et 255
    data = pygame.surfarray.array3d(surface_canvas)

    # Si le pixel le plus clair est très sombre → zone vide → rien à prédire
    if data.max() < 10:
        return None

    # --- Étape 1 : Réduire à 28×28 avec lissage ---
    # smoothscale applique un antialiasing (adoucissement des contours)
    # ce qui imite mieux la façon dont une vraie image MNIST est produite.
    mini = pygame.transform.smoothscale(surface_canvas, (28, 28))

    # Extraire les pixels : forme (28, 28, 3)
    pixels = pygame.surfarray.array3d(mini)

    # --- Étape 2 : Convertir en niveaux de gris ---
    # Nos traits sont blancs (255, 255, 255), la moyenne des canaux R, G, B suffit.
    # Résultat : tableau (28, 28) de float32 avec valeurs 0.0 à 255.0
    gris = pixels.mean(axis=2).astype(np.float32)

    # IMPORTANT — correction d'axe :
    # Pygame stocke les pixels en (x=colonne, y=ligne) → forme (largeur, hauteur)
    # PyTorch attend (ligne, colonne) → on transpose pour corriger l'orientation.
    gris = gris.T   # (28, 28) maintenant indexé [ligne][colonne]

    # --- Étape 3 : Normalisation identique à l'entraînement ---
    # 1. Ramener les valeurs en [0, 1] (pixels originaux : 0 à 255)
    gris = gris / 255.0
    # 2. Centrer et réduire avec les statistiques de MNIST :
    #    moyenne = 0.1307, écart-type = 0.3081
    #    C'est exactement transforms.Normalize((0.1307,), (0.3081,)) de train.py
    gris = (gris - 0.1307) / 0.3081

    # --- Étape 4 : Créer le tenseur et prédire ---
    tenseur = torch.from_numpy(gris).float()   # (28, 28)
    # reshape plutôt que view : après smoothscale + transpose, le tenseur peut être
    # non-contigu en mémoire ; reshape alloue une copie contiguë si nécessaire.
    tenseur = tenseur.reshape(1, 784).to(device)

    # torch.no_grad() : désactive le moteur de gradient de PyTorch.
    # Pendant la prédiction, on n'a pas besoin de calculer les gradients
    # (on ne modifie pas les poids), donc on gagne en vitesse et en mémoire.
    with torch.no_grad():
        logits = modele(tenseur)                        # sorties brutes : (1, 10)
        probas = torch.softmax(logits, dim=1)           # → probabilités qui somment à 1
        probas = probas.squeeze().cpu().numpy()         # → tableau numpy de 10 valeurs

    return probas


# =============================================================================
# RENDU DU PANNEAU DROIT
# =============================================================================
def dessiner_panneau(surface, probas, fonts):
    """
    Dessine le panneau de résultats à droite :
      - En haut  : chiffre prédit en grand + % de confiance
      - En bas   : 10 barres horizontales, une par chiffre (0-9)
    """
    font_titre, font_moyen, font_petit = fonts
    px       = LARGEUR_CANVAS + 20   # x de départ du panneau (marge 20px)
    BARRE_MAX = 235                  # longueur maximale d'une barre (en pixels)

    if probas is None:
        # Pas encore de dessin : afficher un message d'invite
        msg = font_moyen.render("Dessinez un chiffre", True, BLANC)
        surface.blit(msg, (px, 22))
        hint = font_petit.render("dans la zone noire à gauche", True, GRIS_TEXTE)
        surface.blit(hint, (px, 54))
        return

    chiffre   = int(np.argmax(probas))    # indice de la probabilité la plus haute
    confiance = probas[chiffre] * 100     # en pourcentage

    # --- Prédiction principale (en haut à droite) ---
    grand_chiffre = font_titre.render(str(chiffre), True, VERT)
    surface.blit(grand_chiffre, (px, 5))

    pct_texte = font_moyen.render(f"{confiance:.1f} %", True, VERT)
    surface.blit(pct_texte, (px + 60, 18))

    label_conf = font_petit.render("confiance", True, GRIS_TEXTE)
    surface.blit(label_conf, (px + 60, 46))

    # --- Barres de probabilité (une par chiffre) ---
    for i in range(10):
        y = 90 + i * 28        # position verticale de cette ligne
        est_gagnant = (i == chiffre)

        # Étiquette du chiffre (colorée en vert si c'est le gagnant)
        couleur_lbl = VERT if est_gagnant else BLANC
        lbl = font_petit.render(str(i), True, couleur_lbl)
        surface.blit(lbl, (px, y + 2))

        # Barre de fond (grisée, indique le maximum possible)
        pygame.draw.rect(surface, GRIS_BARRE,
                         (px + 20, y, BARRE_MAX, 20), border_radius=4)

        # Barre colorée dont la longueur = probabilité × BARRE_MAX
        largeur = int(probas[i] * BARRE_MAX)
        if largeur > 0:
            couleur_barre = VERT if est_gagnant else BLEU
            pygame.draw.rect(surface, couleur_barre,
                             (px + 20, y, largeur, 20), border_radius=4)

        # Pourcentage affiché à droite de la barre
        pct = font_petit.render(f"{probas[i] * 100:5.1f}%", True, BLANC)
        surface.blit(pct, (px + 20 + BARRE_MAX + 6, y + 2))


# =============================================================================
# PROGRAMME PRINCIPAL
# =============================================================================
def main():
    pygame.init()
    fenetre = pygame.display.set_mode((LARGEUR_FENETRE, HAUTEUR_FENETRE))
    pygame.display.set_caption("Neurolution — Dessine un chiffre !")

    # Chargement des polices système
    font_titre = pygame.font.SysFont("Arial", 54, bold=True)
    font_moyen = pygame.font.SysFont("Arial", 22)
    font_petit = pygame.font.SysFont("Arial", 16)
    fonts = (font_titre, font_moyen, font_petit)

    # Surface de dessin : fond noir, 280×280
    # C'est sur cette surface que l'utilisateur dessine, et qu'on lit les pixels
    canvas = pygame.Surface((LARGEUR_CANVAS, HAUTEUR_CANVAS))
    canvas.fill(NOIR)

    # --- Variables d'état ---
    dessin_actif   = False    # True quand le bouton gauche de la souris est tenu
    pos_precedente = None     # Position de la souris au frame précédent (pour les lignes)
    probas         = None     # Tableau de 10 probabilités (mis à jour en temps réel)
    derniere_maj   = 0        # Timestamp (ms) de la dernière inférence
    DELAI_MAJ      = 80       # Prédire toutes les 80 ms au max (~12 prédictions/sec)

    clock    = pygame.time.Clock()
    continuer = True

    # =========================================================================
    # BOUCLE PRINCIPALE PYGAME
    # =========================================================================
    # Pygame fonctionne avec une boucle infinie qui répète 60×/sec :
    #   1. Lire les événements (souris, clavier)
    #   2. Mettre à jour l'état (dessin, prédiction)
    #   3. Redessiner toute la fenêtre
    # C'est le cœur de toute application Pygame.
    # =========================================================================
    while continuer:

        # --- ÉTAPE 1 : Lecture des événements ---
        for evt in pygame.event.get():

            # L'utilisateur ferme la fenêtre (croix)
            if evt.type == pygame.QUIT:
                continuer = False

            # Touches clavier
            if evt.type == pygame.KEYDOWN:
                if evt.key == pygame.K_ESCAPE:
                    continuer = False          # Échap → quitter
                if evt.key == pygame.K_c:
                    canvas.fill(NOIR)          # C → tout effacer
                    probas = None

            # Clic gauche appuyé : début du tracé
            if evt.type == pygame.MOUSEBUTTONDOWN and evt.button == 1:
                x, y = evt.pos
                if x < LARGEUR_CANVAS and y < HAUTEUR_CANVAS:
                    dessin_actif   = True
                    pos_precedente = (x, y)
                    # Dessiner un cercle au point de clic (pour les points isolés)
                    pygame.draw.circle(canvas, BLANC, (x, y), RAYON_PINCEAU)

            # Clic relâché : fin du tracé
            if evt.type == pygame.MOUSEBUTTONUP and evt.button == 1:
                dessin_actif   = False
                pos_precedente = None

            # Mouvement souris : tracé continu
            if evt.type == pygame.MOUSEMOTION and dessin_actif:
                x, y = evt.pos
                if x < LARGEUR_CANVAS and y < HAUTEUR_CANVAS:
                    # Relier le point précédent au point actuel par une ligne épaisse.
                    # Sans cela, quand la souris bouge vite, on obtient des pointillés.
                    if pos_precedente:
                        pygame.draw.line(canvas, BLANC,
                                         pos_precedente, (x, y),
                                         RAYON_PINCEAU * 2)
                    pygame.draw.circle(canvas, BLANC, (x, y), RAYON_PINCEAU)
                    pos_precedente = (x, y)
                else:
                    # Souris sortie du canvas → on coupe le tracé (évite les lignes parasites)
                    pos_precedente = None

        # --- ÉTAPE 2 : Prédiction en temps réel ---
        # On prédit toutes les DELAI_MAJ ms pour ne pas surcharger le GPU.
        maintenant = pygame.time.get_ticks()   # temps écoulé en millisecondes
        if maintenant - derniere_maj > DELAI_MAJ:
            probas       = predire(canvas)
            derniere_maj = maintenant

        # --- ÉTAPE 3 : Rendu ---
        fenetre.fill(GRIS_FOND)                        # Fond du panneau droit
        fenetre.blit(canvas, (0, 0))                   # Coller le canvas à gauche
        # Bordure fine autour de la zone de dessin
        pygame.draw.rect(fenetre, GRIS_TEXTE,
                         (0, 0, LARGEUR_CANVAS, HAUTEUR_CANVAS), 1)

        dessiner_panneau(fenetre, probas, fonts)        # Barres de probabilité

        # Texte d'aide en bas à droite
        aide = font_petit.render("C = Effacer   |   Échap = Quitter",
                                 True, GRIS_TEXTE)
        fenetre.blit(aide, (LARGEUR_CANVAS + 20, HAUTEUR_FENETRE - 22))

        # Afficher le frame (double-buffering : swap du buffer arrière → écran)
        pygame.display.flip()
        clock.tick(60)   # Limiter à 60 FPS

    pygame.quit()


if __name__ == "__main__":
    main()
