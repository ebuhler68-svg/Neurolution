# Neurolution

Mini réseau de neurones qui reconnaît des chiffres manuscrits (type MNIST).

## Étape actuelle
Etage 1 terminé — réseau entraîné (~97.7% sur MNIST) + interface de dessin en temps réel.

## Prérequis
- Windows 11, GPU NVIDIA RTX 4050 (6 Go VRAM)
- Python 3.14, PyTorch avec CUDA 12.x, pygame-ce

## Vérification GPU
```powershell
.\venv\Scripts\Activate.ps1
python check_gpu.py
```

## Entraîner le modèle
```powershell
.\venv\Scripts\Activate.ps1
python train.py
```
Télécharge MNIST automatiquement dans `./data`, entraîne 5 epochs, sauvegarde les poids dans `model.pt`.

## Dessiner en direct
```powershell
.\venv\Scripts\Activate.ps1
python draw_app.py
```

Une fenêtre s'ouvre avec :
- **Zone noire à gauche** : dessinez un chiffre (0-9) avec la souris
- **Panneau droit** : la prédiction du réseau se met à jour en temps réel

| Touche | Action |
|--------|--------|
| Clic-glissé | Dessiner |
| `C` | Effacer la zone de dessin |
| `Échap` | Quitter |
