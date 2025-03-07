#!/bin/bash
# Script pour lancer le mode Terminal

# 1. Installer les dépendances Python
echo "Installation des dépendances Python..."
pip install -r requirements.txt

# 2. Modifier le fichier de configuration pour activer le mode Terminal
CONFIG_FILE="./.env"  # Adapte ce nom de fichier si besoin
if [ -f "$CONFIG_FILE" ]; then
    echo "Configuration du mode Terminal dans $CONFIG_FILE"
    # Pour chaque ligne contenant "Mode Terminal", on s'assure qu'elle est décommentée
    sed -i '/Mode Terminal/ s/^#//' "$CONFIG_FILE"
    # Pour chaque ligne contenant "Mode Pipeline", on ajoute un "#" si ce n'est pas déjà le cas
    sed -i '/Mode Pipeline/ { /^[^#]/ s/^/#/ }' "$CONFIG_FILE"
else
    echo "Fichier de configuration $CONFIG_FILE non trouvé. Vérifie que le fichier existe."
fi

# 3. Lancer main.py en transmettant les arguments (fichiers à traiter)
echo "Lancement du mode Terminal avec main.py..."
python main.py "$@"
