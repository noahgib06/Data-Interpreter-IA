#!/bin/bash
# Script pour lancer le mode Terminal

# 1. Installer les dépendances Python
echo "Installation des dépendances Python..."
pip install -r requirements.txt

# 2. Modifier le fichier de configuration pour activer le mode Terminal
CONFIG_FILE=".env"
if [ -f "$CONFIG_FILE" ]; then
    echo "Configuration du mode Terminal dans $CONFIG_FILE"

    # Choix du suffixe pour sed selon OS ('' pour macOS, rien pour Linux)
    SED_SUFFIX=""
    if [[ "$OSTYPE" == "darwin"* ]]; then
        SED_SUFFIX=".bak"
    fi

    # Décommenter les lignes contenant "Mode Terminal"
    sed -i "$SED_SUFFIX" '/Mode Terminal/ s/^[[:space:]]*#//' "$CONFIG_FILE"

    # Commenter les lignes contenant "Mode Pipeline" si elles ne sont pas déjà commentées
    sed -i "$SED_SUFFIX" '/Mode Pipeline/ s/^[[:space:]]*\([^#]\)/#\1/' "$CONFIG_FILE"

    # Nettoyage du fichier de sauvegarde temporaire sur macOS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        rm -f "${CONFIG_FILE}.bak"
    fi

else
    echo "Fichier de configuration $CONFIG_FILE non trouvé. Vérifie que le fichier existe."
fi

# 3. Lancer main.py en transmettant les arguments (fichiers à traiter)
echo "Lancement du mode Terminal avec main.py..."
python main.py "$@"

