#!/bin/bash
# Script pour lancer le mode Pipeline

# 1. Modifier le fichier de configuration pour activer le mode Pipeline
CONFIG_FILE="./.env"  # Adapte ce nom de fichier si besoin
if [ -f "$CONFIG_FILE" ]; then
    echo "Configuration du mode Pipeline dans $CONFIG_FILE"
    # Pour chaque ligne contenant "Mode Pipeline", on s'assure qu'elle est décommentée
    sed -i '/Mode Pipeline/ s/^#//' "$CONFIG_FILE"
    # Pour chaque ligne contenant "Mode Terminal", on ajoute un "#" si ce n'est pas déjà le cas
    sed -i '/Mode Terminal/ { /^[^#]/ s/^/#/ }' "$CONFIG_FILE"
else
    echo "Fichier de configuration $CONFIG_FILE non trouvé. Vérifie que le fichier existe."
fi

# 2. Lancer Docker Compose pour démarrer le pipeline
echo "Lancement du mode Pipeline avec docker-compose..."
docker compose up --build
