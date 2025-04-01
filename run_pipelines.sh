#!/bin/bash
# Script pour lancer le mode Pipeline

CONFIG_FILE=".env"
if [ -f "$CONFIG_FILE" ]; then
    echo "Configuration du mode Terminal dans $CONFIG_FILE"

    # Choix du suffixe pour sed selon OS ('' pour macOS, rien pour Linux)
    SED_SUFFIX=""
    if [[ "$OSTYPE" == "darwin"* ]]; then
        SED_SUFFIX=".bak"
    fi

    # Décommenter les lignes contenant "Mode Pipeline"
    sed -i "$SED_SUFFIX" '/Mode Pipeline/ s/^[[:space:]]*#//' "$CONFIG_FILE"

    # Commenter les lignes contenant "Mode Terminal" si elles ne sont pas déjà commentées
    sed -i "$SED_SUFFIX" '/Mode Terminal/ s/^[[:space:]]*\([^#]\)/#\1/' "$CONFIG_FILE"

    # Nettoyage du fichier de sauvegarde temporaire sur macOS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        rm -f "${CONFIG_FILE}.bak"
    fi

else
    echo "Fichier de configuration $CONFIG_FILE non trouvé. Vérifie que le fichier existe."
fi


# 2. Lancer Docker Compose pour démarrer le pipeline
echo "Lancement du mode Pipeline avec docker-compose..."
docker compose up --build
