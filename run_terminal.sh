#!/bin/bash
# Script pour lancer le mode Terminal


# 0. Vérifier si un environnement virtuel existe déjà
echo "Vérification de l'environnement virtuel..."

# Fonction pour détecter un venv actif
venv_actif() {
    [[ -n "$VIRTUAL_ENV" ]]
}

# Fonction pour lister les dossiers contenant potentiellement un venv
detect_venvs() {
    find . -maxdepth 1 -type d -exec test -f "{}/bin/activate" \; -print
}

# Si aucun venv actif
if ! venv_actif; then
    echo "Aucun environnement virtuel actif détecté."

    EXISTING_VENVS=$(detect_venvs)

    if [[ -n "$EXISTING_VENVS" ]]; then
        echo "Environnements virtuels détectés :"
        echo "$EXISTING_VENVS"
        echo "Souhaitez-vous en utiliser un ? (y/n)"
        read -r use_existing
        if [[ "$use_existing" == "y" ]]; then
            echo "Entrez le chemin vers l'environnement à activer :"
            read -r venv_path
            source "$venv_path/bin/activate"
        fi
    fi

    if ! venv_actif; then
        echo "Souhaitez-vous créer un nouvel environnement virtuel ? (y/n)"
        read -r create_venv
        if [[ "$create_venv" == "y" ]]; then
            echo "Nom de l'environnement virtuel à créer (par ex. venv) :"
            read -r venv_name
            python3 -m venv "$venv_name"
            source "$venv_name/bin/activate"
            echo "Environnement $venv_name activé."
        fi
    fi
else
    echo "Environnement virtuel déjà actif : $VIRTUAL_ENV"
fi

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


