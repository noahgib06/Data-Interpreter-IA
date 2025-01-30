import logging
import os
from logging.handlers import RotatingFileHandler

from dotenv import load_dotenv

load_dotenv()

# Configuration du logger global
LOG_LEVEL_ENV = os.getenv(
    "LOG_LEVEL_history_func"
)  # Changez pour INFO, EXCEPTION, DEBUG, ERROR. si nécessaire

# Mappage des niveaux de log
LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def setup_logger(
    log_file=os.getenv("LOG_FILE_history_func"),
    max_size=5 * 1024 * 1024,
    backup_count=3,
):
    """
    Configure un logger global pour suivre les actions de la fonction.
    """
    if not os.path.exists("../Logs"):
        os.makedirs("../Logs", exist_ok=True)
    logger = logging.getLogger("history_logger")
    logger.setLevel(LOG_LEVEL_MAP.get(LOG_LEVEL_ENV))

    # Format des logs
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    )

    # Handler console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOG_LEVEL_MAP.get(LOG_LEVEL_ENV))
    console_handler.setFormatter(formatter)

    # Handler fichier avec rotation
    file_handler = RotatingFileHandler(
        log_file, maxBytes=max_size, backupCount=backup_count
    )
    file_handler.setLevel(LOG_LEVEL_MAP.get(LOG_LEVEL_ENV))
    file_handler.setFormatter(formatter)

    # Ajout des handlers au logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


# Initialisation du logger
logger = setup_logger()


def add_message(history, role, content):
    """
    Ajoute un message à l'historique avec validation et gestion des caractères spéciaux.

    :param history: Liste contenant l'historique des messages.
    :param role: Rôle de l'auteur du message ('user', 'assistant', etc.).
    :param content: Contenu du message.
    :return: Historique mis à jour.
    """
    logger.info("Tentative d'ajout d'un message à l'historique.")

    # Validation des arguments
    if not isinstance(history, list):
        logger.error(
            "Type invalide pour 'history' : attendu list, obtenu %s.",
            type(history).__name__,
        )
        raise ValueError("L'historique doit être une liste.")

    if not isinstance(role, str) or not role.strip():
        logger.error("Rôle invalide : '%s'.", role)
        raise ValueError("Le rôle doit être une chaîne de caractères non vide.")

    if not isinstance(content, str):
        logger.error(
            "Contenu invalide : attendu str, obtenu %s.", type(content).__name__
        )
        raise ValueError("Le contenu doit être une chaîne de caractères.")

    # Normalisation et gestion des caractères spéciaux
    try:
        logger.debug("Validation et nettoyage du contenu.")
        content_cleaned = (
            content.strip()
        )  # Retire les espaces en trop au début et à la fin
    except Exception as e:
        logger.exception("Erreur lors du nettoyage du contenu : %s", e)
        raise ValueError("Impossible de traiter le contenu fourni.")

    # Ajout du message à l'historique
    message = {"role": role, "content": content_cleaned}
    history.append(message)
    logger.info("Message ajouté avec succès : %s", message)

    # Debugging avancé pour inspecter l'historique actuel
    logger.debug("État actuel de l'historique : %s", history)

    return history
