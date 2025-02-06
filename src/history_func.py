import logging
import os
from logging.handlers import RotatingFileHandler

import duckdb
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


def add_message(path, role, content):
    """
    Ajoute un message dans l'historique d'une DB spécifique.
    """
    conn = duckdb.connect(path)

    # Vérifie si la table existe (sinon on la crée)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY,  -- ✅ Supprimé AUTOINCREMENT
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    # Insère le message avec une gestion auto de l'ID
    conn.execute(
        """
        INSERT INTO chat_history (id, role, content) 
        SELECT COALESCE(MAX(id), 0) + 1, ?, ? FROM chat_history
    """,
        (role, content),
    )

    conn.close()


def get_history(path):
    """
    Récupère l'historique des conversations d'une base DuckDB spécifique.
    """
    conn = duckdb.connect(path)

    # Vérifie si la table existe pour éviter une erreur
    result = conn.execute(
        """
        SELECT COUNT(*) FROM information_schema.tables 
        WHERE table_name = 'chat_history'
    """
    ).fetchone()

    if result[0] == 0:
        conn.close()
        return []  # Retourne un historique vide si la table n'existe pas encore

    # Récupère l'historique trié par date
    history = conn.execute(
        """
        SELECT role, content, timestamp 
        FROM chat_history 
        ORDER BY timestamp ASC
    """
    ).fetchall()

    conn.close()
    return history


def setup_history_database(path):
    """
    Initialise la base de données DuckDB et crée la table d'historique des conversations si elle n'existe pas.
    """
    conn = duckdb.connect(path)

    # Vérifier si la table existe déjà
    table_exists = (
        conn.execute(
            """
        SELECT COUNT(*) FROM information_schema.tables 
        WHERE table_name = 'chat_history'
    """
        ).fetchone()[0]
        > 0
    )

    if table_exists:
        print(f"✅ La base de données `{path}` existe déjà avec `chat_history`.")
        conn.close()
        return  # Sortir immédiatement

    # Si la table n'existe pas, la créer
    conn.execute(
        """
        CREATE TABLE chat_history (
            id INTEGER PRIMARY KEY,  -- ✅ Supprimé AUTOINCREMENT
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    conn.close()
    print(f"✅ Base `{path}` initialisée avec la table `chat_history`.")
