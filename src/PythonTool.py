# import matplotlib.pyplot as plt  # plt.show(block=False) plt.savefig(path)
import io
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from contextlib import redirect_stdout

from dotenv import load_dotenv
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

load_dotenv()

# Configuration du logger global
LOG_LEVEL_ENV = os.getenv(
    "LOG_LEVEL_PythonTool"
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
    log_file=os.getenv("LOG_FILE_PythonTool"), max_size=5 * 1024 * 1024, backup_count=3
):
    """
    Configure un logger global pour suivre toutes les actions.
    """
    if not os.path.exists("../Logs"):
        os.makedirs("../Logs", exist_ok=True)
    logger = logging.getLogger("file_operations_logger")
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
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=max_size, backupCount=backup_count
    )
    file_handler.setLevel(LOG_LEVEL_MAP.get(LOG_LEVEL_ENV))
    file_handler.setFormatter(formatter)

    # Ajout des handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


# Initialisation du logger
logger = setup_logger()

active_observers = []
observed_directories = set()
created_paths = []  # Liste pour stocker les chemins des fichiers et dossiers créés


def move_and_create_links(source_files, target_directory, base_url):
    logger.info("Début du déplacement des fichiers et création des liens.")
    os.makedirs(target_directory, exist_ok=True)
    generated_links = []

    for file in source_files:
        try:
            file_name = os.path.basename(file)
            target_path = os.path.join(target_directory, file_name)
            shutil.move(file, target_path)
            file_url = os.path.join(
                base_url, os.path.relpath(target_path, start=target_directory)
            )
            generated_links.append(file_url)

            logger.info(f"Fichier déplacé vers {target_path}")
            logger.info(f"URL locale pour accéder au fichier : {file_url}")
        except Exception as e:
            logger.error(f"Erreur lors du déplacement du fichier {file} : {e}")
    return generated_links


class FileCreationHandler(FileSystemEventHandler):
    def on_any_event(self, event):
        logger.info(f"Événement détecté : {event.event_type} - {event.src_path}")

    def on_created(self, event):
        path = os.path.abspath(event.src_path)
        if not event.is_directory and path not in created_paths:
            created_paths.append(path)
            logger.info(f"Fichier créé : {event.src_path}")

    def on_modified(self, event):
        path = os.path.abspath(event.src_path)
        if not event.is_directory and path not in created_paths:
            created_paths.append(path)
            logger.info(f"Fichier modifié : {event.src_path}")

    def on_modified(self, event):
        # Si c'est un fichier (et non un dossier), on l'ajoute à la liste
        path = os.path.abspath(event.src_path)
        if not event.is_directory and path not in created_paths:
            created_paths.append(
                os.path.abspath(event.src_path)
            )  # Ajouter le chemin absolu du fichier modifié
            print(f"Fichier modifié: {event.src_path}")


def stop_all_observers():
    logger.info("Arrêt de tous les observateurs actifs.")
    global active_observers
    for observer in active_observers:
        observer.stop()
        observer.join()
    active_observers = []
    logger.info("Tous les observateurs ont été arrêtés et réinitialisés.")


# Fonction de surveillance
def watch_directories(directories, stop_event):
    logger.info("Démarrage de la surveillance des répertoires.")
    global active_observers, observed_directories
    stop_all_observers()

    for directory in directories:
        if directory in observed_directories:
            logger.warning(f"Le répertoire {directory} est déjà surveillé.")
            continue

        observed_directories.add(directory)
        event_handler = FileCreationHandler()
        observer = Observer()
        observer.schedule(event_handler, directory, recursive=True)
        observer.start()
        active_observers.append(observer)
        logger.info(f"Surveillance activée pour le répertoire : {directory}")

    try:
        while not stop_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        logger.warning("Surveillance interrompue par l'utilisateur.")
        stop_all_observers()


def parse_code(tool):
    code_match = re.search(r"```python\n([\s\S]*?)```", tool)
    if not code_match:
        logger.warning("Aucun code Python trouvé dans l'outil.")
        return ""
    code = code_match.group(1).strip()
    logger.info("Code Python extrait avec succès.")
    return code


def parse_and_execute_python_code(tool, context, sql_results):
    logger.info("Début de l'analyse et de l'exécution du code Python.")
    global created_paths, observed_directories
    created_paths = []
    observed_directories.clear()
    stop_all_observers()

    code = parse_code(tool)

    stop_event = threading.Event()
    directories_to_watch = ["./"]
    watch_thread = threading.Thread(
        target=watch_directories, args=(directories_to_watch, stop_event)
    )
    watch_thread.daemon = True
    watch_thread.start()

    time.sleep(1)
    imports = re.findall(
        r"^\s*import (\S+)|^\s*from (\S+) import (\S+)", code, re.MULTILINE
    )
    modules = set()
    for imp in imports:
        module = imp[0] or imp[1]
        modules.add(module)

    for module in modules:
        try:
            __import__(module)
        except ImportError:
            logger.info(f"Module {module} manquant, tentative d'installation...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", module])
                logger.info(f"Module {module} installé avec succès.")
            except subprocess.CalledProcessError as e:
                logger.error(f"Échec de l'installation du module {module} : {e}")
                context["error"] = f"Failed to install module {module}. Error: {e}"
                return context, "", []

    exec_context = globals()
    if sql_results:
        exec_context["sql_results"] = sql_results
        logger.info(
            f"Résultats SQL injectés dans le contexte d'exécution : {sql_results}"
        )

    buffer = io.StringIO()
    try:
        with redirect_stdout(buffer):
            exec(code, exec_context)
        output = buffer.getvalue()
        context["python_results"] = output
        logger.info("Exécution du code Python réussie.")
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution du code Python : {e}")
        context["error"] = f"Python error: {e}"
        output = ""

    time.sleep(5)
    stop_event.set()
    watch_thread.join()

    logger.info(f"Fichiers créés/modifiés détectés : {created_paths}")
    context["created_paths"] = move_and_create_links(
        created_paths,
        os.getenv("SAVE_DIRECTORY"),
        "http://localhost:8080/files",
    )
    logger.info(f"Liens créés : {context['created_paths']}")
    return context, output, context["created_paths"]
