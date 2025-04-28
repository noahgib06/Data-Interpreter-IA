# import matplotlib.pyplot as plt  # plt.show(block=False) plt.savefig(path)
import io
import logging
import logging.handlers
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from contextlib import redirect_stdout

import requests
from dotenv import load_dotenv
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

load_dotenv()

# Configuration du logger global
LOG_LEVEL_ENV = os.getenv("LOG_LEVEL_PythonTool", "INFO")  # Valeur par défaut: INFO

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
    # Créer le répertoire de logs s'il n'existe pas
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # Nettoyer le chemin du fichier de log
    if log_file:
        log_file = log_file.strip('"')  # Supprimer les guillemets
        log_file = os.path.join(log_dir, os.path.basename(log_file))

    logger = logging.getLogger("file_operations_logger")
    logger.setLevel(LOG_LEVEL_MAP.get(LOG_LEVEL_ENV, logging.INFO))

    # Format des logs
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    )

    # Handler console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOG_LEVEL_MAP.get(LOG_LEVEL_ENV, logging.INFO))
    console_handler.setFormatter(formatter)

    # Handler fichier avec rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=max_size, backupCount=backup_count
    )
    file_handler.setLevel(LOG_LEVEL_MAP.get(LOG_LEVEL_ENV, logging.INFO))
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
OPENWEBUI_API = os.getenv("OPENWEBUI_API", "").rstrip("/") + "/"
OPENWEBUI_API_KEY = os.getenv("OPENWEBUI_API_KEY")
DOWNLOAD_URL = os.getenv("DOWNLOAD_URL", "").rstrip("/") + "/"


def get_mime_type(file_path):
    """
    Détermine le type MIME d'un fichier basé sur son extension.
    """
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type or "application/octet-stream"


def download_file_openwebui(file: str):
    """
    Télécharge un fichier à partir d'un nom de fichier.
    Utilise un système de mapping pour éviter les téléchargements redondants.
    """
    try:
        file_path = os.path.abspath(file)

        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {OPENWEBUI_API_KEY}",
        }
        url = f"{OPENWEBUI_API}files/"
        logger.info(f"Uploading file to OpenWebUI API: {url}")
        logger.info(f"File path: {file}")

        file_id = ""
        try:
            mime_type = get_mime_type(file_path)
            with open(file, "rb") as f:
                files = {"file": (os.path.basename(file), f, mime_type)}
                logger.info(f"Files: {files}")
                logger.info(f"Headers: {headers}")
                logger.info(f"URL: {url}")

                response = requests.post(url, headers=headers, files=files)
                logger.info(f"Upload response status: {response.status_code}")

                if response.status_code != 200:
                    logger.error(
                        f"File upload failed: {response.status_code} - {response.text}"
                    )
                    return {"error": f"File upload failed: {response.status_code}"}

                response_data = response.json()
                logger.info(f"Upload response: {response_data}")
                file_id = response_data.get("id", "")

                if not file_id:
                    logger.error("No file ID returned from upload")
                    return {"error": "No file ID returned from upload"}

                logger.info(f"Added file mapping: {file_path} -> {file_id}")
        except Exception as e:
            logger.error(f"Error uploading file: {str(e)}")
            return {"error": f"Error uploading file: {str(e)}"}

        download_url = f"{DOWNLOAD_URL}files/{file_id}/content"
        logger.info(f"Download URL: {download_url}")
        return {"download_url": download_url}
    except Exception as e:
        logger.error(f"Error in download_file_openwebui: {str(e)}")
        return {"error": f"Error in download_file_openwebui: {str(e)}"}


def move_and_create_links(source_files, target_directory):
    """
    Moves files to the target directory and generates accessible local URLs.
    Ensures the target directory exists before moving files.
    """
    logger.info("📂 Starting file move and link generation process.")
    os.makedirs(target_directory, exist_ok=True)
    generated_links = []
    processed_files = set()  # Pour éviter les doublons

    for file in source_files:
        try:
            # Éviter les doublons et les fichiers déjà dans uploads
            if file in processed_files or "uploads" in file:
                continue
            processed_files.add(file)

            if OPENWEBUI_API is not None:
                res = download_file_openwebui(file)
                if "download_url" in res:
                    generated_links.append(res["download_url"])
                    logger.info(
                        f"🔗 Generated local URL for file access: {res['download_url']}"
                    )
            else:
                file_name = os.path.basename(file)
                target_path = os.path.join(target_directory, file_name)
                shutil.move(file, target_path)
                logger.info(f"✅ File moved to {target_path}")

        except Exception as e:
            logger.error(f"❌ Error moving file {file}: {e}")

    return generated_links


class FileCreationHandler(FileSystemEventHandler):
    """
    Handles file system events such as file creation and modification.
    Monitors changes and logs detected events.
    """

    def is_valid_file(self, path):
        """
        Vérifie si le fichier doit être traité ou ignoré.
        """
        ignored_extensions = {
            ".log",  # Fichiers log
            ".db",  # Bases de données
            ".db-journal",  # Fichiers temporaires SQLite
            ".tmp",  # Fichiers temporaires
            ".temp",  # Fichiers temporaires
            ".bak",  # Fichiers backup
            ".swp",  # Fichiers swap vim
            ".lock",  # Fichiers lock
            "~",  # Fichiers backup
        }

        # Ignorer les fichiers dans le dossier uploads et les fichiers système
        return (
            not any(path.endswith(ext) for ext in ignored_extensions)
            and "uploads" not in path
            and not os.path.basename(path).startswith(".")
        )

    def on_any_event(self, event):
        """
        Logs any detected file system event.
        """
        # logger.info(
        #    f"📌 Event detected: {event.event_type} - {event.src_path}"
        # )  # INFO: Log general event

    def on_created(self, event):
        """
        Logs newly created files and adds them to the tracking list.
        """
        path = os.path.abspath(event.src_path)
        if (
            not event.is_directory
            and path not in created_paths
            and self.is_valid_file(path)
        ):
            created_paths.append(path)
            logger.info(f"🆕 File created: {event.src_path}")  # INFO: Log file creation

    def on_modified(self, event):
        """
        Logs modified files and adds them to the tracking list.
        """
        path = os.path.abspath(event.src_path)
        if (
            not event.is_directory
            and path not in created_paths
            and self.is_valid_file(path)
        ):
            created_paths.append(path)
            logger.info(
                f"✏️ File modified: {event.src_path}"
            )  # INFO: Log file modification


def stop_all_observers():
    """
    Stops all active file system observers and resets the observer list.
    Ensures that each observer is properly stopped and joined.
    """
    logger.info("🛑 Stopping all active observers.")  # INFO: Process start

    global active_observers
    for observer in active_observers:
        observer.stop()
        observer.join()  # Ensure proper shutdown
    active_observers = []

    logger.info(
        "✅ All observers have been stopped and reset."
    )  # INFO: Process completion


def watch_directories(directories, stop_event):
    """
    Monitors specified directories for file creation and modification events.
    Uses a background observer to track changes until stopped.
    """
    logger.info("👀 Starting directory monitoring.")  # INFO: Process start

    global active_observers, observed_directories
    stop_all_observers()  # Ensure previous observers are stopped before starting new ones

    for directory in directories:
        if directory in observed_directories:
            logger.warning(
                f"⚠️ Directory already being monitored: {directory}"
            )  # WARNING: Avoid duplicate monitoring
            continue

        observed_directories.add(directory)
        event_handler = FileCreationHandler()
        observer = Observer()
        observer.schedule(event_handler, directory, recursive=True)
        observer.start()
        active_observers.append(observer)

        logger.info(
            f"✅ Monitoring activated for directory: {directory}"
        )  # INFO: Monitoring started

    try:
        while not stop_event.is_set():
            time.sleep(1)  # Keep monitoring while stop event is not triggered
    except KeyboardInterrupt:
        logger.warning(
            "⛔ Monitoring interrupted by user."
        )  # WARNING: User interruption detected
        stop_all_observers()


def parse_code(tool):
    """
    Extracts Python code from a given tool response using regex.
    Searches for code blocks formatted with triple backticks (```python ... ```).
    """
    code_match = re.search(r"```python\n([\s\S]*?)```", tool)

    if not code_match:
        logger.warning(
            "⚠️ No Python code found in the tool response."
        )  # WARNING: No code detected
        return ""

    code = code_match.group(1).strip()
    logger.info("✅ Python code successfully extracted.")  # INFO: Extraction successful
    return code


def clean_sql_results(sql_results):
    """
    Nettoie et valide les résultats SQL pour éviter les problèmes de syntaxe.
    """
    if not isinstance(sql_results, list):
        return []

    cleaned_results = []
    for result in sql_results:
        if isinstance(result, dict):
            # S'assurer que les clés 'results' et 'metadata' existent
            if "results" in result and isinstance(result["results"], list):
                cleaned_result = {
                    "results": result["results"],
                    "metadata": result.get("metadata", {}),
                }
                cleaned_results.append(cleaned_result)

    return cleaned_results


def parse_and_execute_python_code(tool, context, sql_results):
    """
    Parses and executes Python code extracted from the given tool response.
    Automatically handles missing module installations and tracks created/modified files.
    """
    logger.info("🚀 Starting Python code analysis and execution.")

    # Reset tracked paths and stop any running observers
    global created_paths, observed_directories
    created_paths = []
    observed_directories.clear()
    stop_all_observers()

    # Extract Python code from the tool response
    code = parse_code(tool)

    # Start a background file system observer
    stop_event = threading.Event()
    directories_to_watch = ["./"]
    watch_thread = threading.Thread(
        target=watch_directories, args=(directories_to_watch, stop_event)
    )
    watch_thread.daemon = True
    watch_thread.start()

    time.sleep(1)  # Allow observer to initialize

    # Identify imported modules from the extracted code
    imports = re.findall(
        r"^\s*import (\S+)|^\s*from (\S+) import (\S+)", code, re.MULTILINE
    )
    modules = set()
    for imp in imports:
        module = imp[0] or imp[1]
        modules.add(module)

    # Attempt to import required modules and install missing ones
    for module in modules:
        try:
            __import__(module)
        except ImportError:
            logger.info(
                f"⚠️ Missing module detected: {module}. Attempting installation..."
            )
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", module])
                logger.info(f"✅ Successfully installed module: {module}")
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Failed to install module {module}: {e}")
                context["error"] = f"Failed to install module {module}. Error: {e}"
                return context, "", []

    # Set up execution context
    exec_context = globals()
    if sql_results:
        # Nettoyer les résultats SQL avant de les injecter dans le contexte
        cleaned_sql_results = clean_sql_results(sql_results)
        exec_context["sql_results"] = cleaned_sql_results
        logger.info(
            f"📊 Injected SQL results into execution context: {cleaned_sql_results}"
        )

    buffer = io.StringIO()
    try:
        # Execute the extracted Python code and capture output
        with redirect_stdout(buffer):
            exec(code, exec_context)
        output = buffer.getvalue()
        context["python_results"] = output
        logger.info("✅ Python code executed successfully.")
    except Exception as e:
        logger.error(f"❌ Error executing Python code: {e}")
        context["error"] = f"Python error: {e}"
        output = ""

    # Stop file system monitoring
    time.sleep(5)
    stop_event.set()
    watch_thread.join()

    logger.info(f"📂 Detected created/modified files: {created_paths}")
    context["created_paths"] = move_and_create_links(
        created_paths, os.getenv("SAVE_DIRECTORY")
    )
    logger.info(f"🔗 Generated file links: {context['created_paths']}")

    return context, output, context["created_paths"]
