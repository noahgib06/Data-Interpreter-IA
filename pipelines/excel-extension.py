import os
import sys

# Add application path to system path
sys.path.append("/app/src")
import logging
import shutil
import typing
from logging.handlers import RotatingFileHandler
from typing import Set

import duckdb
from dotenv import load_dotenv
from fastapi import HTTPException
from langchain_ollama import OllamaLLM
from pydantic import BaseModel
from urllib3.exceptions import NewConnectionError

from history_func import add_message
from LlmGeneration import (command_r_plus_plan,
                           generate_final_response_with_llama,
                           generate_tools_with_llm)
from SetupDatabase import prepare_database, remove_database_file
from SqlTool import get_schema

load_dotenv("/app/.env")

# Set global log level and log file from environment variables
LOG_LEVEL_ENV = os.getenv("LOG_LEVEL_main", "INFO")
LOG_FILE_MAIN = os.getenv("LOG_FILE_main", "Logs/main.log")

# Map log levels to logging module
LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


# Configure logger
def setup_logger(log_file, max_size=10 * 1024 * 1024, backup_count=5):
    """
    Set up a logger with rotating file and console handlers.
    """
    if not os.path.exists("Logs"):
        os.makedirs("Logs", exist_ok=True)

    logger = logging.getLogger("main_logger")
    logger.setLevel(LOG_LEVEL_MAP.get(LOG_LEVEL_ENV, logging.INFO))

    # Log format
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(LOG_LEVEL_MAP.get(LOG_LEVEL_ENV, logging.INFO))
    console_handler.setFormatter(formatter)

    # Rotating file handler
    file_handler = RotatingFileHandler(
        log_file, maxBytes=max_size, backupCount=backup_count
    )
    file_handler.setLevel(LOG_LEVEL_MAP.get(LOG_LEVEL_ENV, logging.INFO))
    file_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


logger = setup_logger(LOG_FILE_MAIN)


# Test connection to Ollama
def test_ollama_connection():
    try:
        model = OllamaLLM(
            model="llama3.2:latest", base_url="http://host.docker.internal:11434"
        )
        response = model.generate(prompts=["Test message"])
        logger.info(f"Test message response: {response}")
    except Exception as e:
        logger.error(f"Failed to connect to Ollama: {e}", exc_info=True)
        raise RuntimeError(
            "Failed to connect to Ollama. Check the service URL and availability."
        )


class Pipeline:
    class Valves(BaseModel):
        LLAMAINDEX_OLLAMA_BASE_URL: str = "http://host.docker.internal:11434"
        LLAMAINDEX_MODEL_NAME: str = os.getenv("REASONING_MODEL")
        LLAMAINDEX_RAG_MODEL_NAME: str = os.getenv("DATABASE_MODEL")
        LLAMAINDEX_CONTEXT_MODEL_NAME: str = os.getenv("CONTEXTUALISATION_MODEL")
        # FICHIERS: str = ""

    def __init__(self):
        self.sql_results = None
        self.python_results = None
        # On stocke deux ensembles pour mieux gérer :
        #  - ceux qu'on connaît déjà dans shared_data
        #  - ceux qu'on connaît déjà dans uploads
        self.shared_data_directory = "/app/shared_data/data/"
        self.upload_directory = "/app/backend/data/uploads/"
        self.known_shared_files: Set[str] = set()
        self.known_upload_files: Set[str] = set()
        self.history = []
        self.valves = self.Valves(
            LLAMAINDEX_OLLAMA_BASE_URL=os.getenv(
                "LLAMAINDEX_OLLAMA_BASE_URL", "http://host.docker.internal:11434"
            ),
            LLAMAINDEX_MODEL_NAME=os.getenv(
                "LLAMAINDEX_MODEL_NAME", os.getenv("REASONING_MODEL")
            ),
            LLAMAINDEX_RAG_MODEL_NAME=os.getenv(
                "LLAMAINDEX_RAG_MODEL_NAME", os.getenv("DATABASE_MODEL")
            ),
            LLAMAINDEX_CONTEXT_MODEL_NAME=os.getenv(
                "LLAMAINDEX_CONTEXT_MODEL_NAME", os.getenv("CONTEXTUALISATION_MODEL")
            ),
        )
        print(f"Valves initialized: {self.valves}")

        try:
            self.database_model = OllamaLLM(
                model=self.valves.LLAMAINDEX_RAG_MODEL_NAME,
                base_url="http://host.docker.internal:11434",
            )
            self.reasoning_model = OllamaLLM(
                model=self.valves.LLAMAINDEX_MODEL_NAME,
                base_url="http://host.docker.internal:11434",
            )
            self.contextualisation_model = OllamaLLM(
                model=self.valves.LLAMAINDEX_CONTEXT_MODEL_NAME,
                base_url="http://host.docker.internal:11434",
            )
            test_ollama_connection()
            print("Models initialized successfully.")
        except NewConnectionError as conn_error:
            print(f"Connection to Ollama failed: {conn_error}")
            raise RuntimeError(
                "Failed to connect to Ollama. Check the service URL and availability."
            )
        except Exception as e:
            print(f"Error initializing models: {e}")
            raise RuntimeError("General error during model initialization.")

    def scan_directory_shared(self) -> set:
        """Scanne le répertoire shared_data et retourne le nom de fichier sans UUID."""
        files = set()
        for f in os.listdir(self.shared_data_directory):
            pathf = os.path.join(self.shared_data_directory, f)
            if os.path.isfile(pathf) and f.endswith(
                (".xls", ".xlsx", ".csv", ".json", ".pdf", ".py")
            ):
                files.add(pathf)
        logging.debug(f"Fichiers scannés dans shared_data: {files}")
        return files

    def scan_directory_uploads(self) -> set:
        """Retourne un set de tuples (nom_fichier_complet, nom_fichier_sans_UUID)."""
        files = set()
        for f in os.listdir(self.upload_directory):
            pathf = os.path.join(self.upload_directory, f)
            if os.path.isfile(pathf) and f.endswith(
                (".xls", ".xlsx", ".csv", ".json", ".pdf", ".py")
            ):
                clean_name = self.extract_filename_without_uuid(f)
                files.add((f, clean_name))
        logging.debug(f"Fichiers scannés dans uploads: {files}")
        return files

    @staticmethod
    def extract_filename_without_uuid(filename: str) -> str:
        """Extrait le nom de fichier sans l'UUID éventuel en préfixe."""
        parts = filename.split("_", 1)
        return parts[1] if len(parts) > 1 else filename

    async def on_startup(self):
        """Au démarrage, charge les fichiers de shared_data et retient les fichiers dans uploads."""
        logging.info("Démarrage du processus de scan des répertoires.")
        remove_database_file()

        shared_files = self.scan_directory_shared()
        if shared_files:
            filepaths = [
                os.path.join(self.shared_data_directory, f) for f in shared_files
            ]
            prepare_database(filepaths)
            logging.info(
                f"Chargement des fichiers suivants dans la base de données: {shared_files}"
            )
        else:
            logging.warning("Aucun fichier à charger dans shared_data.")

        self.known_shared_files = shared_files
        all_uploads = self.scan_directory_uploads()
        self.known_upload_files = {clean for (_, clean) in all_uploads}
        logging.info(f"Fichiers présents dans uploads: {all_uploads}")

    def detect_and_process_changes(self):
        """Détecte les fichiers nouvellement ajoutés dans uploads et gère la BD
        si on en ajoute ou en retire de shared_data.
        """
        # 1) État actuel du dossier shared_data (chemins complets)
        current_shared_files = self.scan_directory_shared()
        print("Current shared files: ", current_shared_files)

        # 2) État actuel du dossier uploads (retourne (raw, clean))
        current_upload_files = self.scan_directory_uploads()
        print("Current upload files: ", current_upload_files)
        current_upload_clean_names = {clean for (_, clean) in current_upload_files}

        # 3) Identifier les nouveaux "clean names" arrivés dans uploads
        new_uploads = current_upload_clean_names - self.known_upload_files
        if new_uploads:
            logging.info(f"Nouveaux fichiers détectés dans uploads: {new_uploads}")
            for raw_file, clean_file in current_upload_files:
                # On traite seulement les clean_file qui sont vraiment nouveaux
                if clean_file in new_uploads:
                    # Chemin complet du fichier dans uploads
                    upload_file_path = os.path.join(self.upload_directory, raw_file)
                    # Chemin final "net" dans shared_data
                    final_path = os.path.join(self.shared_data_directory, clean_file)

                    # Vérifier si ce fichier "propre" n'existe pas déjà en shared_data
                    if final_path not in current_shared_files:
                        try:
                            shutil.copy2(upload_file_path, final_path)
                            if os.path.exists(final_path):
                                # Appeler la fonction de DB sur le fichier copié
                                prepare_database([final_path])
                                logging.info(
                                    f"{raw_file} copié et importé dans la base de données."
                                )
                            else:
                                logging.warning(f"Échec de la copie pour {raw_file}.")
                        except Exception as e:
                            logging.error(f"Erreur en copiant {raw_file}: {e}")
                    else:
                        logging.debug(
                            f"{clean_file} est déjà présent dans shared_data, on ignore."
                        )

        # 4) Identifier si des fichiers ont été supprimés de shared_data
        removed_from_shared = self.known_shared_files - current_shared_files
        if removed_from_shared:
            logging.info(
                f"Fichiers supprimés détectés dans shared_data: {removed_from_shared}"
            )
            conn = duckdb.connect(os.getenv("DB_FILE"))
            for removed_file_path in removed_from_shared:
                removed_filename = os.path.basename(removed_file_path)
                logging.info(
                    f"Suppression des données liées à {removed_filename} de la BD."
                )
                try:
                    existing_tables = conn.execute("SHOW TABLES").fetchall()
                    for (table_name,) in existing_tables:
                        # On compare en fonction du nom de base, ex. "test_excel_for_program"
                        if table_name.startswith(
                            os.path.splitext(removed_filename)[0].lower()
                        ):
                            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
                            logging.info(f"Table {table_name} supprimée de la BD.")
                except Exception as e:
                    logging.error(
                        f"Erreur lors de la suppression de {removed_filename} : {e}"
                    )
            conn.close()

        # 5) Mettre à jour les sets connus pour la prochaine itération
        self.known_shared_files = current_shared_files
        self.known_upload_files = current_upload_clean_names
        logging.debug("Mise à jour des états connus des fichiers terminée.")

    async def inlet(self, body: dict, user: typing.Optional[dict] = None) -> dict:
        if self.valves.LLAMAINDEX_RAG_MODEL_NAME is not None:
            self.database_model = OllamaLLM(
                model=self.valves.LLAMAINDEX_RAG_MODEL_NAME,
                base_url="http://host.docker.internal:11434",
            )
        if self.valves.LLAMAINDEX_MODEL_NAME is not None:
            self.reasoning_model = OllamaLLM(
                model=self.valves.LLAMAINDEX_MODEL_NAME,
                base_url="http://host.docker.internal:11434",
            )
        if self.valves.LLAMAINDEX_CONTEXT_MODEL_NAME is not None:
            self.contextualisation_model = OllamaLLM(
                model=self.valves.LLAMAINDEX_CONTEXT_MODEL_NAME,
                base_url="http://host.docker.internal:11434",
            )
        # On ne fait la détection et le traitement des changements que sur les nouveaux + suppressions de shared_data
        self.detect_and_process_changes()

        # Extraire les fichiers du corps de la requête
        return body

    async def on_shutdown(self):
        print("Server shutting down...")

    def verify_and_reflect(self, context, python_results):
        logger.info("Verifying and reflecting on execution results...")
        if context["sql_results"]:
            invalid_sql_results = [
                res
                for res in context["sql_results"]
                if not isinstance(res, dict) or not res
            ]
            if invalid_sql_results:
                logger.warning(f"Invalid SQL results detected: {invalid_sql_results}")
                return "Résultats SQL incorrects"

            if "requires_python_analysis" in context:
                return "Passer à Python"
            else:
                return "Terminé"

        elif python_results:
            logger.info("Python results found.")
            return "Terminé"

        logger.debug("Continuing...")
        return "Continuer"

    def llm_data_interpreter(self, question, schema, initial_context):
        logger.info(f"Starting LLM data interpreter with question: {question}")
        context = initial_context
        schema = get_schema(duckdb.connect(os.getenv("DB_FILE")))
        self.history = add_message(self.history, "user", question)
        context["sql_results"] = context.get("sql_results", [])
        context["python_results"] = context.get("python_results", [])
        while True:
            logger.debug("Generating plan...")
            self.python_results = None
            self.sql_results = None
            plan, python_code = command_r_plus_plan(
                question, schema, self.contextualisation_model, self.history
            )
            context, self.python_results, self.sql_results, files_generated = (
                generate_tools_with_llm(
                    plan,
                    schema,
                    context,
                    self.sql_results,
                    self.python_results,
                    self.database_model,
                    self.reasoning_model,
                    python_code,
                )
            )
            logger.debug(f"Results: {context['sql_results']}, {self.python_results}")
            reflection = self.verify_and_reflect(context, self.python_results)
            logger.debug(f"Reflection result: {reflection}")

            if "Terminé" in reflection:
                logger.info("Execution process completed.")
                break
            break
        final_response = generate_final_response_with_llama(
            context,
            self.sql_results,
            self.python_results,
            self.reasoning_model,
            files_generated,
            self.history,
        )
        self.history = add_message(self.history, "assistant", final_response)
        return final_response

    def pipe(
        self,
        user_message: str,
        model_id: str = None,
        messages: typing.List[dict] = None,
        body: dict = None,
    ) -> typing.Union[str, typing.Generator, typing.Iterator]:
        try:
            schema = get_schema(duckdb.connect(os.getenv("DB_FILE")))
            initial_context = {"question": user_message}
            return self.llm_data_interpreter(user_message, schema, initial_context)
        except Exception as e:
            print(f"Error executing request: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))


pipeline = Pipeline()
