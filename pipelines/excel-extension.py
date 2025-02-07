import os
import sys

# Add application path to system path
sys.path.append("/app/src")
import logging
import re
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

from history_func import (add_conversation_with_embedding,
                          retrieve_similar_conversations,
                          setup_history_database)
from LlmGeneration import (command_r_plus_plan,
                           generate_final_response_with_llama,
                           generate_tools_with_llm)
from SetupDatabase import prepare_database
from SqlTool import get_schema

load_dotenv("/app/.env")

# Set global log level and log file from environment variables
LOG_LEVEL_ENV = os.getenv("LOG_LEVEL_main", "INFO")
LOG_FILE_MAIN = os.getenv("LOG_FILE_main", "Logs/main.log")
UUID_REGEX = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_"
)


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
    if not os.path.exists("failed/Logs"):
        os.makedirs("failed/Logs", exist_ok=True)

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
        self.shared_data_directory = "/srv/data/"
        self.upload_directory = "/app/backend/data/uploads/"
        self.known_shared_files: Set[str] = set()
        self.known_upload_files: Set[str] = set()
        self.history = []
        self.chat_id = None
        self.custom_db_path = None
        self.custom_history_path = None
        self.latest_files_by_chat = {}
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
        logger.info(f"Valves initialized: {self.valves}")

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
            logger.info("Models initialized successfully.")
        except NewConnectionError as conn_error:
            logger.error(f"Connection to Ollama failed: {conn_error}")
            raise RuntimeError(
                "Failed to connect to Ollama. Check the service URL and availability."
            )
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise RuntimeError("General error during model initialization.")

    def scan_directory_shared(self) -> set:
        """Scanne le répertoire `/srv/data/{chat_id}` et retourne les fichiers valides."""
        files = set()

        # ✅ Vérifier si `chat_id` est bien défini
        if not self.chat_id:
            logging.warning(
                "❌ Aucun `chat_id` défini ! Impossible de scanner les fichiers."
            )
            return files  # Retourne un set vide

        path = os.path.join(self.shared_data_directory, self.chat_id)

        # ✅ Vérifier si le dossier existe avant de le scanner
        if not os.path.exists(path):
            logging.warning(f"❌ Dossier non trouvé : {path}")
            return files  # Retourne un set vide

        # ✅ Scanner uniquement les fichiers autorisés
        for f in os.listdir(path):
            pathf = os.path.join(path, f)
            if os.path.isfile(pathf) and f.endswith(
                (".xls", ".xlsx", ".csv", ".json", ".pdf", ".py")
            ):
                files.add(pathf)

        logging.debug(f"📂 Fichiers scannés dans `{path}` : {files}")
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
        pass

    def detect_and_process_changes(self):
        """
        Ajoute les fichiers à /srv/data/{chat_id} si adding: False,
        et supprime éventuellement de la base de données ceux qui ne sont plus présents.
        (Ici la suppression est omise ; à adapter selon vos besoins.)
        """
        logger.info(f"🔍detect_and_process_changes() lancé")
        logger.debug(f"📌 Valeur actuelle de self.chat_id: {self.chat_id}")

        if not self.chat_id:
            logging.warning(
                "❌ Aucun chat_id défini ! Impossible de traiter les fichiers."
            )
            return

        chat_path = os.path.join(self.shared_data_directory, self.chat_id)
        logger.debug(f"🔍 Chat Path: {chat_path}")

        # ✅ Assurer que le dossier existe
        if not os.path.exists(chat_path):
            os.makedirs(chat_path, exist_ok=True)
            logging.info(f"📁 Dossier créé : {chat_path}")

        # 🔍 Liste des fichiers actuellement présents (si vous voulez gérer une suppression)
        existing_files = set(os.listdir(chat_path))
        existing_uuids = {f.split("_", 1)[0] for f in existing_files if "_" in f}
        logger.debug(f"📂 DEBUG: Fichiers actuels dans {chat_path}: {existing_files}")
        logger.debug(f"📂 DEBUG: UUIDs extraits des fichiers actuels: {existing_uuids}")

        # ✅ Liste à jour après gestion
        updated_files = []
        new_filepaths_to_db = []  # Liste des nouveaux fichiers pour prepare_database()

        logger.info("🔄 Début du traitement des fichiers...")

        # Parcourir la liste des fichiers du dernier message
        for file_data in self.latest_files_by_chat[self.chat_id]:
            file_id = file_data.get("file_id")
            filename = file_data.get("filename")

            logger.info(
                f"📝 Traitement du fichier : file_id={file_id}, filename={filename}"
            )

            if file_id is None or filename is None:
                logger.error(
                    f"🚨 ERREUR: file_id ou filename est None ! Données: {file_data}"
                )
                continue  # On saute ce fichier corrompu

            # Si 'adding' est déjà True, on ne refait pas la copie
            if file_data.get("adding", False) is True:
                logger.info(f"⚠️ Fichier {filename} déjà ajouté, on skip.")
                updated_files.append(file_data)
                continue

            # 🔥 Vérification du chemin source et destination
            source_path = os.path.join(self.upload_directory, filename)
            dest_path = os.path.join(chat_path, filename)

            logger.info(
                f"📥 Vérification de la copie depuis {source_path} vers {dest_path}"
            )

            # Copie du fichier si pas encore ajouté
            if not file_data.get("adding", False):
                if os.path.exists(source_path):
                    shutil.copy(source_path, dest_path)
                    logging.info(f"✅ Fichier copié: {source_path} → {dest_path}")
                    file_data["adding"] = True  # Marquer comme ajouté
                    new_filepaths_to_db.append(dest_path)  # Ajouter pour traitement BD
                else:
                    logging.warning(
                        f"⚠️ Fichier source introuvable: {source_path}, fichier non copié."
                    )

            updated_files.append(file_data)

        # 🛠️ Mettre à jour la liste des fichiers actifs
        self.latest_files_by_chat[self.chat_id] = updated_files

        # 📊 Si de nouveaux fichiers ont été ajoutés, on les envoie à la base de données
        if new_filepaths_to_db:
            logging.info(
                f"📊 Envoi des nouveaux fichiers à la base de données: {new_filepaths_to_db}"
            )
            prepare_database(filepaths=new_filepaths_to_db, collection_id=self.chat_id)

    def get_existing_files_by_uuid(self, chat_path: str) -> set:
        """Retourne un set des UUID des fichiers existants dans `/srv/data/{chat_id}/`."""
        existing_uuids = set()
        for f in os.listdir(chat_path):
            if "_" in f:
                file_uuid = f.split("_", 1)[0]  # Extraire l'UUID du nom de fichier
                existing_uuids.add(file_uuid)
        return existing_uuids

    def get_latest_files(self, chat_id):
        """Retourne tous les fichiers du dernier message pour cette conversation"""
        return self.latest_files_by_chat.get(chat_id, [])

    async def inlet(self, body: dict, user: typing.Optional[dict] = None) -> dict:
        logger.debug(f"📂 DEBUG: Body reçu dans `inlet()` → {body}")

        self.chat_id = body.get("metadata", {}).get("chat_id", "unknown_chat")

        if self.chat_id is None:
            logger.error("🚨 ERREUR: `chat_id` est None, correction en cours...")
            self.chat_id = "unknown_chat"

        path = os.path.join("/srv/data", self.chat_id)

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

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
        logger.debug(f"📂 DEBUG: Body reçu dans inlet() → {body}")

        self.chat_id = body.get("metadata", {}).get("chat_id", "unknown_chat")
        if self.chat_id is None:
            logger.error("🚨 ERREUR: chat_id est None, correction en cours...")
            self.chat_id = "unknown_chat"

        path = os.path.join(self.shared_data_directory, self.chat_id)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        # Définition de LLM fictives si besoin
        if self.valves.LLAMAINDEX_RAG_MODEL_NAME is not None:
            # ...
            pass

        logger.debug(f"📌 DEBUG: chat_id extrait → {self.chat_id}")

        files = body.get("metadata", {}).get("files", [])
        logger.debug(f"📂 DEBUG: files extrait → {files}")

        if not files:
            logger.info("❌ Aucun fichier détecté ! Vérifie la structure du body.")
            return body  # On ne fait rien s'il n'y a pas de fichier

        # On prépare la liste des nouveaux fichiers pour ce chat_id
        if self.chat_id not in self.latest_files_by_chat:
            self.latest_files_by_chat[self.chat_id] = []

        existing_files = self.latest_files_by_chat.get(self.chat_id, [])
        existing_filenames = {fichier["filename"] for fichier in existing_files}

        new_files = []
        for file_data in files:
            file_info = file_data.get("file", {})
            file_id = file_info.get("id")
            original_filename = file_info.get("filename")

            logger.info(
                f"🔎 Traitement du fichier → file_id={file_id}, filename={original_filename}"
            )
            if not file_id or not original_filename:
                logger.error(f"🚨 ERREUR: Problème avec ce fichier: {file_data}")
                continue

            if not UUID_REGEX.match(f"{file_id}_{original_filename}"):
                # On saute si ça ne matche pas le pattern ou si info incomplète
                logger.warning(
                    f"Fichier ignoré (pas d'UUID valide) : {original_filename}"
                )
                continue
            # 🛠️ Générer le nouveau nom du fichier (avec UUID si pas déjà présent)
            # Ici, on vérifie si le filename est déjà de la forme UUID_filename
            if not re.match(
                r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_.+",
                original_filename,
            ):
                new_filename = f"{file_id}_{original_filename}"
            else:
                # On suppose que le fichier a déjà un UUID
                new_filename = original_filename

            if new_filename in existing_filenames:
                logger.debug(
                    f"Le fichier {new_filename} est déjà présent, on conserve son état."
                )
                continue

            source_path = os.path.join(self.upload_directory, original_filename)
            new_path = os.path.join(self.upload_directory, new_filename)

            if os.path.exists(source_path):
                # On renomme localement dans uploads/
                os.rename(source_path, new_path)
                logger.debug(f"✅ Fichier renommé: {source_path} → {new_path}")
            else:
                logger.debug(
                    f"⚠️ Fichier source introuvable: {source_path} (peut-être non encore uploadé)"
                )

            # adding=False -> on va le copier plus tard dans detect_and_process_changes
            new_files.append(
                {"file_id": file_id, "filename": new_filename, "adding": False}
            )

            existing_filenames.add(new_filename)

        self.latest_files_by_chat[self.chat_id] = existing_files + new_files
        logger.info(
            f"📂 Fichiers du dernier message: {self.latest_files_by_chat[self.chat_id]}"
        )

        # Lance la détection et le traitement
        self.detect_and_process_changes()

        self.custom_history_path = os.getenv("HISTORY_DB_FILE")
        self.custom_history_path = self.custom_history_path.replace(
            "id", str(self.chat_id)
        )

        return body

    async def on_shutdown(self):
        pass

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
        setup_history_database(self.custom_history_path)
        logger.info(f"Starting LLM data interpreter with question: {question}")

        context = initial_context

        # 🔹 Rechercher les messages similaires
        similar_messages = retrieve_similar_conversations(
            question, self.custom_history_path
        )

        # 🔹 Ajouter les résultats SQL et Python initiaux
        context["sql_results"] = context.get("sql_results", [])
        context["python_results"] = context.get("python_results", [])

        # 🔹 Construire un résumé des messages trouvés
        if similar_messages:
            history_summary = "\n".join(
                [
                    f"User: {conv['question']}\nAssistant: {conv['response']}"
                    for conv in similar_messages
                ]
            )
            logger.debug(f"🔍 Historique pertinent trouvé : \n{history_summary}")
            # 🔹 Générer la réponse finale
            final_response = generate_final_response_with_llama(
                context,
                None,
                self.reasoning_model,
                None,
                history_summary,  # Utilise le résumé de l'historique
            )
            add_conversation_with_embedding(
                self.custom_history_path, question, final_response
            )

            return final_response

        history_summary = ""

        # 🔹 Ajouter l'historique résumé au contexte
        context["history_summary"] = history_summary

        while True:
            logger.debug("🔄 Génération du plan...")
            self.python_results = None
            self.sql_results = None

            # 🔹 Utiliser history_summary au lieu de self.history
            plan, python_code = command_r_plus_plan(
                question, schema, self.contextualisation_model, history_summary
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
                    self.custom_db_path,
                )
            )

            logger.debug(
                f"📊 Résultats : {context['sql_results']}, {self.python_results}"
            )

            reflection = self.verify_and_reflect(context, self.python_results)
            logger.debug(f"🔁 Résultat de la réflexion : {reflection}")

            if "Terminé" in reflection:
                logger.info("✅ Exécution terminée.")
                break

        # 🔹 Générer la réponse finale
        final_response = generate_final_response_with_llama(
            context,
            self.python_results,
            self.reasoning_model,
            files_generated,
            None,  # Utilise le résumé de l'historique
        )

        add_conversation_with_embedding(
            self.custom_history_path, question, final_response
        )

        return final_response

    def pipe(
        self,
        user_message: str,
        model_id: str = None,
        messages: typing.List[dict] = None,
        body: dict = None,
    ) -> typing.Union[str, typing.Generator, typing.Iterator]:
        try:
            save_directory = os.path.join(self.shared_data_directory, self.chat_id)
            if not os.path.exists(save_directory):
                os.makedirs(save_directory, exist_ok=True)
            logger.debug(f"chat_id available in pipe: {self.chat_id}")
            self.custom_db_path = os.getenv("DB_FILE")
            self.custom_db_path = self.custom_db_path.replace("id", str(self.chat_id))
            self.custom_history_path = os.getenv("HISTORY_DB_FILE")
            self.custom_history_path = self.custom_history_path.replace(
                "id", str(self.chat_id)
            )
            schema = get_schema(duckdb.connect(self.custom_db_path))
            initial_context = {"question": user_message}
            return self.llm_data_interpreter(user_message, schema, initial_context)
        except Exception as e:
            logger.error(f"Error executing request: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))


pipeline = Pipeline()
