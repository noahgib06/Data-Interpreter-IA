import os
import sys
from readline import clear_history

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
from LlmGeneration import (generate_final_response_with_llama, generate_plan,
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

class Pipeline:
    class Valves(BaseModel):
        LLAMAINDEX_OLLAMA_BASE_URL: str = "http://host.docker.internal:11434"
        LLAMAINDEX_REASONING_MODEL_NAME: str = os.getenv("REASONING_MODEL")
        LLAMAINDEX_PLAN_MODEL_NAME: str = os.getenv("PLAN_MODEL")
        LLAMAINDEX_CODE_MODEL_NAME: str = os.getenv("CODE_MODEL")
        LLAMAINDEX_EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL")

        # FICHIERS: str = ""

    def __init__(self):
        """
        Initializes the Pipeline class with model configurations and paths.
        """
        logger.debug(
            "Initializing Pipeline class"
        )  # Debug log: initialization process started

        # Initialize instance variables
        self.sql_results = None
        self.python_results = None
        self.shared_data_directory = "/srv/data/"
        self.upload_directory = "/app/backend/data/uploads/"
        self.known_shared_files: Set[str] = set()
        self.known_upload_files: Set[str] = set()
        self.history = []
        self.chat_id = None
        self.custom_db_path = None
        self.custom_history_path = None
        self.latest_files_by_chat = {}

        # Load model configurations from environment variables
        self.valves = self.Valves(
            LLAMAINDEX_OLLAMA_BASE_URL=os.getenv(
                "LLAMAINDEX_OLLAMA_BASE_URL", "http://host.docker.internal:11434"
            ),
            LLAMAINDEX_REASONING_MODEL_NAME=os.getenv(
                "LLAMAINDEX_REASONING_MODEL_NAME", os.getenv("REASONING_MODEL")
            ),
            LLAMAINDEX_PLAN_MODEL_NAME=os.getenv(
                "LLAMAINDEX_PLAN_MODEL_NAME", os.getenv("PLAN_MODEL")
            ),
            LLAMAINDEX_CODE_MODEL_NAME=os.getenv(
                "LLAMAINDEX_CODE_MODEL_NAME", os.getenv("CODE_MODEL")
            ),
            LLAMAINDEX_EMBEDDING_MODEL_NAME=os.getenv(
                "LLAMAINDEX_EMBEDDING_MODEL_NAME", os.getenv("EMBEDDING_MODEL")
            ),
        )
        logger.info(
            f"Valves initialized: {self.valves}"
        )  # Info log: model configurations loaded

        try:
            # Initialize LLM models
            self.reasoning_model = OllamaLLM(
                model=self.valves.LLAMAINDEX_REASONING_MODEL_NAME,
                base_url="http://host.docker.internal:11434",
            )
            self.plan_model = OllamaLLM(
                model=self.valves.LLAMAINDEX_PLAN_MODEL_NAME,
                base_url="http://host.docker.internal:11434",
            )
            self.code_model = OllamaLLM(
                model=self.valves.LLAMAINDEX_CODE_MODEL_NAME,
                base_url="http://host.docker.internal:11434",
            )
            self.embedding_model = os.getenv("EMBEDDING_MODEL")

        except NewConnectionError as conn_error:
            logger.error(
                f"Connection to Ollama failed: {conn_error}"
            )  # Error log: connection failure
            raise RuntimeError(
                "Failed to connect to Ollama. Check the service URL and availability."
            )
        except Exception as e:
            logger.error(
                f"Error initializing models: {e}"
            )  # Error log: general initialization error
            raise RuntimeError("General error during model initialization.")

    def clear_history(self):
        try:
            logger.info("Running startup cleanup...")

            db_path = f"/app/backend/data/webui.db"
            
            # Check if the database file exists before trying to connect
            if not os.path.exists(db_path):
                logger.warning(f"Database file not found: {db_path}. Skipping cleanup.")
                return
            
            # Use sqlite3 directly instead of DuckDB with SQLite extension
            import sqlite3
            
            try:
                # Connect directly to SQLite database
                con = sqlite3.connect(db_path)
                cursor = con.cursor()
                
                # Query the chat table
                cursor.execute("SELECT id FROM chat")
                chat_ids = cursor.fetchall()
                chat_ids_set = {str(row[0]) for row in chat_ids}

                data_folder = self.shared_data_directory
                if os.path.exists(data_folder):
                    for folder in os.listdir(data_folder):
                        folder_path = os.path.join(data_folder, folder)

                        if os.path.isdir(folder_path) and folder not in chat_ids_set:
                            shutil.rmtree(folder_path)
                            logger.info(f"Folder deleted : {folder_path}")
                else:
                    logger.info(f"Data folder does not exist: {data_folder}")

                logger.info("History Cache Cleanup complete.")
                
            finally:
                if 'con' in locals():
                    con.close()

        except Exception as e:
            logger.error(f"Erreur lors du nettoyage des dossiers: {e}", exc_info=True)

    async def on_startup(self):
        self.clear_history()

    def detect_and_process_changes(self):
        """
        Detects and processes file changes by adding missing files to /srv/data/{chat_id}.
        Removes database entries for missing files if necessary (deletion is currently omitted).
        """
        logger.info(
            "🔍 detect_and_process_changes() started"
        )  # INFO: Function execution begins
        logger.debug(
            f"📌 Current chat_id value: {self.chat_id}"
        )  # DEBUG: Check chat_id value

        if not self.chat_id:
            logger.warning(
                "❌ No chat_id defined! Cannot process files."
            )  # WARNING: chat_id is missing
            return

        chat_path = os.path.join(self.shared_data_directory, self.chat_id)
        logger.debug(f"🔍 Chat Path: {chat_path}")  # DEBUG: Show target chat path

        # Ensure the directory exists
        if not os.path.exists(chat_path):
            os.makedirs(chat_path, exist_ok=True)
            logger.info(
                f"📁 Created directory: {chat_path}"
            )  # INFO: Chat directory created

        # Retrieve existing files in the chat directory
        existing_files = set(os.listdir(chat_path))
        existing_uuids = {f.split("_", 1)[0] for f in existing_files if "_" in f}
        logger.debug(
            f"📂 DEBUG: Existing files in {chat_path}: {existing_files}"
        )  # DEBUG: List current files
        logger.debug(
            f"📂 DEBUG: Extracted UUIDs: {existing_uuids}"
        )  # DEBUG: Show extracted UUIDs

        updated_files = []
        new_filepaths_to_db = []  # List of new files to add to the database

        logger.info("🔄 Starting file processing...")  # INFO: Begin processing files

        # Iterate through the latest files
        for file_data in self.latest_files_by_chat[self.chat_id]:
            file_id = file_data.get("file_id")
            filename = file_data.get("filename")

            logger.info(
                f"📝 Processing file: file_id={file_id}, filename={filename}"
            )  # INFO: Processing a file

            if file_id is None or filename is None:
                logger.error(
                    f"🚨 ERROR: file_id or filename is None! Data: {file_data}"
                )  # ERROR: Invalid file data
                continue

            # Skip if already added
            if file_data.get("adding", False) is True:
                logger.info(
                    f"⚠️ File {filename} already added, skipping."
                )  # INFO: Skipping already added file
                updated_files.append(file_data)
                continue

            source_path = os.path.join(self.upload_directory, filename)
            dest_path = os.path.join(chat_path, filename)

            logger.info(
                f"📥 Checking file copy from {source_path} to {dest_path}"
            )  # INFO: Checking file transfer

            # Check if file already exists at the destination
            if os.path.exists(dest_path):
                logger.info(
                    f"✅ File already exists in {dest_path}, marking as added."
                )  # INFO: File exists, marking added
                file_data["adding"] = True
            else:
                # Copy file if source exists
                if os.path.exists(source_path):
                    shutil.copy(source_path, dest_path)
                    logger.info(
                        f"✅ File copied: {source_path} → {dest_path}"
                    )  # INFO: File successfully copied
                    file_data["adding"] = True
                    new_filepaths_to_db.append(dest_path)  # Add for DB processing
                else:
                    logger.warning(
                        f"⚠️ Source file not found: {source_path}, not copied."
                    )  # WARNING: Source file missing

            updated_files.append(file_data)

        # Update the list of active files
        self.latest_files_by_chat[self.chat_id] = updated_files

        # Add new files to the database if necessary
        if new_filepaths_to_db:
            logger.info(
                f"📊 Sending new files to the database: {new_filepaths_to_db}"
            )  # INFO: Updating database
            prepare_database(filepaths=new_filepaths_to_db, collection_id=self.chat_id)

    async def inlet(self, body: dict, user: typing.Optional[dict] = None) -> dict:
        """
        Processes incoming requests, extracts metadata, and manages file operations.
        Ensures the correct chat_id is assigned and files are processed accordingly.
        """
        logger.info("🔄 inlet() function started")  # INFO: Function execution begins
        logger.debug(
            f"📂 DEBUG: Received body → {body}"
        )  # DEBUG: Log received request body

        # Initialize LLM models if their configurations are available
        if self.valves.LLAMAINDEX_REASONING_MODEL_NAME is not None:
            self.reasoning_model = OllamaLLM(
                model=self.valves.LLAMAINDEX_REASONING_MODEL_NAME,
                base_url="http://host.docker.internal:11434",
            )
        if self.valves.LLAMAINDEX_PLAN_MODEL_NAME is not None:
            self.plan_model = OllamaLLM(
                model=self.valves.LLAMAINDEX_PLAN_MODEL_NAME,
                base_url="http://host.docker.internal:11434",
            )
        if self.valves.LLAMAINDEX_CODE_MODEL_NAME is not None:
            self.code_model = OllamaLLM(
                model=self.valves.LLAMAINDEX_CODE_MODEL_NAME,
                base_url="http://host.docker.internal:11434",
            )
        if self.valves.LLAMAINDEX_EMBEDDING_MODEL_NAME is not None:
            self.embedding_model = self.valves.LLAMAINDEX_EMBEDDING_MODEL_NAME

        clear_history()

        # Extract chat_id from metadata

        self.chat_id = body.get("metadata", {}).get("chat_id")

        print(f"voila le chat id : {self.chat_id}")


        if self.chat_id is None:
            logger.error(
                "🚨 ERROR: chat_id is None, correcting to 'unknown_chat'"
            )  # ERROR: chat_id missing
            self.chat_id = "unknown_chat"

        self.clear_history()

        logger.debug(
            f"📌 DEBUG: Extracted chat_id → {self.chat_id}"
        )  # DEBUG: Log extracted chat_id

        # Ensure the directory for the chat exists
        path = os.path.join(self.shared_data_directory, self.chat_id)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            logger.info(f"📁 Created directory: {path}")  # INFO: Chat directory created

        # Extract file metadata
        files = body.get("metadata", {}).get("files", [])
        logger.debug(
            f"📂 DEBUG: Extracted files → {files}"
        )  # DEBUG: Log extracted file list

        if not files:
            logger.warning(
                "❌ No files detected! Check request body structure."
            )  # WARNING: No files found
            return body  # Exit early if no files are present

        # Prepare a list of new files for this chat_id
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
                f"🔎 Processing file → file_id={file_id}, filename={original_filename}"
            )  # INFO: File processing started

            if not file_id or not original_filename:
                logger.error(
                    f"🚨 ERROR: Issue with file metadata: {file_data}"
                )  # ERROR: File data is incorrect
                continue

            if not UUID_REGEX.match(f"{file_id}_{original_filename}"):
                logger.warning(
                    f"⚠️ File ignored (invalid UUID format): {original_filename}"
                )  # WARNING: Invalid file format
                continue

            # Generate a new filename if it does not already have a UUID
            if not re.match(
                r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_.+",
                original_filename,
            ):
                new_filename = f"{file_id}_{original_filename}"
            else:
                new_filename = original_filename  # Assume the file already has a UUID

            if new_filename in existing_filenames:
                logger.debug(
                    f"File {new_filename} already exists, maintaining its state."
                )  # DEBUG: File already exists
                continue

            source_path = os.path.join(self.upload_directory, original_filename)
            new_path = os.path.join(self.upload_directory, new_filename)

            if os.path.exists(source_path):
                os.rename(source_path, new_path)
                logger.debug(
                    f"✅ File renamed: {source_path} → {new_path}"
                )  # DEBUG: File successfully renamed
            else:
                logger.debug(
                    f"⚠️ Source file not found: {source_path} (possibly not uploaded yet)"
                )  # DEBUG: Source file missing

            # Mark as not added yet for further processing
            new_files.append(
                {"file_id": file_id, "filename": new_filename, "adding": False}
            )
            existing_filenames.add(new_filename)

        self.latest_files_by_chat[self.chat_id] = existing_files + new_files
        logger.info(
            f"📂 Updated latest files for chat_id {self.chat_id}: {self.latest_files_by_chat[self.chat_id]}"
        )  # INFO: Updated file list

        # Trigger file processing
        self.detect_and_process_changes()

        # Update custom history database path
        self.custom_history_path = os.getenv("HISTORY_DB_FILE")
        self.custom_history_path = self.custom_history_path.replace(
            "id", str(self.chat_id)
        )

        return body

    async def on_shutdown(self):
        pass

    def verify_and_reflect(self, context, python_results):
        """
        Verifies and reflects on execution results.
        Determines the next step based on SQL and Python results.
        """
        logger.info(
            "🔍 Verifying and reflecting on execution results..."
        )  # INFO: Process started

        # Check if SQL results exist
        if context["sql_results"]:
            logger.debug(
                f"📊 SQL results found: {context['sql_results']}"
            )  # DEBUG: Log SQL results

            # Identify invalid SQL results
            invalid_sql_results = [
                res
                for res in context["sql_results"]
                if not isinstance(res, dict) or not res
            ]
            if invalid_sql_results:
                logger.warning(
                    f"⚠️ Invalid SQL results detected: {invalid_sql_results}"
                )  # WARNING: Invalid SQL results found
                return "Résultats SQL incorrects"

            # Determine if further Python analysis is required
            if "requires_python_analysis" in context:
                logger.info(
                    "🔄 SQL results require Python analysis, proceeding..."
                )  # INFO: SQL needs Python analysis
                return "Passer à Python"
            else:
                logger.info(
                    "✅ SQL results valid, process complete."
                )  # INFO: SQL processing complete
                return "Terminé"

        # Check if Python results exist
        elif python_results:
            logger.info(
                "✅ Python results found, process complete."
            )  # INFO: Python execution successful
            return "Terminé"

        # If no valid SQL or Python results, continue processing
        logger.debug(
            "🔄 No valid SQL or Python results, continuing execution..."
        )  # DEBUG: Continue execution
        return "Continuer"

    def llm_data_interpreter(self, question, schema, initial_context):
        """
        Interprets data using the LLM model.
        Retrieves relevant conversation history, processes SQL and Python results,
        and generates a final response.
        """
        question = question.lower()
        setup_history_database(self.custom_history_path)
        logger.info(
            f"🚀 Starting LLM data interpreter with question: {question}"
        )  # INFO: Function execution begins

        context = initial_context

        # Retrieve similar past conversations
        similar_messages = retrieve_similar_conversations(
            question, self.custom_history_path, self.embedding_model
        )

        # Initialize SQL and Python results in the context
        context["sql_results"] = context.get("sql_results", [])
        context["python_results"] = context.get("python_results", [])

        history_summary = ""

        # Generate a summary of similar conversation history
        if similar_messages or "#pass" in question:
            history_summary = "\n".join(
                [
                    f"User: {conv['question']}\nAssistant: {conv['response']}"
                    for conv in similar_messages
                ]
            )
            logger.debug(
                f"🔍 Relevant conversation history found:\n{history_summary}"
            )  # DEBUG: Display retrieved history

            if "#force" not in question:
                # Generate a final response using LLM
                final_response = generate_final_response_with_llama(
                    context,
                    None,
                    self.reasoning_model,
                    None,
                    history_summary,  # Use summarized history
                )
                add_conversation_with_embedding(
                    self.custom_history_path, question, final_response, self.embedding_model
                )

                return final_response

        if "#force" in question:
            question = question.replace("#force", "")

        # Store summarized history in context
        context["history_summary"] = history_summary

        while True:
            logger.debug(
                "🔄 Generating execution plan..."
            )  # DEBUG: Plan generation starts
            self.python_results = None
            self.sql_results = None

            # Generate plan and Python code
            plan, python_code = generate_plan(
                question, schema, self.plan_model, history_summary
            )

            # Execute tools and generate results
            context, self.python_results, self.sql_results, files_generated = (
                generate_tools_with_llm(
                    plan,
                    schema,
                    context,
                    self.sql_results,
                    self.python_results,
                    self.code_model,
                    python_code,
                    self.custom_db_path,
                )
            )

            logger.debug(
                f"📊 Results generated - SQL: {context['sql_results']}, Python: {self.python_results}"
            )  # DEBUG: Log results

            # Verify and reflect on execution results
            reflection = self.verify_and_reflect(context, self.python_results)
            logger.debug(
                f"🔁 Reflection result: {reflection}"
            )  # DEBUG: Log reflection output

            if "Terminé" in reflection:
                logger.info(
                    "✅ Execution completed successfully."
                )  # INFO: Execution finished
                break

        # Generate final response from the execution context
        final_response = generate_final_response_with_llama(
            context,
            self.python_results,
            self.reasoning_model,
            files_generated,
            None,  # Use summarized history
        )

        add_conversation_with_embedding(
            self.custom_history_path, question, final_response, self.embedding_model
        )

        return final_response

    def pipe(
        self,
        user_message: str,
        model_id: str = None,
        messages: typing.List[dict] = None,
        body: dict = None,
    ) -> typing.Union[str, typing.Generator, typing.Iterator]:
        """
        Processes the user message, retrieves the schema, and calls the LLM data interpreter.
        Ensures necessary directories and database paths are set before execution.
        """
        try:
            # Ensure the save directory exists for the given chat_id
            save_directory = os.path.join(self.shared_data_directory, self.chat_id)
            if not os.path.exists(save_directory):
                os.makedirs(save_directory, exist_ok=True)
                logger.info(
                    f"📁 Created save directory: {save_directory}"
                )  # INFO: Directory creation log

            logger.debug(
                f"💬 chat_id available in pipe: {self.chat_id}"
            )  # DEBUG: Log chat_id

            # Retrieve and configure database paths
            self.custom_db_path = os.getenv("DB_FILE").replace("id", str(self.chat_id))
            self.custom_history_path = os.getenv("HISTORY_DB_FILE").replace(
                "id", str(self.chat_id)
            )

            logger.debug(
                f"📂 Database path set to: {self.custom_db_path}"
            )  # DEBUG: Log DB path
            logger.debug(
                f"📂 History DB path set to: {self.custom_history_path}"
            )  # DEBUG: Log history path

            # Retrieve the database schema
            schema = get_schema(duckdb.connect(self.custom_db_path))
            logger.info(
                "✅ Database schema retrieved successfully."
            )  # INFO: Schema retrieval successful

            # Prepare the initial context with the user message
            initial_context = {"question": user_message}

            response = self.reasoning_model.invoke(user_message)

            # Call the LLM data interpreter to process the request
            return self.llm_data_interpreter(user_message, schema, initial_context)

        except Exception as e:
            logger.error(
                f"❌ Error executing request: {str(e)}", exc_info=True
            )  # ERROR: Log exception details
            raise HTTPException(status_code=500, detail=str(e))


pipeline = Pipeline()
