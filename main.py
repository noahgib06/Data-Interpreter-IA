import sys

import duckdb
from langchain_ollama import OllamaLLM

sys.path.insert(1, "./src/")
import logging
import os
from logging.handlers import RotatingFileHandler

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from history_func import (add_conversation_with_embedding, get_history,
                          retrieve_similar_conversations,
                          setup_history_database)
from LlmGeneration import (generate_final_response_with_llama, generate_plan,
                           generate_tools_with_llm)
from SetupDatabase import prepare_database, remove_database_file
from SqlTool import get_schema
from version import __version__

load_dotenv()

# Variable globale pour le niveau de log
LOG_LEVEL_ENV = os.getenv("LOG_LEVEL_main", "INFO")  # Valeur par défaut: INFO
REASONING_MODEL = os.getenv("REASONING_MODEL")
PLAN_MODEL = os.getenv("PLAN_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

# Mappage des niveaux de log
LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def setup_logger(
    log_file=os.getenv("LOG_FILE_main"), max_size=10 * 1024 * 1024, backup_count=5
):
    """
    Configure un logger global pour l'application avec des niveaux configurables.
    """
    # Créer le répertoire de logs s'il n'existe pas
    log_dir = os.path.join(os.path.dirname(__file__), "Logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # Nettoyer le chemin du fichier de log
    if log_file:
        log_file = log_file.strip('"')  # Supprimer les guillemets
        log_file = os.path.join(log_dir, os.path.basename(log_file))

    logger = logging.getLogger("main_logger")
    logger.setLevel(LOG_LEVEL_MAP.get(LOG_LEVEL_ENV, logging.INFO))

    # Format des logs
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    )

    # Handler pour la console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(LOG_LEVEL_MAP.get(LOG_LEVEL_ENV, logging.INFO))
    console_handler.setFormatter(formatter)

    # Handler pour les fichiers avec rotation
    file_handler = RotatingFileHandler(
        log_file, maxBytes=max_size, backupCount=backup_count
    )
    file_handler.setLevel(LOG_LEVEL_MAP.get(LOG_LEVEL_ENV, logging.INFO))
    file_handler.setFormatter(formatter)

    # Ajout des handlers au logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


logger = setup_logger()

app = FastAPI()

sql_results = None
python_results = None
history = []


def verify_and_reflect(context, python_results):
    """
    Evaluates execution results and determines the next step based on available SQL and Python results.
    """
    logger.info("🔍 Starting verification and reflection on execution results.")

    # Check if there are SQL results
    if context["sql_results"]:
        invalid_sql_results = [
            res
            for res in context["sql_results"]
            if not isinstance(res, dict) or not res
        ]

        if invalid_sql_results:
            logger.warning(f"⚠️ Invalid SQL results detected: {invalid_sql_results}")
            return "Résultats SQL incorrects"  # Indicate SQL result issues

        # If Python analysis is required, transition to Python execution
        if "requires_python_analysis" in context:
            logger.info("🔄 SQL results indicate a need for Python analysis.")
            return "Passer à Python"

        logger.info("✅ SQL results validated successfully.")
        return "Terminé"  # Execution can be considered complete

    # Check if Python results exist
    elif python_results:
        logger.info("✅ Python results found. Execution considered complete.")
        return "Terminé"

    # If no results are found, continue processing
    logger.debug("⚡ No results found yet, continuing execution...")
    return "Continuer"


def llm_data_interpreter(question, schema, initial_context):
    """
    Interprets user queries using LLM, leveraging historical data and generating responses through SQL and Python execution.
    """
    question = question.lower()
    history_path = os.getenv("HISTORY_DB_FILE")
    setup_history_database(history_path)
    logger.info(f"🚀 Starting LLM data interpreter for question: {question}")

    context = initial_context
    global sql_results
    global python_results

    # Retrieve similar past interactions for context
    similar_messages = retrieve_similar_conversations(
        question, os.getenv("HISTORY_DB_FILE"), EMBEDDING_MODEL
    )
    context["sql_results"] = context.get("sql_results", [])
    context["python_results"] = context.get("python_results", [])

    # Initialize an empty history summary
    history_summary = ""

    if similar_messages or "#pass" in question:
        history_summary = ""
        if similar_messages:
            history_summary = "\n".join(
                [
                    f"User: {conv['question']}\nAssistant: {conv['response']}"
                    for conv in similar_messages
                ]
            )
            logger.debug(f"🔍 Relevant conversation history found: \n{history_summary}")

        # Generate a final response based on historical data
        if "#force" not in question:
            final_response = generate_final_response_with_llama(
                context,
                None,
                reasoning_model,
                None,
                history_summary,
            )
            add_conversation_with_embedding(
                os.getenv("HISTORY_DB_FILE"), question, final_response, EMBEDDING_MODEL
            )

            return final_response

    if "#force" in question:
        question = question.replace("#force", "")

    context["history_summary"] = history_summary

    while True:
        logger.debug("🛠️ Generating execution plan...")
        sql_results = None
        python_results = None

        if history_summary == "":
            full_history = get_history(history_path)
            # on prend les 8 dernières entrées (4 paires)
            recent = full_history[-8:]
            # regrouper en paires User/Assistant
            pairs = []
            for i in range(0, len(recent), 2):
                if i + 1 < len(recent):
                    user_msg = recent[i]["content"]
                    assistant_msg = recent[i + 1]["content"]
                    pairs.append((user_msg, assistant_msg))
            # on garde au plus les 4 dernières paires
            pairs = pairs[-4:]
            # transformer en résumé textuel
            history_summary = "\n".join(
                [f"User: {q}\nAssistant: {a}" for q, a in pairs]
            )
            logger.debug(
                f"🔍 Résumé fallback de l'historique (4 derniers échanges) :\n{history_summary}"
            )
        else:
            # on garde au plus 4 messages similaires trouvés
            similar_messages = similar_messages[:4]
            history_summary = "\n".join(
                [
                    f"User: {conv['question']}\nAssistant: {conv['response']}"
                    for conv in similar_messages
                ]
            )
            logger.debug(
                f"🔍 Résumé historique (similar messages) :\n{history_summary}"
            )

        # Generate an action plan and extract Python code if needed
        plan, python_code = generate_plan(
            question, schema, plan_model, history_summary
        )

        # Execute tools (SQL, Python) based on the generated plan
        context, python_results, sql_results, files_generated = generate_tools_with_llm(
            plan,
            schema,
            context,
            sql_results,
            python_results,
            code_model,
            python_code,
            os.getenv("DB_FILE"),
        )

        logger.debug(
            f"📊 Execution results: {context['sql_results']}, {python_results}"
        )

        # Analyze results and determine next steps
        reflection = verify_and_reflect(context, python_results)
        logger.debug(f"🔄 Reflection result: {reflection}")

        if "Terminé" in reflection:
            logger.info("✅ Execution process completed successfully.")
            break

    # Generate a final response based on gathered data
    final_response = generate_final_response_with_llama(
        context,
        python_results,
        reasoning_model,
        files_generated,
        None,
    )

    # Save the interaction in history
    add_conversation_with_embedding(
        os.getenv("HISTORY_DB_FILE"), question, final_response, EMBEDDING_MODEL
    )

    return final_response


class QueryRequest(BaseModel):
    complex_query: str


def run_help_command():
    """
    Affiche l'aide du programme ou lance le serveur avec le fichier spécifié.
    """
    logger.info("Displaying help command")
    print("Usage: python script.py [FILEPATH(S)]\n")
    print("Options:")
    print("  --help        Affiche ce message d'aide et quitte le programme.\n")
    print("Arguments:")
    print(
        "  FILEPATH(S)   Chemin(s) vers un ou plusieurs fichiers de données à traiter."
    )
    print("                Les fichiers acceptés incluent :")
    print("                - Excel (.xls, .xlsx)")
    print("                - CSV (.csv)")
    print("                - JSON (.json)")
    print("                - PDF (.pdf)")
    print("                - Python (.py)\n")
    print("Description:")
    print(
        "  Ce programme sert à extraire, traiter et analyser des données provenant de divers types de fichiers."
    )
    print("  Il se compose de plusieurs étapes :\n")
    print("  1. Extraction et préparation des données:")
    print(
        "      - Les fichiers fournis sont analysés et importés dans une base de données DuckDB."
    )
    print(
        "      - Les fichiers PDF, par exemple, auront leur texte et leurs images extraits, tandis que les"
    )
    print(
        "        fichiers Python verront leurs fonctions, classes, et importations enregistrées dans des tables distinctes.\n"
    )
    print("  2. Interprétation et génération de requêtes:")
    print(
        "      - Le programme inclut une interface API, accessible via l'URL `/query/`, permettant d'exécuter des"
    )
    print("        requêtes complexes.")
    print(
        "      - Lorsqu'une requête est envoyée à l'API, elle est interprétée, et une requête SQL ou un code de"
    )
    print("        traitement est généré pour obtenir les informations souhaitées.\n")
    print("  3. Modèles LLM intégrés:")
    print(
        "      - Plusieurs modèles LLM (Large Language Models) sont utilisés pour interpréter les requêtes, générer"
    )
    print(
        "        des requêtes SQL ou du code, et formuler des réponses basées sur les résultats de la base de données.\n"
    )
    print("Exemples d'utilisation :")
    print("  Pour charger et préparer les données issues de plusieurs fichiers :")
    print("      python script.py file1.xlsx file2.json file3.py\n")
    print("API Endpoint:")
    print(
        "  - POST /query/: Envoie une requête JSON contenant la clé `complex_query` pour poser une question complexe"
    )
    print(
        "    sur les données, et reçoit en retour une analyse de la requête et des résultats de la base de données.\n"
    )
    print(
        "Ce programme vous aide ainsi à traiter des ensembles de données variés et à effectuer des analyses avancées à"
    )
    print("l'aide de modèles de langage.")
    return


@app.post("/query/")
async def query_endpoint(request: QueryRequest):
    """
    Handles incoming queries, retrieves the database schema, and processes the request using LLM.
    """
    try:
        complex_query = request.complex_query
        logger.info(f"📥 Received query: {complex_query}")

        # Establish a connection to DuckDB to retrieve the schema
        conn = duckdb.connect(os.getenv("DB_FILE"))
        schema = get_schema(conn)
        logger.debug(f"📊 Retrieved schema: {schema}")

        # Close the database connection after retrieving schema information
        conn.close()
        logger.info("✅ Database connection closed.")

        # Prepare the initial context for LLM processing
        initial_context = {"question": complex_query}

        # Process the query using the LLM data interpreter
        response = llm_data_interpreter(complex_query, schema, initial_context)

        logger.info("✅ Query processed successfully.")
        return {"analysis_result": response}

    except Exception as e:
        logger.error("❌ Error processing query", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    """
    Main entry point of the application. Handles argument parsing, database preparation,
    model initialization, and starts the FastAPI server.
    """
    try:
        if len(sys.argv) <= 1:
            logger.info("No arguments provided. Displaying help command.")
            run_help_command()
            exit(84)

        if len(sys.argv) == 2 and sys.argv[1] == "--help":
            logger.info("Displaying help command.")
            run_help_command()
            exit(0)

        if len(sys.argv) == 2 and sys.argv[1] == "--v":
            logger.info("Displaying script version.")
            print(__version__)
            exit(0)

        filepath = sys.argv[1:]
        if filepath is not None:
            logger.info("🔄 Removing old database file...")
            remove_database_file()

        logger.info(f"📂 Preparing database with files: {filepath}")
        prepare_database(filepath)
        setup_history_database(os.getenv("HISTORY_DB_FILE"))

        # Initializing LLM models
        reasoning_model = OllamaLLM(model=REASONING_MODEL)
        logger.info(f"🧠 Reasoning model initialized: {REASONING_MODEL}")
        plan_model = OllamaLLM(model=PLAN_MODEL)
        logger.info(f"📜 Contextualisation model initialized: {PLAN_MODEL}")
        code_model = OllamaLLM(model=os.getenv("CODE_MODEL"))
        logger.info(f"💻 Code model initialized: {os.getenv('CODE_MODEL')}")

        # Ensure output directory exists
        if not os.path.exists("output"):
            os.makedirs("output", exist_ok=True)
            logger.info("📁 'output' directory created.")

        # Start the FastAPI server
        logger.info("🚀 Starting FastAPI server...")
        uvicorn.run(app, host=os.getenv("ADDRESS"), port=int(os.getenv("PORT")))

    except Exception:
        logger.error("❌ Fatal error during application startup", exc_info=True)
        sys.exit(1)
