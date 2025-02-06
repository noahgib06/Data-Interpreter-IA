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

from history_func import add_message, get_history, setup_history_database
from LlmGeneration import (command_r_plus_plan,
                           generate_final_response_with_llama,
                           generate_tools_with_llm)
from SetupDatabase import prepare_database, remove_database_file
from SqlTool import get_schema
from version import __version__

load_dotenv()

# Variable globale pour le niveau de log
LOG_LEVEL_ENV = os.getenv(
    "LOG_LEVEL_main"
)  # Changez ce niveau pour DEBUG, INFO, WARNING, ERROR, CRITICAL
DATABASE_MODEL = os.getenv("DATABASE_MODEL")
REASONING_MODEL = os.getenv("REASONING_MODEL")
CONTEXTUALISATION_MODEL = os.getenv("CONTEXTUALISATION_MODEL")

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
    if not os.path.exists("Logs"):
        os.makedirs("Logs", exist_ok=True)
    logger = logging.getLogger("main_logger")
    logger.setLevel(
        LOG_LEVEL_MAP.get(LOG_LEVEL_ENV)
    )  # Niveau global basé sur la variable LOG_LEVEL

    # Format des logs
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    )

    # Handler pour la console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(
        LOG_LEVEL_MAP.get(LOG_LEVEL_ENV)
    )  # Niveau basé sur LOG_LEVEL
    console_handler.setFormatter(formatter)

    # Handler pour les fichiers avec rotation
    file_handler = RotatingFileHandler(
        log_file, maxBytes=max_size, backupCount=backup_count
    )
    file_handler.setLevel(LOG_LEVEL_MAP.get(LOG_LEVEL_ENV))  # Niveau basé sur LOG_LEVEL
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


def llm_data_interpreter(question, schema, initial_context):
    logger.info(f"Starting LLM data interpreter with question: {question}")
    context = initial_context
    global sql_results
    global python_results
    add_message(os.getenv("HISTORY_DB_FILE"), "user", question)
    context["sql_results"] = context.get("sql_results", [])
    context["python_results"] = context.get("python_results", [])
    history = get_history(os.getenv("HISTORY_DB_FILE"))
    while True:
        logger.debug("Generating plan...")
        sql_results = None
        python_results = None
        plan, python_code = command_r_plus_plan(
            question, schema, contextualisation_model, history
        )

        context, python_results, sql_results, files_generated = generate_tools_with_llm(
            plan,
            schema,
            context,
            sql_results,
            python_results,
            database_model,
            reasoning_model,
            python_code,
            os.getenv("DB_FILE"),
        )
        logger.debug(f"Results: {context['sql_results']}, {python_results}")
        reflection = verify_and_reflect(context, python_results)
        logger.debug(f"Reflection result: {reflection}")

        if "Terminé" in reflection:
            logger.info("Execution process completed.")
            break

    history = get_history(os.getenv("HISTORY_DB_FILE"))
    final_response = generate_final_response_with_llama(
        context, sql_results, python_results, reasoning_model, files_generated, history
    )
    add_message(history, "assistant", final_response)
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
    try:
        complex_query = request.complex_query
        logger.info(f"Received query: {complex_query}")
        conn = duckdb.connect(os.getenv("DB_FILE"))
        schema = get_schema(conn)
        logger.debug(f"Schema: {schema}")
        conn.close()  # Fermez la connexion DuckDB
        initial_context = {"question": complex_query}
        response = llm_data_interpreter(complex_query, schema, initial_context)
        return {"analysis_result": response}
    except Exception as e:
        logger.error("Error processing query", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    try:
        if len(sys.argv) <= 1:
            run_help_command()
            exit(84)

        if len(sys.argv) == 2 and sys.argv[1] == "--help":
            run_help_command()
            exit(0)

        if len(sys.argv) == 2 and sys.argv[1] == "--v":
            logger.info("Showing script version")
            print(__version__)
            exit(0)
        filepath = sys.argv[1:]
        if filepath is not None:
            logger.info("Suppression de l'ancienne base de données...")
            remove_database_file()
        logger.info(f"Preparing database with files: {filepath}")
        prepare_database(filepath)
        setup_history_database(os.getenv("HISTORY_DB_FILE"))

        database_model = OllamaLLM(model=DATABASE_MODEL)
        logger.info(f"Database model: {DATABASE_MODEL}")
        reasoning_model = OllamaLLM(model=REASONING_MODEL)
        logger.info(f"Reasoning model: {REASONING_MODEL}")
        contextualisation_model = OllamaLLM(model=CONTEXTUALISATION_MODEL)
        logger.info(f"Contextualiser model: {CONTEXTUALISATION_MODEL}")

        if not os.path.exists("output"):
            os.makedirs("output", exist_ok=True)

        logger.info("Starting FastAPI server...")
        uvicorn.run(app, host=os.getenv("ADDRESS"), port=int(os.getenv("PORT")))

    except Exception:
        logger.error("Fatal error during application startup", exc_info=True)
        sys.exit(1)
