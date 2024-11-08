import duckdb
from langchain_community.llms import Ollama
import sys
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from SetupDatabase import prepare_database
from SqlTool import get_schema
from LlmGeneration import (
    generate_tools_with_llm,
    command_r_plus_plan,
    generate_final_response_with_llama,
)

app = FastAPI()

sql_results = None
python_results = None


def verify_and_reflect(context, schema):
    global sql_results
    global python_results
    print("Verifying and reflecting on execution results...")
    if sql_results:
        print(f"SQL results found: {sql_results}")
        if sql_results and sql_results is not None:
            if "requires_python_analysis" in context:
                return "Passer à Python"
            else:
                return "Terminé"
        else:
            return "Résultats SQL incorrects"
    elif python_results:
        print("Python results found.")
        return "Terminé"
    print("Continuing...")
    return "Continuer"


def llm_data_interpreter(question, schema, initial_context):
    print(f"Starting LLM data interpreter with question: {question}")
    context = initial_context
    global sql_results
    global python_results
    while True:
        print("Generating plan...")
        sql_results = None
        python_results = None
        plan = command_r_plus_plan(question, schema, contextualisation_model)

        context, python_results, sql_results = generate_tools_with_llm(
            plan,
            schema,
            context,
            sql_results,
            python_results,
            database_model,
            reasoning_model,
        )

        reflection = verify_and_reflect(context, schema)
        print(f"Reflection result: {reflection}")

        if "Terminé" in reflection:
            print("Execution process completed.")
            break

    final_response = generate_final_response_with_llama(
        context, sql_results, python_results, reasoning_model
    )
    print(f"Final response from Llama: {final_response}")
    return final_response


class QueryRequest(BaseModel):
    complex_query: str


def run_help_command():
    """
    Affiche l'aide du programme ou lance le serveur avec le fichier spécifié.
    """
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
        schema = get_schema(duckdb.connect("my_database.duckdb"))
        initial_context = {"question": complex_query}
        response = llm_data_interpreter(complex_query, schema, initial_context)
        return {"analysis_result": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    filepath = sys.argv[1:] if len(sys.argv) > 1 else None
    if sys.argv[1] == "--help":
        run_help_command()
        exit(0)
    prepare_database(filepath)

    database_model = Ollama(model="duckdb-nsql:latest")
    reasoning_model = Ollama(model="llama3.2:latest")
    contextualisation_model = Ollama(model="command-r-plus:latest")
    uvicorn.run(app, host="0.0.0.0", port=8000)
