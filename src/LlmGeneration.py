import json
import logging
import os
import re
from datetime import datetime
from logging.handlers import RotatingFileHandler

from dotenv import load_dotenv

from PythonTool import parse_and_execute_python_code, parse_code
from SqlTool import execute_sql_query

load_dotenv()

# Configuration du logger global
LOG_LEVEL_ENV = os.getenv("LOG_LEVEL_LlmGeneration", "INFO")  # Valeur par défaut: INFO

# Mappage des niveaux de log
LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

# Limite maximale d'entrées SQL à inclure dans le prompt (pour les performances)
MAX_SQL_RESULTS_IN_PROMPT = int(os.getenv("MAX_SQL_RESULTS_IN_PROMPT", "20"))  # Valeur par défaut: 20


def serialize_datetime(obj):
    """
    Fonction de sérialisation personnalisée pour les objets datetime.
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def safe_json_dumps(obj, **kwargs):
    """
    Version sécurisée de json.dumps qui gère les objets datetime.
    """
    try:
        return json.dumps(obj, default=serialize_datetime, **kwargs)
    except Exception as e:
        logger.warning(f"Erreur lors de la sérialisation JSON: {e}")
        # En cas d'erreur, on essaie de convertir en string
        return str(obj)


def setup_logger(
    log_file=os.getenv("LOG_FILE_LlmGeneration"),
    max_size=5 * 1024 * 1024,
    backup_count=3,
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

    logger = logging.getLogger("action_plan_logger")
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
    file_handler = RotatingFileHandler(
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


def truncate_large_sql_results(sql_results, max_entries=MAX_SQL_RESULTS_IN_PROMPT):
    """
    Tronque et résume les résultats SQL s'ils sont trop volumineux.
    Garantit que toutes les tables sont représentées équitablement.

    Args:
        sql_results: Liste de résultats SQL à traiter
        max_entries: Nombre maximum d'entrées à conserver au total

    Returns:
        Résultats tronqués avec un résumé des données
    """
    logger.debug("Vérification de la taille des résultats SQL pour optimisation")

    if not sql_results or not isinstance(sql_results, list):
        return sql_results

    # 1. Compter le nombre total d'entrées et identifier les tables
    total_entries = 0
    tables_info = []

    for i, result_set in enumerate(sql_results):
        if not isinstance(result_set, dict) or "results" not in result_set:
            tables_info.append(
                {
                    "index": i,
                    "entries": 0,
                    "source_table": "unknown_table",
                    "needs_truncation": False,
                }
            )
            continue

        results = result_set["results"]
        source_table = result_set.get("source_table", f"table_{i}")

        tables_info.append(
            {
                "index": i,
                "entries": len(results),
                "source_table": source_table,
                "needs_truncation": len(results)
                > (max_entries // max(1, len(sql_results))),
            }
        )

        total_entries += len(results)

    # Si le nombre total d'entrées est inférieur à la limite, aucune troncature n'est nécessaire
    if total_entries <= max_entries:
        logger.debug(
            f"Pas de troncature nécessaire: {total_entries} entrées au total (max: {max_entries})"
        )
        return sql_results

    # 2. Déterminer combien d'entrées allouer à chaque table
    logger.info(
        f"Troncature nécessaire: {total_entries} entrées au total (max: {max_entries})"
    )

    # Garantir un minimum d'entrées par table (au moins 5, ou moins si la table est plus petite)
    min_entries_per_table = min(5, max_entries // max(1, len(tables_info)))
    remaining_entries = max_entries - (
        min_entries_per_table * len([t for t in tables_info if t["entries"] > 0])
    )

    # Répartir les entrées restantes proportionnellement à la taille de chaque table
    for table_info in tables_info:
        if table_info["entries"] == 0:
            table_info["allocated_entries"] = 0
            continue

        # Garantir le minimum
        table_info["allocated_entries"] = min(
            table_info["entries"], min_entries_per_table
        )

    # Distribuer les entrées restantes proportionnellement
    if remaining_entries > 0 and sum(t["entries"] for t in tables_info) > 0:
        # Calculer le facteur de proportion pour chaque table
        total_remaining_entries = sum(
            max(0, t["entries"] - t["allocated_entries"]) for t in tables_info
        )

        if total_remaining_entries > 0:
            for table_info in tables_info:
                if table_info["entries"] <= table_info["allocated_entries"]:
                    continue

                proportion = (
                    table_info["entries"] - table_info["allocated_entries"]
                ) / total_remaining_entries
                additional_entries = min(
                    table_info["entries"] - table_info["allocated_entries"],
                    max(0, int(remaining_entries * proportion)),
                )
                table_info["allocated_entries"] += additional_entries
                remaining_entries -= additional_entries

        # Distribuer les entrées restantes si nécessaire
        i = 0
        while remaining_entries > 0 and i < len(tables_info):
            table_info = tables_info[i]
            if table_info["entries"] > table_info["allocated_entries"]:
                table_info["allocated_entries"] += 1
                remaining_entries -= 1
            i = (i + 1) % len(tables_info)
            if i == 0:  # Si on a fait un tour complet sans pouvoir allouer plus
                break

    # 3. Tronquer chaque table selon l'allocation et ajouter des métadonnées
    processed_results = []

    for table_info in tables_info:
        i = table_info["index"]
        original_result = sql_results[i]

        # Si ce n'est pas un ensemble de résultats valide ou s'il est déjà assez petit
        if (
            not isinstance(original_result, dict)
            or "results" not in original_result
            or len(original_result.get("results", []))
            <= table_info["allocated_entries"]
        ):
            processed_results.append(original_result)
            continue

        # Tronquer les résultats selon l'allocation
        results = original_result["results"]
        allocated = table_info["allocated_entries"]

        truncated_result = {
            "results": results[:allocated],
            "metadata": {
                "total_rows": len(results),
                "shown_rows": allocated,
                "truncated": True,
                "source_table": table_info["source_table"],
                "summary": f"Résultats tronqués: {allocated} sur {len(results)} lignes affichées",
            },
        }

        # Essayer de créer un résumé basique des données
        try:
            if results and isinstance(results[0], dict):
                # Identifier les colonnes potentiellement catégoriques
                categorical_summaries = {}
                for key in results[0].keys():
                    # Compter les valeurs uniques pour chaque colonne
                    values = [r.get(key) for r in results if r.get(key) is not None]
                    unique_values = set(values)

                    # Si la colonne a moins de 20 valeurs uniques et moins de 30% du nombre total d'entrées,
                    # on la considère comme catégorique
                    if (
                        len(unique_values) < 20
                        and len(unique_values) < len(values) * 0.3
                    ):
                        value_counts = {}
                        for v in values:
                            value_counts[v] = value_counts.get(v, 0) + 1

                        # Trier par fréquence décroissante
                        sorted_counts = sorted(
                            value_counts.items(), key=lambda x: x[1], reverse=True
                        )
                        categorical_summaries[key] = sorted_counts[
                            :10
                        ]  # Top 10 catégories

                if categorical_summaries:
                    truncated_result["metadata"][
                        "categorical_summary"
                    ] = categorical_summaries

        except Exception as e:
            logger.warning(f"Erreur lors de la création du résumé des données: {e}")

        processed_results.append(truncated_result)
        logger.info(
            f"Table {table_info['source_table']}: Résultats tronqués de {len(results)} à {allocated} entrées"
        )

    return processed_results


def generate_plan(question, schema, plan_model, history):
    """
    Generates an action plan based on the given question and schema.
    Uses an LLM to determine the appropriate steps, SQL queries, or Python code required.
    """
    logger.debug(
        f"🔄 Starting `generate_plan` with question: {question}, schema: {schema}, history: {history}"
    )  # DEBUG: Function initiation

    # Construct database schema description
    schema_description = "Database schema details:\n"
    for table_name, columns in schema.items():
        schema_description += f"Table '{table_name}' contains the following columns:\n"
        for column in columns:
            schema_description += f"  - '{column['name']}' (type: {column['type']})\n"
        schema_description += "\n"

    logger.info(
        f"📜 Generated schema description: {schema_description}"
    )  # INFO: Schema details logged

    # Define the LLM prompt for action plan generation
    prompt = (
        f"History (used to guide and refine follow‑up queries on the same topic):\n"
        f"{history}\n\n"
        f'The request is: "{question}"\n\n'
        "**Instructions to generate the action plan:**\n"
        "1. Determine whether data can be directly extracted from the columns in the schema. If possible, outline a plan to retrieve this data.\n"
        f"2. If data extraction is required, generate a simple and precise SQL query that retrieves only the relevant data, ensuring full compliance with the provided schema. Do not assume any additional columns or data outside of the following schema: {schema_description}\n"
        "3. If the request involves interpretation, calculation, visualization, or content generation (e.g., charts, mathematical computations, or documents), generate appropriate Python code to perform these tasks, but only if Python usage is strictly necessary.\n"
        "4. If the request involves correcting or improving existing code, provide the exact corrections or enhancements needed without referencing the technological context.\n"
        "5. Avoid suggesting unnecessary technologies or methods unless explicitly required. For example, when analyzing a document or extracting textual data, limit your response to essential extraction steps unless a specific output format (chart, plot, calculation, etc.) is requested.\n"
        "6. If a type conversion or adjustment (e.g., INTEGER to VARCHAR) is required to prevent errors, explicitly include these adjustments in the plan.\n"
        "7. If the schema does not contain relevant information, retrieve data from all tables separately without performing joins. There are no restrictions on the number of DuckDB queries you can execute.\n\n"
        "8. When you have WHERE clauses, ensure that the arguments are in lower case format.\n"
        "9. Use the history **only** to guide or refine searches when the user asks for complementary information on the same subject (e.g.:\n"
        "   #force Quels sont les individus ayant un domaine d'expertise lié à la discrétion ?\n"
        "   puis\n"
        "   #force Quels sont les individus cités ayant le niveau expert ?"
        "   — l'historique aide alors à affiner la requête SQL existante, en ajoutant des filtres si nécessaire).\n"
        "   If the user changes topic entirely, ignore the history.\n\n"
        "10. If the user asks a question that demands the consultation of multiple tables, generate a plan to retrieve data from all tables separately without performing joins. There are no restrictions on the number of DuckDB queries you can execute. You can consult every table that seems relevant for you to answer the question.\n\n"
        "11. Sometimes, in tables, just few lines are interesting but it can make the difference and improve the questions. Consult all of the tables that you want."
        "12. **Generic full-text search:** consider any VARCHAR or TEXT column as a candidate and use:`WHERE lower(<column>) LIKE '%keyword%'`."
        "Examples : "
        "« Parmi les personnes listées dans l'annuaire, quelles sont celles citées au moins deux fois dans les documents liés au réglement ? »"
        "**Expected Plan:**\n"
        "- Clearly define the approach (SQL query or action steps) based on the nature of the question.\n"
        "- If SQL is sufficient to answer the question, do not propose other unnecessary methods.\n"
        "- If the question requires visualization or computations, include a method to produce the expected output.\n"
        "- If the request involves code correction or improvement, only provide necessary steps without discussing the technological context.\n"
        "- Do not mention Python unless it is required and Python code is explicitly generated; otherwise, avoid using the term.\n"
        "- Ensure the plan is concise, strictly adhering to available schema information and the specific question.\n"
        "- When you have WHERE clauses, ensure that the arguments are in lower case format.\n"
    )

    try:
        logger.debug(
            "📡 Sending prompt to contextualization model..."
        )  # DEBUG: Sending request to LLM
        plan = plan_model.invoke(input=prompt)
        logger.info(
            f"📋 Action plan generated: {plan}"
        )  # INFO: LLM-generated plan received

        # Extract Python code from the generated plan if applicable
        python_code = parse_code(plan)
        logger.debug(
            f"📝 Extracted Python code: {python_code}"
        )  # DEBUG: Extracted Python snippet

    except Exception as e:
        logger.error(
            f"❌ Error generating action plan or extracting Python code: {e}"
        )  # ERROR: Log exceptions
        raise

    return plan, python_code


def adjust_sql_query_with_duckdb(sql_query, schema):
    """
    Adjusts an SQL query for compatibility with the DuckDB engine,
    handling potential type errors and ensuring schema compliance.
    """
    logger.debug(
        f"🔄 Starting `adjust_sql_query_with_duckdb` with SQL query: {sql_query}"
    )  # DEBUG: Function initiation

    # Construct database schema description
    schema_description = "Database schema for DuckDB:\n"
    for table_name, columns in schema.items():
        schema_description += f"Table '{table_name}' contains the following columns:\n"
        for column in columns:
            schema_description += f"  - '{column['name']}' (type: {column['type']})\n"
        schema_description += "\n"

    try:
        # Adjust SQL query by converting unsupported data types
        sql_query = re.sub(
            r"CAST\((.*?) AS UNSIGNED\)",
            r"CAST(\1 AS INTEGER)",
            sql_query,
            flags=re.IGNORECASE,
        )
        logger.info(
            f"✅ SQL query adjusted for DuckDB: {sql_query}"
        )  # INFO: Successfully adjusted SQL query

    except Exception as e:
        logger.error(
            f"❌ Error adjusting SQL query: {e}"
        )  # ERROR: Log exception details
        raise

    print("Adjusting SQL query with DuckDB model...")

    try:
        logger.debug(
            "📡 Sending query adjustment request to DuckDB model..."
        )  # DEBUG: Sending request to model
        logger.info(
            f"🛠️ Adjusted SQL query from model: {sql_query}"
        )  # INFO: Log model-adjusted query
        return sql_query

    except Exception as e:
        logger.error(
            f"❌ Error executing DuckDB model adjustment: {e}"
        )  # ERROR: Log execution failure
        raise


def validate_sql_with_schema(schema, query):
    """
    Validates an SQL query against the provided schema.
    Ensures that all referenced tables and columns exist within the schema.
    """
    logger.debug(
        f"🔍 Validating SQL query against schema: {schema}"
    )  # DEBUG: Start validation process

    # Create a mapping of table names to their respective columns
    column_map = {
        table: [col["name"] for col in cols] for table, cols in schema.items()
    }

    # Extract tables referenced in the query
    tables_in_query = set(
        re.findall(r"\bFROM\s+(\w+)|\bJOIN\s+(\w+)", query, re.IGNORECASE)
    )
    tables_in_query = {table for match in tables_in_query for table in match if table}

    missing_elements = []

    # Validate referenced tables and columns
    for table in tables_in_query:
        if table not in column_map:
            missing_elements.append(
                f"❌ Unknown table referenced: {table}"
            )  # Log missing table
        else:
            for column in re.findall(rf"{table}\.(\w+)", query):
                if column not in column_map[table]:
                    missing_elements.append(
                        f"❌ Unknown column referenced: {table}.{column}"
                    )  # Log missing column

    # Raise an error if missing elements are detected
    if missing_elements:
        logger.error(
            f"⚠️ Missing tables or columns in schema: {missing_elements}"
        )  # ERROR: Missing elements found
        raise ValueError(f"Missing tables or columns in schema: {missing_elements}")

    logger.info(
        "✅ SQL query validation successful."
    )  # INFO: SQL query validated successfully
    return True


def clean_sql_query(sql_query, schema):
    """
    Cleans an SQL query by ensuring proper table-column references.
    Replaces standalone column names with fully qualified names (table.column).
    """
    logger.debug("🧹 Cleaning SQL query...")  # DEBUG: Start SQL cleaning process

    if schema:
        # Create a mapping of tables to their respective column names
        column_map = {
            table: [col["name"] for col in cols] for table, cols in schema.items()
        }

        # Replace standalone column names with fully qualified names
        for table, columns in column_map.items():
            for column in columns:
                pattern = rf"(?<!\.)\b{column}\b(?!\.)"
                replacement = f"{table}.{column}"
                sql_query = re.sub(pattern, replacement, sql_query)

    # Normalize whitespace
    sql_query = re.sub(r"\s+", " ", sql_query).strip()

    logger.info(f"✅ Cleaned SQL query: {sql_query}")  # INFO: Log cleaned query
    return sql_query


def extract_sql_from_plan(plan_text):
    """
    Extracts all SQL queries from a given plan text.
    - Récupère les requêtes dans des blocs ```sql ... ```
    - Récupère les requêtes classiques SELECT ... ;
    """
    logger = logging.getLogger("action_plan_logger")
    logger.debug(f"🔍 Starting `extract_sql_from_plan` with plan text: {plan_text!r}")

    try:
        queries = []

        # 1) Blocs Markdown ```sql ... ```
        md_blocks = re.findall(r"```sql\s*([\s\S]*?)```", plan_text, re.IGNORECASE)
        logger.info(f"📌 Found Markdown SQL blocks: {md_blocks}")
        for block in md_blocks:
            # On enlève les indentations inutiles
            sql = "\n".join(line.strip() for line in block.strip().splitlines())
            queries.append(sql)

        # 2) Requêtes classiques SELECT ... ;
        classic = re.findall(r"(SELECT[\s\S]*?;)", plan_text, re.IGNORECASE)
        logger.info(f"📌 Found classic SQL queries: {classic}")
        queries.extend([q.strip() for q in classic])

        # Dédupliquer
        unique_queries = list(dict.fromkeys(queries))
        if unique_queries:
            logger.info(f"✅ Unique SQL queries extracted: {unique_queries}")
            return unique_queries
        else:
            logger.warning("⚠️ No SQL queries found in the plan.")
            raise ValueError("No SQL queries found in the plan.")

    except Exception as e:
        logger.error(f"❌ Error extracting SQL queries: {e}")
        raise


def generate_tools_with_llm(
    plan,
    schema,
    context,
    sql_results,
    python_results,
    code_model,
    python_code,
    custom_sql_path=None,
):
    """
    Generates the necessary tools based on the given plan.
    Handles SQL query extraction, validation, execution, and Python code generation.
    """
    logger.debug(
        "🔄 Starting `generate_tools_with_llm` function."
    )  # DEBUG: Function initiation
    files_generated = []
    results = []

    # Process SQL queries if present in the plan
    if "SQL" in plan:
        logger.info(
            "🛠 Processing SQL queries from the plan."
        )  # INFO: SQL processing started
        try:
            sql_queries = extract_sql_from_plan(plan)
            logger.info(
                f"📌 Extracted SQL queries: {sql_queries}"
            )  # INFO: Log extracted queries

            for i, sql_query in enumerate(sql_queries):
                try:
                    logger.debug(
                        f"🔍 Processing SQL query #{i + 1}: {sql_query}"
                    )  # DEBUG: Log query processing

                    # Clean and validate SQL query
                    sql_query = clean_sql_query(sql_query, schema)
                    logger.info(
                        f"✅ Cleaned SQL query: {sql_query}"
                    )  # INFO: Log cleaned query

                    validate_sql_with_schema(schema, sql_query)
                    logger.info(
                        "✅ SQL query validated against schema."
                    )  # INFO: Schema validation success

                    # Adjust SQL query for DuckDB compatibility
                    sql_query = adjust_sql_query_with_duckdb(
                        sql_query, schema
                    )
                    logger.info(
                        f"🔧 Adjusted SQL query: {sql_query}"
                    )  # INFO: Log adjusted query

                    # Execute SQL query
                    if custom_sql_path is not None:
                        sql_results = execute_sql_query(sql_query, custom_sql_path)
                    else:
                        sql_results = execute_sql_query(sql_query)
                    logger.info(
                        f"📊 SQL query results: {sql_results}"
                    )  # INFO: Log query results

                    if "sql_results" not in context:
                        context["sql_results"] = []

                    if sql_results:
                        context["sql_results"].append({"results": sql_results})

                except Exception as e:
                    logger.error(
                        f"❌ Error processing SQL query #{i + 1}: {e}"
                    )  # ERROR: Query execution failure
                    continue

            logger.info(
                f"📌 Updated context with SQL results: {context['sql_results']}"
            )  # INFO: Context updated

        except Exception as e:
            logger.error(
                f"❌ Error extracting or processing SQL queries: {e}"
            )  # ERROR: Extraction failure

    # Process Python code generation if required
    if "Python" in plan or "python" in plan:
        logger.info(
            "🖥️ Generating Python code based on the plan."
        )  # INFO: Python processing started
        try:
            logger.debug(
                f"📊 Available SQL results for Python code: {context['sql_results']}"
            )  # DEBUG: Log available SQL results

            # Tronquer les résultats SQL pour le prompt Python si nécessaire
            truncated_sql_for_python = truncate_large_sql_results(
                context["sql_results"]
            )

            prompt = (
                f'The initial request is: "{context["question"]}"\n\n'
                f"Here is an example of Python code that could fulfill the request: {python_code}\n"
                "**Instructions for code generation:**\n"
                "1. Use only the exact data provided in the results below. Do not generate any additional or fictitious values.\n"
                "2. Do not assume default values for missing data; **strictly use only the given data**.\n"
                "3. Ensure that the code is **complete, functional, and executable without manual intervention**.\n"
                "4. Avoid unnecessary **conditional logic** or **assumptions**. If data is missing, do not attempt to complete or guess it; only use the available results.\n"
                "5. You are working in a **Docker environment without a graphical interface**. Any visualizations (e.g., Matplotlib graphs) must be **saved to a file** (e.g., PNG for graphs).\n"
                "6. **Do not use plt.show()**, as graphical results cannot be displayed directly.\n"
                "7. If the task involves **simple calculations or non-visual operations** (e.g., computing averages), generate appropriate code without attempting to produce files.\n"
                "8. For graphical results, ensure files are saved, ignoring format or naming conventions (use default names).\n\n"
                "9. Whether the request involves a graph, a calculation, or another operation, generate the code using only the extracted values, maximizing included elements to provide a full view, and without inventing any data.\n"
                f"Here are the available SQL results:\n{safe_json_dumps(truncated_sql_for_python)}\n\n"
                "**Generate complete Python code that utilizes these results as static data.** The code should directly address the request (graph, calculation, or other) and **never** make calls to databases such as SQLite or external services for retrieving data."
            )

            logger.debug(
                "📡 Sending prompt for Python code generation to reasoning model."
            )  # DEBUG: Sending request to LLM
            python_tool = code_model.invoke(prompt)

            context, python_results, files_generated = parse_and_execute_python_code(
                python_tool, context, truncated_sql_for_python
            )

            logger.info(
                f"📌 Python results: {python_results}"
            )  # INFO: Log Python execution results
            logger.info(
                f"📂 Generated files: {files_generated}"
            )  # INFO: Log generated files

        except Exception as e:
            logger.error(
                f"❌ Error generating or executing Python code: {e}"
            )  # ERROR: Python execution failure

    logger.debug(
        "✅ Finished `generate_tools_with_llm` function."
    )  # DEBUG: Function completion
    return context, python_results, sql_results, files_generated


def generate_manual_sources(sql_results):
    """
    Génère des sources manuelles à partir des résultats SQL complets.

    Args:
        sql_results: Liste des résultats SQL non tronqués

    Returns:
        Une chaîne de caractères contenant les sources formatées
    """
    if not sql_results:
        return ""

    sources = []
    sources.append("\n\n--- SOURCES DÉTAILLÉES ---\n")

    for i, result_set in enumerate(sql_results):
        if not isinstance(result_set, dict) or "results" not in result_set:
            continue

        results = result_set.get("results", [])
        source_table = result_set.get("source_table", f"table_{i+1}")

        sources.append(f"\nSource: {source_table} ({len(results)} entrées)\n")

        # Limiter à 100 entrées max dans les sources détaillées pour éviter des réponses trop longues
        display_limit = min(100, len(results))

        if display_limit > 0:
            for j, row in enumerate(results[:display_limit]):
                if isinstance(row, dict):
                    row_str = " | ".join([f"{k}: {v}" for k, v in row.items()])
                else:
                    row_str = str(row)

                sources.append(f"  - Ligne {j+1}: {row_str}\n")

            if len(results) > display_limit:
                sources.append(
                    f"  ... et {len(results) - display_limit} autres entrées\n"
                )

    return "".join(sources)


def generate_final_response_with_llama(
    context, python_results, reasoning_model, files_generated, history
):
    """
    Generates the final response using the LLM model.
    Summarizes results from SQL queries, Python execution, and file generation.
    """
    logger = logging.getLogger("action_plan_logger")
    logger.debug(
        "🔄 Starting `generate_final_response_with_llama`."
    )  # DEBUG: Function initiation

    # Conserver une copie des résultats SQL originaux pour les sources manuelles
    original_sql_results = context.get("sql_results", [])

    # Tronquer les résultats SQL s'ils sont trop volumineux
    truncated_sql_results = None
    if "sql_results" in context and context["sql_results"]:
        logger.info("Optimisation des résultats SQL pour le prompt final")
        truncated_sql_results = truncate_large_sql_results(context["sql_results"])
    else:
        truncated_sql_results = context.get("sql_results", [])

    # Create the section for generated files
    files_section = ""
    if files_generated:
        logger.info("📂 Generated files are available.")  # INFO: Files detected
        files_section = "\nGenerated Files:\n" + "\n".join(
            [f"- {file}" for file in files_generated]
        )
        logger.debug(f"📌 Files section added: {files_section}")  # DEBUG: Log file list
    else:
        logger.info("📂 No files were generated.")  # INFO: No files available

    logger.debug(
        f"Voila les resultats SQL finaux : {safe_json_dumps(truncated_sql_results, indent=2)}"
    )

    # Construct the prompt for the reasoning model
    prompt = (
        f"Final Context:\n\n"
        f"Question: \"{context['question']}\"\n"
        f"SQL Results: {safe_json_dumps(truncated_sql_results)} \n"
        f"Python Results: {python_results}\n\n"
        f'History: "{history}"\n'
        f"{files_section}\n\n"
        "**Final Response Guidelines:**\n"
        "- Respond DIRECTLY to the question, focusing precisely on what was asked without generalizing.\n"
        "- For the query example 'Quels sont les individus ayant un libellé de niveau d'expertise expert sur le site de Brest?', respond with exact names and details of people matching ALL criteria (both expert level AND Brest location).\n"
        "- Always include detailed source information using this format: [Source: table_name, row X] after each piece of information.\n"
        "- List ALL results matching the criteria exactly as they appear in the data, without summarizing or omitting entries.\n"
        "- If a history is present and non-null, extract useful elements from it to answer the initial question.\n"
        "- If the keyword '#pass' is included in the question and no SQL or Python results are available, you are allowed to respond freely.\n"
        "**Specific Directives:**\n"
        "1. Answer in the SAME LANGUAGE as the question was asked.\n"
        "2. If multiple filters are in the query (e.g., 'expertise expert' AND 'site de Brest'), ensure ALL criteria are applied.\n"
        "3. Present ALL matching data explicitly - avoid vague summaries like 'The document contains a list of individuals...'.\n"
        "4. For each fact presented, cite its source using [Source: table_name, row X] format.\n"
        "5. If files have been generated, provide links and explain their relevance to the request.\n"
        "6. For numerical results, provide exact numbers with proper context.\n"
        "7. If no results match ALL criteria, clearly state this rather than providing partial matches.\n"
        "8. For LARGE RESULT SETS (>50 entries): Present the first 20 results in detail, then provide a summary with total count and key categories. Still cite sources for the summary, including the total row count from the source table.\n\n"
        "**EXAMPLES OF GOOD RESPONSES:**\n\n"
        "Example 1:\n"
        "Voici les individus ayant un niveau d'expertise 'Expert' sur le site de Brest :\n\n"
        "1. Martin Dupont - Niveau: Expert - Site: Brest [Source: experts_table, ligne 12]\n"
        "2. Sophie Laurent - Niveau: Expert - Site: Brest [Source: experts_table, ligne 15]\n"
        "3. Jean Durand - Niveau: Expert - Site: Brest [Source: experts_table, ligne 23]\n\n"
        "Example 2:\n"
        "Les personnes ayant participé à plus de 3 projets en 2022 sont :\n\n"
        "1. Thomas Bernard - 5 projets [Source: projets_table, lignes 34-38]\n"
        "2. Marie Lefevre - 4 projets [Source: projets_table, lignes 45-48]\n"
        "Chaque personne est indiquée avec le nombre exact de projets et la source précise des données.\n\n"
        "Example 3:\n"
        "Aucun individu ne correspond aux critères 'niveau d'expertise Senior' ET 'site de Marseille'. [Source: recherche complète dans experts_table]\n\n"
        "Example 4 (pour grands volumes de données):\n"
        "Voici les 20 premiers documents référencés dans la base de données (sur un total de 412) [Source: documents_table, 412 lignes au total]:\n\n"
        "1. Rapport 2022-01 - Catégorie: Finances - Date: 2022-01-15 [Source: documents_table, ligne 1]\n"
        "2. Rapport 2022-02 - Catégorie: RH - Date: 2022-01-22 [Source: documents_table, ligne 2]\n"
        "...\n"
        "20. Rapport 2022-20 - Catégorie: Finances - Date: 2022-05-15 [Source: documents_table, ligne 20]\n\n"
        "Résumé des 412 documents par catégorie:\n"
        "- Finances: 189 documents [Source: documents_table, comptage des entrées catégorie='Finances']\n"
        "- RH: 103 documents [Source: documents_table, comptage des entrées catégorie='RH']\n"
        "- Technique: 120 documents [Source: documents_table, comptage des entrées catégorie='Technique']\n\n"
        "NE JAMAIS RÉPONDRE AVEC DES RÉSUMÉS GÉNÉRAUX. TOUJOURS DONNER DES RÉPONSES PRÉCISES AVEC SOURCES."
    )

    logger.debug(
        "📜 Constructed prompt for reasoning model."
    )  # DEBUG: Prompt creation logged

    try:
        # Call the reasoning model to generate the final response
        logger.info(
            "📡 Sending prompt to the reasoning model."
        )  # INFO: Model invocation started
        final_response = reasoning_model.invoke(input=prompt)
        logger.info(
            "✅ Final response successfully generated."
        )  # INFO: Model response received

    except Exception as e:
        logger.error(
            f"❌ Error generating final response: {e}"
        )  # ERROR: Log model failure
        raise

    # Append generated file links to the response if available
    if files_generated:
        links_section = "\n\nGenerated File Links:\n" + "\n".join(
            [f"- {file}" for file in files_generated]
        )
        final_response += links_section
        logger.debug(
            f"📂 Added file links to final response: {links_section}"
        )  # DEBUG: File links added

    # Ajouter les sources manuelles à la réponse
    manual_sources = generate_manual_sources(original_sql_results)
    if manual_sources:
        final_response += manual_sources
        logger.info("📚 Added manual sources to final response")

    # Log and return the final response
    logger.debug(
        f"📝 Final response: {final_response}"
    )  # DEBUG: Log final response content
    logger.debug(
        "✅ Completed `generate_final_response_with_llama`."
    )  # DEBUG: Function completion

    return final_response
