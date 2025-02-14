import logging
import os
import re
from logging.handlers import RotatingFileHandler

from dotenv import load_dotenv

from PythonTool import parse_and_execute_python_code, parse_code
from SqlTool import execute_sql_query

load_dotenv()

# Configuration du logger global
LOG_LEVEL_ENV = os.getenv(
    "LOG_LEVEL_LlmGeneration"
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
    log_file=os.getenv("LOG_FILE_LlmGeneration"),
    max_size=5 * 1024 * 1024,
    backup_count=3,
):
    """
    Configure un logger global pour suivre toutes les actions.
    """
    if not os.path.exists("../Logs"):
        os.makedirs("../Logs", exist_ok=True)
    logger = logging.getLogger("action_plan_logger")
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

    # Ajout des handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


# Initialisation du logger
logger = setup_logger()


def command_r_plus_plan(question, schema, plan_model, history):
    """
    Generates an action plan based on the given question and schema.
    Uses an LLM to determine the appropriate steps, SQL queries, or Python code required.
    """
    logger.debug(
        f"🔄 Starting `command_r_plus_plan` with question: {question}, schema: {schema}, history: {history}"
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
        f'The request is: "{question}"\n\n'
        "**Instructions to generate the action plan:**\n"
        "1. Determine whether data can be directly extracted from the columns in the schema. If possible, outline a plan to retrieve this data.\n"
        f"2. If data extraction is required, generate a simple and precise SQL query that retrieves only the relevant data, ensuring full compliance with the provided schema. Do not assume any additional columns or data outside of the following schema: {schema_description}\n"
        "3. If the request involves interpretation, calculation, visualization, or content generation (e.g., charts, mathematical computations, or documents), generate appropriate Python code to perform these tasks, but only if Python usage is strictly necessary.\n"
        "4. If the request involves correcting or improving existing code, provide the exact corrections or enhancements needed without referencing the technological context.\n"
        "5. Avoid suggesting unnecessary technologies or methods unless explicitly required. For example, when analyzing a document or extracting textual data, limit your response to essential extraction steps unless a specific output format (chart, plot, calculation, etc.) is requested.\n"
        "6. If a type conversion or adjustment (e.g., INTEGER to VARCHAR) is required to prevent errors, explicitly include these adjustments in the plan.\n"
        "7. If the schema does not contain relevant information, retrieve data from all tables separately without performing joins. There are no restrictions on the number of DuckDB queries you can execute.\n\n"
        "**Expected Plan:**\n"
        "- Clearly define the approach (SQL query or action steps) based on the nature of the question.\n"
        "- If SQL is sufficient to answer the question, do not propose other unnecessary methods.\n"
        "- If the question requires visualization or computations, include a method to produce the expected output.\n"
        "- If the request involves code correction or improvement, only provide necessary steps without discussing the technological context.\n"
        "- Do not mention Python unless it is required and Python code is explicitly generated; otherwise, avoid using the term.\n"
        "- Ensure the plan is concise, strictly adhering to available schema information and the specific question.\n"
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


def adjust_sql_query_with_duckdb(sql_query, schema, duckdb_model):
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
    Uses regex to identify and retrieve SQL SELECT statements.
    """
    logger = logging.getLogger("action_plan_logger")
    logger.debug(
        f"🔍 Starting `extract_sql_from_plan` with plan text: {plan_text}"
    )  # DEBUG: Function initiation

    try:
        # Extract SQL queries using regex pattern
        queries = re.findall(r"SELECT .*?;", plan_text, re.DOTALL)
        logger.info(
            f"📌 Extracted SQL queries: {queries}"
        )  # INFO: Log extracted queries

        # Remove duplicate queries
        unique_queries = list(set(queries))
        if unique_queries:
            logger.info(
                f"✅ Unique SQL queries extracted: {unique_queries}"
            )  # INFO: Log unique queries
            return unique_queries
        else:
            logger.warning(
                "⚠️ No SQL queries found in the plan."
            )  # WARNING: No SQL queries detected
            raise ValueError("No SQL queries found in the plan.")

    except Exception as e:
        logger.error(
            f"❌ Error extracting SQL queries: {e}"
        )  # ERROR: Log extraction failure
        raise


def generate_tools_with_llm(
    plan,
    schema,
    context,
    sql_results,
    python_results,
    database_model,
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
                        sql_query, schema, database_model
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
                f"Here are the available SQL results:\n{context['sql_results']}\n\n"
                "**Generate complete Python code that utilizes these results as static data.** The code should directly address the request (graph, calculation, or other) and **never** make calls to databases such as SQLite or external services for retrieving data."
            )

            logger.debug(
                "📡 Sending prompt for Python code generation to reasoning model."
            )  # DEBUG: Sending request to LLM
            python_tool = code_model.invoke(prompt)

            context, python_results, files_generated = parse_and_execute_python_code(
                python_tool, context, context["sql_results"]
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

    # Construct the prompt for the reasoning model
    prompt = (
        f"Final Context:\n\n"
        f"Question: \"{context['question']}\"\n"
        f"SQL Results: {context['sql_results']}\n"
        f"Python Results: {python_results}\n\n"
        f'History: "{history}"\n'
        f"{files_section}\n\n"
        "**Final Response Guidelines:**\n"
        "- Summarize the content concisely, explaining the purpose of the document or directly answering the request.\n"
        "- Do not include intermediate reasoning or speculative additions. Use only the provided final context to construct the response.\n\n"
        "- If a history is present and non-null, extract useful elements from it to answer the initial question. Otherwise, rely solely on the SQL and Python results.\n"
        "- If the keyword '#pass' is included in the question and no SQL or Python results are available, you are allowed to respond freely.\n"
        "**Specific Directives:**\n"
        "1. If files have been generated (mentioned above), briefly explain their content and relevance to the request.\n"
        "2. If the response contains numerical results, ensure they are well-contextualized for immediate understanding.\n"
        "3. Do not provide unnecessary technical explanations unless explicitly required by the initial question. Keep the response user-friendly.\n"
        "4. Explicitly mention the links to any generated files (listed above) in the response.\n"
        "5. Always respond in the same language as the initial question.\n"
        "6. If history is available and non-null, use it to refine the response.\n"
        "7. If the keyword '#pass' is in the question and no SQL or Python results exist, you may generate a response as you see fit. In this case, prioritize historical context for coherence. Using general knowledge is permitted if it helps answer the question accurately.\n"
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

    # Log and return the final response
    logger.debug(
        f"📝 Final response: {final_response}"
    )  # DEBUG: Log final response content
    logger.debug(
        "✅ Completed `generate_final_response_with_llama`."
    )  # DEBUG: Function completion

    return final_response
