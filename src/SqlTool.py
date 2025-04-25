import logging
import os

import duckdb
from dotenv import load_dotenv

load_dotenv()

# Configuration du logger global
LOG_LEVEL_ENV = os.getenv("LOG_LEVEL_SqlTool", "INFO")  # Valeur par défaut: INFO

# Mappage des niveaux de log
LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def setup_logger(
    log_file=os.getenv("LOG_FILE_SqlTool"), max_size=5 * 1024 * 1024, backup_count=3
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

    logger = logging.getLogger("duckdb_logger")
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


def execute_sql_query(query, db_path):
    """
    Executes an SQL query on a DuckDB database and returns the formatted results.
    Establishes a connection, runs the query, and retrieves structured output.
    """
    print("📂 Database path: ", db_path)
    logger.info(
        f"🚀 Starting SQL query execution: {query}"
    )  # INFO: Query execution start

    try:
        # Connect to the DuckDB database
        connection = duckdb.connect(db_path)
        logger.info(
            f"✅ Connected to the database: {db_path}"
        )  # INFO: Connection success

        # Execute the query and fetch results
        results = connection.execute(query).fetchall()
        column_names = [desc[0] for desc in connection.description]
        formatted_results = [dict(zip(column_names, row)) for row in results]

        logger.info(
            f"📊 SQL query results: {formatted_results}"
        )  # INFO: Log retrieved results
        return formatted_results
    except Exception as e:
        logger.error(
            f"❌ Error executing SQL query: {e}"
        )  # ERROR: Query execution failure
        return None


def get_schema(con):
    """
    Retrieves the schema of all tables in the database.
    Queries the database to list all tables and their column details.
    """
    logger.info("📂 Starting database schema retrieval.")  # INFO: Process start
    schema_info = {}

    try:
        # Fetch all table names
        table_names = con.execute("SHOW TABLES").fetchall()
        logger.info(
            f"✅ Tables found in the database: {table_names}"
        )  # INFO: Tables detected

        # Retrieve schema for each table
        for table in table_names:
            table_name = table[0]
            columns_info = con.execute(f"PRAGMA table_info('{table_name}')").fetchall()

            schema_info[table_name] = [
                {"name": column[1], "type": column[2]} for column in columns_info
            ]

            logger.info(
                f"📑 Schema for table '{table_name}': {schema_info[table_name]}"
            )  # INFO: Table schema logged

    except Exception as e:
        logger.error(
            f"❌ Error retrieving schema: {e}"
        )  # ERROR: Schema retrieval failed

    return schema_info
