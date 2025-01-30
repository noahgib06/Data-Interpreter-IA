import json
import logging
import os
import re
from logging.handlers import RotatingFileHandler

import duckdb
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from unidecode import unidecode

from PdfExtension import extract_pdf
from PythonExtension import extract_python

load_dotenv()

# Configuration du logger global
LOG_LEVEL_ENV = os.getenv(
    "LOG_LEVEL_SetupDatabase"
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
    log_file=os.getenv("LOG_FILE_SetupDatabase"),
    max_size=10 * 1024 * 1024,
    backup_count=5,
):
    if not os.path.exists("../Logs"):
        os.makedirs("../Logs", exist_ok=True)
    logger = logging.getLogger("database_logger")
    logger.setLevel(LOG_LEVEL_MAP.get(LOG_LEVEL_ENV))
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

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


logger = setup_logger()


def remove_database_file():
    database_path = os.getenv("DB_FILE")
    if os.path.exists(database_path):
        os.remove(database_path)
        logger.info(f"Fichier '{database_path}' supprimé avec succès.")
    else:
        logger.warning(f"Le fichier '{database_path}' n'existe pas.")


def clean_column_name(column_name):
    if pd.isna(column_name):
        return "unnamed"
    column_name = unidecode(column_name)
    column_name = column_name.replace("'", "_")
    cleaned_name = re.sub(r"[^a-zA-Z0-9_]", "_", column_name).lower()
    logger.debug(f"Nettoyage du nom de colonne '{column_name}' -> '{cleaned_name}'")
    return cleaned_name


def prepare_database(filepaths=None):

    all_filepaths = []
    for path in filepaths:
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for file in files:
                    all_filepaths.append(os.path.join(root, file))
        else:
            all_filepaths.append(path)

    conn = duckdb.connect(os.getenv("DB_FILE"))
    logger.info(f"Fichiers à traiter : {all_filepaths}")
    if all_filepaths:
        for filepath in all_filepaths:
            logger.info(f"Traitement du fichier : {filepath}")
            try:
                data = load_file_data(filepath)
                for sheet_name, df in data.items():
                    table_name = generate_table_name(filepath, sheet_name)
                    logger.debug(
                        f"Création de la table '{table_name}' pour la feuille '{sheet_name}'..."
                    )
                    create_table_from_dataframe(conn, df, table_name)
                    handle_nested_data(conn, df, table_name)
            except ValueError as e:
                logger.error(f"Erreur de format pour le fichier '{filepath}': {e}")
            except Exception as e:
                logger.error(
                    f"Erreur inattendue lors du traitement du fichier '{filepath}': {e}"
                )
    return conn


def load_file_data(filepath):
    logger.info(f"Chargement des données pour le fichier : {filepath}")
    if filepath.endswith(".xls"):
        return pd.read_excel(filepath, sheet_name=None, engine="xlrd")
    elif filepath.endswith(".xlsx") or filepath.endswith(".xlsm"):
        return pd.read_excel(filepath, sheet_name=None, engine="openpyxl")
    elif filepath.endswith(".csv"):
        return {"sheet1": pd.read_csv(filepath, sep=";", encoding="utf-8")}
    elif filepath.endswith(".json"):
        with open(filepath, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        return {"main": pd.json_normalize(json_data, sep="_")}
    elif filepath.endswith(".pdf"):
        return process_pdf_file(filepath)
    elif filepath.endswith(".py"):
        return process_python_file(filepath)
    else:
        raise ValueError("Format de fichier non pris en charge.")


def process_pdf_file(filepath):
    logger.info(f"Traitement du fichier PDF : {filepath}")
    extracted_text, images_data = extract_pdf(filepath)
    data = {}
    if extracted_text:
        data["text"] = pd.DataFrame([{"content": t} for t in extracted_text])
    if images_data:
        data["images"] = pd.DataFrame(images_data)
    return data


def process_python_file(filepath):
    logger.info(f"Traitement du fichier Python : {filepath}")
    extracted_data = extract_python(filepath)
    data = {}
    if "functions" in extracted_data:
        data["functions"] = pd.DataFrame(extracted_data["functions"])
    if "classes" in extracted_data:
        data["classes"] = pd.DataFrame(extracted_data["classes"])
    if "imports" in extracted_data:
        data["imports"] = pd.DataFrame(extracted_data["imports"])
    if "module_code" in extracted_data:
        data["module_code"] = pd.DataFrame(
            [{"module_code": extracted_data["module_code"]}]
        )
    return data


def generate_table_name(filepath, sheet_name):
    base_name = re.sub(
        r"[^a-zA-Z0-9_]", "_", os.path.splitext(os.path.basename(filepath))[0]
    ).lower()
    table_name = f"{base_name}_{clean_column_name(sheet_name)}"
    logger.debug(f"Génération du nom de table : {table_name}")
    return table_name


def create_table_from_dataframe(conn, df, table_name):
    logger.info(f"Création de la table '{table_name}' dans DuckDB...")
    column_definitions = []
    for column_name, dtype in df.dtypes.items():
        column_type = map_dtype_to_duckdb_type(dtype, df[column_name])
        column_definitions.append(f'"{clean_column_name(column_name)}" {column_type}')

    column_definitions_str = ", ".join(column_definitions)
    create_table_query = (
        f"CREATE TABLE IF NOT EXISTS {table_name} ({column_definitions_str})"
    )
    conn.execute(create_table_query)

    chunks = np.array_split(df, np.ceil(len(df) / 500))
    for chunk in chunks:
        conn.register("temp_chunk", chunk)
        conn.execute(f"INSERT INTO {table_name} SELECT * FROM temp_chunk")
    logger.info(f"Table '{table_name}' créée et données insérées.")


def handle_nested_data(conn, df, base_table_name):
    logger.info(f"Gestion des données imbriquées pour la table '{base_table_name}'...")
    for column in df.columns:
        if df[column].apply(lambda x: isinstance(x, (list, dict))).any():
            logger.debug(f"Colonne imbriquée détectée : {column}")
            nested_entries = []
            for index, row in df.iterrows():
                value = row[column]
                if isinstance(value, list):
                    nested_entries.extend(value)
                elif isinstance(value, dict):
                    nested_entries.append(value)

            if nested_entries:
                nested_df = pd.json_normalize(nested_entries, sep="_")
                nested_table_name = f"{base_table_name}_{clean_column_name(column)}"
                create_table_from_dataframe(conn, nested_df, nested_table_name)


def map_dtype_to_duckdb_type(dtype, column_data):
    type_mapping = {
        np.integer: "INTEGER",
        np.floating: "DOUBLE",
        np.bool_: "BOOLEAN",
        np.datetime64: "TIMESTAMP",
        object: lambda: (
            "DOUBLE"
            if column_data.apply(is_float).all()
            else (
                "INTEGER"
                if column_data.apply(is_integer).all()
                else "TIMESTAMP" if column_data.apply(is_date).all() else "TEXT"
            )
        ),
    }
    base_type = dtype.type
    return (
        type_mapping.get(base_type, "TEXT")()
        if callable(type_mapping.get(base_type))
        else type_mapping.get(base_type, "TEXT")
    )


def is_float(value):
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def is_integer(value):
    try:
        return float(value).is_integer()
    except (ValueError, TypeError):
        return False


def is_date(value):
    try:
        pd.to_datetime(value)
        return True
    except (ValueError, TypeError):
        return False
