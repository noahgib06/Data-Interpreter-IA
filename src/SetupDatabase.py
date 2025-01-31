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

UUID_REGEX = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_"
)

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
    """
    Nettoie le nom d'une colonne pour être compatible avec DuckDB.
    """
    if pd.isna(column_name) or column_name.strip() == "":
        return "unnamed_column"
    column_name = unidecode(column_name)  # Supprime les accents
    column_name = column_name.replace("'", "_")
    column_name = re.sub(r"[^a-zA-Z0-9_]", "_", column_name).lower()
    # Supprimer les UUID éventuels du nom
    column_name = re.sub(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
        "",
        column_name,
    )
    column_name = re.sub(r"_+", "_", column_name).strip("_")
    # Ajouter un préfixe si le nom commence par un chiffre
    if column_name and column_name[0].isdigit():
        column_name = "col_" + column_name
    return column_name


def prepare_database(filepaths=None, collection_id=None):
    """
    Parcourt la liste des fichiers, en extrait les dataframes,
    et crée les tables dans DuckDB sans recréer celles déjà identiques.
    """
    if not filepaths:
        return

    all_filepaths = []
    for path in filepaths:
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for file in files:
                    all_filepaths.append(os.path.join(root, file))
        else:
            all_filepaths.append(path)

    if collection_id is not None:
        path = os.getenv("DB_FILE")
        path = path.replace("id", str(collection_id))
        conn = duckdb.connect(path)
    else:
        conn = duckdb.connect(os.getenv("DB_FILE"))
    logger.info(f"Fichiers à traiter : {all_filepaths}")

    for filepath in all_filepaths:
        logger.info(f"Traitement du fichier : {filepath}")
        try:
            data = load_file_data(filepath)
            for sheet_name, df in data.items():
                table_name = generate_table_name(filepath, sheet_name)
                logger.debug(
                    f"Création éventuelle de la table '{table_name}' pour la feuille '{sheet_name}'..."
                )
                create_table_from_dataframe(conn, df, table_name)
                handle_nested_data(conn, df, table_name)
        except ValueError as e:
            logger.error(f"Erreur de format pour le fichier '{filepath}': {e}")
        except Exception as e:
            logger.error(
                f"Erreur inattendue lors du traitement du fichier '{filepath}': {e}"
            )

    conn.close()


def load_file_data(filepath):
    """
    Charge les données selon l'extension du fichier.
    """
    logger.info(f"Chargement des données pour le fichier : {filepath}")
    if filepath.endswith(".xls"):
        return pd.read_excel(filepath, sheet_name=None, engine="xlrd")
    elif filepath.endswith(".xlsx") or filepath.endswith(".xlsm"):
        return pd.read_excel(filepath, sheet_name=None, engine="openpyxl")
    elif filepath.endswith(".csv"):
        return {
            "sheet1": pd.read_csv(filepath, sep=";", encoding="utf-8", engine="python")
        }
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
    """
    Génère un nom de table SANS UUID, même si le fichier se nomme
    {uuid}_trucchose.pdf.
    """
    base = os.path.splitext(os.path.basename(filepath))[0]
    # Supprimer l'UUID éventuel au début (ex: "8a2da82f-1cd5-42b6-91bb-d0c2282f57d0_...")
    base = re.sub(UUID_REGEX, "", base)  # retire l'UUID + underscore
    base = re.sub(r"[^A-Za-z0-9_]+", "_", base).strip("_").lower()

    sheet = re.sub(r"[^A-Za-z0-9_]+", "_", sheet_name).strip("_").lower()

    table_name = f"{base}_{sheet}"
    # on évite que ça commence par un chiffre
    if table_name and table_name[0].isdigit():
        table_name = "t_" + table_name

    logger.debug(f"Nom de table final: {table_name}")
    return table_name


def table_exists_with_same_schema(conn, df, table_name):
    """
    Vérifie si la table `table_name` existe déjà dans la base
    ET si son schéma correspond exactement aux colonnes de `df`.
    S'il y a correspondance parfaite, on considère que c'est un doublon.
    """
    # Récupérer la liste des tables existantes
    tables = [x[0] for x in conn.execute("SHOW TABLES").fetchall()]
    if table_name not in tables:
        return False  # La table n'existe pas encore

    # Décrire la table existante
    # duckdb DESC <table> renvoie un tableau du type (column_name, column_type, null, key, default, extra)
    schema_info = conn.execute(f"DESCRIBE {table_name}").fetchall()

    # Extraire seulement les noms de colonnes et types DuckDB
    existing_cols = [(row[0].lower(), row[1].lower()) for row in schema_info]

    # Construire la liste (colName, duckdbType) attendue
    desired_cols = []
    for col in df.columns:
        colname_clean = clean_column_name(col)
        coltype = map_dtype_to_duckdb_type(df[col].dtype, df[col])
        desired_cols.append((colname_clean.lower(), coltype.lower()))

    # Comparaison stricte (même nombre de colonnes, mêmes noms, mêmes types, dans le même ordre)
    if len(existing_cols) != len(desired_cols):
        return False

    return existing_cols == desired_cols


def create_table_from_dataframe(conn, df, table_name):
    logger.info(f"Création éventuelle de la table '{table_name}' dans DuckDB...")

    # Vérifier si la table existe déjà à l'identique
    if table_exists_with_same_schema(conn, df, table_name):
        logger.info(
            f"Table '{table_name}' existe déjà avec le même schéma, pas de création."
        )
        return  # On ne recrée pas

    # Sinon, on crée la table (ou on la recrée si le schéma a changé)
    # Pour éviter un conflit, on peut DROP la table s'il existe un "table_name" au schéma différent
    tables = [x[0] for x in conn.execute("SHOW TABLES").fetchall()]
    if table_name in tables:
        logger.info(
            f"Table '{table_name}' existe déjà mais le schéma diffère, on la DROP avant recréation."
        )
        conn.execute(f"DROP TABLE {table_name}")

    # Prépare la requête de CREATE TABLE
    column_definitions = []
    for column_name, dtype in df.dtypes.items():
        col_clean = clean_column_name(column_name)
        column_type = map_dtype_to_duckdb_type(dtype, df[column_name])
        column_definitions.append(f'"{col_clean}" {column_type}')

    column_definitions_str = ", ".join(column_definitions)
    create_table_query = f"CREATE TABLE {table_name} ({column_definitions_str})"

    try:
        conn.execute(create_table_query)
        logger.info(f"Table '{table_name}' créée avec succès.")
    except Exception as e:
        logger.error(f"Erreur SQL lors de la création de la table {table_name}: {e}")
        return

    # Insertion par paquets
    try:
        if not df.empty:
            batch_size = max(1, len(df) // 500)  # découpage en ~500 lignes par lot
            for chunk in np.array_split(df, batch_size):
                conn.register("temp_chunk", chunk)
                conn.execute(f"INSERT INTO {table_name} SELECT * FROM temp_chunk")
        logger.info(f"Données insérées dans '{table_name}' avec succès.")
    except Exception as e:
        logger.error(f"Erreur SQL lors de l'insertion dans {table_name}: {e}")
    logger.info(f"Table '{table_name}' opération terminée.")


def handle_nested_data(conn, df, base_table_name):
    """
    Gère les colonnes contenant des données imbriquées en créant des tables supplémentaires.
    """
    logger.info(f"Gestion des données imbriquées pour la table '{base_table_name}'...")

    # Supprime un éventuel UUID
    base_table_name = re.sub(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
        "",
        base_table_name,
    ).strip("_")

    for column in df.columns:
        # On check si la colonne contient parfois dict ou list
        if df[column].apply(lambda x: isinstance(x, (list, dict))).any():
            logger.debug(f"Colonne imbriquée détectée : {column}")
            nested_entries = []
            for _, row in df.iterrows():
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
