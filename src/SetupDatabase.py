import csv
import json
import logging
import os
import re
from logging.handlers import RotatingFileHandler

import duckdb
import markdown
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from unidecode import unidecode
from docx import Document

from PdfExtension import extract_pdf
from PythonExtension import extract_python

load_dotenv()

# Configuration du logger global
LOG_LEVEL_ENV = os.getenv("LOG_LEVEL_SetupDatabase", "INFO")  # Valeur par défaut: INFO

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
    # Créer le répertoire de logs s'il n'existe pas
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # Nettoyer le chemin du fichier de log
    if log_file:
        log_file = log_file.strip('"')  # Supprimer les guillemets
        log_file = os.path.join(log_dir, os.path.basename(log_file))

    logger = logging.getLogger("database_logger")
    logger.setLevel(LOG_LEVEL_MAP.get(LOG_LEVEL_ENV, logging.INFO))

    # Format des logs
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

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


logger = setup_logger()


def remove_database_file():
    """
    Removes the database file if it exists.
    Retrieves the database file path from environment variables and deletes it if found.
    """
    database_path = os.getenv("DB_FILE")

    if os.path.exists(database_path):
        os.remove(database_path)
        logger.info(
            f"🗑️ Successfully deleted database file: '{database_path}'"
        )  # INFO: File deletion success
    else:
        logger.warning(
            f"⚠️ Database file '{database_path}' does not exist."
        )  # WARNING: File not found


def clean_column_name(column_name):
    """
    Cleans a column name to ensure compatibility with DuckDB.
    Handles empty names, removes accents, replaces special characters, and standardizes format.
    """
    if pd.isna(column_name) or column_name.strip() == "":
        return "unnamed_column"  # Return default name for empty columns

    column_name = unidecode(column_name)  # Remove accents
    column_name = column_name.replace("'", "_")  # Replace apostrophes with underscores
    column_name = re.sub(
        r"[^a-zA-Z0-9_]", "_", column_name
    ).lower()  # Standardize format

    # Remove any UUID patterns from column names
    column_name = re.sub(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
        "",
        column_name,
    )

    column_name = re.sub(r"_+", "_", column_name).strip("_")  # Remove extra underscores

    # Prefix column name if it starts with a digit
    if column_name and column_name[0].isdigit():
        column_name = "col_" + column_name

    logger.info(
        f"🛠️ Cleaned column name: {column_name}"
    )  # INFO: Log cleaned column name
    return column_name


def prepare_database(filepaths=None, collection_id=None):
    """
    Processes a list of files, extracts dataframes, and creates tables in DuckDB.
    Ensures that identical tables are not recreated.
    """
    if not filepaths:
        return

    # Collect all file paths, including those inside directories
    all_filepaths = []
    for path in filepaths:
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for file in files:
                    all_filepaths.append(os.path.join(root, file))
        else:
            all_filepaths.append(path)

    # Determine the database path
    if collection_id is not None:
        path = os.getenv("DB_FILE").replace("id", str(collection_id))
        conn = duckdb.connect(path)
    else:
        conn = duckdb.connect(os.getenv("DB_FILE"))

    logger.info(
        f"📂 Files to process: {all_filepaths}"
    )  # INFO: Logging files being processed

    for filepath in all_filepaths:
        logger.info(
            f"📄 Processing file: {filepath}"
        )  # INFO: Log each file being processed
        try:
            data = load_file_data(filepath)  # Load file data into dataframes
            for sheet_name, df in data.items():
                table_name = generate_table_name(filepath, sheet_name)
                logger.debug(
                    f"🔧 Checking or creating table '{table_name}' for sheet '{sheet_name}'..."  # DEBUG: Table creation process
                )
                create_table_from_dataframe(
                    conn, df, table_name
                )  # Create table from dataframe
                handle_nested_data(
                    conn, df, table_name
                )  # Handle nested data structures
        except ValueError as e:
            logger.error(
                f"❌ Format error in file '{filepath}': {e}"
            )  # ERROR: File format issue
        except Exception as e:
            logger.error(
                f"🚨 Unexpected error while processing file '{filepath}': {e}"  # ERROR: General processing failure
            )

    conn.close()


def convert_to_lowercase(df):
    """
    Convertit les valeurs de type chaîne en minuscules dans un DataFrame.
    """
    return df.map(lambda x: x.lower() if isinstance(x, str) else x)


def detect_delimiter(filepath):
    """
    Détecte automatiquement le délimiteur du fichier CSV en analysant les premières lignes.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        sample = f.read(2048)
        if not sample.strip():
            raise ValueError("CSV file is empty or unreadable.")
        sniffer = csv.Sniffer()
        try:
            delimiter = sniffer.sniff(sample).delimiter
            return delimiter
        except csv.Error:
            return ","  # Par défaut, on suppose une virgule si la détection échoue


def infer_column_types(df, date_threshold=0.8, numeric_threshold=0.8):
    for col in df.columns:
        sample_values = df[col].dropna()

        if sample_values.empty:
            continue

        str_vals = sample_values.astype(str)

        # Try date inference
        parsed_dates = pd.to_datetime(str_vals, errors="coerce")
        valid_dates_ratio = parsed_dates.notna().mean()
        if valid_dates_ratio >= date_threshold:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            continue

        # Try numeric inference
        numeric_series = pd.to_numeric(str_vals.str.replace(",", "."), errors="coerce")
        valid_numeric_ratio = numeric_series.notna().mean()
        if valid_numeric_ratio >= numeric_threshold:
            # Check if all numeric values are integer-like
            # --- Fix here: use a lambda instead of float.is_integer ---
            is_all_integers = (
                numeric_series.dropna().apply(lambda x: x.is_integer()).all()
            )
            if is_all_integers:
                df[col] = pd.to_numeric(
                    df[col].str.replace(",", "."), errors="coerce"
                ).astype("Int64")
            else:
                df[col] = pd.to_numeric(
                    df[col].str.replace(",", "."), errors="coerce", downcast="float"
                )
        else:
            # Otherwise keep as string
            pass

    return df


def load_file_data(filepath):
    """
    Loads data based on the file extension.
    Supports Excel, CSV, JSON, PDF, and Python files.
    """
    logger.info(
        f"📂 Loading data from file: {filepath}"
    )  # INFO: Log file loading attempt

    if filepath.endswith(".xls"):
        try:
            data = pd.read_excel(filepath, sheet_name=None, engine="xlrd")  # Load .xls file
            return {sheet: convert_to_lowercase(df) for sheet, df in data.items()}
        except Exception as e:
            logger.warning(f"⚠️ Arrow compatibility issue with .xls file, trying alternative method: {e}")
            # Fallback method without arrow
            data = pd.read_excel(filepath, sheet_name=None, engine="xlrd", dtype=str)
            return {sheet: convert_to_lowercase(df) for sheet, df in data.items()}
    elif filepath.endswith(".xlsx") or filepath.endswith(".xlsm"):
        try:
            data = pd.read_excel(
                filepath, sheet_name=None, engine="openpyxl"
            )  # Load .xlsx or .xlsm
            return {sheet: convert_to_lowercase(df) for sheet, df in data.items()}
        except Exception as e:
            logger.warning(f"⚠️ Arrow compatibility issue with .xlsx/.xlsm file, trying alternative method: {e}")
            # Fallback method without arrow
            data = pd.read_excel(filepath, sheet_name=None, engine="openpyxl", dtype=str)
            return {sheet: convert_to_lowercase(df) for sheet, df in data.items()}
    elif filepath.endswith(".csv"):
        delimiter = detect_delimiter(filepath)
        logger.info(f"🔍 Detected delimiter: '{delimiter}' for {filepath}")
        # Read all columns as string initially for robust inference
        data = pd.read_csv(
            filepath,
            sep=delimiter,
            encoding="utf-8",
            engine="python",
            dtype=str,
        )
        # Normalize column names (strip spaces, lower)
        data.columns = [col.strip().replace(" ", "_").lower() for col in data.columns]
        # Convert to lowercase and then attempt type inference
        data = convert_to_lowercase(data)
        data = infer_column_types(data)
        return {"data": data}  # Single key for CSV
    elif filepath.endswith(".json"):
        with open(filepath, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        return {
            "main": pd.json_normalize(json_data, sep="_")
        }  # Load JSON data as a DataFrame
    elif filepath.endswith(".pdf"):
        return process_pdf_file(filepath)  # Process PDF file
    elif filepath.endswith(".py"):
        return process_python_file(filepath)  # Process Python script
    elif filepath.endswith(".md"):
        return process_text_file(filepath)  # Process markdown file
    elif filepath.endswith((".doc", ".docx")):
        return process_text_file(filepath)  # Process Word document
    else:
        logger.error(
            f"❌ Unsupported file format: {filepath}"
        )  # ERROR: Unsupported file type
        raise ValueError("Unsupported file format.")


def process_pdf_file(filepath):
    """
    Processes a PDF file by extracting text and images.
    Converts extracted data into DataFrames for further analysis.
    """
    logger.info(f"📄 Processing PDF file: {filepath}")  # INFO: Start processing PDF

    # Extract text and images from the PDF
    extracted_text, images_data = extract_pdf(filepath)
    data = {}

    # Store extracted text in a DataFrame if available
    if extracted_text:
        data["text"] = pd.DataFrame([{"content": t} for t in extracted_text])
        logger.info(
            f"📝 Extracted text from PDF: {len(extracted_text)} entries"
        )  # INFO: Text extraction summary

    # Store extracted images with OCR results in a DataFrame if available
    if images_data:
        data["images"] = pd.DataFrame(images_data)
        logger.info(
            f"🖼️ Extracted images from PDF: {len(images_data)} images"
        )  # INFO: Image extraction summary

    return data


def process_python_file(filepath):
    """
    Processes a Python (.py) file by extracting functions, classes, imports, and module code.
    Converts extracted data into DataFrames for structured analysis.
    """
    logger.info(
        f"🐍 Processing Python file: {filepath}"
    )  # INFO: Start processing Python file

    # Extract structured data from the Python script
    extracted_data = extract_python(filepath)
    data = {}

    # Store extracted functions in a DataFrame if available
    if "functions" in extracted_data:
        data["functions"] = pd.DataFrame(extracted_data["functions"])
        logger.info(
            f"🛠️ Extracted functions: {len(extracted_data['functions'])}"
        )  # INFO: Function extraction summary

    # Store extracted classes in a DataFrame if available
    if "classes" in extracted_data:
        data["classes"] = pd.DataFrame(extracted_data["classes"])
        logger.info(
            f"📦 Extracted classes: {len(extracted_data['classes'])}"
        )  # INFO: Class extraction summary

    # Store extracted imports in a DataFrame if available
    if "imports" in extracted_data:
        data["imports"] = pd.DataFrame(extracted_data["imports"])
        logger.info(
            f"📦 Extracted imports: {len(extracted_data['imports'])}"
        )  # INFO: Imports extraction summary

    # Store full module code in a DataFrame if available
    if "module_code" in extracted_data:
        data["module_code"] = pd.DataFrame(
            [{"module_code": extracted_data["module_code"]}]
        )
        logger.info("📜 Extracted full module code.")  # INFO: Module code extraction

    return data


def process_text_file(filepath):
    """
    Processes text-based files (.md, .doc, .docx) by extracting their content.
    Converts extracted data into DataFrames for further analysis.
    """
    logger.info(
        f"📄 Processing text file: {filepath}"
    )  # INFO: Start processing text file

    data = {}

    try:
        if filepath.endswith(".md"):
            # For markdown files, we'll extract both raw and rendered content
            with open(filepath, "r", encoding="utf-8") as f:
                raw_content = f.read()
                html_content = markdown.markdown(raw_content)
                data["raw_content"] = pd.DataFrame([{"content": raw_content}])
                data["html_content"] = pd.DataFrame([{"content": html_content}])
        elif filepath.endswith(".docx"):
            # For Word documents (.docx), extract text using python-docx
            doc = Document(filepath)
            text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
            data["content"] = pd.DataFrame([{"content": text}])
        elif filepath.endswith(".doc"):
            # For legacy Word documents (.doc), log warning as we can't process them without textract
            logger.warning(f"⚠️ Legacy .doc files are not supported. Please convert '{filepath}' to .docx format.")
            data["content"] = pd.DataFrame([{"content": f"Legacy .doc file: {os.path.basename(filepath)} - conversion to .docx required"}])

        logger.info(f"📝 Extracted content from file: {filepath}")
        return data
    except Exception as e:
        logger.error(f"❌ Error processing text file '{filepath}': {e}")
        raise ValueError(f"Error processing text file: {str(e)}")


def generate_table_name(filepath, sheet_name):
    """
    Generates a table name WITHOUT UUID, even if the file is named {uuid}_something.pdf.
    Ensures the name is compatible with DuckDB by removing special characters.
    """
    base = os.path.splitext(os.path.basename(filepath))[
        0
    ]  # Extract file name without extension

    # Remove potential UUID at the beginning (e.g., "8a2da82f-1cd5-42b6-91bb-d0c2282f57d0_something")
    base = re.sub(UUID_REGEX, "", base)  # Remove UUID and underscore
    base = (
        re.sub(r"[^A-Za-z0-9_]+", "_", base).strip("_").lower()
    )  # Normalize base name

    # Normalize sheet name
    sheet = re.sub(r"[^A-Za-z0-9_]+", "_", sheet_name).strip("_").lower()

    # Combine file name and sheet name
    table_name = f"{base}_{sheet}"

    # Ensure the table name does not start with a digit
    if table_name and table_name[0].isdigit():
        table_name = "t_" + table_name

    logger.debug(
        f"🛠️ Final table name generated: {table_name}"
    )  # DEBUG: Log final table name
    return table_name


def table_exists_with_same_schema(conn, df, table_name):
    """
    Checks if the table `table_name` already exists in the database
    AND if its schema matches exactly with the columns of `df`.
    If there is a perfect match, the table is considered a duplicate.
    """
    logger.debug(
        f"🔍 Checking if table '{table_name}' exists with the same schema."
    )  # DEBUG: Start schema check

    # Retrieve the list of existing tables
    tables = [x[0] for x in conn.execute("SHOW TABLES").fetchall()]
    if table_name not in tables:
        logger.info(
            f"✅ Table '{table_name}' does not exist. It can be created."
        )  # INFO: Table does not exist
        return False

    # Retrieve schema of existing table
    schema_info = conn.execute(f"DESCRIBE {table_name}").fetchall()

    # Extract column names and types from the existing table
    existing_cols = [(row[0].lower(), row[1].lower()) for row in schema_info]

    # Build the expected column structure based on the DataFrame
    desired_cols = []
    for col in df.columns:
        colname_clean = clean_column_name(col)  # Clean column name
        coltype = map_dtype_to_duckdb_type(
            df[col].dtype, df[col]
        )  # Convert to DuckDB type
        desired_cols.append((colname_clean.lower(), coltype.lower()))

    # Compare schema: Same number of columns, same names, same types, in the same order
    if len(existing_cols) != len(desired_cols):
        logger.info(
            f"⚠️ Schema mismatch: Table '{table_name}' exists but has a different structure."
        )  # INFO: Schema mismatch
        return False

    if existing_cols == desired_cols:
        logger.info(
            f"🔄 Table '{table_name}' already exists with the same schema. Skipping creation."
        )  # INFO: Table already exists with the same schema
        return True

    return False


def create_table_from_dataframe(conn, df, table_name):
    """
    Creates a table in DuckDB from a DataFrame, ensuring schema compatibility.
    If the table exists but with a different schema, it is dropped and recreated.
    """
    logger.info(
        f"🛠️ Attempting to create table '{table_name}' in DuckDB..."
    )  # INFO: Table creation start

    # Check if the table already exists with the same schema
    if table_exists_with_same_schema(conn, df, table_name):
        logger.info(
            f"✅ Table '{table_name}' already exists with the same schema. Skipping creation."
        )  # INFO: Table exists
        return

    # Drop the table if it exists with a different schema
    tables = [x[0] for x in conn.execute("SHOW TABLES").fetchall()]
    if table_name in tables:
        logger.info(
            f"⚠️ Table '{table_name}' exists but has a different schema. Dropping before recreation."
        )  # INFO: Schema mismatch
        conn.execute(f"DROP TABLE {table_name}")

    # Construct the CREATE TABLE query
    column_definitions = []
    for column_name, dtype in df.dtypes.items():
        col_clean = clean_column_name(column_name)  # Clean column name
        column_type = map_dtype_to_duckdb_type(
            dtype, df[column_name]
        )  # Convert to DuckDB type
        column_definitions.append(f'"{col_clean}" {column_type}')

    column_definitions_str = ", ".join(column_definitions)
    create_table_query = f"CREATE TABLE {table_name} ({column_definitions_str})"

    # Execute table creation query
    try:
        conn.execute(create_table_query)
        logger.info(
            f"🎉 Table '{table_name}' successfully created."
        )  # INFO: Table created successfully
    except Exception as e:
        logger.error(
            f"❌ SQL error while creating table '{table_name}': {e}"
        )  # ERROR: SQL execution failed
        return

    # Insert data into the newly created table
    try:
        if not df.empty:
            # Convert DataFrame to native Python types completely
            df_clean = df.copy()
            
            # Ensure all data is in native Python types
            for col in df_clean.columns:
                if df_clean[col].dtype == 'object':
                    df_clean[col] = df_clean[col].astype(str).fillna('')
                elif 'int' in str(df_clean[col].dtype).lower():
                    df_clean[col] = df_clean[col].astype('int64').fillna(0)
                elif 'float' in str(df_clean[col].dtype).lower():
                    df_clean[col] = df_clean[col].astype('float64').fillna(0.0)
                elif 'bool' in str(df_clean[col].dtype).lower():
                    df_clean[col] = df_clean[col].astype(bool).fillna(False)
                else:
                    df_clean[col] = df_clean[col].astype(str).fillna('')
            
            # Get clean column names
            clean_columns = [clean_column_name(col) for col in df_clean.columns]
            columns_str = ', '.join([f'"{col}"' for col in clean_columns])
            
            # Insert data row by row using direct SQL to avoid arrow issues
            batch_size = 1000
            for i in range(0, len(df_clean), batch_size):
                batch = df_clean.iloc[i:i + batch_size]
                
                # Build VALUES clauses for batch insert
                values_list = []
                for _, row in batch.iterrows():
                    row_values = []
                    for col in df_clean.columns:
                        value = row[col]
                        if pd.isna(value) or value is None:
                            row_values.append('NULL')
                        elif isinstance(value, str):
                            # Escape single quotes in strings
                            escaped_value = value.replace("'", "''")
                            row_values.append(f"'{escaped_value}'")
                        elif isinstance(value, (int, float)):
                            row_values.append(str(value))
                        elif isinstance(value, bool):
                            row_values.append('TRUE' if value else 'FALSE')
                        else:
                            # Convert to string as fallback
                            escaped_value = str(value).replace("'", "''")
                            row_values.append(f"'{escaped_value}'")
                    
                    values_list.append(f"({', '.join(row_values)})")
                
                if values_list:
                    values_str = ', '.join(values_list)
                    insert_query = f"INSERT INTO {table_name} ({columns_str}) VALUES {values_str}"
                    conn.execute(insert_query)
                
        logger.info(
            f"📊 Data successfully inserted into '{table_name}'."
        )  # INFO: Data insertion success
    except Exception as e:
        logger.error(
            f"❌ SQL error during data insertion into '{table_name}': {e}"
        )  # ERROR: Data insertion failed

    logger.info(
        f"✅ Table '{table_name}' operation completed."
    )  # INFO: Operation finished


def handle_nested_data(conn, df, base_table_name):
    """
    Handles nested data columns (lists or dictionaries) by creating separate tables.
    """
    logger.info(
        f"🔍 Handling nested data for table '{base_table_name}'..."
    )  # INFO: Start processing

    # Remove any potential UUID from the table name
    base_table_name = re.sub(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
        "",
        base_table_name,
    ).strip("_")

    for column in df.columns:
        # Check if the column contains nested data (list or dict)
        if df[column].apply(lambda x: isinstance(x, (list, dict))).any():
            logger.debug(
                f"📌 Nested column detected: {column}"
            )  # DEBUG: Found nested data

            nested_entries = []
            for _, row in df.iterrows():
                value = row[column]
                if isinstance(value, list):
                    nested_entries.extend(value)  # Flatten list entries
                elif isinstance(value, dict):
                    nested_entries.append(value)  # Store dictionary entries

            if nested_entries:
                nested_df = pd.json_normalize(
                    nested_entries, sep="_"
                )  # Convert to DataFrame
                nested_table_name = f"{base_table_name}_{clean_column_name(column)}"
                logger.info(
                    f"🆕 Creating nested table '{nested_table_name}'..."
                )  # INFO: Creating a new table
                create_table_from_dataframe(
                    conn, nested_df, nested_table_name
                )  # Create the nested table


def map_dtype_to_duckdb_type(dtype, column_data):
    """
    Maps a Pandas dtype to the appropriate DuckDB data type,
    including Pandas extension dtypes (Int64, Float32, etc.).
    """
    logger.info(f"🔄 Mapping dtype '{dtype}' to DuckDB type.")

    # Convert the dtype object to a string, e.g. "Int64", "float32", "object"
    dtype_str = str(dtype).lower()

    # 1) Check for common extension dtypes or built-in names
    #    (This captures "int64", "int32", "float32", etc.)
    if "int" in dtype_str:
        mapped_type = "INTEGER"  # Or "BIGINT" if you prefer
        logger.debug(
            f"✅ Mapped extension dtype '{dtype}' to DuckDB type '{mapped_type}'."
        )
        return mapped_type
    if "float" in dtype_str:
        mapped_type = "DOUBLE"
        logger.debug(
            f"✅ Mapped extension dtype '{dtype}' to DuckDB type '{mapped_type}'."
        )
        return mapped_type
    if "bool" in dtype_str:
        mapped_type = "BOOLEAN"
        logger.debug(
            f"✅ Mapped extension dtype '{dtype}' to DuckDB type '{mapped_type}'."
        )
        return mapped_type
    if "datetime" in dtype_str:
        mapped_type = "TIMESTAMP"
        logger.debug(
            f"✅ Mapped extension dtype '{dtype}' to DuckDB type '{mapped_type}'."
        )
        return mapped_type

    # 2) If not matched above, fall back to the "base_type" approach
    #    (handles raw NumPy dtypes like np.integer, np.floating, np.datetime64, etc.)
    base_type = getattr(dtype, "type", None)

    # Create a mapping dictionary for base NumPy types
    type_mapping = {
        np.integer: "INTEGER",
        np.floating: "DOUBLE",
        np.bool_: "BOOLEAN",
        np.datetime64: "TIMESTAMP",
        # For object columns, we call the lambda that checks if all float, all int, etc.
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

    # 3) Handle the mapping or default to "TEXT"
    if base_type in type_mapping:
        result = type_mapping[base_type]
        # If it's a callable (like the `object` case), call it
        mapped_type = result() if callable(result) else result
        logger.debug(f"✅ Mapped '{dtype}' to DuckDB type '{mapped_type}'.")
        return mapped_type

    # 4) Fallback to TEXT if nothing matched
    logger.debug(f"✅ Fallback: mapped '{dtype}' to TEXT.")
    return "TEXT"


def is_float(value):
    """
    Checks if a given value can be converted to a float.
    """
    try:
        float(value)  # Attempt to convert to float
        return True
    except (ValueError, TypeError):  # Catch conversion errors
        logger.debug(f"❌ Value '{value}' is not a float.")  # DEBUG: Failed conversion
        return False


def is_integer(value):
    """
    Checks if a given value can be converted to an integer without losing precision.
    """
    try:
        result = float(
            value
        ).is_integer()  # Convert to float and check if it's an integer
        return result
    except (ValueError, TypeError):  # Catch conversion errors
        logger.debug(
            f"❌ Value '{value}' is not an integer."
        )  # DEBUG: Failed conversion
        return False


def is_date(value):
    """
    Checks if a given value can be converted to a valid date using Pandas.
    """
    try:
        pd.to_datetime(value)  # Attempt to convert the value to a datetime format
        return True
    except (ValueError, TypeError):  # Catch conversion errors
        logger.debug(
            f"❌ Value '{value}' is not a valid date."
        )  # DEBUG: Failed conversion
        return False
