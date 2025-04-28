# import matplotlib.pyplot as plt  # plt.show(block=False) plt.savefig(path)
import ast
import logging
import os
from logging.handlers import RotatingFileHandler

from dotenv import load_dotenv

load_dotenv()

# Configuration du logger global
LOG_LEVEL_ENV = os.getenv(
    "LOG_LEVEL_PythonExtension", "INFO"
)  # Valeur par défaut: INFO

# Mappage des niveaux de log
LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def setup_logger(
    log_file=os.getenv("LOG_FILE_PythonExtension"),
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

    logger = logging.getLogger("python_extraction_logger")
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


def extract_python(filepath):
    """
    Extracts detailed information from a Python (.py) file using the `ast` module.
    Collects module-level docstrings, functions, classes, and import statements.
    """
    logger.info(
        f"📂 Starting extraction for Python file: {filepath}"
    )  # INFO: File processing start

    try:
        # Read the Python file content
        with open(filepath, "r", encoding="utf-8") as f:
            file_content = f.read()
        logger.info(
            f"✅ Successfully loaded Python file: {filepath}"
        )  # INFO: File read successfully
    except Exception as e:
        logger.error(
            f"❌ Error reading Python file '{filepath}': {e}"
        )  # ERROR: File read failure
        return None

    try:
        # Parse the Python code using the `ast` module
        parsed_code = ast.parse(file_content)
        logger.info("✅ Successfully parsed Python code.")  # INFO: AST parsing success
    except SyntaxError as e:
        logger.error(
            f"❌ Syntax error in Python file '{filepath}': {e}"
        )  # ERROR: Syntax issue detected
        return None

    # Initialize extracted data structure
    extracted_data = {
        "module_code": file_content,
        "functions": [],
        "classes": [],
        "imports": [],
        "docstrings": ast.get_docstring(parsed_code),
    }

    # Log module-level docstring if present
    if extracted_data["docstrings"]:
        logger.info("📝 Extracted module docstring.")  # INFO: Log module docstring
    else:
        logger.info(
            "⚠️ No module-level docstring found."
        )  # INFO: No docstring available

    # Traverse the abstract syntax tree (AST)
    for node in ast.walk(parsed_code):
        if isinstance(node, ast.FunctionDef):
            try:
                # Extract function details
                function_info = {
                    "name": node.name,
                    "arguments": [arg.arg for arg in node.args.args],
                    "docstring": ast.get_docstring(node),
                    "line_number": node.lineno,
                    "content": ast.get_source_segment(file_content, node),
                }
                extracted_data["functions"].append(function_info)
                logger.info(
                    f"🔧 Extracted function: {function_info['name']} (line {function_info['line_number']})."
                )  # INFO: Log function extraction
            except Exception as e:
                logger.error(
                    f"❌ Error extracting function: {e}"
                )  # ERROR: Function extraction failure

        elif isinstance(node, ast.ClassDef):
            try:
                # Extract class details
                class_info = {
                    "name": node.name,
                    "methods": [],
                    "docstring": ast.get_docstring(node),
                    "line_number": node.lineno,
                    "content": ast.get_source_segment(file_content, node),
                }
                # Extract methods inside the class
                for class_node in node.body:
                    if isinstance(class_node, ast.FunctionDef):
                        method_info = {
                            "name": class_node.name,
                            "arguments": [arg.arg for arg in class_node.args.args],
                            "docstring": ast.get_docstring(class_node),
                            "line_number": class_node.lineno,
                            "content": ast.get_source_segment(file_content, class_node),
                        }
                        class_info["methods"].append(method_info)
                        logger.info(
                            f"🔹 Extracted method: {method_info['name']} (line {method_info['line_number']})."
                        )  # INFO: Log method extraction

                extracted_data["classes"].append(class_info)
                logger.info(
                    f"🏛️ Extracted class: {class_info['name']} (line {class_info['line_number']})."
                )  # INFO: Log class extraction
            except Exception as e:
                logger.error(
                    f"❌ Error extracting class: {e}"
                )  # ERROR: Class extraction failure

        elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
            try:
                # Extract import statements
                import_info = {
                    "module": getattr(node, "module", None),
                    "names": [alias.name for alias in node.names],
                    "line_number": node.lineno,
                }
                extracted_data["imports"].append(import_info)
                logger.info(
                    f"📦 Extracted import: {import_info['names']} (line {import_info['line_number']})."
                )  # INFO: Log import extraction
            except Exception as e:
                logger.error(
                    f"❌ Error extracting import: {e}"
                )  # ERROR: Import extraction failure

    logger.info(
        f"✅ Extraction completed for Python file: {filepath}"
    )  # INFO: Extraction finished
    return extracted_data
