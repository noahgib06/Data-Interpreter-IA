# import matplotlib.pyplot as plt  # plt.show(block=False) plt.savefig(path)
import ast
import logging
import os
from logging.handlers import RotatingFileHandler

from dotenv import load_dotenv

load_dotenv()

# Configuration du logger global
LOG_LEVEL_ENV = os.getenv(
    "LOG_LEVEL_PythonExtension"
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
    log_file=os.getenv("LOG_FILE_PythonExtension"),
    max_size=5 * 1024 * 1024,
    backup_count=3,
):
    """
    Configure un logger global pour suivre toutes les actions.
    """
    if not os.path.exists("../Logs"):
        os.makedirs("../Logs", exist_ok=True)
    logger = logging.getLogger("python_extraction_logger")
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


def extract_python(filepath):
    """Extrait des informations complètes d'un fichier Python (.py) en utilisant le module ast et en lisant le fichier directement."""
    logger.info(f"Début de l'extraction pour le fichier Python : {filepath}")

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            file_content = f.read()
        logger.info(f"Fichier Python chargé avec succès : {filepath}")
    except Exception as e:
        logger.error(f"Erreur lors de la lecture du fichier Python '{filepath}' : {e}")
        return None

    try:
        # Utiliser ast pour parser le fichier Python
        parsed_code = ast.parse(file_content)
        logger.info("Code Python parsé avec succès.")
    except SyntaxError as e:
        logger.error(f"Erreur de syntaxe dans le fichier Python '{filepath}' : {e}")
        return None

    extracted_data = {
        "module_code": file_content,
        "functions": [],
        "classes": [],
        "imports": [],
        "docstrings": ast.get_docstring(parsed_code),
    }

    if extracted_data["docstrings"]:
        logger.info("Docstring du module extrait.")
    else:
        logger.info("Aucun docstring de module trouvé.")

    # Parcourir l'arbre de syntaxe abstraite (AST)
    for node in ast.walk(parsed_code):
        if isinstance(node, ast.FunctionDef):
            try:
                function_info = {
                    "name": node.name,
                    "arguments": [arg.arg for arg in node.args.args],
                    "docstring": ast.get_docstring(node),
                    "line_number": node.lineno,
                    "content": ast.get_source_segment(
                        file_content, node
                    ),  # Récupérer le contenu exact de la fonction
                }
                extracted_data["functions"].append(function_info)
                logger.info(
                    f"Fonction extraite : {function_info['name']} (ligne {function_info['line_number']})."
                )
            except Exception as e:
                logger.error(f"Erreur lors de l'extraction d'une fonction : {e}")

        elif isinstance(node, ast.ClassDef):
            try:
                class_info = {
                    "name": node.name,
                    "methods": [],
                    "docstring": ast.get_docstring(node),
                    "line_number": node.lineno,
                    "content": ast.get_source_segment(
                        file_content, node
                    ),  # Récupérer le contenu exact de la classe
                }
                for class_node in node.body:
                    if isinstance(class_node, ast.FunctionDef):
                        method_info = {
                            "name": class_node.name,
                            "arguments": [arg.arg for arg in class_node.args.args],
                            "docstring": ast.get_docstring(class_node),
                            "line_number": class_node.lineno,
                            "content": ast.get_source_segment(
                                file_content, class_node
                            ),  # Récupérer le contenu exact de la méthode
                        }
                        class_info["methods"].append(method_info)
                        logger.info(
                            f"Méthode extraite : {method_info['name']} (ligne {method_info['line_number']})."
                        )
                extracted_data["classes"].append(class_info)
                logger.info(
                    f"Classe extraite : {class_info['name']} (ligne {class_info['line_number']})."
                )
            except Exception as e:
                logger.error(f"Erreur lors de l'extraction d'une classe : {e}")

        elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
            try:
                import_info = {
                    "module": getattr(node, "module", None),
                    "names": [alias.name for alias in node.names],
                    "line_number": node.lineno,
                }
                extracted_data["imports"].append(import_info)
                logger.info(
                    f"Import extrait : {import_info['names']} (ligne {import_info['line_number']})."
                )
            except Exception as e:
                logger.error(f"Erreur lors de l'extraction d'un import : {e}")

    logger.info(f"Extraction terminée pour le fichier Python : {filepath}")
    return extracted_data
