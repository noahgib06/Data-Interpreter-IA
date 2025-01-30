import io
import logging
import os
from logging.handlers import RotatingFileHandler

import fitz
import pytesseract
from dotenv import load_dotenv
from PIL import Image, ImageEnhance, ImageFilter

load_dotenv()

# Configuration du logger global
LOG_LEVEL_ENV = os.getenv("LOG_LEVEL_PdfExtension")

# Mappage des niveaux de log
LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def setup_logger(
    log_file=os.getenv("LOG_FILE_PdfExtension"),
    max_size=5 * 1024 * 1024,
    backup_count=3,
):
    """
    Configure un logger global pour suivre toutes les actions.
    """
    if not os.path.exists("../Logs"):
        os.makedirs("../Logs", exist_ok=True)
    logger = logging.getLogger("pdf_extraction_logger")
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


def extract_pdf(filepath):
    """Extrait le texte et les images d'un PDF, y compris le texte OCR des images."""
    logger.info(f"Début de l'extraction pour le fichier PDF : {filepath}")

    # Extraction du texte et des images du PDF
    extracted_text = []
    images_data = []

    try:
        # Ouvrir le document PDF avec PyMuPDF (fitz)
        pdf_document = fitz.open(filepath)
        logger.info(f"Nombre de pages dans le PDF '{filepath}' : {len(pdf_document)}")
    except Exception as e:
        logger.error(f"Erreur lors de l'ouverture du PDF '{filepath}' : {e}")
        return extracted_text, images_data

    for page_num in range(len(pdf_document)):
        try:
            # Extraction du texte de la page avec PyMuPDF
            page = pdf_document.load_page(page_num)
            pdf_text = page.get_text()
            if pdf_text.strip():
                logger.info(
                    f"Texte extrait de la page {page_num} : {pdf_text[:200]}..."
                )  # Premier 200 caractères
                extracted_text.append({"page": page_num, "content": pdf_text})
            else:
                logger.info(
                    f"Aucun texte extrait de la page {page_num} du PDF '{filepath}'."
                )
        except Exception as e:
            logger.error(
                f"Erreur lors de l'extraction du texte de la page {page_num} : {e}"
            )
            continue

        # Extraction des images de la page
        try:
            images = page.get_images(full=True)
            logger.info(
                f"Nombre d'images trouvées sur la page {page_num} : {len(images)}"
            )

            for img_index, img in enumerate(images):
                try:
                    xref = img[0]
                    base_image = pdf_document.extract_image(xref)
                    image_bytes = base_image["image"]
                    image = Image.open(io.BytesIO(image_bytes))

                    # Prétraitement de l'image pour l'OCR
                    image = image.convert("L")
                    image = ImageEnhance.Contrast(image).enhance(2)
                    image = image.filter(ImageFilter.SHARPEN)
                    image = image.resize((image.width * 2, image.height * 2))

                    # Utilisation d'OCR pour extraire le texte des images
                    ocr_text = pytesseract.image_to_string(
                        image, config="--psm 6", lang="eng"
                    )
                    if ocr_text.strip():
                        logger.info(
                            f"Texte OCR extrait de l'image {img_index} sur la page {page_num} : {ocr_text[:200]}..."
                        )
                    else:
                        logger.info(
                            f"Aucun texte OCR extrait de l'image {img_index} sur la page {page_num}."
                        )

                    images_data.append(
                        {
                            "page": page_num,
                            "image_index": img_index,
                            "ocr_text": ocr_text,
                        }
                    )
                except Exception as e:
                    logger.error(
                        f"Erreur lors du traitement de l'image {img_index} sur la page {page_num} : {e}"
                    )
                    continue
        except Exception as e:
            logger.error(
                f"Erreur lors de l'extraction des images de la page {page_num} : {e}"
            )
            continue

    logger.info(f"Extraction terminée pour le fichier PDF : {filepath}")
    return extracted_text, images_data
