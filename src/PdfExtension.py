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
LOG_LEVEL_ENV = os.getenv("LOG_LEVEL_PdfExtension", "INFO")  # Valeur par défaut: INFO

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
    # Créer le répertoire de logs s'il n'existe pas
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # Nettoyer le chemin du fichier de log
    if log_file:
        log_file = log_file.strip('"')  # Supprimer les guillemets
        log_file = os.path.join(log_dir, os.path.basename(log_file))

    logger = logging.getLogger("pdf_extraction_logger")
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


def extract_pdf(filepath):
    """
    Extracts text and images from a PDF, including OCR-processed text from images.
    Uses PyMuPDF (fitz) for text extraction and pytesseract for OCR on images.
    """
    logger.info(
        f"📂 Starting extraction for PDF file: {filepath}"
    )  # INFO: File processing start

    # Initialize storage for extracted text and images
    extracted_text = []
    images_data = []

    try:
        # Open the PDF document with PyMuPDF (fitz)
        pdf_document = fitz.open(filepath)
        logger.info(
            f"📄 Number of pages in PDF '{filepath}': {len(pdf_document)}"
        )  # INFO: Log page count
    except Exception as e:
        logger.error(
            f"❌ Error opening PDF '{filepath}': {e}"
        )  # ERROR: PDF opening failure
        return extracted_text, images_data

    # Iterate through each page of the PDF
    for page_num in range(len(pdf_document)):
        try:
            # Extract text from the current page
            page = pdf_document.load_page(page_num)
            pdf_text = page.get_text()
            if pdf_text.strip():
                logger.info(
                    f"📝 Extracted text from page {page_num}: {pdf_text[:200]}..."
                )  # INFO: Log first 200 characters
                extracted_text.append({"page": page_num, "content": pdf_text})
            else:
                logger.info(
                    f"⚠️ No text extracted from page {page_num} in '{filepath}'."
                )  # INFO: No text found
        except Exception as e:
            logger.error(
                f"❌ Error extracting text from page {page_num}: {e}"
            )  # ERROR: Text extraction failure
            continue

        # Extract images from the current page
        try:
            images = page.get_images(full=True)
            logger.info(
                f"🖼️ Number of images found on page {page_num}: {len(images)}"
            )  # INFO: Log image count

            for img_index, img in enumerate(images):
                try:
                    # Extract and preprocess the image for OCR
                    xref = img[0]
                    base_image = pdf_document.extract_image(xref)
                    image_bytes = base_image["image"]
                    image = Image.open(io.BytesIO(image_bytes))

                    # Enhance image for better OCR results
                    image = image.convert("L")
                    image = ImageEnhance.Contrast(image).enhance(2)
                    image = image.filter(ImageFilter.SHARPEN)
                    image = image.resize((image.width * 2, image.height * 2))

                    # Perform OCR on the image
                    ocr_text = pytesseract.image_to_string(
                        image, config="--psm 6", lang="eng"
                    )
                    if ocr_text.strip():
                        logger.info(
                            f"🔠 OCR extracted text from image {img_index} on page {page_num}: {ocr_text[:200]}..."
                        )  # INFO: Log first 200 characters
                    else:
                        logger.info(
                            f"⚠️ No OCR text extracted from image {img_index} on page {page_num}."
                        )  # INFO: No OCR text found

                    images_data.append(
                        {
                            "page": page_num,
                            "image_index": img_index,
                            "ocr_text": ocr_text,
                        }
                    )
                except Exception as e:
                    logger.error(
                        f"❌ Error processing image {img_index} on page {page_num}: {e}"
                    )  # ERROR: Image processing failure
                    continue
        except Exception as e:
            logger.error(
                f"❌ Error extracting images from page {page_num}: {e}"
            )  # ERROR: Image extraction failure
            continue

    logger.info(
        f"✅ Extraction completed for PDF file: {filepath}"
    )  # INFO: Extraction finished
    return extracted_text, images_data
