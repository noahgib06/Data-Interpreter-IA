import json
import logging
import os
import uuid
from logging.handlers import RotatingFileHandler

import duckdb
import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

# Global logger configuration
LOG_LEVEL_ENV = os.getenv("LOG_LEVEL_history_func")
LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


# Logger setup function
def setup_logger(
    log_file=os.getenv("LOG_FILE_history_func"),
    max_size=5 * 1024 * 1024,
    backup_count=3,
):
    # Create log directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # Clean up the log file path
    if log_file:
        log_file = log_file.strip('"')  # Remove any quotes
        log_file = os.path.join(log_dir, os.path.basename(log_file))

    # Initialize logger
    logger = logging.getLogger("history_logger")
    logger.setLevel(LOG_LEVEL_MAP.get(LOG_LEVEL_ENV, logging.INFO))

    # Define log message format
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    )

    # Console handler setup
    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOG_LEVEL_MAP.get(LOG_LEVEL_ENV, logging.INFO))
    console_handler.setFormatter(formatter)

    # File handler setup with rotation
    file_handler = RotatingFileHandler(
        log_file, maxBytes=max_size, backupCount=backup_count
    )
    file_handler.setLevel(LOG_LEVEL_MAP.get(LOG_LEVEL_ENV, logging.INFO))
    file_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# Initialize global logger
logger = setup_logger()

# Retrieve OLLAMA API URL from environment variables
OLLAMA_URL = os.getenv("OLLAMA_URL")


# Function to obtain embedding using Ollama API
def get_embedding(text, model="all-minilm:33m"):
    logger.info(f"Fetching embedding for text: {text}")
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": text},
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        return response.json()["embedding"]
    except requests.exceptions.RequestException as e:
        logger.exception(f"Ollama API error: {e}")
        raise


# Function to calculate cosine similarity between two vectors
def cosine_similarity(vec1, vec2):
    logger.debug("Calculating cosine similarity between two vectors")
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# Function to configure the DuckDB database and `chat_history` table
def setup_history_database(path):
    logger.info(f"Initializing database at: {path}")
    conn = duckdb.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_history (
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding TEXT, 
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (conversation_id, role)  -- Composite primary key
        )
    """
    )
    conn.close()
    logger.info("✅ Database and `chat_history` table successfully initialized.")


# Function to add a conversation entry with embedding storage
def add_conversation_with_embedding(path, question, response, model="all-minilm:33m"):
    logger.info("Adding a new conversation to the database")
    try:
        conn = duckdb.connect(path)
        conversation_id = str(uuid.uuid4())  # Generate unique conversation ID

        # Generate embeddings for question and response
        question_embedding = get_embedding(question, model)
        response_embedding = get_embedding(response, model)

        # Convert embeddings to JSON format
        question_embedding_json = json.dumps(question_embedding)
        response_embedding_json = json.dumps(response_embedding)

        # Escape single quotes in text content
        question_escaped = question.replace("'", "''")
        response_escaped = response.replace("'", "''")
        question_embedding_escaped = question_embedding_json.replace("'", "''")
        response_embedding_escaped = response_embedding_json.replace("'", "''")

        # Insert user question into chat history using direct SQL
        user_query = f"""
            INSERT INTO chat_history (conversation_id, role, content, embedding) 
            VALUES ('{conversation_id}', 'user', '{question_escaped}', '{question_embedding_escaped}')
        """
        conn.execute(user_query)

        # Insert assistant response into chat history using direct SQL
        assistant_query = f"""
            INSERT INTO chat_history (conversation_id, role, content, embedding) 
            VALUES ('{conversation_id}', 'assistant', '{response_escaped}', '{response_embedding_escaped}')
        """
        conn.execute(assistant_query)
        
        conn.close()
        logger.info("✅ Conversation successfully added.")
    except Exception as e:
        logger.exception("Error adding conversation")
        raise


# Function to retrieve conversation history
def get_history(path):
    logger.info("Fetching conversation history")
    conn = duckdb.connect(path)
    history = conn.execute(
        """
        SELECT role, content, embedding FROM chat_history ORDER BY timestamp ASC
        """
    ).fetchall()
    conn.close()
    logger.info(f"Retrieved {len(history)} entries from history")
    return [
        {
            "role": role,
            "content": content,
            "embedding": json.loads(embedding) if embedding else None,
        }
        for role, content, embedding in history
    ]


# Function to retrieve similar conversations based on embedding similarity
def retrieve_similar_conversations(
    question, path, model="all-minilm:33m", min_k=1, max_k=5, threshold=0.70
):
    logger.info(f"Searching for similar conversations for question: {question}")
    try:
        question_embedding = get_embedding(question, model)

        # Fetch stored user messages
        conn = duckdb.connect(path)
        user_messages = conn.execute(
            "SELECT conversation_id, content, embedding FROM chat_history WHERE role = 'user'"
        ).fetchall()
        conn.close()

        similarities = []
        conversations = {}

        # Compute similarity between input question and stored user questions
        for conversation_id, content, embedding in user_messages:
            if embedding:
                msg_embedding = json.loads(embedding)
                sim = cosine_similarity(question_embedding, msg_embedding)
                if sim >= threshold:
                    similarities.append((conversation_id, sim))
                    conversations[conversation_id] = {
                        "question": content,
                        "response": None,
                    }

        # Fetch stored assistant responses
        conn = duckdb.connect(path)
        assistant_messages = conn.execute(
            "SELECT conversation_id, content FROM chat_history WHERE role = 'assistant'"
        ).fetchall()
        conn.close()

        # Match responses to corresponding user questions
        for conv_id, content in assistant_messages:
            if conv_id in conversations:
                conversations[conv_id]["response"] = content

        similarities.sort(key=lambda x: x[1], reverse=True)
        logger.debug(f"Found {len(similarities)} similar conversations")
        return [
            conversations[conv_id]
            for conv_id, _ in similarities[: max(min_k, min(len(similarities), max_k))]
        ]
    except Exception as e:
        logger.exception("Error retrieving similar conversations")
        raise
