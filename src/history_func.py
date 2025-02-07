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

# Configuration du logger global
LOG_LEVEL_ENV = os.getenv("LOG_LEVEL_history_func")
LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


# Initialisation du logger
def setup_logger(
    log_file=os.getenv("LOG_FILE_history_func"),
    max_size=5 * 1024 * 1024,
    backup_count=3,
):
    if not os.path.exists("../Logs"):
        os.makedirs("../Logs", exist_ok=True)
    logger = logging.getLogger("history_logger")
    logger.setLevel(LOG_LEVEL_MAP.get(LOG_LEVEL_ENV))
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOG_LEVEL_MAP.get(LOG_LEVEL_ENV))
    console_handler.setFormatter(formatter)
    file_handler = RotatingFileHandler(
        log_file, maxBytes=max_size, backupCount=backup_count
    )
    file_handler.setLevel(LOG_LEVEL_MAP.get(LOG_LEVEL_ENV))
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


logger = setup_logger()

OLLAMA_URL = "http://host.docker.internal:11434/api/embeddings"


# 🔹 Fonction pour obtenir un embedding avec Ollama
def get_embedding(text, model="all-minilm:33m"):
    response = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": text},
        headers={"Content-Type": "application/json"},
    )
    if response.status_code == 200:
        return response.json()["embedding"]
    else:
        raise Exception(f"Erreur Ollama: {response.text}")


# 🔹 Fonction pour calculer la similarité cosinus entre deux vecteurs
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# 🔹 Fonction pour configurer la base DuckDB et la table `chat_history`
def setup_history_database(path):
    conn = duckdb.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_history (
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding TEXT, 
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (conversation_id, role)  -- Clé composite
        )
    """
    )
    conn.close()
    print(f"✅ Base `{path}` initialisée avec `chat_history` et `embedding`.")


# 🔹 Fonction pour ajouter un message et stocker l'embedding
def add_conversation_with_embedding(path, question, response, model="all-minilm:33m"):
    conn = duckdb.connect(path)

    # Générer un identifiant unique pour la conversation
    conversation_id = str(
        uuid.uuid4()
    )  # Utilisation d'un UUID pour éviter les conflits

    # Générer les embeddings pour la question et la réponse
    question_embedding = get_embedding(question, model)
    response_embedding = get_embedding(response, model)

    # Convertir en JSON string
    question_embedding_json = json.dumps(question_embedding)
    response_embedding_json = json.dumps(response_embedding)

    # Insérer la question
    conn.execute(
        """
        INSERT INTO chat_history (conversation_id, role, content, embedding) 
        VALUES (?, ?, ?, ?)
    """,
        (conversation_id, "user", question, question_embedding_json),
    )

    # Insérer la réponse associée
    conn.execute(
        """
        INSERT INTO chat_history (conversation_id, role, content, embedding) 
        VALUES (?, ?, ?, ?)
    """,
        (conversation_id, "assistant", response, response_embedding_json),
    )

    conn.close()


# 🔹 Fonction pour récupérer l'historique
def get_history(path):
    conn = duckdb.connect(path)
    history = conn.execute(
        """
        SELECT role, content, embedding FROM chat_history ORDER BY timestamp ASC
    """
    ).fetchall()
    conn.close()

    # Convertir les embeddings JSON stringifiés en liste
    formatted_history = []
    for role, content, embedding in history:
        embedding_vector = json.loads(embedding) if embedding else None
        formatted_history.append(
            {"role": role, "content": content, "embedding": embedding_vector}
        )

    return formatted_history


# 🔹 Fonction pour récupérer les messages similaires à une question donnée
def retrieve_similar_conversations(
    question, path, model="all-minilm:33m", min_k=1, max_k=5, threshold=0.70
):
    # Générer l'embedding de la question posée
    question_embedding = get_embedding(question, model)

    # Connexion à la DB pour récupérer uniquement les questions (role "user")
    conn = duckdb.connect(path)
    user_messages = conn.execute(
        "SELECT conversation_id, content, embedding FROM chat_history WHERE role = 'user'"
    ).fetchall()
    conn.close()

    if not user_messages:
        return []

    similarities = []
    conversations = {}

    # Calculer la similarité entre l'embedding de la question posée et celles des messages utilisateurs stockés
    for conversation_id, content, embedding in user_messages:
        if embedding:
            msg_embedding = json.loads(embedding)
            sim = cosine_similarity(question_embedding, msg_embedding)
            if sim >= threshold:
                similarities.append((conversation_id, sim))
                # On enregistre la question et on prépare la place pour la réponse
                conversations[conversation_id] = {"question": content, "response": None}

    # Si aucune question ne correspond, retourner une liste vide
    if not similarities:
        return []

    # Pour chaque conversation qui a passé le seuil, récupérer la réponse associée (role "assistant")
    conn = duckdb.connect(path)
    assistant_messages = conn.execute(
        "SELECT conversation_id, content FROM chat_history WHERE role = 'assistant'"
    ).fetchall()
    conn.close()

    for conv_id, content in assistant_messages:
        if conv_id in conversations:
            conversations[conv_id]["response"] = content

    # Trier les résultats par similarité décroissante
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Sélectionner les Top-K conversations (en s'assurant que la réponse existe)
    top_conversations = []
    for conv_id, sim in similarities[: max(min_k, min(len(similarities), max_k))]:
        if conversations[conv_id]["question"] and conversations[conv_id]["response"]:
            top_conversations.append(conversations[conv_id])
    return top_conversations
