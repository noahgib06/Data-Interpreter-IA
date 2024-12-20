def add_message(history, role, content):
    """Ajoute un message à l'historique."""
    history.append({"role": role, "content": content})
    return history
