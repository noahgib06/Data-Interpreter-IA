from SqlTool import execute_sql_query
from PythonTool import parse_and_execute_python_code


def command_r_plus_plan(question, schema, contextualisation_model):
    schema_description = "Voici le schéma de la base de données :\n"
    # Mise en forme du schéma de la base de donnée
    for table_name, columns in schema.items():
        schema_description += (
            f"Table '{table_name}' contient les colonnes suivantes :\n"
        )
        for column in columns:
            schema_description += f"  - '{column['name']}' (type: {column['type']})\n"
        schema_description += "\n"

    prompt = (
        f"{schema_description}\n"
        f'La demande est : "{question}"\n\n'
        "**Plan d'action :**\n"
        "- Étape 1 : Identifiez si des informations sont disponibles directement dans les colonnes, et précisez les valeurs de colonne ou types d’information à extraire.\n"
        "- Étape 2 : Si la demande requiert une extraction de données (comme texte, OCR, code), suggérez des requêtes SQL simples et précises pour obtenir un échantillon représentatif de chaque type de donnée disponible, ou pour répondre à des questions spécifiques. Cette requete doit uniquement se baser sur le schéma que je te fournis. Tu ne dois rien ajouter qui ne soit pas mentionné dans le schéma de la base de donnée.\n"
        "- Étape 3 : Si la demande implique une interprétation (par exemple, analyser le contenu ou trouver des mots-clés), expliquez brièvement comment interpréter les résultats SQL sans utiliser d'étapes de réflexion intermédiaires ou de raisonnement complexe.\n\n"
        "Répondez uniquement aux besoins précis de la question sans suggérer de code Python, sauf si spécifiquement requis pour traiter un type de donnée extrait. **Si la demande inclut des termes comme chart, plot, graph, ou fait référence à un calcul ou une visualisation, générez du code pour créer un graphique ou effectuer le calcul.  Le plan doit contenir une seule méthode (SQL ou autre) en fonction de ce qui est nécessaire pour traiter la demande."
    )

    print(
        f"Generating plan from Command R Plus for question: {question} with schema: {schema_description}"
    )
    plan = contextualisation_model.invoke(prompt)
    print(f"Plan généré par Command R Plus : {plan}")
    return plan


def generate_tools_with_llm(
    plan, schema, context, sql_results, python_results, database_model, reasoning_model
):
    print("Génération des outils en fonction du plan...")
    schema_description = "Voici le schéma de la base de données :\n"
    # Mise en forme du schéma de la base de donnée
    for table_name, columns in schema.items():
        schema_description += (
            f"Table '{table_name}' contient les colonnes suivantes :\n"
        )
        for column in columns:
            schema_description += f"  - '{column['name']}' (type: {column['type']})\n"
        schema_description += "\n"

    if "SQL" in plan:
        print("Génération d'une requête SQL...")
        prompt = (
            f"{schema_description}\n\n"
            f"Plan d’action :\n{plan}\n\n"
            f"Demande : \"{context['question']}\"\n\n"
            "**Requête SQL :**\n"
            "- Formulez une requête SQL simple qui extrait uniquement les informations nécessaires du schéma de la base de données pour répondre à la question ou aux étapes du plan.\n"
            "- La requête doit être directe, sans clauses complexes (comme des agrégations avancées ou des jointures inutiles), sauf si spécifiquement nécessaire.\n"
            "- Utilisez les valeurs de `page`, `ocr_text` ou `content` selon le type de document, ou filtrez les résultats pour fournir des exemples clairs de chaque catégorie de données disponibles.\n\n"
            "Si la requête nécessite des optimisations, appliquez-les, mais restez fidèle au plan d'action."
        )
        sql_tool = database_model.invoke(prompt)
        sql_results = execute_sql_query(sql_tool)
        context["sql_results"] = sql_results

    if "Python" in plan:
        print("Génération de code Python...")
        print("les voila:", sql_results)
        prompt = (
            f"La demande initiale est : \"{context['question']}\"\n\n"
            f"Le plan d’action défini est le suivant :\n{plan}\n\n"
            "**Instructions pour le code :**\n"
            "1. Utilisez uniquement les données exactes fournies dans les résultats ci-dessous, sans générer de valeurs fictives ou supplémentaires. "
            "Si des valeurs spécifiques ne sont pas disponibles, n’ajoutez pas de valeurs par défaut ; limitez-vous strictement aux données fournies.\n"
            "2. Que la demande porte sur un graphique, un calcul, ou une autre opération, générez le code en utilisant exclusivement les valeurs extraites, en maximisant les éléments inclus pour offrir une vue complète, et sans inventer de données.\n"
            "3. Assurez-vous que le code est complet et fonctionnel, sans sections incomplètes ni parties laissées pour une intervention manuelle.\n"
            "4. Limitez toute logique conditionnelle ou hypothèse ; si une donnée manque, n’essayez pas de la compléter ou de la deviner.\n\n"
            f"Voici les résultats SQL disponibles, s’ils sont extraits : {sql_results}\n\n"
            "Générez le code en fonction de ces résultats exacts comme données statiques. Le code ne doit contenir que des valeurs provenant de ces résultats et répondre directement à la demande (graphique, calcul, ou autre)."
        )

        python_tool = reasoning_model.invoke(prompt)
        context, python_results = parse_and_execute_python_code(
            python_tool, context, sql_results
        )

    return context, python_results, sql_results


def generate_final_response_with_llama(
    context, sql_results, python_results, reasoning_model
):
    print(f"Generating final response with context: {context}")
    prompt = (
        f"Contexte final :\n\n"
        f"Question : \"{context['question']}\"\n"
        f"Résultats SQL : {sql_results}\n"
        f"Résultats Python : {python_results}\n\n"
        "**Réponse finale :**\n"
        "- Résumez le contenu de manière concise en expliquant de quoi traite le document ou en répondant précisément à la demande.\n"
        "- Ne faites pas de raisonnement intermédiaire ni d'ajouts spéculatifs. Utilisez uniquement les informations fournies dans le contexte final pour formuler la réponse.\n\n"
        "Si nécessaire, reformulez les résultats pour qu’ils répondent directement à la demande sans ajout d’informations non essentielles."
    )
    final_response = reasoning_model.invoke(prompt)
    print(f"Final response: {final_response}")
    return final_response
