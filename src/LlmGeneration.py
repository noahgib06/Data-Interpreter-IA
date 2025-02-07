import logging
import os
import re
from logging.handlers import RotatingFileHandler

from dotenv import load_dotenv

from PythonTool import parse_and_execute_python_code, parse_code
from SqlTool import execute_sql_query

load_dotenv()

# Configuration du logger global
LOG_LEVEL_ENV = os.getenv(
    "LOG_LEVEL_LlmGeneration"
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
    log_file=os.getenv("LOG_FILE_LlmGeneration"),
    max_size=5 * 1024 * 1024,
    backup_count=3,
):
    """
    Configure un logger global pour suivre toutes les actions.
    """
    if not os.path.exists("../Logs"):
        os.makedirs("../Logs", exist_ok=True)
    logger = logging.getLogger("action_plan_logger")
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


def summarize_model(history, contextualisation_model):
    formatted_history = "\n".join(
        f"{entry[0].capitalize()}: {entry[1]}" for entry in history
    )
    prompt = (
        f"voici un historique de conversation :\n{formatted_history}\n\n"
        "Résume cet historique en gardant uniquement les éléments essentiels pour répondre à une nouvelle question."
    )

    try:
        summary = contextualisation_model.invoke(prompt)
        return summary
    except Exception as e:
        logger.error(e)
        return "Résumé non disponible."


def command_r_plus_plan(question, schema, contextualisation_model, history):
    """
    Génère un plan d'action basé sur la question et le schéma fourni.
    """
    logger.debug(
        f"Début de `command_r_plus_plan` avec question: {question}, schéma: {schema}, historique: {history}"
    )

    schema_description = "Voici le schéma de la base de données :\n"
    for table_name, columns in schema.items():
        schema_description += (
            f"Table '{table_name}' contient les colonnes suivantes :\n"
        )
        for column in columns:
            schema_description += f"  - '{column['name']}' (type: {column['type']})\n"
        schema_description += "\n"

    logger.info(f"Description du schéma générée : {schema_description}")

    prompt = (
        f'La demande est : "{question}"\n\n'
        "**Instructions pour générer le plan d'action :**\n"
        "1. Identifiez si des informations peuvent être extraites directement des colonnes mentionnées dans le schéma. Si c'est possible, fournissez un plan pour extraire ces données directement.\n"
        f"2. Si une extraction de données est nécessaire, proposez une requête SQL simple et précise pour obtenir uniquement les données pertinentes et prête à être exécutée sans modification préalable. Assurez-vous que cette requête respecte strictement le schéma fourni, sans faire d'hypothèses sur des colonnes ou des données non mentionnées. Utilisez uniquement ce schéma : {schema_description} \n"
        "3. Si la demande implique une interprétation, un calcul, une visualisation ou une génération de contenu (par exemple, graphiques, calculs mathématiques ou documents), générez un code Python adapté pour réaliser ces tâches uniquement si l'usage de python est vraiment nécessaire pour répondre à la demande.\n"
        "4. Si la demande concerne la correction ou l'amélioration d'un code existant, fournissez directement les corrections ou améliorations nécessaires sans mentionner le contexte technologique. Concentrez-vous sur les ajustements précis nécessaires pour répondre à la demande.\n"
        "5. Ne proposez pas de technologies ou de méthodes inutiles si ce n'est pas explicitement requis. Par exemple, pour analyser un document ou extraire des informations textuelles, limitez-vous aux étapes d'extraction ou de traitement nécessaires, sauf si la demande précise un type de sortie spécifique (chart, plot, graph, calcul, etc.).\n"
        "6. Si une conversion ou un ajustement de type (par exemple entre INTEGER et VARCHAR) est nécessaire pour résoudre des erreurs, incluez explicitement ces ajustements dans le plan.\n\n"
        "7. Si tu ne trouves pas d'informations qui te conviennent dans le schéma de la base de donnée, tu peux collecter les informations de toutes les tables. Ne fais pas de jointures, questionne les tables une à une. Tu n'es pas limité en nombre de requetes Duckdb à exécuter."
        "**Plan attendu :**\n"
        "- Fournissez une méthode (SQL ou étapes d'action) basée sur la nature de la question.\n"
        "- Si SQL suffit pour répondre à la question, ne proposez pas d'autres méthodes inutilement.\n"
        "- Si la question implique une visualisation ou un calcul, incluez une méthode appropriée pour produire le résultat final.\n"
        "- Si la demande inclut une correction ou amélioration de code, fournissez uniquement les étapes nécessaires pour corriger ou améliorer le code.\n"
        "- Ne mentionnes python dans le code uniquement si tu ne peux pas répondre à la question sans l'usage de ce langage et que surtout tu as fourni du code python, sinon ce mot est interdit."
        "- Le plan doit être clair, concis et strictement limité aux informations disponibles dans le schéma et la question.\n"
    )
    prompt2 = (
        f'The request is: "{question}"\n\n'
        "**Instructions to generate the action plan:**\n"
        "1. Identify if information can be directly extracted from the columns mentioned in the schema. If possible, provide a plan to directly extract this data.\n"
        f"2. If data extraction is necessary, propose a simple and precise SQL query to retrieve only the relevant data, ensuring it is fully executable without requiring prior modifications. Make sure this query strictly adheres to the provided schema, without making assumptions about unspecified columns or data. You must not invent tables or columns; use only this schema: {schema_description}\n"
        "3. If the request involves interpretation, calculation, visualization, or content generation (e.g., charts, mathematical calculations, or documents), produce only executable code without including unnecessary explanations or comments.\n"
        "4. If the request involves correcting or improving existing code, provide the necessary corrections or improvements directly without referencing the technological context (e.g., Python). Focus on the precise adjustments needed to address the request.\n"
        "5. Do not propose unnecessary technologies or methods unless explicitly required. For instance, when analyzing a document or extracting textual information, limit the steps to necessary extraction or processing, unless the request specifies a specific type of output (chart, plot, graph, calculation, etc.).\n"
        "6. If a type conversion or adjustment (e.g., between INTEGER and VARCHAR) is required to resolve errors, explicitly include these adjustments in the plan.\n\n"
        "**Expected Plan:**\n"
        "- Provide a method (SQL or action steps) based on the nature of the question.\n"
        "- If SQL is sufficient to answer the question, do not propose other unnecessary methods.\n"
        "- If the question involves visualization or calculation, include an appropriate method to produce the final result.\n"
        "- If the request includes code correction or improvement, provide only the necessary steps to correct or improve the code, without referencing the technological context.\n"
        "- The plan must be clear, concise, and strictly limited to the information available in the schema and the question.\n"
    )
    try:
        logger.debug("Envoi du prompt au modèle de contextualisation.")
        plan = contextualisation_model.invoke(input=prompt)
        logger.info(f"Plan généré par le modèle : {plan}")

        python_code = parse_code(plan)
        logger.debug(f"Code Python extrait : {python_code}")
    except Exception as e:
        logger.error(
            f"Erreur lors de la génération du plan ou extraction du code : {e}"
        )
        raise

    return plan, python_code


def adjust_sql_query_with_duckdb(sql_query, schema, duckdb_model):
    """
    Ajuste une requête SQL en fonction du moteur DuckDB, en gérant les erreurs de types.
    """
    logger.debug(
        f"Début de `adjust_sql_query_with_duckdb` avec la requête SQL : {sql_query}"
    )

    schema_description = "Voici le schéma de la base de données pour DuckDB :\n"
    for table_name, columns in schema.items():
        schema_description += (
            f"Table '{table_name}' contient les colonnes suivantes :\n"
        )
        for column in columns:
            schema_description += f"  - '{column['name']}' (type: {column['type']})\n"
        schema_description += "\n"

    try:
        sql_query = re.sub(
            r"CAST\((.*?) AS UNSIGNED\)",
            r"CAST(\1 AS INTEGER)",
            sql_query,
            flags=re.IGNORECASE,
        )
        logger.info(f"Requête SQL ajustée pour DuckDB : {sql_query}")
    except Exception as e:
        logger.error(f"Erreur lors de l'ajustement de la requête SQL : {e}")
        raise

    prompt = (
        f"{schema_description}\n\n"
        f"Voici une requête SQL générée initialement :\n```sql\n{sql_query}\n```\n\n"
        "**Instructions pour DuckDB :**\n"
        "- Corrigez les erreurs éventuelles en validant les colonnes et les relations entre les tables.\n"
        "- Si une incompatibilité de types est détectée (par exemple, INTEGER vs VARCHAR), ajoutez un casting explicite.\n"
        "- Fournissez uniquement une requête SQL corrigée mais tu ne dois pas la modifier."
        "- Tu ne dois pas la modifier sauf si tu es sur que la requete actuelle va renvoyer une erreur d'exécution.\n"
    )

    """prompt = (
        f"{schema_description}\n\n"
        f"Here is an initially generated SQL query:\n```sql\n{sql_query}\n```\n\n"
        "**Instructions for DuckDB:**\n"
        "- Fix any errors by validating the columns and relationships between tables.\n"
        "- If a type mismatch is detected (e.g., INTEGER vs VARCHAR), add explicit casting.\n"
        "- Provide only a corrected and optimized SQL query within a ```sql``` block."
    )"""

    print("Adjusting SQL query with DuckDB model...")
    try:
        logger.debug("Envoi du prompt pour ajustement au modèle DuckDB.")
        # adjusted_query = duckdb_model.invoke(prompt)
        logger.info(f"Requête SQL ajustée par le modèle : {sql_query}")
        return sql_query
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution du modèle DuckDB : {e}")
        raise


def validate_sql_with_schema(schema, query):
    logger.debug(f"Validation de la requête SQL contre le schéma : {schema}")
    column_map = {
        table: [col["name"] for col in cols] for table, cols in schema.items()
    }
    tables_in_query = set(
        re.findall(r"\bFROM\s+(\w+)|\bJOIN\s+(\w+)", query, re.IGNORECASE)
    )
    tables_in_query = {table for match in tables_in_query for table in match if table}

    missing_elements = []
    for table in tables_in_query:
        if table not in column_map:
            missing_elements.append(f"Table inconnue référencée : {table}")
        else:
            for column in re.findall(rf"{table}\.(\w+)", query):
                if column not in column_map[table]:
                    missing_elements.append(f"{table}.{column}")

    if missing_elements:
        logger.error(
            f"Colonnes ou tables manquantes dans le schéma : {missing_elements}"
        )
        raise ValueError(
            f"Colonnes ou tables manquantes dans le schéma : {missing_elements}"
        )

    logger.info("Validation de la requête SQL réussie.")
    return True


def clean_sql_query(sql_query, schema):
    logger.debug("Nettoyage de la requête SQL.")
    if schema:
        column_map = {
            table: [col["name"] for col in cols] for table, cols in schema.items()
        }
        for table, columns in column_map.items():
            for column in columns:
                pattern = rf"(?<!\.)\b{column}\b(?!\.)"
                replacement = f"{table}.{column}"
                sql_query = re.sub(pattern, replacement, sql_query)
    sql_query = re.sub(r"\s+", " ", sql_query).strip()
    logger.info(f"Requête SQL nettoyée : {sql_query}")
    return sql_query


def extract_sql_from_plan(plan_text):
    """
    Extrait toutes les requêtes SQL d'un plan.
    """
    logger = logging.getLogger("action_plan_logger")
    logger.debug(
        f"Début de `extract_sql_from_plan` avec le texte du plan : {plan_text}"
    )

    try:
        # Extraction des requêtes SQL avec une expression régulière
        queries = re.findall(r"SELECT .*?;", plan_text, re.DOTALL)
        logger.info(f"Requêtes SQL extraites : {queries}")

        # Suppression des doublons
        unique_queries = list(set(queries))
        if unique_queries:
            logger.info(f"Requêtes SQL uniques extraites : {unique_queries}")
            return unique_queries
        else:
            logger.warning("Aucune requête SQL trouvée dans le plan.")
            raise ValueError("Aucune requête SQL trouvée dans le plan.")
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction des requêtes SQL : {e}")
        raise


def generate_tools_with_llm(
    plan,
    schema,
    context,
    sql_results,
    python_results,
    database_model,
    code_model,
    python_code,
    custom_sql_path=None,
):
    """
    Génère les outils nécessaires en fonction du plan.
    """
    print("Generating tools based on the plan...")
    logger.debug("Début de la fonction `generate_tools_with_llm`.")
    files_generated = []
    results = []

    if "SQL" in plan:
        print("voila le path vers l'historique : ", custom_sql_path)
        logger.info("Traitement des requêtes SQL dans le plan.")
        try:
            sql_queries = extract_sql_from_plan(plan)
            logger.info(f"Requêtes SQL extraites : {sql_queries}")

            for i, sql_query in enumerate(sql_queries):
                try:
                    logger.debug(f"Traitement de la requête SQL #{i + 1} : {sql_query}")

                    # Nettoyage et validation des étapes SQL
                    sql_query = clean_sql_query(sql_query, schema)
                    logger.info(f"Requête SQL nettoyée : {sql_query}")

                    validate_sql_with_schema(schema, sql_query)
                    logger.info(f"Requête SQL validée contre le schéma.")

                    # Ajustement avec DuckDB (type casting ou corrections spécifiques)
                    sql_query = adjust_sql_query_with_duckdb(
                        sql_query, schema, database_model
                    )
                    logger.info(f"Requête SQL ajustée : {sql_query}")

                    # Exécuter la requête ajustée
                    if custom_sql_path is not None:
                        sql_results = execute_sql_query(sql_query, custom_sql_path)
                    else:
                        sql_results = execute_sql_query(sql_query)
                    logger.info(f"Résultats de la requête SQL : {sql_results}")

                    if "sql_results" not in context:
                        context["sql_results"] = []

                    if sql_results:
                        context["sql_results"].append({"results": sql_results})
                except Exception as e:
                    logger.error(
                        f"Erreur lors du traitement de la requête SQL #{i + 1} : {e}"
                    )
                    continue

            logger.info(
                f"Contexte mis à jour avec les résultats SQL : {context['sql_results']}"
            )
        except Exception as e:
            logger.error(
                f"Erreur lors de l'extraction ou du traitement des requêtes SQL : {e}"
            )

    if "Python" in plan or "python" in plan:
        logger.info("Génération de code Python basée sur le plan.")
        try:
            logger.debug(
                f"Résultats SQL disponibles pour le code Python : {context['sql_results']}"
            )
            prompt = (
                f"La demande initiale est : \"{context['question']}\"\n\n"
                f"Voici un exemple de code python qui pourrait permettre de répondre à la demande : {python_code}\n"
                "**Instructions pour le code :**\n"
                "1. Utilisez uniquement les données exactes fournies dans les résultats ci-dessous, sans générer de valeurs fictives ou supplémentaires. "
                "2. Ne générez aucune valeur par défaut pour compenser des données manquantes ; **limitez-vous strictement aux données fournies**.\n"
                "3. Assurez-vous que le code est **complet, fonctionnel et prêt à l'emploi**, sans sections incomplètes ou nécessitant une intervention manuelle.\n"
                "4. Limitez toute **logique conditionnelle** ou **supposition**. Si une donnée est manquante, ne tentez pas de la compléter ou de la deviner ; utilisez uniquement les résultats fournis.\n"
                "5. Vous travaillez dans un **environnement Docker sans interface graphique**. Toute visualisation, comme un graphique avec matplotlib, doit être **sauvegardée dans un fichier** (par exemple, PNG pour les graphiques).\n"
                "6. **Aucune utilisation de plt.show()** n'est autorisée, car les résultats graphiques ne peuvent pas être affichés directement.\n"
                "7. Si la tâche implique des **calculs simples ou des opérations non visuelles** (par exemple, calcul de moyennes), générez simplement le code approprié sans tenter de produire des fichiers.\n"
                "8. Pour les résultats graphiques, assurez-vous que les fichiers sont sauvegardés sans vous soucier du format ou du nom (ex. utilisez des noms par défaut).\n\n"
                "9. Que la demande porte sur un graphique, un calcul, ou une autre opération, générez le code en utilisant exclusivement les valeurs extraites, en maximisant les éléments inclus pour offrir une vue complète, et sans inventer de données.\n"
                f"Voici les résultats SQL disponibles :\n{context['sql_results']}\n\n"
                "Le code doit etre complété avec les résultats que je te donne et doit pouvoir s'exécuter sans intervention au préalable avec ces statiques. Tu ne dois pas faire un code différent de celui que je te donne. Ton role c'est de le compléter pour qu'il s'execute en fonction notamment de ma demande."
                "**Générez un code Python complet qui exploite ces résultats comme données statiques**. Le code doit répondre directement à la demande (graphique, calcul, ou autre) et **ne jamais** faire d'appels à des bases de données comme SQLite ou des services externes pour récupérer des données."
            )

            """prompt = (
                f'The initial request is: "{context["question"]}"\n\n'
                f"The defined action plan is as follows:\n{plan}\n\n"
                "**Instructions for the code:**\n"
                "1. Use only the exact data provided in the results below, without generating any fictitious or additional values.\n"
                "2. Do not generate default values to compensate for missing data; **strictly limit yourself to the provided data**.\n"
                "3. Ensure the code is **complete, functional, and ready to use**, with no incomplete sections or requiring manual intervention.\n"
                "4. Limit any **conditional logic** or **assumptions**. If data is missing, do not attempt to complete or guess it; use only the provided results.\n"
                "5. You are working in a **Docker environment without a graphical interface**. Any visualization, such as a graph using matplotlib, must be **saved to a file** (e.g., PNG for graphs).\n"
                "6. **No use of plt.show()** is allowed, as graphical results cannot be displayed directly.\n"
                "7. If the task involves **simple calculations or non-visual operations** (e.g., calculating averages), generate the appropriate code without attempting to produce files.\n"
                "8. For graphical results, ensure that files are saved without worrying about format or naming (e.g., use default names).\n\n"
                "9. Whether the request involves a graph, a calculation, or another operation, generate the code using only the extracted values, maximizing the included elements to provide a complete view, without inventing data.\n"
                "10. Whether the request involves a graph, a calculation, or another operation, generate the code using only the extracted values, maximizing the included elements to provide a complete view, without inventing data.\n"
                f"Here are the available SQL results:\n{context["sql_results"]}\n\n"
                "**Generate complete Python code that uses these results as static data.** The code must directly address the request (graph, calculation, or other) and **never** make calls to databases such as SQLite or external services to retrieve data."
            )"""

            """old prompt = (
                f"Demande initiale : \"{context['question']}\"\n\n"
                f"Plan d’action défini :\n{plan}\n\n"
                "**Instructions strictes pour le code Python :**\n"
                "1. Utilisez uniquement les **données exactes** fournies dans les résultats SQL ci-dessous, sans générer de valeurs fictives ou supplémentaires.\n"
                "2. Ne générez aucune valeur par défaut pour compenser des données manquantes ; **limitez-vous strictement aux données fournies**.\n"
                "3. Assurez-vous que le code est **complet, fonctionnel et prêt à l'emploi**, sans sections incomplètes ou nécessitant une intervention manuelle.\n"
                "4. Limitez toute **logique conditionnelle** ou **supposition**. Si une donnée est manquante, ne tentez pas de la compléter ou de la deviner ; utilisez uniquement les résultats fournis.\n"
                "5. Vous travaillez dans un **environnement Docker sans interface graphique**. Toute visualisation, comme un graphique avec matplotlib, doit être **sauvegardée dans un fichier** (par exemple, PNG pour les graphiques).\n"
                "6. **Aucune utilisation de plt.show()** n'est autorisée, car les résultats graphiques ne peuvent pas être affichés directement.\n"
                "7. Si la tâche implique des **calculs simples ou des opérations non visuelles** (par exemple, calcul de moyennes), générez simplement le code approprié sans tenter de produire des fichiers.\n"
                "8. Pour les résultats graphiques, assurez-vous que les fichiers sont sauvegardés sans vous soucier du format ou du nom (ex. utilisez des noms par défaut).\n\n"
                f"Voici les résultats SQL disponibles :\n{sql_results}\n\n"
                "**Générez un code Python complet qui exploite ces résultats comme données statiques**. Le code doit répondre directement à la demande (graphique, calcul, ou autre) et **ne jamais** faire d'appels à des bases de données comme SQLite ou des services externes pour récupérer des données."
            )"""

            logger.debug(
                "Envoi du prompt pour génération de code Python au modèle de raisonnement python."
            )
            python_tool = code_model.invoke(prompt)

            context, python_results, files_generated = parse_and_execute_python_code(
                python_tool, context, context["sql_results"]
            )

            logger.info(f"Résultats Python : {python_results}")
            logger.info(f"Fichiers générés : {files_generated}")

        except Exception as e:
            logger.error(
                f"Erreur lors de la génération ou de l'exécution du code Python : {e}"
            )

    logger.debug("Fin de la fonction `generate_tools_with_llm`.")
    return context, python_results, sql_results, files_generated


def generate_final_response_with_llama(
    context, python_results, reasoning_model, files_generated, history
):
    logger = logging.getLogger("action_plan_logger")
    logger.debug("Début de la fonction `generate_final_response_with_llama`.")

    # Créer la section des fichiers générés
    files_section = ""
    if files_generated:
        logger.info("Des fichiers générés sont disponibles.")
        files_section = "\nFichiers générés :\n" + "\n".join(
            [f"- {file}" for file in files_generated]
        )
        logger.debug(f"Section des fichiers générés : {files_section}")
    else:
        logger.info("Aucun fichier généré.")

    # Construction du prompt
    prompt = (
        f"Contexte final :\n\n"
        f"Question : \"{context['question']}\"\n"
        f"Résultats SQL : {context['sql_results']}\n"
        f"Résultats Python : {python_results}\n\n"
        f'Historique : "{history}"\n'
        f"{files_section}\n\n"
        "**Réponse finale :**\n"
        "- Résumez le contenu de manière concise en expliquant de quoi traite le document ou en répondant précisément à la demande.\n"
        "- Ne faites pas de raisonnement intermédiaire ni d'ajouts spéculatifs. Utilisez uniquement les informations fournies dans le contexte final pour formuler la réponse.\n\n"
        "- Si un historique est présent et non null, tu peux collecter des éléments réponse à l'intérieur pour répondre à la question initiale. Dans le cas contraire, tu t'appuies simplement sur les résultats SQL et Python.\n"
        "**Directives spécifiques :**\n"
        "1. Si des fichiers ont été générés (mentionnés ci-dessus), expliquez brièvement leur contenu et leur utilité en lien avec la demande.\n"
        "2. Si la réponse contient des résultats chiffrés, assurez-vous qu'ils sont bien contextualisés pour une compréhension immédiate.\n"
        "3. Ne donnez aucune explication technique non demandée par la question initiale. Limitez-vous à une explication compréhensible pour l'utilisateur final.\n"
        "4. Mentionnez les liens des fichiers créés (listés ci-dessus) de manière explicite dans la réponse.\n"
        "5. Répond toujours dans la meme langue que celle utilisée pour la question initiale.\n"
        "6. Si un historique est présent et non null, tu peux t'en servir pour répondre à la question initiale.\n"
    )

    logger.debug("Prompt construit pour le modèle de raisonnement.")

    try:
        # Appel au modèle pour générer la réponse finale
        logger.info("Envoi du prompt au modèle de raisonnement finale.")
        final_response = reasoning_model.invoke(input=prompt)
        logger.info("Réponse finale générée par le modèle.")
    except Exception as e:
        logger.error(f"Erreur lors de la génération de la réponse finale : {e}")
        raise

    # Ajouter les liens des fichiers générés à la fin de la réponse, s'ils existent
    if files_generated:
        links_section = "\n\nLiens des fichiers générés :\n" + "\n".join(
            [f"- {file}" for file in files_generated]
        )
        final_response += links_section
        logger.debug(
            f"Liens des fichiers ajoutés à la réponse finale : {links_section}"
        )

    # Afficher la réponse finale
    logger.debug(f"Réponse finale : {final_response}")
    logger.debug("Fin de la fonction `generate_final_response_with_llama`.")

    return final_response
