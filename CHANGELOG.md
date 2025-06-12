# Récapitulatif des versions du Data Interpreter

## [1.7.0] - 2025-06-12

## Ajouté :
- Ajout de la possibilité de modifier le modèle d'embedding par défaut.

## Modifié :

- Suppression du modèle duckdb de la pipeline pour l'agent SQL.

## [1.6.0] - 2025-04-28

## Modifié :
- Suppression de la dépendance Filebrowser pour télécharger les fichiers générés, tout passe par l'api OpenWebui désormais.

## [1.5.7] - 2025-04-25

## Ajouté : 
- Prise en charge des formats de fichiers Doc, Docx ainsi que Markdown

## Modifié :
- Amélioration de la gestion des résultats SQL générés en acceptant que les 20 premiers pour les donner à l'agent Final.
- Modification des prompts des Agents pour consulter différents formats de fichiers et exploiter au mieux les sources d'informations.

## [1.5.6] - 2025-04-22

## Ajouté : 
- Mise en place d'un système d'historique à l'agent Planification afin de garder du contexte tout au long d'une conversation

## Modifié :
- Changement du modèle LLM par défaut pour le raisonnement et la Planifcation.
- Amélioration du parser de requetes SQL pour prendre en compte le format Markdown.
- Modification du prompt de l'agent LLM

## [1.5.5] - 2025-04-02
## Ajouté
- Mise en place d'un cleaner afin de pouvoir vider le cache du Data Interpreter concernant lié aux conversations qui ont été supprimé sur l'interface OpenWEBI.

## [1.5.4] - 2025-03-07
## Ajouté
- Mise en place de scripts sh afin de pouvoir aisément lancer les modes pipelines et terminal incluant une modification automatisée des variables d'environnements du fichier .env.

## [1.5.3] - 2025-02-28
## Modifié
- Conversion des caractères alphabétiques de la question ainsi que du dataFrame des fichiers Excel en minuscule.
- Correction de la détection des séparateurs ainsi que des types de données dans les fichiers CSV.

## [1.5.2] - 2025-02-27
## Modifié
- Répercution des modifications de l'ajout du Bypass sur le script main de la pipeline OpenWebui

## Ajouté 
- Mise en place d'un flag #force afin de recommencer un processus de planification si telle est la volonté.

## [1.5.1] - 2025-02-14
## Ajouté
- Mise en place d'un Bypass afin d'exclure le processus de planification pour une question spécifique. (#pass)

## [1.4.1] - 2025-02-06
## Modifié
- Correction de la gestion des fichiers en entrée afin de ne pas traiter de nouveau un fichier ayant déja été traité.

## Ajouté
- Mise en place d'un traitement par similarité afin d'éviter de refaire des recherches pour une question déja posé
- Ajout de la possibilité d'utiliser un modèle spécial pour le code.

## [1.3.2] - 2025-01-31
## Modifié
- Correction de la gestion de path custom pour l'appel de db

## Ajouté
- Mise en place d'une gestion d'historique sous forme de db.

## [1.3.1] - 2025-01-31
## Modifié
- Correction du Filebrowser afin d'avoir de nouveau accès aux fichiers générés par le data interpreter.
- Correction du processus de gestion de base de donnée afin de ne pas créer des tables qui existent déja.
- Suppression de la possibilité pour openwebui de supprimer un document de la base de donnée à la main. 

## Ajouté
- Mise en place d'un système de collection afin de pouvoir créer des sessions de chat vierge à volonté dans openwebui avec le data interpreter 
- Mise en place d'une gestion d'historique des documents pour chaque session chat créée.

## [1.2.1] - 2025-01-30
### Modifié
- Corriger l'enchainement d'execution des requetes sql générées par le LLM de planification.
- Modifier les prompts pour répondre dans la langue de la question.
- Correction de prompt afin d'avoir une génération de code python toujours fonctionnel. 
- Correction du mode retry dans le processus de génération de réponse afin de pouvoir retenter sa chance automatique si la réponse n'est pas convenable
- Correction du parcer des noms des colones dans des fichiers excel afin de traiter les accents.

### Ajouté
- Mise en place d'un logger dans le projet.
- Possibilité d'utiliser le bouton upload sur l'interface openwebui afin de pouvoir ajouter des documents.
- Possibilité de customiser les paramètres du data interpreter via un fichier .env

## [1.1.2] - 2025-01-03
### Modifié
- Correction de la détection des requêtes SQL afin de pouvoir exécuter plusieurs requêtes SQL
- Correction de la fonction de détection des noms de colones au moment de la lecture d'un fichier afin de pouvoir utiliser des nombres en guise de nom de colonne
- Correction du prompt pour le modèle de planification afin de s'assurer d'obtenir des requêtes exécutables dans l'immédiat.

## [1.1.1] - 2025-01-03
### Ajouté
- Initialisation du projet.
- Fonctionnalité de versioning.

### Modifié
- Amélioration de la gestion des erreurs pour la détection des arguments de la ligne de commande.
- Passage de Llama 3.2 à Llama 3.3 pour le modèle de raisonnement et de planification 
- Correction du parsing de la requête SQL pour ne pas supprimer les lignes faisant référence à un alias. 
- Correction dans la gestion des requêtes SQL pour toujours récupérer uniquement la dernière requête SQL générée par le modèle de planification.
- Traduction du prompt des modèles en anglais afin de gagner en efficacité.
- Modification du prompt du modèle de planification pour lui indiquer de ne jamais insérer dans la requête une table ou une colonne inexistante dans le schéma de la base de données.   
