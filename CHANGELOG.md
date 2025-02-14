# Récapitulatif des versions du Data Interpreter

## [1.5.1] - 2025-02-14
## Ajouté
- Mise en place d'un Bypass afin d'exclure le processus de planification pour une question spécifique.

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
