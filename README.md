# Maillage Pré-Publication

## Description

Maillage Pré-Publication est un outil d'analyse et de génération de plan de maillage interne pour des articles qui ne sont pas encore publiés. Contrairement aux outils traditionnels d'analyse de maillage qui nécessitent que le contenu soit déjà en ligne, cet outil permet de planifier votre stratégie de maillage interne avant même la publication de vos articles.

## Fonctionnalités

- **Analyse sémantique des titres d'articles** : Utilise des modèles de langage avancés pour comprendre la similarité entre les titres d'articles.
- **Suggestions de maillage basées sur la similarité** : Génère des suggestions de liens internes pertinents entre vos articles.
- **Règles de maillage personnalisables** : Définissez des règles spécifiques pour le maillage entre différents types d'articles.
- **Analyse de l'impact sur le PageRank** : Évalue l'impact des suggestions de maillage sur le PageRank de vos articles.
- **Suggestions d'ancres pertinentes** : Propose des textes d'ancrage optimisés pour vos liens internes.
- **Visualisation du maillage** : Représentation graphique de la structure de maillage proposée.

## Prérequis

- Python 3.10 ou supérieur
- Les dépendances listées dans `requirements.txt`

## Installation

1. Clonez ce dépôt :
```bash
git clone https://github.com/votre-utilisateur/maillage-prepub.git
cd maillage-prepub
```

2. Créez et activez un environnement virtuel :
```bash
python -m venv venv
source venv/bin/activate  # Sur Windows : venv\Scripts\activate
```

3. Installez les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

1. Lancez l'application :
```bash
python -m app.main
```
ou
```bash
cd app && python main.py
```

2. Ouvrez votre navigateur à l'adresse `http://localhost:8000`

3. Suivez les instructions à l'écran pour télécharger vos fichiers et générer votre plan de maillage.

### Format des fichiers d'entrée

#### Fichier des titres (obligatoire)
Un fichier Excel (.xlsx) contenant au minimum deux colonnes :
- `Titre` : Le titre de l'article
- `Type` : Le type d'article (ex: blog, produit, catégorie, etc.)

#### Fichier de contenu (optionnel)
Un fichier Excel (.xlsx) contenant au minimum deux colonnes :
- `Titre` : Le titre de l'article (doit correspondre aux titres du fichier précédent)
- `Contenu` : Le contenu de l'article

#### Fichier de règles de maillage (optionnel)
Un fichier JSON définissant les règles de maillage entre les différents types d'articles. Exemple :
```json
{
  "blog": {
    "produit": {
      "min_links": 1,
      "max_links": 3,
      "min_similarity": 0.3,
      "enabled": true
    }
  }
}
```

## Structure du projet

```
maillage-prepub/
├── app/
│   ├── data/
│   │   └── jobs/
│   ├── models/
│   │   ├── similarity.py
│   │   ├── linking_rules.py
│   │   └── pagerank.py
│   ├── results/
│   ├── static/
│   │   └── css/
│   │       └── style.css
│   ├── templates/
│   │   ├── index.html
│   │   ├── upload.html
│   │   └── report.html
│   └── main.py
├── requirements.txt
└── README.md
```

## Technologies utilisées

- **Backend** : FastAPI (Python)
- **Frontend** : HTML, CSS, JavaScript, Bootstrap
- **Analyse sémantique** : sentence-transformers
- **Analyse de graphe** : NetworkX
- **Traitement des données** : Pandas, NumPy
- **Visualisation** : vis.js

## Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.

## Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou à soumettre une pull request.

## Auteur

Benoit - [GitHub](https://github.com/BenSlashr)
