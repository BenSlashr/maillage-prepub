"""
Script pour générer des exemples de fichiers Excel pour tester l'application Maillage Pré-Publication.
"""

import pandas as pd
import random
import os

# Définir les types d'articles
TYPES = ["article", "produit", "categorie"]

# Définir des thèmes pour générer des titres cohérents
THEMES = {
    "technologie": [
        "Les dernières tendances en matière de {tech}",
        "Comment {action} avec {tech}",
        "Guide complet sur {tech}",
        "Comparatif des meilleurs {tech} en 2025",
        "{tech} vs {tech2} : lequel choisir ?",
        "L'avenir de {tech} dans le monde professionnel",
        "5 façons d'utiliser {tech} pour {benefit}",
        "Pourquoi {tech} va révolutionner {industry}",
        "Les erreurs à éviter avec {tech}",
        "Optimiser votre {tech} pour de meilleures performances"
    ],
    "marketing": [
        "Stratégies de {marketing_type} pour {business_type}",
        "Comment {action_marketing} pour augmenter vos conversions",
        "Guide de {marketing_type} pour débutants",
        "Les tendances {marketing_type} à suivre en 2025",
        "Améliorer votre ROI grâce à {marketing_type}",
        "{marketing_type} : les meilleures pratiques",
        "Comment mesurer l'efficacité de votre {marketing_type}",
        "Créer une stratégie de {marketing_type} efficace",
        "Les erreurs de {marketing_type} qui coûtent cher",
        "{marketing_type} vs {marketing_type2} : quelle stratégie choisir ?"
    ],
    "seo": [
        "Optimiser votre {content_type} pour le SEO",
        "Les facteurs de classement SEO en 2025",
        "Comment {action_seo} pour améliorer votre référencement",
        "Guide complet du {seo_aspect}",
        "Audit SEO : comment analyser votre site web",
        "SEO local : stratégies pour {business_type}",
        "Comment battre vos concurrents sur {keyword_type}",
        "Les erreurs SEO qui pénalisent votre site",
        "Stratégie de maillage interne pour améliorer votre SEO",
        "L'importance de {seo_aspect} pour votre référencement"
    ]
}

# Définir des variables pour les templates
VARIABLES = {
    "tech": ["intelligence artificielle", "réalité virtuelle", "blockchain", "IoT", "cloud computing", 
             "5G", "cybersécurité", "big data", "machine learning", "robotique"],
    "tech2": ["cloud computing", "edge computing", "5G", "4G", "intelligence artificielle", 
              "machine learning", "deep learning", "réalité augmentée", "réalité virtuelle"],
    "action": ["améliorer votre productivité", "sécuriser vos données", "optimiser vos processus", 
               "réduire vos coûts", "innover", "transformer votre entreprise", "automatiser vos tâches"],
    "benefit": ["gagner du temps", "économiser de l'argent", "améliorer la sécurité", 
                "augmenter la productivité", "satisfaire vos clients", "battre la concurrence"],
    "industry": ["la finance", "la santé", "l'éducation", "le commerce", "l'industrie", 
                 "les services", "le marketing", "la logistique", "l'agriculture"],
    "marketing_type": ["content marketing", "email marketing", "social media marketing", 
                       "inbound marketing", "marketing d'influence", "marketing automation", 
                       "video marketing", "SEO", "SEM", "marketing mobile"],
    "marketing_type2": ["marketing traditionnel", "marketing digital", "growth hacking", 
                        "marketing de contenu", "marketing par email", "marketing sur les réseaux sociaux"],
    "business_type": ["e-commerce", "startup", "PME", "entreprise locale", "B2B", "B2C", 
                      "SaaS", "marketplace", "site de contenu"],
    "action_marketing": ["utiliser les réseaux sociaux", "créer du contenu de qualité", 
                         "segmenter votre audience", "personnaliser vos emails", 
                         "optimiser votre tunnel de conversion", "utiliser la vidéo"],
    "content_type": ["site web", "blog", "fiche produit", "landing page", "article de blog", 
                     "page catégorie", "page à propos", "FAQ"],
    "action_seo": ["optimiser vos meta-descriptions", "améliorer la vitesse de votre site", 
                   "créer du contenu de qualité", "obtenir des backlinks", "optimiser pour le mobile", 
                   "utiliser les données structurées", "optimiser votre maillage interne"],
    "seo_aspect": ["SEO on-page", "SEO off-page", "SEO technique", "SEO local", "SEO mobile", 
                   "SEO e-commerce", "SEO international", "SEO voice search"],
    "keyword_type": ["mots-clés longue traîne", "requêtes locales", "recherches vocales", 
                     "mots-clés de marque", "mots-clés commerciaux", "mots-clés informationnels"]
}

def generate_title(theme):
    """Génère un titre aléatoire basé sur un thème."""
    template = random.choice(THEMES[theme])
    
    # Remplacer les variables dans le template
    for var in VARIABLES:
        if "{" + var + "}" in template:
            template = template.replace("{" + var + "}", random.choice(VARIABLES[var]))
        if "{" + var + "2}" in template:
            # S'assurer que la deuxième variable est différente de la première
            options = [x for x in VARIABLES[var] if x != template.split(var)[0].split("{")[-1].split("}")[0]]
            if options:
                template = template.replace("{" + var + "2}", random.choice(options))
            else:
                template = template.replace("{" + var + "2}", random.choice(VARIABLES[var]))
    
    return template

def generate_example_data(num_articles=30):
    """Génère des données d'exemple pour l'application."""
    titles = []
    types = []
    contents = []
    
    themes = list(THEMES.keys())
    
    for _ in range(num_articles):
        # Choisir un thème aléatoire
        theme = random.choice(themes)
        
        # Générer un titre
        title = generate_title(theme)
        
        # Choisir un type d'article
        article_type = random.choice(TYPES)
        
        # Générer un contenu fictif (pour l'exemple)
        content = f"Ceci est un exemple de contenu pour l'article '{title}'. "
        content += f"Il s'agit d'un article de type '{article_type}' sur le thème '{theme}'. "
        content += "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed non risus. "
        content += "Suspendisse lectus tortor, dignissim sit amet, adipiscing nec, ultricies sed, dolor. "
        content += "Cras elementum ultrices diam. Maecenas ligula massa, varius a, semper congue, euismod non, mi. "
        content *= 3  # Répéter pour avoir plus de contenu
        
        titles.append(title)
        types.append(article_type)
        contents.append(content)
    
    # Créer les DataFrames
    titles_df = pd.DataFrame({
        "Titre": titles,
        "Type": types
    })
    
    content_df = pd.DataFrame({
        "Titre": titles,
        "Contenu": contents
    })
    
    return titles_df, content_df

def main():
    """Fonction principale pour générer et sauvegarder les fichiers d'exemple."""
    # Créer le dossier de sortie s'il n'existe pas
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Générer les données
    titles_df, content_df = generate_example_data(num_articles=30)
    
    # Sauvegarder les fichiers
    titles_file = os.path.join(output_dir, "example_titles.xlsx")
    content_file = os.path.join(output_dir, "example_content.xlsx")
    
    titles_df.to_excel(titles_file, index=False)
    content_df.to_excel(content_file, index=False)
    
    print(f"Fichiers d'exemple générés avec succès :")
    print(f"- Titres : {titles_file}")
    print(f"- Contenu : {content_file}")

if __name__ == "__main__":
    main()
