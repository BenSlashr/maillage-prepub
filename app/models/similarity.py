"""
Module pour l'analyse de similarité entre les titres d'articles.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch

class TitleSimilarityAnalyzer:
    """
    Classe pour analyser la similarité entre les titres d'articles et générer
    des suggestions de maillage interne.
    """
    
    def __init__(self, model_name: str = "distiluse-base-multilingual-cased-v2"):
        """
        Initialise l'analyseur de similarité avec le modèle BERT spécifié.
        
        Args:
            model_name: Nom du modèle SentenceTransformer à utiliser
        """
        self.model_name = model_name
        
        # Initialiser le modèle BERT
        try:
            self.model = SentenceTransformer(model_name)
            device_name = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model.to(device_name)
            logging.info(f"Utilisation du périphérique PyTorch: {device_name}")
            logging.info(f"Modèle SentenceTransformer chargé: {model_name}")
        except Exception as e:
            logging.error(f"Erreur lors du chargement du modèle: {str(e)}")
            raise e
    
    def calculate_similarity_matrix(self, titles: List[str]) -> np.ndarray:
        """
        Calcule la matrice de similarité entre tous les titres.
        
        Args:
            titles: Liste des titres d'articles
            
        Returns:
            Matrice de similarité (numpy array)
        """
        logging.info(f"Calcul de la matrice de similarité pour {len(titles)} titres")
        
        # Générer les embeddings pour tous les titres
        embeddings = self.model.encode(titles, show_progress_bar=True)
        
        # Calculer la similarité cosinus entre tous les embeddings
        similarity_matrix = cosine_similarity(embeddings)
        
        logging.info(f"Matrice de similarité calculée: {similarity_matrix.shape}")
        return similarity_matrix
    
    def generate_suggestions(
        self,
        titles: List[str],
        types: List[str],
        similarity_matrix: np.ndarray,
        min_similarity: float = 0.3,
        max_suggestions_per_title: int = 5,
        linking_rules: Optional[Dict[str, Dict[str, Dict[str, int]]]] = None,
        content_df: Optional[pd.DataFrame] = None
    ) -> List[Dict[str, Any]]:
        """
        Génère des suggestions de maillage interne basées sur la similarité des titres.
        
        Args:
            titles: Liste des titres d'articles
            types: Liste des types d'articles (même longueur que titles)
            similarity_matrix: Matrice de similarité entre les titres
            min_similarity: Score minimum de similarité pour les suggestions
            max_suggestions_per_title: Nombre maximum de suggestions par titre
            linking_rules: Règles de maillage entre types d'articles
            content_df: DataFrame contenant le contenu des articles (optionnel)
            
        Returns:
            Liste de dictionnaires contenant les suggestions de maillage
        """
        logging.info(f"Génération des suggestions de maillage pour {len(titles)} titres")
        
        suggestions = []
        
        # Pour chaque titre, trouver les titres les plus similaires
        for i, source_title in enumerate(titles):
            source_type = types[i]
            
            # Obtenir les scores de similarité pour ce titre
            similarities = similarity_matrix[i]
            
            # Trier les indices par score de similarité (du plus élevé au plus bas)
            sorted_indices = np.argsort(similarities)[::-1]
            
            # Filtrer les titres avec une similarité suffisante
            suggestions_for_title = []
            
            for j in sorted_indices:
                # Ignorer le titre lui-même ou les titres identiques (doublons)
                if i == j or source_title.strip().lower() == titles[j].strip().lower():
                    logging.debug(f"Ignoré auto-lien: {source_title} -> {titles[j]}")
                    continue
                
                # Vérifier si la similarité est suffisante
                if similarities[j] < min_similarity:
                    continue
                
                target_title = titles[j]
                target_type = types[j]
                
                # Vérifier les règles de maillage si elles sont définies
                if linking_rules and source_type in linking_rules and target_type in linking_rules[source_type]:
                    rule = linking_rules[source_type][target_type]
                    max_links = rule.get("max_links", 10)
                    
                    # Compter le nombre de suggestions déjà faites pour ce type
                    existing_count = sum(1 for s in suggestions_for_title if s["target_type"] == target_type)
                    
                    # Vérifier si on a déjà atteint le nombre maximum de liens
                    if existing_count >= max_links:
                        continue
                
                # Extraire des suggestions d'ancres basées sur le titre cible
                anchor_suggestions = self._extract_anchor_suggestions(target_title)
                
                # Ajouter la suggestion
                suggestions_for_title.append({
                    "target_title": target_title,
                    "similarity": float(similarities[j]),
                    "target_type": target_type,
                    "anchor_suggestions": anchor_suggestions
                })
                
                # Limiter le nombre de suggestions par titre
                if len(suggestions_for_title) >= max_suggestions_per_title:
                    break
            
            # Ajouter les suggestions pour ce titre
            for suggestion in suggestions_for_title:
                suggestions.append({
                    "source_title": source_title,
                    "source_type": source_type,
                    "target_title": suggestion["target_title"],
                    "target_type": suggestion["target_type"],
                    "similarity_score": suggestion["similarity"],
                    "anchor_text": suggestion["anchor_suggestions"][0] if suggestion["anchor_suggestions"] else suggestion["target_title"]
                })
        
        logging.info(f"Génération terminée: {len(suggestions)} suggestions de maillage")
        return suggestions
    
    def _extract_anchor_suggestions(self, title: str, max_suggestions: int = 3) -> List[str]:
        """
        Extrait des suggestions d'ancres à partir du titre cible.
        
        Args:
            title: Titre cible
            max_suggestions: Nombre maximum de suggestions d'ancres
            
        Returns:
            Liste de suggestions d'ancres
        """
        # Pour simplifier, on utilise le titre complet comme première suggestion
        suggestions = [title]
        
        # Ajouter des variantes du titre (par exemple, sans les mots communs)
        words = title.split()
        if len(words) > 3:
            # Suggestion: les 3-4 premiers mots si le titre est long
            suggestions.append(" ".join(words[:min(4, len(words) - 1)]))
            
            # Suggestion: les mots-clés (en excluant les mots communs)
            common_words = ["le", "la", "les", "un", "une", "des", "et", "ou", "pour", "dans", "sur", "avec", "sans", "par"]
            keywords = [word for word in words if word.lower() not in common_words and len(word) > 3]
            if len(keywords) > 1:
                suggestions.append(" ".join(keywords[:min(3, len(keywords))]))
        
        return suggestions[:max_suggestions]
