"""
Module pour le calcul du PageRank et l'évaluation de l'impact des suggestions de maillage.
"""

import logging
import networkx as nx
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional

def calculate_pagerank(
    links_df: pd.DataFrame, 
    damping_factor: float = 0.85, 
    max_iterations: int = 100
) -> Dict[str, float]:
    """
    Calcule le PageRank pour chaque titre d'article.
    
    Args:
        links_df: DataFrame avec colonnes 'Source' et 'Destination'
        damping_factor: Facteur d'amortissement (généralement 0.85)
        max_iterations: Nombre maximum d'itérations pour la convergence
        
    Returns:
        Dictionnaire associant chaque titre à son score PageRank
    """
    try:
        # Créer un graphe dirigé à partir des liens
        G = nx.DiGraph()
        
        # Ajouter les liens au graphe
        for _, row in links_df.iterrows():
            source = row["Source"]
            destination = row["Destination"]
            
            if isinstance(source, str) and isinstance(destination, str):
                G.add_edge(source, destination)
        
        # Si le graphe est vide (pas de liens), retourner un dictionnaire vide
        if G.number_of_edges() == 0:
            logging.warning("Aucun lien trouvé pour le calcul du PageRank")
            # Créer un PageRank uniforme pour tous les nœuds
            return {node: 1.0/len(links_df["Source"].unique()) for node in links_df["Source"].unique()}
        
        # Calculer le PageRank
        pagerank = nx.pagerank(G, alpha=damping_factor, max_iter=max_iterations)
        
        logging.info(f"PageRank calculé pour {len(pagerank)} titres")
        
        return pagerank
    
    except Exception as e:
        logging.error(f"Erreur lors du calcul du PageRank: {str(e)}")
        return {}

def calculate_pagerank_with_suggestions(
    existing_links_df: pd.DataFrame, 
    suggested_links_df: pd.DataFrame,
    damping_factor: float = 0.85,
    max_iterations: int = 100
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Calcule le PageRank avant et après l'ajout des liens suggérés.
    
    Args:
        existing_links_df: DataFrame des liens existants
        suggested_links_df: DataFrame des liens suggérés
        damping_factor: Facteur d'amortissement
        max_iterations: Nombre maximum d'itérations
        
    Returns:
        Tuple de deux dictionnaires (pagerank_actuel, pagerank_optimisé)
    """
    try:
        # Pour les articles pré-publication, il n'y a généralement pas de liens existants
        # Mais nous créons quand même un PageRank "actuel" comme référence
        current_pagerank = calculate_pagerank(
            existing_links_df,
            damping_factor=damping_factor,
            max_iterations=max_iterations
        )
        
        # Créer un DataFrame combiné avec les liens existants et suggérés
        combined_links = pd.concat([existing_links_df, suggested_links_df]).drop_duplicates()
        
        # Calculer le PageRank optimisé
        optimized_pagerank = calculate_pagerank(
            combined_links,
            damping_factor=damping_factor,
            max_iterations=max_iterations
        )
        
        # Ajouter les titres qui pourraient être dans l'un mais pas dans l'autre
        all_titles = set(current_pagerank.keys()) | set(optimized_pagerank.keys())
        
        # S'assurer que tous les titres sont dans les deux dictionnaires
        for title in all_titles:
            if title not in current_pagerank:
                # Valeur par défaut pour les titres manquants
                current_pagerank[title] = min(current_pagerank.values()) if current_pagerank else 0.01
            
            if title not in optimized_pagerank:
                # Valeur par défaut pour les titres manquants
                optimized_pagerank[title] = min(optimized_pagerank.values()) if optimized_pagerank else 0.01
        
        logging.info(f"PageRank calculé pour {len(current_pagerank)} titres (avant et après suggestions)")
        
        return current_pagerank, optimized_pagerank
    
    except Exception as e:
        logging.error(f"Erreur lors du calcul du PageRank avec suggestions: {str(e)}")
        return {}, {}

def prepare_graph_data(
    titles: List[str],
    links_df: pd.DataFrame,
    suggested_links_df: Optional[pd.DataFrame] = None,
    pagerank_current: Optional[Dict[str, float]] = None,
    pagerank_optimized: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Prépare les données de graphe pour la visualisation.
    
    Args:
        titles: Liste des titres d'articles
        links_df: DataFrame des liens existants
        suggested_links_df: DataFrame des liens suggérés (optionnel)
        pagerank_current: Scores PageRank actuels (optionnel)
        pagerank_optimized: Scores PageRank optimisés (optionnel)
        
    Returns:
        Dictionnaire avec les données formatées pour la visualisation
    """
    try:
        # Créer les nœuds du graphe
        nodes = []
        for title in titles:
            node = {
                "id": title,
                "label": title[:30] + "..." if len(title) > 30 else title,
                "title": title
            }
            
            # Ajouter les scores PageRank si disponibles
            if pagerank_current and title in pagerank_current:
                node["pagerank_current"] = pagerank_current[title]
            
            if pagerank_optimized and title in pagerank_optimized:
                node["pagerank_optimized"] = pagerank_optimized[title]
                
                # Calculer l'amélioration du PageRank
                if pagerank_current and title in pagerank_current:
                    improvement = (pagerank_optimized[title] - pagerank_current[title]) / pagerank_current[title] * 100
                    node["improvement"] = improvement
            
            nodes.append(node)
        
        # Créer les liens du graphe
        edges = []
        
        # Ajouter les liens existants
        for _, row in links_df.iterrows():
            source = row["Source"]
            destination = row["Destination"]
            
            if isinstance(source, str) and isinstance(destination, str):
                edges.append({
                    "from": source,
                    "to": destination,
                    "type": "existing"
                })
        
        # Ajouter les liens suggérés
        if suggested_links_df is not None:
            for _, row in suggested_links_df.iterrows():
                source = row["Source"]
                destination = row["Destination"]
                
                if isinstance(source, str) and isinstance(destination, str):
                    edges.append({
                        "from": source,
                        "to": destination,
                        "type": "suggested"
                    })
        
        return {
            "nodes": nodes,
            "edges": edges
        }
    
    except Exception as e:
        logging.error(f"Erreur lors de la préparation des données de graphe: {str(e)}")
        return {"nodes": [], "edges": []}
