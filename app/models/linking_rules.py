"""
Module pour la gestion des règles de maillage entre types d'articles.
"""

import json
import logging
from typing import Dict, List, Optional, Any

class LinkingRulesManager:
    """
    Classe pour gérer les règles de maillage entre différents types d'articles.
    """
    
    def __init__(self, rules_file: Optional[str] = None):
        """
        Initialise le gestionnaire de règles de maillage.
        
        Args:
            rules_file: Chemin vers un fichier JSON contenant les règles de maillage
        """
        self.rules = {}
        
        if rules_file:
            self.load_rules(rules_file)
    
    def load_rules(self, rules_file: str) -> None:
        """
        Charge les règles de maillage depuis un fichier JSON.
        
        Args:
            rules_file: Chemin vers un fichier JSON contenant les règles de maillage
        """
        try:
            with open(rules_file, 'r', encoding='utf-8') as f:
                self.rules = json.load(f)
            
            logging.info(f"Règles de maillage chargées depuis {rules_file}")
        except Exception as e:
            logging.error(f"Erreur lors du chargement des règles de maillage: {str(e)}")
            self.rules = {}
    
    def save_rules(self, rules_file: str) -> None:
        """
        Sauvegarde les règles de maillage dans un fichier JSON.
        
        Args:
            rules_file: Chemin vers un fichier JSON où sauvegarder les règles
        """
        try:
            with open(rules_file, 'w', encoding='utf-8') as f:
                json.dump(self.rules, f, ensure_ascii=False, indent=2)
            
            logging.info(f"Règles de maillage sauvegardées dans {rules_file}")
        except Exception as e:
            logging.error(f"Erreur lors de la sauvegarde des règles de maillage: {str(e)}")
    
    def get_rule(self, source_type: str, target_type: str) -> Dict[str, Any]:
        """
        Récupère la règle de maillage entre deux types d'articles.
        
        Args:
            source_type: Type de l'article source
            target_type: Type de l'article cible
            
        Returns:
            Dictionnaire contenant la règle de maillage, ou règle par défaut si non trouvée
        """
        # Règle par défaut
        default_rule = {
            "min_links": 0,
            "max_links": 5,
            "min_similarity": 0.3,
            "enabled": True
        }
        
        # Vérifier si la règle existe
        if source_type in self.rules and target_type in self.rules[source_type]:
            return self.rules[source_type][target_type]
        
        return default_rule
    
    def set_rule(self, source_type: str, target_type: str, rule: Dict[str, Any]) -> None:
        """
        Définit une règle de maillage entre deux types d'articles.
        
        Args:
            source_type: Type de l'article source
            target_type: Type de l'article cible
            rule: Dictionnaire contenant la règle de maillage
        """
        if source_type not in self.rules:
            self.rules[source_type] = {}
        
        self.rules[source_type][target_type] = rule
        
        logging.info(f"Règle de maillage définie: {source_type} -> {target_type}")
    
    def get_all_rules(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Récupère toutes les règles de maillage.
        
        Returns:
            Dictionnaire contenant toutes les règles de maillage
        """
        return self.rules
    
    def get_all_types(self) -> List[str]:
        """
        Récupère tous les types d'articles définis dans les règles.
        
        Returns:
            Liste des types d'articles
        """
        types = set()
        
        # Ajouter les types sources
        for source_type in self.rules:
            types.add(source_type)
            
            # Ajouter les types cibles
            for target_type in self.rules[source_type]:
                types.add(target_type)
        
        return sorted(list(types))
    
    def is_link_allowed(self, source_type: str, target_type: str) -> bool:
        """
        Vérifie si un lien est autorisé entre deux types d'articles.
        
        Args:
            source_type: Type de l'article source
            target_type: Type de l'article cible
            
        Returns:
            True si le lien est autorisé, False sinon
        """
        rule = self.get_rule(source_type, target_type)
        return rule.get("enabled", True)
