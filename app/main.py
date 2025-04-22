"""
Application de génération de plan de maillage interne pour articles pré-publication.
Cette application permet de générer un plan de maillage basé sur les titres d'articles
qui ne sont pas encore publiés.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

# Configuration du logging
logging.basicConfig(
    level=logging.DEBUG,  # Changer INFO en DEBUG pour plus de détails
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Déterminer le chemin de base de l'application
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Création des dossiers nécessaires s'ils n'existent pas
os.makedirs(os.path.join(BASE_DIR, "data/jobs"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "results"), exist_ok=True)

# Import des modèles
from models.similarity import TitleSimilarityAnalyzer
from models.linking_rules import LinkingRulesManager
from models.pagerank import calculate_pagerank, calculate_pagerank_with_suggestions



# Initialiser le gestionnaire de règles de maillage
rules_manager = LinkingRulesManager()

# Initialisation de l'application FastAPI
app = FastAPI(
    title="Maillage Pré-Publication",
    description="Outil de génération de plan de maillage interne pour articles pré-publication",
    version="1.0.0"
)

# Configuration des templates et des fichiers statiques
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Dictionnaire pour stocker les jobs en cours
jobs = {}

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Page d'accueil"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/rules", response_class=HTMLResponse)
async def rules_page(request: Request):
    """Page de configuration des règles de maillage"""
    # Récupérer tous les types d'articles et les règles
    types = rules_manager.get_all_types()
    rules = rules_manager.get_all_rules()
    
    # Si aucun type n'est défini, ajouter des types par défaut
    if not types:
        default_types = ["Article", "Guide", "Tutoriel", "Actualité"]
        for type_name in default_types:
            types.append(type_name)
    
    return templates.TemplateResponse("rules.html", {
        "request": request,
        "types": types,
        "rules": rules
    })

@app.post("/rules/add_type")
async def add_type(request: Request, new_type: str = Form(...)):
    """Ajouter un nouveau type d'article"""
    # Récupérer tous les types existants
    types = rules_manager.get_all_types()
    
    # Vérifier si le type existe déjà
    if new_type in types:
        return JSONResponse(
            status_code=400,
            content={"error": f"Le type '{new_type}' existe déjà"}
        )
    
    # Ajouter une règle par défaut pour ce type
    default_rule = {
        "min_links": 0,
        "max_links": 5,
        "min_similarity": 0.3,
        "enabled": True
    }
    rules_manager.set_rule(new_type, new_type, default_rule)
    
    # Sauvegarder les règles dans un fichier
    rules_file = os.path.join(BASE_DIR, "data/default_rules.json")
    rules_manager.save_rules(rules_file)
    
    # Rediriger vers la page des règles
    return JSONResponse({"redirect": "/rules", "message": f"Type '{new_type}' ajouté avec succès"})

@app.post("/rules/update")
async def update_rule(
    request: Request,
    source_type: str = Form(...),
    target_type: str = Form(...),
    min_links: int = Form(...),
    max_links: int = Form(...),
    min_similarity: float = Form(...),
    enabled: bool = Form(False)
):
    """Mettre à jour une règle de maillage"""
    
    # Créer la règle
    rule = {
        "min_links": min_links,
        "max_links": max_links,
        "min_similarity": min_similarity,
        "enabled": enabled
    }
    
    # Mettre à jour la règle
    rules_manager.set_rule(source_type, target_type, rule)
    
    # Sauvegarder les règles dans un fichier
    rules_file = os.path.join(BASE_DIR, "data/default_rules.json")
    rules_manager.save_rules(rules_file)
    
    # Rediriger vers la page des règles
    return JSONResponse({"redirect": "/rules", "message": f"Règle '{source_type} → {target_type}' mise à jour"})

@app.post("/rules/save")
async def save_rules(request: Request):
    """Sauvegarder toutes les règles"""
    # Sauvegarder les règles dans un fichier
    rules_file = os.path.join(BASE_DIR, "data/default_rules.json")
    rules_manager.save_rules(rules_file)
    
    # Rediriger vers la page des règles
    return JSONResponse({"redirect": "/rules", "message": "Règles sauvegardées avec succès"})

@app.get("/rules/export")
async def export_rules():
    """Exporter les règles au format JSON"""
    # Sauvegarder les règles dans un fichier temporaire
    rules_file = os.path.join(BASE_DIR, "data/export_rules.json")
    rules_manager.save_rules(rules_file)
    
    # Renvoyer le fichier
    return FileResponse(
        path=rules_file,
        filename="maillage_rules.json",
        media_type="application/json"
    )

@app.post("/rules/import")
async def import_rules(rules_file: UploadFile = File(...)):
    """Importer des règles depuis un fichier JSON"""
    try:
        # Lire le contenu du fichier
        content = await rules_file.read()
        
        # Sauvegarder le fichier temporairement
        temp_file = os.path.join(BASE_DIR, "data/import_rules.json")
        with open(temp_file, "wb") as f:
            f.write(content)
        
        # Charger les règles
        rules_manager.load_rules(temp_file)
        
        # Sauvegarder les règles dans le fichier par défaut
        default_rules_file = os.path.join(BASE_DIR, "data/default_rules.json")
        rules_manager.save_rules(default_rules_file)
        
        # Rediriger vers la page des règles
        return JSONResponse({"redirect": "/rules", "message": "Règles importées avec succès"})
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"Erreur lors de l'importation des règles: {str(e)}"}
        )

@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    """Page d'upload des fichiers"""
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/upload")
async def upload_files(
    background_tasks: BackgroundTasks,
    titles_file: UploadFile = File(...),
    content_file: Optional[UploadFile] = File(None),
    rules_file: Optional[UploadFile] = File(None)
):
    try:
        logging.info(f"Début de l'upload - Fichier des titres: {titles_file.filename}, Taille: {titles_file.size} bytes")
        logging.debug(f"Métadonnées du fichier des titres: {titles_file.__dict__}")
        if content_file:
            logging.info(f"Fichier de contenu: {content_file.filename}, Taille: {content_file.size} bytes")
            logging.debug(f"Métadonnées du fichier de contenu: {content_file.__dict__}")
        if rules_file:
            logging.info(f"Fichier de règles: {rules_file.filename}, Taille: {rules_file.size} bytes")
            logging.debug(f"Métadonnées du fichier de règles: {rules_file.__dict__}")
    except Exception as e:
        logging.error(f"Erreur lors de la lecture des métadonnées des fichiers: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=400,
            content={"error": f"Erreur lors de la lecture des métadonnées des fichiers: {str(e)}"})
    """
    Endpoint pour uploader les fichiers nécessaires à l'analyse:
    - Fichier des titres (obligatoire)
    - Fichier de contenu (optionnel)
    - Fichier de règles de maillage (optionnel)
    """
    try:
        # Vérifier les extensions de fichiers
        logging.info(f"Vérification de l'extension du fichier des titres: {titles_file.filename}")
        if not titles_file.filename or not titles_file.filename.endswith(('.xlsx', '.xls')):
            error_msg = f"Le fichier des titres doit être au format Excel (.xlsx ou .xls). Fichier reçu: {titles_file.filename}"
            logging.error(error_msg)
            return JSONResponse(
                status_code=400,
                content={"error": error_msg})
        
        if content_file and content_file.filename and not content_file.filename.endswith(('.xlsx', '.xls')):
            error_msg = f"Le fichier de contenu doit être au format Excel (.xlsx ou .xls). Fichier reçu: {content_file.filename}"
            logging.error(error_msg)
            return JSONResponse(
                status_code=400,
                content={"error": error_msg})
        
        if rules_file and rules_file.filename and not rules_file.filename.endswith('.json'):
            error_msg = f"Le fichier de règles doit être au format JSON (.json). Fichier reçu: {rules_file.filename}"
            logging.error(error_msg)
            return JSONResponse(
                status_code=400,
                content={"error": error_msg})
        
        # Générer un ID unique pour ce job
        job_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Créer un dossier pour ce job
        job_dir = os.path.join(BASE_DIR, f"data/jobs/{job_id}")
        os.makedirs(job_dir, exist_ok=True)
        
        # Sauvegarder les fichiers
        titles_path = os.path.join(job_dir, "titles.xlsx")
        content = await titles_file.read()
        with open(titles_path, "wb") as buffer:
            buffer.write(content)
            
        # Créer le répertoire data s'il n'existe pas
        os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
        
        # Vérifier que le fichier Excel est valide
        logging.info(f"Vérification de la validité du fichier Excel: {titles_path}")
        try:
            # Essayer d'abord avec openpyxl
            try:
                logging.info("Tentative de lecture avec engine='openpyxl'")
                df = pd.read_excel(titles_path, engine='openpyxl', nrows=5)
                logging.debug(f"Lecture réussie avec openpyxl. Taille du DataFrame: {df.shape}")
            except Exception as openpyxl_error:
                logging.warning(f"Erreur avec openpyxl: {str(openpyxl_error)}", exc_info=True)
                # Si ça échoue, essayer avec xlrd
                logging.info("Tentative de lecture avec engine='xlrd'")
                try:
                    df = pd.read_excel(titles_path, engine='xlrd', nrows=5)
                    logging.debug(f"Lecture réussie avec xlrd. Taille du DataFrame: {df.shape}")
                except Exception as xlrd_error:
                    logging.error(f"Erreur avec xlrd: {str(xlrd_error)}", exc_info=True)
                    raise Exception(f"Impossible de lire le fichier Excel avec openpyxl ({str(openpyxl_error)}) ou xlrd ({str(xlrd_error)})")
            
            logging.info(f"Fichier Excel lu avec succès. Colonnes détectées: {list(df.columns)}")
            
            # Vérifier que les colonnes requises sont présentes
            # Convertir les noms de colonnes en minuscules pour une comparaison insensible à la casse
            original_columns = list(df.columns)
            logging.debug(f"Colonnes originales: {original_columns}")
            df.columns = [str(col).strip().lower() for col in df.columns]
            logging.debug(f"Colonnes après conversion en minuscules: {list(df.columns)}")
            required_columns = ["titre", "type"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            logging.debug(f"Colonnes manquantes: {missing_columns}")
            
            if missing_columns:
                # Essayer de trouver des colonnes similaires
                column_mapping = {}
                for col in df.columns:
                    for req_col in required_columns:
                        if req_col in col or col in req_col:
                            column_mapping[req_col] = col
                            logging.debug(f"Colonne similaire trouvée: {req_col} -> {col}")
                
                if len(column_mapping) == len(required_columns):
                    logging.info(f"Colonnes similaires trouvées: {column_mapping}")
                    # Renommer les colonnes
                    df = df.rename(columns={v: k for k, v in column_mapping.items()})
                    logging.debug(f"Colonnes après renommage: {list(df.columns)}")
                    # Sauvegarder le fichier avec les colonnes renommées
                    try:
                        df.to_excel(titles_path, index=False)
                        logging.info("Fichier sauvegardé avec les colonnes renommées")
                    except Exception as save_error:
                        logging.error(f"Erreur lors de la sauvegarde du fichier avec les colonnes renommées: {str(save_error)}", exc_info=True)
                        os.remove(titles_path)  # Supprimer le fichier invalide
                        return JSONResponse(
                            status_code=400,
                            content={"error": f"Erreur lors de la sauvegarde du fichier: {str(save_error)}"})
                else:
                    os.remove(titles_path)  # Supprimer le fichier invalide
                    error_msg = f"Colonnes manquantes dans le fichier des titres: {missing_columns}. Colonnes requises: {required_columns}. Colonnes trouvées: {list(df.columns)}"
                    logging.error(error_msg)
                    return JSONResponse(
                        status_code=400,
                        content={"error": error_msg})
            
            # Afficher un aperçu des données pour le débogage
            logging.info(f"Aperçu des données:\n{df.head().to_string()}")
                    
        except Exception as e:
            if os.path.exists(titles_path):
                os.remove(titles_path)  # Supprimer le fichier invalide
            error_msg = f"Le fichier des titres n'est pas un fichier Excel valide: {str(e)}"
            logging.error(error_msg, exc_info=True)
            # Afficher plus d'informations sur le fichier
            try:
                file_size = os.path.getsize(titles_path) if os.path.exists(titles_path) else "fichier supprimé"
                logging.debug(f"Taille du fichier avant suppression: {file_size} bytes")
                with open(titles_path, "rb") as f:
                    header = f.read(100).hex()
                    logging.debug(f"En-tête du fichier (100 premiers octets): {header}")
            except Exception as debug_error:
                logging.debug(f"Erreur lors du débogage du fichier: {str(debug_error)}")
            return JSONResponse(
                status_code=400,
                content={"error": error_msg})
        
        content_path = None
        if content_file:
            content_path = os.path.join(job_dir, "content.xlsx")
            content = await content_file.read()
            with open(content_path, "wb") as buffer:
                buffer.write(content)
            
            # Vérifier que le fichier Excel est valide
            try:
                pd.read_excel(content_path, engine='openpyxl', nrows=5)
            except Exception as e:
                os.remove(content_path)  # Supprimer le fichier invalide
                content_path = None
                logging.warning(f"Le fichier de contenu n'est pas valide et sera ignoré: {str(e)}")
        
        rules_path = None
        if rules_file:
            rules_path = os.path.join(job_dir, "rules.json")
            content = await rules_file.read()
            with open(rules_path, "wb") as buffer:
                buffer.write(content)
            
            # Vérifier que le fichier JSON est valide
            try:
                with open(rules_path, 'r') as f:
                    json.load(f)
            except Exception as e:
                os.remove(rules_path)  # Supprimer le fichier invalide
                rules_path = None
                logging.warning(f"Le fichier de règles n'est pas valide et sera ignoré: {str(e)}")
                
    except Exception as e:
        logging.error(f"Erreur lors du traitement des fichiers: {str(e)}", exc_info=True)
        # Ajouter plus de détails sur l'erreur
        import traceback
        error_details = traceback.format_exc()
        logging.debug(f"Détails de l'erreur:\n{error_details}")
        
        # Vérifier si le job_dir existe et contient des fichiers
        if 'job_dir' in locals():
            try:
                if os.path.exists(job_dir):
                    files_in_dir = os.listdir(job_dir)
                    logging.debug(f"Fichiers dans {job_dir}: {files_in_dir}")
            except Exception as dir_error:
                logging.debug(f"Erreur lors de la vérification du répertoire: {str(dir_error)}")
        
        return JSONResponse(
            status_code=500,
            content={"error": f"Erreur lors du traitement des fichiers: {str(e)}", "details": error_details})
    
    # Créer un job
    job = {
        "id": job_id,
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "titles_file": titles_path,
        "content_file": content_path,
        "rules_file": rules_path,
        "progress": 0,
        "message": "Job créé, en attente de traitement"
    }
    
    logging.info(f"Job créé avec succès: {job_id}")
    
    # Sauvegarder le job
    jobs[job_id] = job
    with open(f"{job_dir}/job.json", "w") as f:
        json.dump(job, f)
    
    # Lancer l'analyse en arrière-plan
    background_tasks.add_task(run_analysis, job_id, titles_path, content_path, rules_path)
    
    return {"job_id": job_id, "status": "pending"}

@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Récupère le statut d'un job"""
    if job_id not in jobs:
        # Essayer de charger le job depuis le fichier
        job_file = os.path.join(BASE_DIR, f"data/jobs/{job_id}/job.json")
        if os.path.exists(job_file):
            with open(job_file, "r") as f:
                job = json.load(f)
                jobs[job_id] = job
        else:
            raise HTTPException(status_code=404, detail="Job non trouvé")
    
    return jobs[job_id]

@app.get("/results/{job_id}")
async def get_results(job_id: str):
    """Récupère les résultats d'un job"""
    if job_id not in jobs:
        # Essayer de charger le job depuis le fichier
        job_file = os.path.join(BASE_DIR, f"data/jobs/{job_id}/job.json")
        if os.path.exists(job_file):
            with open(job_file, "r") as f:
                job = json.load(f)
                jobs[job_id] = job
        else:
            raise HTTPException(status_code=404, detail="Job non trouvé")
    
    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job non terminé")
    
    result_file = job.get("result_file")
    if not result_file or not os.path.exists(result_file):
        raise HTTPException(status_code=404, detail="Résultats non trouvés")
    
    return FileResponse(result_file, filename=f"maillage_prepub_{job_id}.xlsx")

@app.get("/report/{job_id}", response_class=HTMLResponse)
async def view_report(request: Request, job_id: str):
    """Affiche le rapport d'analyse"""
    if job_id not in jobs:
        # Essayer de charger le job depuis le fichier
        job_file = os.path.join(BASE_DIR, f"data/jobs/{job_id}/job.json")
        if os.path.exists(job_file):
            with open(job_file, "r") as f:
                job = json.load(f)
                jobs[job_id] = job
        else:
            raise HTTPException(status_code=404, detail="Job non trouvé")
    
    job = jobs[job_id]
    return templates.TemplateResponse(
        "report.html", 
        {"request": request, "job": job, "job_id": job_id}
    )

@app.get("/api/suggestions/{job_id}")
async def get_suggestions(job_id: str):
    """Récupère les suggestions de maillage pour un job"""
    if job_id not in jobs:
        # Essayer de charger le job depuis le fichier
        job_file = os.path.join(BASE_DIR, f"data/jobs/{job_id}/job.json")
        if os.path.exists(job_file):
            with open(job_file, "r") as f:
                job = json.load(f)
                jobs[job_id] = job
        else:
            raise HTTPException(status_code=404, detail="Job non trouvé")
    
    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job non terminé")
    
    result_file = job.get("result_file")
    if not result_file or not os.path.exists(result_file):
        raise HTTPException(status_code=404, detail="Résultats non trouvés")
    
    # Charger les suggestions
    suggestions_df = pd.read_excel(result_file, engine='openpyxl')
    
    # Convertir en format JSON
    suggestions = suggestions_df.to_dict(orient="records")
    
    return {"suggestions": suggestions}

@app.get("/api/pagerank/{job_id}")
async def get_pagerank(job_id: str):
    """Calcule et retourne les scores PageRank pour les articles"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job non trouvé")

    job = jobs[job_id]
    
    # Vérifier que les fichiers nécessaires existent
    titles_file = job.get("titles_file")
    result_file = job.get("result_file")
    
    if not titles_file or not os.path.exists(titles_file):
        raise HTTPException(status_code=404, detail="Fichier de titres non trouvé")
        
    try:
        # Charger les titres
        titles_df = pd.read_excel(titles_file)
        
        # Charger les suggestions de liens si disponibles
        suggested_links_df = None
        if result_file and os.path.exists(result_file) and job["status"] == "completed":
            suggested_links_df = pd.read_excel(result_file, engine='openpyxl')
            # Convertir au format attendu
            suggested_links_df = pd.DataFrame({
                "Source": suggested_links_df["source_title"],
                "Destination": suggested_links_df["target_title"]
            })
        
        # Créer un DataFrame de liens vide (pas de liens existants pour des articles non publiés)
        existing_links_df = pd.DataFrame(columns=["Source", "Destination"])
        
        # Calculer le PageRank actuel et optimisé
        if suggested_links_df is not None:
            current_pagerank, optimized_pagerank = calculate_pagerank_with_suggestions(
                existing_links_df, suggested_links_df
            )
            
            # Préparer les résultats
            results = {
                "current": {
                    title: {"pagerank": score, "rank": i+1} 
                    for i, (title, score) in enumerate(sorted(current_pagerank.items(), key=lambda x: x[1], reverse=True))
                },
                "optimized": {
                    title: {"pagerank": score, "rank": i+1} 
                    for i, (title, score) in enumerate(sorted(optimized_pagerank.items(), key=lambda x: x[1], reverse=True))
                },
                "improvement": {}
            }
            
            # Calculer les améliorations
            for title in current_pagerank:
                if title in optimized_pagerank:
                    current_score = current_pagerank[title]
                    optimized_score = optimized_pagerank[title]
                    improvement_pct = ((optimized_score - current_score) / current_score) * 100 if current_score > 0 else 0
                    
                    results["improvement"][title] = {
                        "absolute": optimized_score - current_score,
                        "percentage": round(improvement_pct, 2),
                        "current_rank": results["current"][title]["rank"],
                        "optimized_rank": results["optimized"][title]["rank"],
                        "rank_change": results["current"][title]["rank"] - results["optimized"][title]["rank"]
                    }
            
            return results
        else:
            # Seulement le PageRank actuel (qui sera uniforme sans liens existants)
            current_pagerank = calculate_pagerank(existing_links_df)
            
            # Préparer les résultats
            results = {
                "current": {
                    title: {"pagerank": score, "rank": i+1} 
                    for i, (title, score) in enumerate(sorted(current_pagerank.items(), key=lambda x: x[1], reverse=True))
                }
            }
            
            return results
            
    except Exception as e:
        logging.error(f"Erreur lors du calcul du PageRank: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur lors du calcul du PageRank: {str(e)}")

async def run_analysis(job_id: str, titles_file: str, content_file: Optional[str], rules_file: Optional[str]):
    """Fonction pour exécuter l'analyse en arrière-plan"""
    try:
        # Mettre à jour le statut du job
        job = jobs[job_id]
        job["status"] = "running"
        job["start_time"] = datetime.now().isoformat()
        job["message"] = "Analyse en cours..."
        job["progress"] = 10
        
        # Sauvegarder le job
        with open(os.path.join(BASE_DIR, f"data/jobs/{job_id}/job.json"), "w") as f:
            json.dump(job, f)
        
        # Charger les titres
        try:
            titles_df = pd.read_excel(titles_file, engine='openpyxl')
        except Exception as e:
            logging.warning(f"Erreur lors de la lecture avec openpyxl: {str(e)}")
            titles_df = pd.read_excel(titles_file, engine='xlrd')
        
        # Normaliser les noms de colonnes
        titles_df.columns = [str(col).strip().lower() for col in titles_df.columns]
        
        # Vérifier que les colonnes requises sont présentes
        required_columns = ["titre", "type"]
        for col in required_columns:
            if col not in titles_df.columns:
                # Chercher une colonne similaire
                for existing_col in titles_df.columns:
                    if col in existing_col or existing_col in col:
                        titles_df = titles_df.rename(columns={existing_col: col})
                        break
        
        logging.info(f"Fichier de titres chargé: {len(titles_df)} articles")
        logging.info(f"Colonnes disponibles: {list(titles_df.columns)}")
        job["progress"] = 20
        job["message"] = f"Fichier de titres chargé: {len(titles_df)} articles"
        
        # Charger le contenu si disponible
        content_df = None
        if content_file and os.path.exists(content_file):
            try:
                content_df = pd.read_excel(content_file, engine='openpyxl')
                logging.info(f"Fichier de contenu chargé: {len(content_df)} articles")
                job["progress"] = 30
                job["message"] = f"Fichier de contenu chargé: {len(content_df)} articles"
            except Exception as e:
                logging.warning(f"Impossible de charger le fichier de contenu: {str(e)}")
                job["progress"] = 30
                job["message"] = f"Avertissement: Impossible de charger le fichier de contenu, l'analyse continuera sans lui"
        
        # Charger les règles si disponibles
        linking_rules = None
        if rules_file and os.path.exists(rules_file):
            try:
                # Utiliser le gestionnaire de règles pour charger le fichier
                rules_manager.load_rules(rules_file)
                linking_rules = rules_manager.get_all_rules()
                logging.info("Règles de maillage chargées")
                job["progress"] = 40
                job["message"] = "Règles de maillage chargées"
            except Exception as e:
                logging.warning(f"Erreur lors du chargement des règles de maillage: {str(e)}")
                # Essayer de charger directement le fichier JSON comme avant
                with open(rules_file, "r") as f:
                    linking_rules = json.load(f)
                logging.info("Règles de maillage chargées (méthode alternative)")
                job["progress"] = 40
                job["message"] = "Règles de maillage chargées"
        
        # Initialiser l'analyseur de similarité
        analyzer = TitleSimilarityAnalyzer()
        job["progress"] = 50
        job["message"] = "Analyseur de similarité initialisé"
        
        # Analyser les titres
        job["progress"] = 60
        job["message"] = "Analyse des titres en cours..."
        
        # Préparer les données pour l'analyse
        # Utiliser les noms de colonnes en minuscules pour être cohérent avec le traitement dans upload_files
        titles = titles_df["titre"].tolist()
        types = titles_df["type"].tolist() if "type" in titles_df.columns else ["Article"] * len(titles)
        
        # Calculer la matrice de similarité
        similarity_matrix = analyzer.calculate_similarity_matrix(titles)
        job["progress"] = 70
        job["message"] = "Matrice de similarité calculée"
        
        # Générer les suggestions de maillage
        suggestions = analyzer.generate_suggestions(
            titles=titles,
            types=types,
            similarity_matrix=similarity_matrix,
            min_similarity=0.3,
            linking_rules=linking_rules,
            content_df=content_df
        )
        job["progress"] = 80
        job["message"] = "Suggestions de maillage générées"
        
        # Créer un DataFrame avec les suggestions
        suggestions_df = pd.DataFrame(suggestions)
        
        # Sauvegarder les résultats
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(BASE_DIR, f"results/maillage_prepub_{job_id}.xlsx")
        suggestions_df.to_excel(result_file, index=False)
        job["progress"] = 90
        job["message"] = "Résultats sauvegardés"
        
        # Mettre à jour le statut du job
        job["status"] = "completed"
        job["end_time"] = datetime.now().isoformat()
        job["result_file"] = result_file
        job["progress"] = 100
        job["message"] = "Analyse terminée avec succès"
        
        # Sauvegarder le job
        with open(os.path.join(BASE_DIR, f"data/jobs/{job_id}/job.json"), "w") as f:
            json.dump(job, f)
        
        logging.info(f"Analyse terminée pour le job {job_id}")
        
    except Exception as e:
        # En cas d'erreur, mettre à jour le statut du job
        job = jobs[job_id]
        job["status"] = "failed"
        job["end_time"] = datetime.now().isoformat()
        job["message"] = f"Erreur lors de l'analyse: {str(e)}"
        
        # Sauvegarder le job
        with open(os.path.join(BASE_DIR, f"data/jobs/{job_id}/job.json"), "w") as f:
            json.dump(job, f)
        
        logging.error(f"Erreur lors de l'analyse pour le job {job_id}: {str(e)}", exc_info=True)

if __name__ == "__main__":
    import sys
    import os
    
    # Déterminer si nous sommes exécutés depuis le répertoire app ou depuis le répertoire racine
    current_dir = os.path.basename(os.getcwd())
    
    # Utiliser le port depuis la variable d'environnement ou 8004 par défaut
    port = int(os.environ.get("PORT", 8004))
    
    # Démarrer l'application
    print("Démarrage de l'application Maillage Pré-Publication...")
    print(f"Accédez à l'application dans votre navigateur: http://localhost:{port}")
    print("Appuyez sur CTRL+C pour arrêter le serveur")
    
    # Lancer l'application avec la bonne configuration selon le répertoire d'exécution
    if current_dir == "app":
        # Exécuté depuis le répertoire app
        uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
    else:
        # Exécuté depuis un autre répertoire (probablement le répertoire racine)
        uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
