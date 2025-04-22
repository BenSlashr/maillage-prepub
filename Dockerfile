FROM python:3.10-slim

WORKDIR /app

# Installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier l'application
COPY ./app ./app

# Définir les variables d'environnement
ENV HOST=0.0.0.0
ENV PORT=8004

# Lancer l'application via main.py
CMD ["python", "-m", "app.main"]

