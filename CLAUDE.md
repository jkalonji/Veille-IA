##  Vision du Projet
Veille-IA est un outil automatisé (ou un repo de curation) structuré pour agréger, traiter et diffuser les actualités liées à l'Intelligence Artificielle. Le but est la clarté, la rapidité de lecture et l'automatisation.

##  Commandes Utiles
- **Installation :** `npm install` ou `pip install -r requirements.txt` (selon la stack détectée)
- **Lancement :** `npm start` ou `python main.py`
- **Tests :** `npm test` ou `pytest`
- **Linting :** `npm run lint` ou `flake8`

##  Règles de Style & Standards
- **Langue :** Documentation en Français, code/commentaires en Anglais.
- **Format des données :** Les sources de veille doivent être au format JSON ou YAML dans le dossier `data/`.
- **Markdown :** Les rapports générés doivent utiliser des headers clairs, des emojis pour catégoriser les news et des liens sources systématiques.
- **Git :** Commits courts et explicites (ex: `feat: add scraper for OpenAI blog`).

##  Intentions de "Vibe"
- Prioriser la simplicité : "Don't over-engineer".
- Toujours vérifier la validité des URLs lors de l'ajout de sources.
- Garder une structure de fichier plate autant que possible pour faciliter la navigation.

##  Structure Clé
- `/src` : Scripts de scraping et traitement.
- `/data` : Liste des flux RSS/Twitter/Blogs à surveiller.
- `/output` : Rapports de veille générés.