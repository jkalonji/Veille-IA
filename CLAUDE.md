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

## Catégories d'articles
Six catégories actives (la catégorie `Geopolitique` a été fusionnée dans `Politique / Regulation`) :
- `Innovation / Tech` 🚀
- `Politique / Regulation` ⚖️ — inclut géopolitique IA, diplomatie tech, régulation internationale
- `Business / Industrie` 💼
- `Societe / Ethique` 🤝
- `Recherche Academique` 🎓
- `Drama / Controverses` 💥

Les articles déjà en base avec `Geopolitique` sont remappés automatiquement à l'affichage via `_CATEGORY_ALIAS` dans `dashboard.py`.

## Méthodologie de détection des sujets vifs

Un article est tagué `hot_topic = True` si son titre/description contient au moins un keyword issu de la liste enrichie ci-dessous.

### Sources de keywords (ordre de priorité)

**1. Google Trends (`fetch_hot_keywords`)**
- Requête pytrends sur `"generative AI"`, fenêtre 7 jours
- Récupère les top & rising related queries
- Fallback statique `HOT_KEYWORDS_FALLBACK` si pytrends indisponible

**2. HN Debate (`_fetch_hn_debate_keywords`)**
- Articles HN (Hacker News Algolia API) sur les 48 dernières heures
- Filtre : `num_comments > 20` — le nombre de commentaires est un proxy de débat actif
- Requêtes : `AI`, `LLM`, `OpenAI`, `Claude`, `machine learning`, `AGI`
- Mots-clés extraits des titres des stories les plus discutées

**3. GitHub Trending (`_fetch_github_trending_keywords`)**
- GitHub Search API : repos pushés dans les 24h, topics `artificial-intelligence`, `large-language-model`, `llm`
- Triés par stars desc — un repo qui explose en stars = un paper ou outil viral
- Mots-clés extraits du nom + description du repo

**4. DB Self-bootstrap (`_fetch_db_trending_keywords`)**
- Requête Supabase : titres de tous les articles collectés aujourd'hui
- Mots apparaissant dans ≥ 3 titres différents = signal de saturation éditoriale
- Auto-alimenté : plus on collecte de sources, plus ce signal est précis

### Groupage visuel dans le dashboard
Chaque article hot est assigné à un groupe selon sa source prioritaire (ordre : hn > trends > github > db).
Les groupes s'affichent dans cet ordre avec des couleurs et icônes distinctes :
- 💬 **Sujets en débat** (orange `#ff6600`) — HN >20 commentaires
- 🔮 **Tendances montantes** (violet `#7b2ff7`) — Google Trends rising
- ⭐ **Tech viral** (vert `#238636`) — GitHub trending repos
- 📡 **Signal éditorial** (cyan `#0e7490`) — auto-bootstrap DB

La colonne `hot_source` en Supabase stocke les sources sous forme de chaîne pipe-séparée (ex: `"trends|hn"`).

**Migration SQL à exécuter une fois en Supabase :**
```sql
ALTER TABLE articles ADD COLUMN IF NOT EXISTS hot_source TEXT DEFAULT '';
```
Le code est backward-compatible : si la colonne n'existe pas, l'upsert se fait sans elle (fallback automatique).

### Seuil Supa Hot Topic
Un article `hot_topic` est promu `supa_hot` si :
- `mention_count > 5` (≥ 5 autres articles du jour partagent ≥ 2 mots-clés du titre)
- ET `published == today`

Visuellement : fond dégradé rouge-orange, badge `🌋 SUPA HOT · N mentions`, affiché en premier.

### Règle d'évolution des keywords
Si un sujet majeur n'est pas capté par les sources dynamiques, l'ajouter manuellement dans `HOT_KEYWORDS_FALLBACK` dans `main.py`. Cette liste est le filet de sécurité.