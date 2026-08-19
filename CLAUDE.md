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
Chaque article hot est catégorisé par **Groq** via le champ `hot_reason` (classification sémantique du contenu).
Les 4 onglets toujours visibles :
- 💬 **Sujets en débat** (`debat`) — controverse, opinions polarisées, drama, licenciements, procès
- ⭐ **Tech viral** (`tech`) — lancement modèle, outil dev, benchmark, sortie produit
- 📡 **Sujets de société** (`societe`) — régulation, emploi, éthique, droits, impact sociétal
- 🔮 **Tendances montantes** (`tendance`) — concept émergent, nouveau paradigme, recherche en hausse

`hot_source` reste en base comme métadonnée de détection (Trends/HN/GitHub/DB), mais n'est plus utilisé pour le groupage — c'est `hot_reason` qui pilote les onglets.

**Migration SQL à exécuter une fois en Supabase :**
```sql
ALTER TABLE articles ADD COLUMN IF NOT EXISTS hot_source TEXT DEFAULT '';
ALTER TABLE articles ADD COLUMN IF NOT EXISTS hot_reason TEXT DEFAULT '';
```
Le code est backward-compatible (fallback automatique si colonnes absentes).

> ⚠️ **Note (2026-07-20) : la section ci-dessus (sources de keywords, 4 signaux, onglets `debat/tech/societe/tendance`) décrit un système qui a depuis été remplacé** par un clustering dynamique par n-grammes (`extract_topic_clusters`/`name_topic_clusters` dans `main.py`, commit `5bd5f54`) — voir `SPECS_MULTIDOMAINE.md` §6 pour la description à jour. `hot_reason` est désormais un label de cluster libre, pas une catégorie fixe. `dashboard.py` traite les anciennes valeurs (`debat`/`tech`/`societe`/`tendance`) comme obsolètes via `_OLD_HOT_REASONS`.

**Migration SQL — extension multi-domaines (Phase 0, voir `SPECS_MULTIDOMAINE.md`) :**
```sql
ALTER TABLE articles ADD COLUMN IF NOT EXISTS domain TEXT DEFAULT 'ia';

CREATE TABLE IF NOT EXISTS market_data (
    id BIGSERIAL PRIMARY KEY,
    domain TEXT NOT NULL,
    symbol TEXT NOT NULL,
    label TEXT NOT NULL,
    value NUMERIC NOT NULL,
    unit TEXT,
    variation_pct NUMERIC,
    source TEXT,
    collected_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```
Le code est backward-compatible pour `domain` (fallback automatique dans `save_to_supabase` si la colonne est absente). `market_data` n'est pas encore utilisée par le pipeline (arrivera en Phase 2).

### Seuil Supa Hot Topic
Un article `hot_topic` est promu `supa_hot` si :
- `mention_count > 5` (≥ 5 autres articles du jour partagent ≥ 2 mots-clés du titre)
- ET `published == today`

Visuellement : fond dégradé rouge-orange, badge `🌋 SUPA HOT · N mentions`, affiché en premier.

### Règle d'évolution des keywords
Si un sujet majeur n'est pas capté par les sources dynamiques, l'ajouter manuellement dans `HOT_KEYWORDS_FALLBACK` dans `main.py`. Cette liste est le filet de sécurité.

## Suivi d'histoires sur plusieurs jours (story tracking)

Les clusters hot (`extract_topic_clusters`/`name_topic_clusters`) sont recalculés **chaque jour, à partir des seuls articles du jour** — sans lien avec les clusters de la veille. Le "suivi d'histoires" ajoute une identité persistante (`stories`) qui traverse les runs quotidiens, pour qu'une actu qui dure plusieurs jours (annonce → réactions → conséquences) reste une seule histoire au lieu de N clusters isolés au fil des jours.

### Fonctionnement (dans `main.py`, après le clustering quotidien)
1. Pour chaque domaine, après `name_topic_clusters`, on récupère les stories `open` de ce domaine (`_fetch_open_stories`).
2. `match_clusters_to_stories` demande à Groq (un seul appel groupé, même pattern que le nommage des clusters) si chaque cluster du jour continue une story ouverte ou en démarre une nouvelle. En cas d'échec Groq, repli automatique sur `_match_clusters_ngram` (recouvrement de bigrammes/trigrammes avec les titres récents de la story — même logique que le clustering intra-jour).
3. `_apply_story_matches` met à jour (`last_seen`, `article_count`, `recent_titles`) ou crée la story en Supabase, et renvoie le `story_id` à appliquer à chaque article matché.
4. `_close_stale_stories` referme (`status = 'closed'`) les stories sans nouvel article depuis `STORY_IDLE_DAYS` (4 jours par défaut).

### Dashboard
- `_build_story_timelines` (dashboard.py) regroupe les articles déjà chargés par `story_id` et ne garde que les stories actives sur **≥ 2 jours distincts** — une story mono-jour est déjà visible dans les onglets Hot Articles, la dupliquer ici n'apporterait rien.
- Rendu dans un nouveau panneau "📖 Suivi d'histoires" : liste des stories (actives aujourd'hui en premier, puis par activité récente), chacune dépliable en timeline jour par jour. Présent à la fois côté Streamlit (`_render_stories`, `st.expander`) et export statique GitHub Pages (`_stories_html`, `<details>` natif, zéro JS).

### Migration SQL à exécuter une fois en Supabase
```sql
CREATE TABLE IF NOT EXISTS stories (
    id BIGSERIAL PRIMARY KEY,
    domain TEXT NOT NULL DEFAULT 'ia',
    label TEXT NOT NULL,
    first_seen DATE NOT NULL,
    last_seen DATE NOT NULL,
    article_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'open',
    recent_titles TEXT DEFAULT ''
);
ALTER TABLE articles ADD COLUMN IF NOT EXISTS story_id BIGINT REFERENCES stories(id);
```
Le code est backward-compatible : tant que la migration n'est pas appliquée, `save_to_supabase` retire automatiquement `story_id` des lignes envoyées (même mécanisme que pour `hot_source`/`mention_count`), et le panneau "Suivi d'histoires" reste vide sans erreur.