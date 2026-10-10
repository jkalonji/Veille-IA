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

> ⚠️ **Note (2026-07-20) : la section ci-dessus (sources de keywords, 4 signaux, onglets `debat/tech/societe/tendance`) décrit un système qui a depuis été remplacé** par un clustering dynamique par n-grammes (`extract_topic_clusters`/`name_topic_clusters` dans `main.py`, commit `5bd5f54`), lui-même remplacé le 2026-10-06 par les **histoires sémantiques** — voir la section « Histoires sémantiques » plus bas. `hot_reason` est désormais un label de cluster libre, pas une catégorie fixe. `dashboard.py` traite les anciennes valeurs (`debat`/`tech`/`societe`/`tendance`) comme obsolètes via `_OLD_HOT_REASONS`.

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

## Histoires sémantiques (sujets Groq, recalculées à chaque collecte)

Depuis le 2026-10-06, les clusters hot et le suivi d'histoires ne font plus qu'un : une **histoire** regroupe tous les articles des 30 derniers jours qui parlent d'un même sujet — même événement, même personne ou organisation, ou même problématique. Le regroupement se fait sur le **sens** (sujets extraits par Groq), plus sur les mots des titres. Un article peut appartenir à **plusieurs** histoires. Tout est dans `main.py`, section « 0b. Semantic stories ».

### Fonctionnement (à chaque run, par domaine — `refresh_stories`)
1. **Sujets** (`extract_topics`) : chaque nouvel article reçoit de Groq (modèle `GROQ_MODEL`, lots de 20 articles par appel) 1 à 4 sujets, stockés dans `articles.topics` : l'**événement** précis, l'**acteur** principal, la **problématique** (débat public, jamais un domaine de recherche). Pour que des articles de jours différents se retrouvent, Groq reçoit la liste des noms déjà utilisés (`_known_names` : les plus fréquents, puis ceux créés pendant le run, puis les plus récents) et doit les réutiliser tels quels. Les exemples et termes interdits par domaine sont dans `DOMAIN_TOPIC_HINTS`.
2. **Regroupement** (`build_story_candidates`) sur les `STORY_WINDOW_DAYS` (30) derniers jours : un sujet devient une histoire s'il est partagé par ≥ 3 articles de ≥ 2 sources, et s'il ne dépasse pas 5 % des articles de la fenêtre (au-delà, il est trop large — ex. « OpenAI » seul). Deux sujets couvrant presque les mêmes articles (≥ 80 %) sont fusionnés.
3. **Revue** (`review_stories`, un appel Groq) : écarte les histoires incohérentes (articles qui ne partagent qu'un thème vague) et écrit le `summary` du dernier développement.
4. **Sauvegarde** (`_save_stories`) : upsert dans `stories` sur la clé `(domain, topic_key)` — une histoire garde donc son `id` d'un run à l'autre tant que son sujet existe (nécessaire pour la newsletter : sujets cochés le dimanche, édition construite le lundi). `stories.article_urls` liste ses articles. Toute autre histoire ouverte du domaine passe en `closed`.
5. **Drapeaux des articles du run** (`_apply_story_flags`) : `hot_topic` si une de ses histoires a ≥ 2 articles sur les 2 derniers jours ; `hot_reason`/`story_id` = son histoire la plus active ; `supa_hot` si cette histoire a ≥ 5 articles aujourd'hui. Ces champs servent à la classification (modèle complet + résumé pour les articles hot) et à Telegram.

### Titre d'une histoire
Affiché sous la forme **`"{label} — {summary}"`** :
- `stories.label` — le nom du sujet, posé à la création de l'histoire et **jamais modifié ensuite** (ancrage stable).
- `stories.summary` — 3 à 6 mots sur le dernier développement, réécrit à chaque run.

### Dashboard et newsletter
- `_fetch_stories_lookup(domain)` lit `id, label, summary, status, article_urls` (histoires ouvertes + anciennes histoires sans `topic_key`) et `_story_members` en déduit les articles de chaque histoire.
- **Hot Articles** (`_extract_hot_topics`) : les histoires avec ≥ 2 articles sur les 2 derniers jours (15 onglets max).
- **📖 Suivi d'histoires** (`_build_story_timelines`) : les histoires actives sur ≥ 2 jours distincts (40 max), dépliables en timeline jour par jour — Streamlit (`_render_stories`) et export statique (`_stories_html`).
- **Newsletter** (`select_candidates`) : les histoires avec ≥ 2 articles dans la semaine.
- Tant qu'aucune histoire n'a d'`article_urls` (avant le premier run), tout retombe sur l'ancien regroupement par `story_id`/`hot_reason`.

### Rattrapage des sujets (`python main.py --backfill-topics`)
Les articles collectés avant le 2026-10-06 n'ont pas de sujets. Le workflow `AI Radar - Backfill Topics` (tous les jours à 14h UTC, ou à la main) en traite au plus `BACKFILL_MAX_ARTICLES` (500) par run, des plus récents aux plus anciens, puis recalcule les histoires. Il s'arrête proprement si le quota Groq est atteint et reprend au run suivant ; une fois tout traité, un run ne fait plus rien et le workflow peut être supprimé.

**Quota Groq :** le palier gratuit de `openai/gpt-oss-120b` est limité à 200 000 tokens/jour, partagés entre la collecte (classification + sujets + revue) et le rattrapage. `gpt-oss-20b` a été testé pour les sujets : nettement moins bon (sujets trop larges, regroupements faux).

### Migration SQL (appliquée le 2026-10-06)
```sql
ALTER TABLE articles ADD COLUMN IF NOT EXISTS topics TEXT[] DEFAULT '{}';
ALTER TABLE stories  ADD COLUMN IF NOT EXISTS topic_key TEXT;
ALTER TABLE stories  ADD COLUMN IF NOT EXISTS article_urls TEXT[] DEFAULT '{}';
CREATE UNIQUE INDEX IF NOT EXISTS stories_domain_topic_key ON stories(domain, topic_key);
```
Pré-requis plus anciens, toujours nécessaires : la table `stories` (`id, domain, label, summary, first_seen, last_seen, article_count, status, recent_titles`) et `articles.story_id`.

## Colonne "Publié" — date de collecte vs date de publication réelle

`Article.published` est censé être la vraie date/heure de publication de la source (récupérée depuis le flux RSS/l'API), pas l'heure du run GitHub Actions. Mais certaines sources ne fournissent parfois aucune date exploitable (flux RSS incomplet, `seendate` GDELT manquant, etc.) — dans ce cas le code de collecte (`fetch_rss`/`fetch_reddit`/`fetch_hackernews`/`fetch_gdelt_all`/`fetch_usgs`) bascule sur l'heure de collecte (`datetime.now(timezone.utc)`) et pose `published_is_estimated = True` sur l'`Article`.

Le dashboard affiche cette estimation avec un préfixe `~` (ex: `~20h`, `~Hier 09:00`) au lieu de prétendre à une précision qu'il n'a pas — voir `_time_ago`/`_fmt_pub_date` dans `dashboard.py`, appliqué au tableau "Derniers articles", aux cartes Hot Articles et donc aussi au panneau "Suivi d'histoires" (qui réutilise `_render_hot_card_html`).

**Migration SQL à exécuter une fois en Supabase :**
```sql
ALTER TABLE articles ADD COLUMN IF NOT EXISTS published_is_estimated BOOLEAN DEFAULT FALSE;
```
Le code est backward-compatible (fallback automatique dans `save_to_supabase` et `load_articles` si la colonne est absente — la marque `~` reste simplement désactivée jusqu'à la migration).

## « Who really invests in AI? » — indicateurs pays (domaine IA)

Depuis le 2026-10-11, le radar « Répartition par catégorie » est remplacé, **pour le domaine IA uniquement**, par 4 onglets (textes en anglais) à droite du globe. Chaque classement est un top 10 rapporté au PIB, pour voir les pays qui font un effort proportionnel (« put their money where their mouth is ») :

| Onglet | Mesure | Source | Mise à jour |
|---|---|---|---|
| 💰 Investment | Investissement privé IA / PIB | Stanford AI Index (Quid), top 15 pays publiés | Manuelle, 1×/an (avril) |
| 📦 Hardware | Importations nettes HS 847150 + 847330 / PIB | UN Comtrade (API publique) | Auto, mensuelle |
| 🖥️ Supercomputers | Rmax des systèmes TOP500 accélérés (GPU) des sites Research/Academic/Government, PFlop/s par 1 000 Md$ de PIB | TOP500 (scraping) | Auto (listes de juin et novembre) |
| 📢 Announcements | Budgets IA **publics** annoncés, étalés sur la durée du plan / PIB | Documents officiels, une URL par chiffre | Manuelle, tous les 3 mois |

- `country_indicators.py` : récupère les sources auto + PIB/taux de change Banque mondiale, fusionne avec `data/country_indicators_manual.json` et écrit `data/country_indicators.json` (lu par `dashboard.py`). Lancement : `python country_indicators.py`.
- Workflow `AI Radar - Country Indicators` : le 1er de chaque mois, rafraîchit et commite le JSON. Une source en échec garde ses dernières données et est marquée `error`.
- Données à revoir : bandeau orange sous le graphique + message Telegram le lundi (`send_data_reminders` dans `main.py`), dès qu'un `next_review` manuel est dépassé ou qu'une source auto a plus de 45 jours.
- Les valeurs `confidence: "low"` sont affichées en plus clair avec le préfixe `≈`.
- Exclus volontairement : Project Transcendence (Arabie saoudite) et MGX (Émirats), qui sont des objectifs rapportés par la presse et non des budgets publics. Les Émirats ne déclarent pas leurs données douanières à Comtrade.
