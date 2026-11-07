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
- **📖 Suivi d'histoires** (`_build_story_timelines`) : les histoires actives sur ≥ 2 jours distincts (40 max), dépliables en timeline jour par jour — Streamlit (`_render_stories`) et export statique (`_stories_html`). Au-dessus de la liste, un **nuage d'histoires** (`_story_cloud_html`) ne montre que celles qui ont de l'actu aujourd'hui : taille et couleur selon le nombre d'articles du jour, la plus active au centre, les autres autour. Dans l'export, un clic sur un nom ouvre la timeline de l'histoire.
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

Depuis le 2026-10-11, le radar « Répartition par catégorie » est remplacé, **pour le domaine IA uniquement**, par des onglets (5 depuis le 2026-10-27) (textes en anglais) à droite du globe. Chaque classement est un top 10 ; les trois classements de pays sont rapportés au PIB, pour voir les pays qui font un effort proportionnel (« put their money where their mouth is »). Depuis le 2026-10-27, le premier onglet (ex-« 💰 Investment », investissement privé IA par pays selon le Stanford AI Index) classe des **entreprises** :

| Onglet | Mesure | Source | Mise à jour |
|---|---|---|---|
| 🏢 Valuations | Valorisation des entreprises qui développent leurs propres modèles d'IA de pointe, en $ (valeur de toute l'entreprise, y compris les activités hors IA des géants). Nvidia, TSMC et Apple sont exclus. | Cotées : capitalisation boursière (scraping de companiesmarketcap.com, `market_cap_slug`). Non cotées : valorisation post-money de la dernière levée close, avec une URL par chiffre. | Cotées : auto, mensuelle. Non cotées : manuelle, tous les 3 mois et à chaque nouvelle levée ou introduction en bourse. |
| 💵 Revenue | Chiffre d'affaires des 12 derniers mois des mêmes entreprises (celles sans chiffre connu, comme Mistral, sont absentes) | Cotées : companiesmarketcap.com (TTM). Non cotées : bloc `revenue` de chaque entreprise dans le JSON manuel — comptes certifiés si disponibles (S-1 d'Anthropic), sinon presse (`confidence: "low"`). Jamais de *run-rate* annualisé. | Comme Valuations |
| 📦 Hardware | Importations nettes HS 847150 + 847330 / PIB | UN Comtrade (API publique) | Auto, mensuelle |
| 🖥️ Supercomputers | Rmax des systèmes TOP500 accélérés (GPU) des sites Research/Academic/Government, PFlop/s par 1 000 Md$ de PIB | TOP500 (scraping) | Auto (listes de juin et novembre) |
| 📢 Announcements | Budgets IA **publics** annoncés, étalés sur la durée du plan / PIB | Documents officiels, une URL par chiffre | Manuelle, tous les 3 mois |

- `country_indicators.py` : récupère les sources auto + PIB/taux de change Banque mondiale, fusionne avec `data/country_indicators_manual.json` et écrit `data/country_indicators.json` (lu par `dashboard.py`). Lancement : `python country_indicators.py`.
- Workflow `AI Radar - Country Indicators` : le 1er de chaque mois, rafraîchit et commite le JSON. Une source en échec garde ses dernières données et est marquée `error`.
- Données à revoir : bandeau orange sous le graphique + message Telegram le lundi (`send_data_reminders` dans `main.py`), dès qu'un `next_review` manuel est dépassé ou qu'une source auto a plus de 45 jours.
- Les valeurs `confidence: "low"` sont affichées en plus clair avec le préfixe `≈`.
- Exclus volontairement : Project Transcendence (Arabie saoudite) et MGX (Émirats), qui sont des objectifs rapportés par la presse et non des budgets publics. Les Émirats ne déclarent pas leurs données douanières à Comtrade.

## 🎂 AI layer cake (domaine IA)

Lecture de l'actualité IA par couche — énergie → puces → infrastructure → modèles → applications — pour repérer les **mouvements inter-couches** (un acteur qui entre dans une autre couche). Spécification complète : `SPECS_LAYER_CAKE.md`. Configuration unique : `data/cake_layers.json` (couches, mots-clés, vocabulaire de rareté, paniers, seuils et poids). Logique : `cake.py` ; rendu du dashboard : `cake_dashboard.py`. Tests : `pytest` (dossier `tests/`).

### Phase A — Étiquetage des articles
- `sources.json` : champ optionnel `"cake_layer"` sur les sources spécialisées d'une couche (énergie, puces, infrastructure). Un article d'une telle source passe le filtre IA de `fetch_all` s'il passe le filtre IA **ou** s'il contient un mot-clé de `relevance_keywords[cake_layer]` (cherché en début de mot).
- `_classify_one` (domaine `ia` uniquement) demande à Groq trois clés en plus : `layers` (0 à 3 couches), `cross_layer` (un acteur entre dans une couche où il n'est pas encore actif) et `scarcity` (`tension` / `detente` / `aucune`), validées par `cake.parse_cake_fields`. **Tous les articles `ia` passent par le modèle complet `GROQ_MODEL`** (avant : seulement les hot), pour la qualité de `cross_layer`.
- Quota : la fin de chaque run logue `Groq tokens used this run: {...}` par étape (classification, sujets, revue des histoires). À surveiller sur les premiers runs (200 000 tokens/jour).
- Dashboard : section « 🎂 AI layer cake » en tête du panneau `ia` (Streamlit et export, ancre `#cake`).

**Migration SQL — Phase A (à exécuter une fois en Supabase) :**
```sql
ALTER TABLE articles ADD COLUMN IF NOT EXISTS layers TEXT[] DEFAULT '{}';
ALTER TABLE articles ADD COLUMN IF NOT EXISTS cross_layer BOOLEAN DEFAULT FALSE;
ALTER TABLE articles ADD COLUMN IF NOT EXISTS scarcity TEXT DEFAULT 'aucune';     -- 'tension' | 'detente' | 'aucune'
ALTER TABLE articles ADD COLUMN IF NOT EXISTS sublayers TEXT[] DEFAULT '{}';      -- v2, vide en v1
```
Le code est backward-compatible (fallback dans `save_to_supabase` et `load_articles` si les colonnes sont absentes : la section cake reste alors masquée).

### Phase B — Mouvements inter-couches et acteurs
- `data/actors_manual.json` : 104 acteurs générés depuis l'Annexe A de la spec (lignes « à discuter » → hors panier), tous `needs_review: true` tant qu'ils ne sont pas relus. `home_layers` conditionne l'origine et la distance des mouvements : c'est la partie à relire en priorité. `market_cap_slug` (companiesmarketcap.com) a été vérifié par ticker ; un acteur coté sans slug n'a pas de taille tant que la Phase C ne la complète pas.
- À chaque run (`cake.process_layer_events`, après la sauvegarde des articles) : synchronisation du fichier manuel vers la table `actors` → extraction Groq (`extract_layer_events`, modèle complet, lots de 10) sur les articles `cross_layer` du run → origine / direction / distance calculées en Python (`locate_move` : couche d'origine la plus proche de la cible, en dessous en cas d'égalité) → fusion sur 90 jours et progression annonce → contrat → mise en service → score (`event_score`) → sauvegarde dans `layer_events` et `actor_relations`.
- Un acteur inconnu est créé dans `actors` avec les couches que Groq lui attribue (`actor_home_layers`) et `needs_review = true` ; il apparaît dans le rappel Telegram du lundi.
- `home_layers` en base = couches du fichier manuel + `auto_home_layers` (cibles ajoutées 30 jours après une mise en service). Éditer le fichier manuel ne perd donc jamais une promotion automatique.
- Poids de l'acteur (`refresh_actor_weights`) : le lundi, ou dès qu'un acteur n'a pas de poids. Taille = revenus TTM companiesmarketcap (cotés) ou bloc `revenue` (non cotés) ; influence = démarrage à froid (percentile des valorisations de `country_indicators.json`) en attendant la Phase C ; longévité = `founded_year` (filiale : `parent_founded_year` × 0,8) ; implantation = PageRank des co-mentions d'acteurs dans `articles.topics` sur 180 jours.
- Telegram : un message « 🎂 Mouvement majeur » par événement majeur (score ≥ 7) avant le digest `ia`, puis un par progression (`notified_at`) ; une ligne « 🎂 Cake : … » dans le digest.
- Dashboard : bandeau des mouvements majeurs sur 7 jours, vue gâteau SVG (7 / 30 / 90 j, flèche pleine = descente, pointillée = montée, clic sur une flèche = fiche, clic sur une bande = filtre des signaux forts), fiches événements. Streamlit l'affiche via `components.html`.

**Migration SQL — Phase B (à exécuter une fois en Supabase) :** celle de la spec (§8), avec deux colonnes en plus dans `actors` : `auto_home_layers` et `parent_founded_year`.
```sql
CREATE TABLE IF NOT EXISTS actors (
    id BIGSERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    aliases TEXT[] DEFAULT '{}',
    home_layers TEXT[] NOT NULL DEFAULT '{}',
    auto_home_layers TEXT[] DEFAULT '{}',        -- couches ajoutées après une mise en service
    home_sublayers TEXT[] DEFAULT '{}',          -- v2
    parent TEXT,
    parent_founded_year INT,
    founded_year INT,
    ticker TEXT,
    market_cap_slug TEXT,
    revenue_usd_bn NUMERIC,
    revenue_source TEXT,
    size NUMERIC, influence NUMERIC, longevity NUMERIC, implantation NUMERIC,
    weight NUMERIC,
    weight_breakdown JSONB,
    weight_confidence TEXT DEFAULT 'low',
    needs_review BOOLEAN DEFAULT FALSE,
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS layer_events (
    id BIGSERIAL PRIMARY KEY,
    event_type TEXT NOT NULL DEFAULT 'integration',  -- 'integration' | 'goulot'
    actor_id BIGINT REFERENCES actors(id),
    origin_layer TEXT,
    target_layer TEXT NOT NULL,
    origin_sublayer TEXT,
    target_sublayer TEXT,
    direction TEXT,
    distance INT,
    engagement TEXT,
    engagement_history JSONB DEFAULT '[]',
    what TEXT,
    bypassed TEXT[] DEFAULT '{}',
    winners TEXT[] DEFAULT '{}',
    losers TEXT[] DEFAULT '{}',
    article_urls TEXT[] DEFAULT '{}',
    score NUMERIC,
    score_breakdown JSONB,
    is_major BOOLEAN DEFAULT FALSE,
    first_seen DATE NOT NULL,
    last_seen DATE NOT NULL,
    progressed_at DATE,
    notified_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS layer_events_actor_target ON layer_events(actor_id, target_layer, last_seen);

CREATE TABLE IF NOT EXISTS actor_relations (
    id BIGSERIAL PRIMARY KEY,
    from_actor TEXT NOT NULL,
    relation TEXT NOT NULL,
    to_actor TEXT NOT NULL,
    article_url TEXT,
    seen_at DATE NOT NULL
);
```
Sans ces tables, l'étape est sautée avec un avertissement et le dashboard garde le compteur de la Phase A.

### Phase C — Marché et rareté (`cake_market.py`)
- **Cours** (`collect_prices`, à chaque run) : clôtures yfinance de tous les tickers d'acteurs + `QQQ` → `market_data` (`symbol` = ticker), et l'indice équipondéré de chaque couche, base 100, calculé en rendements → `market_data` (`symbol` = `CAKE_ENERGIE`, `CAKE_PUCES`…). Paniers dans `cake_layers.json` (`baskets`), générés depuis `actors_manual.json` + `IGV` ; les 88 tickers ont été validés sur Yahoo Finance le 2026-11-07 (cours **et** nom de l'entreprise). Un ticker sans cours est logué et sorti de son panier ce jour-là. Un échec yfinance ne bloque rien : les dernières valeurs restent et le bandeau du gâteau affiche « ⚠️ cours au … » au-delà de 4 jours.
- **Premier lancement, une fois** : `python main.py --backfill-baskets` (un an d'historique, ~5 s), nécessaire pour les courbes et l'influence.
- **Indice de prix des tokens** (`collect_token_prices`) : médiane des prix entrée / sortie ($/M tokens) des 20 modèles payants les plus utilisés la veille sur OpenRouter (`TOKEN_PRICE_IDX_IN` / `_OUT`). L'API publique ne donne pas la popularité : le classement est lu dans les données embarquées de la page `openrouter.ai/rankings` (fragile ; en cas d'échec, pas de point ce jour-là, warning dans les logs), les prix viennent de `/api/v1/models`.
- **Influence complète** (§5.2) : médiane des rendements anormaux absolus J0–J+1 (titre de l'acteur vs `QQQ` s'il est coté, et paniers des couches concernées) autour de ses événements et de ses articles hot ; démarrage à froid sous 3 observations. Taille des cotés sans slug companiesmarketcap : chiffre d'affaires Yahoo Finance converti en USD.
- **Rareté et goulots** (`update_scarcity`, après la sauvegarde des articles) : recalcule `layer_daily` sur les 30 derniers jours à chaque run. Solde 7 jours (tension − détente) / articles, z-score contre les 90 jours précédents **de la même couche** (au moins 30 jours de base requis, donc pas de goulot avant ~1 mois de données après la Phase A, sauf rattrapage Phase E). Goulot signalé = couche au z le plus haut, > 1,5 sur 3 jours ; confirmé par un signal de prix de la couche (§6.4). Migration du goulot confirmé tenue 7 jours → événement `goulot` + message Telegram. Empreinte des sources par couche : un changement met la couche en « normale en reconstruction » 30 jours (jauge pointillée, pas de goulot).
- Dashboard : jauges de rareté, sparklines des paniers (et de l'indice tokens pour Modèles), étiquette de goulot, matrice origine → cible (nombre / scores / évolution), courbes des paniers et des z-scores.

**Migration SQL — Phase C :**
```sql
CREATE TABLE IF NOT EXISTS layer_daily (
    day DATE NOT NULL,
    layer TEXT NOT NULL,
    article_count INT,
    tension_count NUMERIC,
    detente_count NUMERIC,
    scarcity_balance NUMERIC,        -- solde lissé 7 j
    scarcity_z NUMERIC,
    sources_fingerprint TEXT,
    baseline_rebuilding BOOLEAN DEFAULT FALSE,
    bottleneck_status TEXT,          -- NULL | 'signale' | 'confirme'
    PRIMARY KEY (day, layer)
);
CREATE INDEX IF NOT EXISTS market_data_symbol_time ON market_data(domain, symbol, collected_at);
```
(`tension_count` / `detente_count` en NUMERIC et non INT : le repli sur le vocabulaire compte une tension faible pour 0,5.)
