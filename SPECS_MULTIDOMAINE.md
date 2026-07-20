# Spécifications — Extension Multi-Domaines de Veille-IA

## 1. Vision

Aujourd'hui, Veille-IA ne surveille que le domaine de l'IA. Or les décisions d'investissement (où va l'argent) réagissent à des chocs qui surviennent dans d'autres domaines : prix du pétrole, mouvements de marchés, événements politiques majeurs. L'objectif de cette extension est de transformer le pipeline en **plusieurs radars thématiques** tournant dans la même infrastructure, chacun collectant ses propres actualités et données chiffrées, avec la même logique de classification et de détection de sujets chauds que le radar IA actuel.

**Explicitement hors périmètre pour cette phase** : la corrélation automatique entre domaines (ex: détecter que le prix du pétrole et une actu IA sont liés). On se concentre d'abord sur la collecte propre de chaque domaine, avec une donnée bien structurée (`domain` + `category`) qui rendra la corrélation possible plus tard.

## 2. Décisions d'architecture

| Sujet | Décision |
|---|---|
| Structure pipeline | **Un seul pipeline** (`main.py` étendu), pas de repos séparés. Un seul `sources.json` avec un champ `domain` par source. |
| Domaine vs catégorie | **Deux niveaux** : `domain` (IA, Matières Premières, Finance, Services, Politique/Événements Majeurs) + `category`, dont la **liste est propre à chaque domaine** (pas de catégories génériques partagées). |
| Données chiffrées (prix, indices) | Nouvelle table Supabase dédiée **`market_data`** (time series), séparée de `articles`. |
| Fréquence de collecte market_data | Alignée sur le run quotidien existant (8h CET), pas de workflow intraday séparé pour l'instant. |
| Hot topics / Supa Hot | **Même logique conservée**, mais calculée **par domaine** (le `mention_count` compare les articles du jour au sein d'un même domaine, pas cross-domaine). |
| Dashboard | Sélecteur de domaine (un niveau de filtre au-dessus des catégories actuelles), tout reste dans le même Streamlit. |
| Digest Telegram | **Un message séparé par domaine et par jour**, même format compact que l'existant (pas de liste d'articles, juste les compteurs + hot topics + lien dashboard filtré sur ce domaine). |
| Nom du projet | Reste **Veille-IA** comme ombrelle ; pas de renommage de repo/URL. |
| Ordre de construction | 1) **Politique / Événements Majeurs** (priorité choisie, la plus alignée avec la motivation initiale) → 2) Matières Premières → 3) Finance/Marchés → 4) Services/Économie. |

## 3. Modèle de données

### 3.1 Table `articles` (existante, étendue)

Ajout d'un champ `domain` (texte, ex: `ia`, `matieres_premieres`, `finance`, `services`, `politique_evenements`). Le champ `category` existant reste, mais sa liste de valeurs valides dépend désormais du `domain` de l'article (voir taxonomies §4). Les colonnes `hot_topic`, `hot_reason`, `hot_source`, `mention_count`, `supa_hot` sont conservées telles quelles, mais tous les calculs de cross-mention (mention_count) se font **filtrés par domain**.

```sql
ALTER TABLE articles ADD COLUMN IF NOT EXISTS domain TEXT DEFAULT 'ia';
```

Les articles déjà en base sont rétrocompatibles : `domain = 'ia'` par défaut.

### 3.2 Nouvelle table `market_data`

```sql
CREATE TABLE IF NOT EXISTS market_data (
    id BIGSERIAL PRIMARY KEY,
    domain TEXT NOT NULL,              -- ex: 'matieres_premieres', 'finance'
    symbol TEXT NOT NULL,              -- ex: 'BRENT', 'WTI', 'SP500', 'VIX'
    label TEXT NOT NULL,               -- nom lisible, ex: 'Pétrole Brent'
    value NUMERIC NOT NULL,
    unit TEXT,                         -- ex: 'USD/baril', 'points', 'index'
    variation_pct NUMERIC,             -- variation vs la veille, si dispo
    source TEXT,                       -- ex: 'EIA', 'yfinance', 'FRED'
    collected_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

Une ligne par symbole suivi et par jour de collecte. Le dashboard pourra tracer des sparklines par symbole, indépendamment des articles.

## 4. Domaines et taxonomies (proposition initiale — à valider ensemble avant implémentation)

### 🌍 Politique / Événements Majeurs *(Phase 1)*
- Conflits / Guerres ⚔️
- Soulèvements / Manifestations ✊
- Catastrophes naturelles 🌪️
- Changements de régime / Coups d'État 🏛️
- Diplomatie / Sommets internationaux 🤝
- Sanctions / Guerre économique 💣

### 🛢️ Matières Premières *(Phase 2)*
- Pétrole / Gaz 🛢️
- Métaux / Mines ⛏️
- Agriculture / Denrées 🌾
- Énergie (production/renouvelable) ⚡
- Terres rares / Semi-conducteurs (chaîne d'approvisionnement) 💎

### 📈 Finance / Marchés *(Phase 3)*
- Marchés actions 📈
- Taux / Banques centrales 🏦
- Crypto-actifs ₿
- Fusions-acquisitions / IPO 🤝
- Dette / Obligations 📉
- Nouveaux actifs IA 🧮 *(hors crypto : économie des tokens — trackers de prix de tokens, routers d'optimisation/économie de tokens, marketplaces de compute/inférence)*

### 💼 Services / Économie *(Phase 4)*
- Emploi / Marché du travail 👷
- Consommation / Retail 🛒
- Indicateurs macro (PIB, inflation) 📊
- Immobilier 🏠
- Adoption IA (particuliers & entreprises) 🤖 *(taux d'usage, intégration en entreprise, outils grand public)*

> Note : les catégories IA existantes ne changent pas et deviennent le domaine `ia`.

## 5. Sources par domaine

### Phase 1 — Politique / Événements Majeurs ✅ (implémenté)

> **Correction (constatée à l'implémentation)** : Reuters World et AP Top News ont discontinué leurs flux RSS publics — remplacés par The Guardian World et NPR World News. Et le GDELT Event Database brut (CSV 15 min, Goldstein Scale/CAMEO) a été jugé trop lourd/risqué pour un pipeline qui ne tourne qu'1x/jour (risque de dépasser le timeout CI de 20 min) — remplacé par l'**API GDELT DOC 2.0** (recherche d'articles par mot-clé, une requête HTTP légère par thème), voir détails ci-dessous.

| Source | Type | Détails |
|---|---|---|
| BBC World, Al Jazeera, The Guardian World, France24 (EN), NPR World News | RSS | Même logique que les flux existants |
| **GDELT DOC 2.0 API** (`fetch_gdelt_all` dans `main.py`) | API JSON (`api.gdeltproject.org/api/v2/doc/doc`), gratuite, pas de clé | 5 requêtes thématiques (Conflits, Manifestations, Coups d'État, Diplomatie, Sanctions) filtrées `sourcelang:english`, exécutées **séquentiellement** (10s entre chaque + 1 retry après 15s) car l'API rate-limite plus agressivement que documenté (~1 req/5s annoncé, mais des réponses vides ont été observées même en respectant ce rythme) |
| **USGS Earthquake Feed** (`fetch_usgs`) | API GeoJSON gratuite, pas de clé (`significant_day.geojson`) | Catastrophes naturelles (séismes) — flux déjà pré-filtré "significatif" par l'USGS, pas de seuil de magnitude à gérer côté code |
| **ACLED** | API (clé gratuite sur inscription, usage académique/ONG/presse) | Pas encore implémenté — accès à demander séparément, prévu en 1b |

Indicateur `market_data` optionnel pour ce domaine (**VIX**) : reporté, pas encore implémenté (arrivera avec `market_data` en Phase 2/3).

Dashboard : sélecteur de domaine ajouté à la fois côté Streamlit interactif (`run_streamlit`) et côté export statique GitHub Pages (`run_export` — toggle JS entre sections pré-générées par domaine, pas de re-render serveur).

### Phase 2 — Matières Premières
- RSS spécialisés (OilPrice.com, Reuters Commodities)
- `market_data` via **EIA API** (Energy Information Administration, gratuite sur inscription) : pétrole (WTI/Brent), gaz naturel

### Phase 3 — Finance / Marchés
- RSS (Reuters Business, Investing.com)
- `market_data` via **yfinance** (gratuit, sans clé, non officiel) ou **Alpha Vantage** (clé gratuite, quota limité) : indices (S&P 500, Nasdaq), taux directeurs

### Phase 4 — Services / Économie
- RSS macro (Eurostat, INSEE, Trading Economics)
- `market_data` via **FRED API** (Federal Reserve Economic Data, clé gratuite) : PIB, inflation, chômage (données principalement US, à évaluer pour équivalents zone euro)

## 6. Détection de sujets chauds (hot topics) par domaine

**Correction (constatée en explorant le code lors de la Phase 0)** : la description ci-dessous a été mise à jour car la doc initiale se basait sur un système obsolète (4 signaux Trends/HN/GitHub/DB + tags `hot_reason` fixes `debat/tech/societe/tendance`). Ce système a été remplacé par un **clustering dynamique par n-grammes** (commit `5bd5f54`) : `extract_topic_clusters` regroupe les articles qui partagent des bigrammes/trigrammes significatifs dans leur titre, puis Groq (`name_topic_clusters`) attribue un label libre à chaque cluster (ex: "GPT-5 Launch", "Sudan Coup Attempt"). `hot_reason` stocke ce label, pas une catégorie fixe.

Depuis la Phase 0, ce clustering est **scopé par domaine** : les articles sont groupés par `domain` avant clustering, donc un cluster ne mélange jamais deux domaines, et `mention_count`/`supa_hot` restent implicitement calculés par domaine. Deux éléments du clustering sont paramétrés par domaine :
- Les stopwords de filtrage des n-grammes (`DOMAIN_NGRAM_STOPWORDS`, `DOMAIN_GENERIC_NGRAMS` dans `main.py`) : une base générique commune + une extension spécifique au domaine `ia` (mots comme "artificial", "intelligence", "model"... trop génériques dans ce domaine). Les autres domaines utilisent la base générique pour l'instant.
- Le prompt Groq de nommage des clusters (`DOMAIN_CLUSTER_NAMING`) : mots interdits + exemples de bons labels, adaptés à chaque domaine (ex: pour Politique, exemples comme "Sudan Coup Attempt" au lieu de "OpenAI GPT-5").

Le seuil Supa Hot (`article_count >= 5` dans le cluster, même jour) reste identique, calculé par domaine.

Le dashboard n'a plus d'onglets fixes (`debat`/`tech`/`societe`/`tendance` — ces valeurs sont désormais traitées comme obsolètes dans `dashboard.py` via `_OLD_HOT_REASONS`) : les sujets chauds sont groupés dynamiquement par leur label de cluster (`hot_reason`).

## 7. Restitution

### Dashboard
Ajout d'un sélecteur de domaine en haut de page (avant les filtres catégorie actuels). Chaque domaine affiche ses propres catégories, ses hot topics, et ses graphiques `market_data` (sparklines par symbole) le cas échéant.

### Telegram
Un message par domaine par jour, même format compact que l'existant, ex :

```
🌍 Radar Politique — 20/07/2026
📰 12 articles collectés · 🔥 2 hot topics
⚔️ Conflits / Guerres : 5
✊ Soulèvements / Manifestations : 4
🌪️ Catastrophes naturelles : 3
📊 Voir le Dashboard (lien filtré domain=politique)
```

## 8. Feuille de route

1. **Phase 0 — Fondations ✅ (code fait, migration SQL à exécuter par l'utilisateur — voir `CLAUDE.md`)** : migration schéma (`domain` sur `articles`, table `market_data`), refactor classification Groq pour prendre une taxonomie par domaine (`DOMAIN_TAXONOMY` dans `main.py`), refactor hot-topic detection (clustering n-grammes) pour filtrer par domaine, refactor digest Telegram pour émettre un message par domaine (`DOMAIN_META`/`_send_domain_digest`). Le dashboard n'a pas encore de sélecteur de domaine (reporté en Phase 1, aucune donnée réelle à filtrer pour l'instant).
2. **Phase 1 — Politique / Événements Majeurs ✅ (code fait et déployé)** : sources RSS + GDELT DOC 2.0 API + USGS, dashboard (Streamlit + export statique) et Telegram pour ce domaine, tous vérifiés fonctionnels. ACLED en 1b si accès obtenu. **Reste ouvert : justesse de la classification des articles dans ce domaine** (voir §9) — à retravailler avant de considérer la Phase 1 vraiment terminée.
3. **Phase 2 — Matières Premières** : RSS + EIA `market_data`.
4. **Phase 3 — Finance / Marchés** : RSS + yfinance/Alpha Vantage `market_data`.
5. **Phase 4 — Services / Économie** : RSS + FRED `market_data`.
6. **Hors périmètre (V2 future)** : moteur de corrélation cross-domaine (narratif LLM, dashboard de corrélation, ou alertes de co-occurrence — à redécider une fois les 5 domaines en place).

## 9. Risques et points ouverts

- **Justesse de la classification `politique_evenements` (ouvert, constaté après déploiement Phase 1)** : les articles remontés (RSS + GDELT + USGS) ne sont pas toujours classés dans la bonne catégorie de `DOMAIN_TAXONOMY["politique_evenements"]` par Groq. Cause probable : les requêtes GDELT DOC 2.0 matchent sur mots-clés dans le texte complet de l'article (pas juste le titre), ce qui ramène parfois des articles hors-sujet (ex: un article économique contenant incidemment le mot "conflict") que Groq classe ensuite tant bien que mal faute de signal clair. Pistes à explorer : resserrer les requêtes GDELT (opérateurs de proximité, exclusion de thèmes bruyants comme la finance/sport), enrichir les notes de classification du prompt Groq pour ce domaine, ou ajouter un filtre de pertinence a priori comme celui existant pour le domaine `ia` (`AI_STRONG/WEAK_KEYWORDS` dans `fetch_all`) plutôt que de laisser passer tout ce qui n'est pas `ia` sans filtre.
- **ACLED** : accès sur inscription (académique/ONG/presse), délai incertain — ne doit pas bloquer la Phase 1.
- **GDELT** : l'API DOC 2.0 utilisée (voir §5) rate-limite plus agressivement que documenté — `fetch_gdelt_all` espace les requêtes de 10s et retente une fois après 15s, mais il reste possible qu'un ou plusieurs des 5 thèmes échouent silencieusement un jour donné (logué, non bloquant pour le reste du pipeline).
- **yfinance** : librairie non officielle qui scrape Yahoo Finance, peut casser sans préavis — prévoir un fallback (Alpha Vantage) si instable.
- **Charge Groq / durée du workflow GitHub Actions** : 5 domaines en parallèle vont multiplier le volume d'articles à classifier — surveiller le timeout du job (actuellement 20 min) et le quota Groq.
- **Secrets à ajouter** : `EIA_API_KEY`, `ALPHAVANTAGE_API_KEY` (ou rien si yfinance), `FRED_API_KEY`, `ACLED_API_KEY` (phase 1b).
- **Taxonomies proposées (§4)** : à valider/ajuster ensemble avant de coder la classification Groq.
