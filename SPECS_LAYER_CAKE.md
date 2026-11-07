# Spécifications — Dimension « AI layer cake » du dashboard

## 1. Vision

Jensen Huang décrit l'IA comme un gâteau à cinq étages (billet NVIDIA « AI Is a 5-Layer Cake », 10 mars 2026) : **énergie → puces → infrastructure → modèles → applications**. Chaque application tire sur toutes les couches en dessous.

Cette extension ajoute au domaine `ia` une lecture **par couche** de l'actualité. L'objectif est d'**anticiper les mouvements de marché et les shifts soudains** en observant les interactions entre couches. Le signal le plus important est le **mouvement inter-couches** : un acteur d'une couche qui entre dans une autre.

- Une montée (conquête) est rapide et peu chère, mais l'acteur entre en concurrence avec ses propres clients. Exemple : un énergéticien comme EDF publie un modèle d'IA. Lecture : la France cherche la souveraineté sur son intelligence artificielle, et au lieu d'exporter des MWh, elle exporterait des tokens, qui se vendent beaucoup plus cher.
- Une descente (sécurisation) est lente et coûteuse. L'acteur cherche à échapper à la marge de la couche du dessous et à sécuriser son approvisionnement. Exemple : OpenAI produit sa propre électricité. Lecture : les énergéticiens court-circuités perdent des marges, mais l'événement confirme que l'énergie est le goulot, donc il est haussier pour ceux qui vendent les intrants (turbines, nucléaire modulaire, équipement réseau). **La valeur se déplace, elle ne disparaît pas.**

Tout mouvement inter-couches doit apparaître comme **événement majeur** dans le dashboard et sur Telegram. On combine du **qualitatif** (événements, fiches, lecture) et du **quantitatif** (paniers boursiers, indice de rareté, valeur du MWh à chaque couche).

### Les quatre flux entre couches

| Flux | Sens | Ce qu'il transporte | Exemple de signal |
|---|---|---|---|
| Demande | Descend | Usage → tokens → capex → commandes de puces → MW | Révision à la hausse du capex d'un hyperscaler |
| Contraintes | Monte | Pénuries, prix, délais | Hausse du prix du GPU-heure, liste d'attente de raccordement réseau |
| Intégration | Les deux | Un acteur entre dans une autre couche | Un labo de modèles conçoit sa puce |
| Capital | Les deux | Investissements croisés, financements circulaires, dette adossée aux GPU | Un fabricant de puces investit dans son client |

Les quatre flux sont représentés, mais **le flux d'intégration est le cœur de la v1** : c'est lui qui produit les événements majeurs.

### Hors périmètre de cette spec

- **Sous-couches** (voir §10) : le modèle de données les prévoit dès maintenant, mais l'extraction et les vues arrivent en v2.
- Les domaines autres que `ia` : le gâteau ne s'applique qu'au domaine `ia`.
- Toute prédiction automatique de cours. Le dashboard montre des signaux ; l'interprétation reste humaine.

## 2. Décisions d'architecture

| Sujet | Décision |
|---|---|
| Configuration | Un seul fichier `data/cake_layers.json` : couches, sous-couches, mots-clés de pertinence, vocabulaire de rareté, tickers des paniers, seuils et poids. Pas de constantes dispersées dans le code. |
| Étiquetage des articles | On **étend la sortie JSON de `_classify_one`** (clés `layers`, `cross_layer` et `scarcity`), sans nouvel appel Groq par article. |
| Extraction des événements | Nouvel appel Groq groupé (`extract_layer_events`, modèle complet `GROQ_MODEL`), **uniquement** sur les articles `cross_layer = true` du run. |
| Acteurs | Nouvelle table `actors` + fichier manuel `data/actors_manual.json` (même logique que `country_indicators_manual.json` : `updated`, `next_review`, une URL par chiffre, `confidence`). |
| Données chiffrées | Table `market_data` existante (enfin utilisée) pour les paniers et les indicateurs de couche. Collecte dans le run quotidien. |
| Valeur par MWh | Script dédié `layer_value.py` + `data/layer_value_manual.json` → `data/layer_value.json`, rafraîchi par un workflow mensuel, sur le modèle de `country_indicators.py`. |
| Dashboard | Nouvelle section « 🎂 AI layer cake » en tête du panneau `ia`, dans le Streamlit (`run_streamlit`) **et** dans l'export statique (`_render_domain_export_section`). La vue gâteau est un SVG généré en Python (comme `_story_cloud_html`), les autres graphiques sont en Plotly avec le template `cobalt`. |
| Telegram | Un message « 🎂 Mouvement majeur » par événement majeur, envoyé **avant** le digest du domaine `ia`, plus une ligne de synthèse dans le digest. |
| Philosophie | Comme le dit `CLAUDE.md` : « Don't over-engineer ». Chaque phase doit être utile seule et rétrocompatible (fallback si colonnes ou tables absentes, comme pour `domain` ou `published_is_estimated`). |

## 3. Configuration — `data/cake_layers.json`

```json
{
  "_comment": "Configuration de la dimension AI layer cake. Ordre des couches = du bas vers le haut.",
  "layers": [
    {"code": "energie",        "label": "Énergie",        "emoji": "⚡", "rank": 1},
    {"code": "puces",          "label": "Puces",          "emoji": "🔲", "rank": 2},
    {"code": "infrastructure", "label": "Infrastructure", "emoji": "🏗️", "rank": 3},
    {"code": "modeles",        "label": "Modèles",        "emoji": "🧠", "rank": 4},
    {"code": "applications",   "label": "Applications",   "emoji": "📱", "rank": 5}
  ],
  "sublayers": { "...": "voir §10, non utilisé en v1" },
  "relevance_keywords": { "energie": ["data center", "datacenter", "grid", "nuclear", "smr", "ppa", "power demand", "transformer", "turbine", "uranium", "réseau électrique", "raccordement", "..."] },
  "scarcity_vocabulary": ["shortage", "sold out", "waitlist", "allocation", "rationing", "backlog", "bottleneck", "lead time", "pénurie", "rupture", "liste d'attente", "goulot", "délai de livraison", "..."],
  "baskets": {
    "energie":        ["CEG", "VST", "GEV", "..."],
    "puces":          ["NVDA", "TSM", "000660.KS", "..."],
    "infrastructure": ["CRWV", "VRT", "SU.PA", "..."],
    "modeles":        ["2513.HK", "0100.HK"],
    "applications":   ["IGV", "PLTR", "..."]
  },
  "actor_weight_coefficients": {"size": 0.30, "influence": 0.30, "longevity": 0.15, "implantation": 0.25},
  "event_score": {"major_threshold": 7, "novelty_window_days": 180, "novelty_bonus": 1.2, "merge_window_days": 90},
  "bottleneck": {
    "z_threshold": 1.5, "min_days": 3, "migration_days": 7, "baseline_days": 90, "rebuild_days": 30,
    "price_confirmation": {"window_days": 30, "basket_outperformance_pts": 5, "electricity_rise_pct": 10}
  }
}
```

Notes :
- **La liste complète des acteurs et des paniers est en Annexe A** : c'est elle qui fait foi, et Jérémie l'amende avant l'implémentation. Les paniers de `cake_layers.json` sont générés à partir des lignes de l'annexe marquées « Panier : oui ».
- Les géants présents sur plusieurs couches (Microsoft, Alphabet, Amazon, Meta, Tesla, SpaceX…) sont **exclus des paniers** : leur cours mélange trop de couches pour donner un signal propre. Ils restent des acteurs (table `actors`), avec plusieurs `home_layers`.
- **Couche Modèles** : deux pure players sont cotés à Hong Kong depuis janvier 2026 (Zhipu AI `2513.HK`, MiniMax `0100.HK`). Un panier de deux titres chinois est un signal étroit : il est complété par l'indice de prix des tokens (§6.3) et les valorisations déjà présentes dans `data/country_indicators_manual.json`. Anthropic a déposé un S-1 public en octobre 2026 avec le ticker proposé `ANT` : l'ajouter au panier dès la première cotation effective, pas avant.
- EDF n'est pas cotée (détenue à 100 % par l'État). Ce n'est pas un problème : le poids d'un acteur ne dépend pas que de la bourse (§5).

## 4. Étiquetage et extraction (pipeline `main.py`)

### 4.1 Sources et filtre de pertinence

Constat : `sources.json` contient déjà des flux énergie (Canary Media, Energy Monitor, Connaissance des Énergies, Révolution Énergétique, Electrek) et puces (Semiconductor Engineering, EE Times, DigiTimes Asia, Tom's Hardware). Mais **le filtre `AI_STRONG_KEYWORDS` / `AI_WEAK_KEYWORDS` de `fetch_all` écarte les articles qui ne parlent pas d'IA**, alors que les signaux des couches basses (prix de l'uranium, file d'attente de raccordement, pénurie de transformateurs) n'en parlent souvent pas.

- Ajouter dans `sources.json` un champ optionnel `"cake_layer": "<code>"` sur les sources spécialisées d'une couche.
- Dans `fetch_all`, un article d'une source `cake_layer` passe le filtre s'il passe le filtre IA **ou** s'il contient au moins un mot-clé de `relevance_keywords[cake_layer]`.
- Ajouter des sources sur l'infrastructure physique des datacenters (quasi absente aujourd'hui) : par exemple Data Center Dynamics, Data Center Knowledge, Utility Dive. **Vérifier la validité de chaque URL de flux** avant ajout (règle de `CLAUDE.md`).

### 4.2 Étiquetage par article (dans `_classify_one`, domaine `ia` uniquement)

Ajouter trois clés au JSON demandé à Groq (dans `_build_groq_prompt` quand `domain == "ia"`) :

- `"layers"` : liste de 0 à 3 codes parmi `["energie", "puces", "infrastructure", "modeles", "applications"]`, les couches dont parle l'article. Liste vide si aucune ne s'applique (ex. un débat éthique général).
- `"cross_layer"` : `true` si l'article décrit un acteur qui **entre dans une couche où il n'est pas encore actif** (il construit, achète, conçoit, produit, investit ou signe pour y opérer). `false` pour une activité dans sa couche habituelle (Nvidia sort un GPU, OpenAI sort un modèle).
- `"scarcity"` : une valeur parmi `["tension", "detente", "aucune"]`. `tension` si l'article décrit une pénurie, un rationnement, un allongement des délais ou une capacité saturée dans une des couches de `layers` ; `detente` si une telle tension se résorbe (« la pénurie de HBM s'atténue », « les délais de raccordement raccourcissent ») ; `aucune` sinon. Cette clé remplace le simple comptage de mots, qui confondait une pénurie et sa résorption (§6.4).

Le prompt doit rester court : définition d'une ligne par couche, plus deux exemples positifs et deux négatifs. Valider la sortie (codes inconnus ignorés, valeurs par défaut `[]` / `false` / `"aucune"`). Augmenter `max_tokens` de ~50 pour absorber les clés supplémentaires, et **mesurer l'impact sur le quota Groq** (200 000 tokens/jour partagés) sur 3 runs avant de passer à la suite.

Colonnes : `articles.layers TEXT[] DEFAULT '{}'`, `articles.cross_layer BOOLEAN DEFAULT FALSE`, `articles.scarcity TEXT DEFAULT 'aucune'` (voir §8).

### 4.3 Extraction des événements (`extract_layer_events`)

Appelée dans `main()` après la classification et `refresh_stories`, sur les articles du run avec `cross_layer = true`, par lots de 10, modèle `GROQ_MODEL`, `response_format=json_object`, mêmes réessais que les appels d'histoires (`GROQ_RETRIES`).

Pour chaque article, Groq renvoie 0 ou 1 événement :

```json
{
  "actor": "EDF",
  "target_layer": "modeles",
  "engagement": "annonce",
  "what": "EDF publie un modèle de fondation entraîné en France",
  "bypassed": ["fournisseurs de modèles étrangers"],
  "winners": ["fournisseurs de GPU", "hébergeurs souverains"],
  "losers": ["labos de modèles américains sur le marché français"],
  "relations": [{"from": "EDF", "relation": "utilise", "to": "Nvidia"}]
}
```

- `engagement` ∈ `annonce` (intention, communiqué, rumeur), `contrat` (accord signé, commande ferme, investissement versé, acquisition conclue), `mise_en_service` (produit livré, centrale raccordée, modèle disponible, usine qui produit).
- `relations` ∈ `fournit`, `utilise`, `partenaire`, `investit` : alimentent l'implantation (§5.4).
- `what` est en français, 15 mots maximum.

**Résolution de l'acteur.** Le nom est rapproché de `actors.name` / `actors.aliases` (normalisation comme `_topic_key`). Un acteur inconnu est créé avec `needs_review = true` et apparaît dans le rappel Telegram du lundi (étendre `send_data_reminders`).

**Origine, direction, distance (calculées en Python, pas par Groq).**

Deux notions à ne pas confondre :
- `home_layers` est une **propriété de l'acteur** : les couches où il opère déjà. Il peut y en avoir plusieurs, elle évolue lentement, elle est stockée dans `actors`.
- `origin_layer` est une **propriété de l'événement** : la couche d'où part ce mouvement précis, choisie parmi les `home_layers` de l'acteur. Elle est stockée dans `layer_events`.

Règles :
- Chaque acteur a un ensemble `home_layers` (une ou plusieurs couches où il est déjà significativement actif). Microsoft, par exemple : `infrastructure`, `modeles`, `applications`.
- Si `target_layer ∈ home_layers`, ce n'est pas un mouvement inter-couches : on ignore l'événement (et on logue, pour améliorer le prompt).
- Sinon, `origin_layer` = la couche de `home_layers` **la plus proche** de la cible. `distance = |rank(target) − rank(origin)|`, `direction = "montee"` si la cible est au-dessus, `"descente"` sinon. On prend la plus proche pour ne pas surestimer la rupture : Microsoft opère déjà des datacenters, donc un contrat d'électricité le fait descendre de 2 couches (infrastructure → énergie), pas de 4 (applications → énergie).
- En cas d'égalité (deux couches d'origine à la même distance, une au-dessus et une en dessous de la cible), prendre la couche du dessous : l'événement est alors une montée. Loguer ces cas, ils sont rares et méritent une relecture.
- Quand un événement atteint `mise_en_service`, la cible est ajoutée à `home_layers` de l'acteur 30 jours plus tard (les actus suivantes sur cette activité ne sont plus des mouvements).

**Fusion et progression.** Un même acteur avec la même cible sur `merge_window_days` (90 j) = **un seul événement**. Le nouvel article est ajouté à `article_urls`. Si l'engagement progresse (annonce → contrat → mise en service), on met à jour l'engagement, on recalcule le score, et on marque `progressed_at`. Une progression d'un événement majeur est elle-même notifiée sur Telegram.

### 4.4 Cas de test (fictifs, à mettre dans des tests unitaires de la logique Python)

| Entrée | Attendu |
|---|---|
| « EDF publie un modèle de fondation » | `cross_layer`, origine `energie`, cible `modeles`, montée, distance 3, engagement `annonce` |
| « OpenAI met en service sa propre centrale électrique » | origine `modeles`, cible `energie`, descente, distance 3, `mise_en_service` ; `bypassed` contient les énergéticiens |
| « Microsoft signe un contrat d'achat d'électricité nucléaire » | origine `infrastructure` (couche d'origine la plus proche), cible `energie`, descente, distance 2, `contrat` |
| « Nvidia présente son nouveau GPU » | `cross_layer = false` |
| « OpenAI lance un nouveau modèle » | `cross_layer = false` |
| Deux articles « annonce » puis « contrat » du même acteur vers la même cible à 40 jours d'écart | un seul événement, engagement `contrat`, `progressed_at` renseigné |

## 5. Poids de l'acteur

Le poids `W ∈ [0, 1]` combine quatre composantes, chacune normalisée dans `[0, 1]`. Coefficients dans `actor_weight_coefficients`. Si une composante est inconnue, on calcule la moyenne pondérée **sur les composantes disponibles** (coefficients renormalisés) et on marque le poids `confidence: "low"`.

### 5.1 Taille (revenus annuels)

`size = min(1, log10(1 + revenus_12m_md$) / log10(1 + 500))`

L'échelle log évite qu'un seul géant écrase tout. Sources : companiesmarketcap.com pour les cotées (réutiliser `_cmc_usd_bn(slug, "revenue")` de `country_indicators.py`) ; pour les non cotées, le bloc `revenue` du JSON manuel (comptes certifiés si disponibles, sinon presse avec `confidence: "low"`, jamais de run-rate annualisé, même règle que les indicateurs pays).

### 5.2 Influence (mouvement de marché)

On mesure **combien les marchés bougent quand l'acteur fait quelque chose** :

- Pour chaque événement passé de l'acteur (et pour ses articles `hot_topic` en v1, pour avoir assez d'historique), calculer le **rendement anormal absolu** à J0–J+1 :
  - de son propre titre s'il est coté (rendement du titre − rendement du Nasdaq-100 `QQQ`) ;
  - du panier de la **couche cible** (et de la couche d'origine) dans tous les cas, ce qui permet de mesurer l'influence d'un acteur non coté comme OpenAI ou EDF.
- `influence` = rang percentile de la médiane de ces rendements anormaux parmi tous les acteurs ayant au moins 3 observations.
- Démarrage à froid (moins de 3 observations) : percentile de la capitalisation ou de la valorisation (déjà disponibles via `country_indicators`), avec `confidence: "low"`.

### 5.3 Longévité

`longevity = clip((année_courante − année_création) / 15, 0.2, 1)`

Une entreprise qui vient d'arriver garde un plancher de 0,2 : elle compte, mais moins. Le cas d'une filiale ou d'une scission (ex. une nouvelle entité créée par un groupe ancien) se gère avec un champ `parent` dans `actors` : on prend alors la longévité du parent multipliée par 0,8.

### 5.4 Implantation (logique de citation)

Même mécanique que les citations d'un article scientifique : plus d'autres acteurs utilisent ou mentionnent les services d'une entreprise, plus elle pèse.

- On construit un graphe orienté entre acteurs sur 180 jours glissants.
  - **v1 (sans coût Groq supplémentaire)** : arêtes de co-mention entre acteurs, à partir des sujets de type acteur déjà extraits dans `articles.topics`. Poids d'arête = nombre d'articles distincts.
  - **v1.1** : arêtes dirigées issues de `relations` (§4.3). `A utilise B` et `A fournit à B` donnent une arête `A → B` ou `B → A` selon le sens « qui dépend de qui ». Ces arêtes pèsent 3 fois plus qu'une co-mention.
- `implantation` = PageRank (facteur 0,85) normalisé en rang percentile. Implémentation en Python pur (quelques dizaines de lignes), sans ajouter networkx.

### 5.5 Stockage

Recalcul hebdomadaire (le lundi, dans le run quotidien) des quatre composantes et de `W`, stockés dans `actors` avec `weight_breakdown JSONB` pour pouvoir afficher le détail dans les fiches.

## 6. Le quantitatif

### 6.1 Score d'un événement

```
D = {1: 0.40, 2: 0.60, 3: 0.85, 4: 1.00}[distance]
E = {"annonce": 0.40, "contrat": 0.70, "mise_en_service": 1.00}[engagement]
W = poids de l'acteur (§5)
N = novelty_bonus (1.2) si aucun événement avec la même paire (origine → cible) sur novelty_window_days (180 j), sinon 1

score = min(10, round(10 × (0.35·D + 0.30·E + 0.35·W) × N, 1))
```

- **La direction n'entre pas dans le score** : monter et descendre sont deux lectures différentes, pas deux intensités. Elle détermine l'affichage et la lecture (conquête / sécurisation).
- Le bonus de nouveauté sert à capter les shifts : la première fois qu'une paire de couches est franchie depuis longtemps, c'est plus informatif que la dixième.
- `score ≥ major_threshold` (7) ⇒ événement **majeur**.
- `score_breakdown JSONB` stocke D, E, W, N pour l'affichage.

### 6.2 Paniers boursiers par couche

- Nouvelle dépendance `yfinance` (non officielle, peut casser : prévoir de loguer l'erreur sans bloquer le pipeline, comme pour GDELT).
- Chaque jour : clôture de chaque ticker → `market_data` (`domain='ia'`, `symbol=<ticker>`, `unit='USD'` ou devise locale, `variation_pct` vs veille, `source='yfinance'`).
- Panier équipondéré par couche, base 100 au premier jour d'historique, calculé en rendements (devises sans importance) → `market_data` avec `symbol='CAKE_<CODE>'`, `label='Panier <couche>'`, `unit='index'`.
- Premier lancement : historique d'un an via `yfinance.download` pour avoir des courbes et pouvoir calculer l'influence (§5.2). Option `python main.py --backfill-baskets`.
- Ajouter `QQQ` comme référence pour les rendements anormaux.

### 6.3 Indicateurs propres aux couches

| Couche | Indicateur | Source | Mise à jour |
|---|---|---|---|
| Énergie | Prix de gros de l'électricité ($/MWh), États-Unis et Europe | Manuel mensuel avec URL dans un premier temps ; API EIA (clé gratuite, secret `EIA_API_KEY` déjà prévu dans `SPECS_MULTIDOMAINE.md`) en v1.1 | Mensuelle |
| Infrastructure | Prix de location du GPU-heure (H100 puis génération courante, $/h) | **À sourcer** : identifier un indice public et gratuit ; à défaut, relevé manuel mensuel avec URL | Mensuelle |
| Modèles | Indice de prix des tokens : prix médian ($/M tokens, entrée et sortie) des 20 modèles les plus utilisés | API publique OpenRouter `GET https://openrouter.ai/api/v1/models` (vérifier les champs de prix et la manière d'obtenir la popularité au moment de l'implémentation) | Quotidienne |
| Toutes | Indice de rareté (§6.4) | Calculé à partir des articles | Quotidienne |

Ces séries vont dans `market_data` (`symbol` du type `ELEC_US`, `GPU_HOUR_H100`, `TOKEN_PRICE_IDX`).

### 6.4 Indice de rareté et goulots (où est le goulot ?)

**Étape 1 — Solde de rareté par couche (presse).**
- Pour chaque couche et chaque jour : `solde = (nb articles "tension" − nb articles "detente") / nb articles de la couche`, à partir de la clé `scarcity` (§4.2) des articles dont `layers` contient le code. `scarcity_vocabulary` ne sert plus qu'en secours, pour les articles classés avant l'ajout de la clé (comptés uniquement comme tension faible, poids 0,5).
- Lissage sur 7 jours.
- z-score du solde lissé par rapport aux 90 jours précédents **de la même couche**. On compare chaque couche à sa propre normale : la presse énergie parle de contraintes en permanence, donc le solde brut désignerait toujours l'énergie. Le z-score mesure ce qui se tend *anormalement*.
- La jauge de rareté de la vue gâteau affiche ce z-score, ramené à 0–1 (z ≤ 0 → 0 ; z ≥ 3 → 1).

**Étape 2 — Goulot signalé.** La couche au z-score le plus élevé, à condition qu'il dépasse 1,5 pendant au moins 3 jours d'affilée. Sinon, pas de goulot net.

**Étape 3 — Goulot confirmé.** Un goulot signalé devient confirmé si **au moins un signal de prix** de la même couche va dans le même sens sur les 30 derniers jours :

| Couche | Signal de prix qui confirme |
|---|---|
| Énergie | Prix de gros de l'électricité en hausse de plus de 10 % sur 30 j, ou panier énergie qui surperforme `QQQ` de plus de 5 points |
| Puces | Panier puces qui surperforme `QQQ` de plus de 5 points sur 30 j |
| Infrastructure | Prix du GPU-heure en hausse sur 30 j, ou panier infrastructure qui surperforme `QQQ` de plus de 5 points |
| Modèles | Indice de prix des tokens en **hausse** sur 30 j (la tendance de fond est à la baisse, donc une hausse est un signal fort) |
| Applications | Pas de signal de prix fiable en v1 : la couche ne peut être que « signalée » |

Les seuils sont dans `cake_layers.json` (`bottleneck.z_threshold`, `bottleneck.min_days`, `bottleneck.price_confirmation`).

**Étape 4 — Migration.** Il y a migration quand le goulot **confirmé** change de couche et que ça tient 7 jours. C'est un événement majeur synthétique (`event_type = "goulot"`, sans acteur), notifié sur Telegram. Les changements de goulot seulement signalés sont affichés dans le dashboard mais ne sont pas notifiés.

**Changement de sources.** Ajouter ou retirer des flux d'une couche modifie sa normale. On stocke, par couche et par jour, une empreinte de l'ensemble des sources actives (hash des noms triés). Si l'empreinte change, la couche passe en « normale en reconstruction » pendant 30 jours : son z-score est affiché en pointillé avec une mention, et elle ne peut pas déclencher de goulot signalé ou confirmé.

### 6.5 Valeur du MWh à chaque couche (`layer_value.py`)

Question : combien vaut un MWh une fois transformé à chaque étage, et quelle couche crée le plus de valeur aujourd'hui ?

Méthode **bottom-up** : à chaque étage, on convertit un prix unitaire observable en $ par MWh d'électricité. La **valeur ajoutée** d'une couche = son prix − celui de la couche du dessous, et le **multiplicateur** = son prix / celui de la couche du dessous.

**Base commune : le MWh au compteur.** Tous les étages sont ramenés au même dénominateur : l'électricité facturée à l'entrée du datacenter, refroidissement et pertes compris. L'énergie consommée par la puce seule (TDP) est donc toujours multipliée par le PUE, et les estimations d'énergie par token ou par utilisateur (étages 4 et 5) doivent être des valeurs au compteur ; si la source donne une valeur au niveau de la puce, la multiplier par le PUE et le noter dans `note`.

**Pourquoi P3, P4 et P5 n'additionnent pas les étages du dessous.** Ce sont des prix de marché, qui contiennent déjà les coûts des couches inférieures : un loueur fixe son prix du GPU-heure pour couvrir l'électricité, l'amortissement des puces, le datacenter et sa marge ; le prix du million de tokens couvre de même le calcul sous-jacent. Les additionner compterait l'électricité et les puces deux fois. P2 est le seul étage sans prix de marché (personne ne vend du « calcul sur puce, sans datacenter ») : on le reconstruit par les coûts et on y ajoute P1 explicitement.

| Étage | Formule ($/MWh au compteur) | Entrées |
|---|---|---|
| 1. Énergie | `P1 = prix_gros_electricite` | §6.3 |
| 2. Puces | `P2 = P1 + prix_accelerateur / (TDP_kW × PUE × duree_vie_h × taux_utilisation / 1000)` | prix catalogue d'un accélérateur de référence, TDP, PUE moyen, durée d'amortissement, taux d'utilisation |
| 3. Infrastructure | `P3 = prix_gpu_heure / (TDP_kW × PUE / 1000)` | prix GPU-heure (§6.3), même accélérateur de référence et même PUE qu'en P2 |
| 4. Modèles | `P4 = prix_par_million_tokens / energie_par_million_tokens_MWh` | indice de prix des tokens (§6.3), estimation sourcée de l'énergie par million de tokens, au compteur |
| 5. Applications | `P5 = revenu_applicatif_par_utilisateur_mois / energie_par_utilisateur_mois_MWh` | ARPU d'abonnements IA, consommation estimée par utilisateur et par mois, au compteur |

L'accélérateur de référence, son TDP et le PUE sont **partagés** entre P2 et P3 (une seule entrée dans le JSON), sinon les deux étages ne sont pas comparables.

**Valeur ajoutée négative.** Rien ne garantit que `P_k > P_(k-1)`. Si un étage vend sous ses coûts (guerre des prix sur les tokens, capacité louée à perte, abonnements subventionnés par des levées de fonds), sa valeur ajoutée devient négative. Ce n'est pas une erreur de calcul à masquer, c'est un signal : la couche est en compression de marge ou vit sur du capital. `layer_value.py` pose alors `value_added_negative: true` sur l'étage. Le dashboard l'affiche distinctement (§7.1), et le rapport mensuel le signale sur Telegram.

- Toutes les entrées manuelles sont dans `data/layer_value_manual.json`, au format de `country_indicators_manual.json` : valeur, unité, `source_url`, `confidence`, `note`, `updated`, `next_review`.
- **Les étages 4 et 5 reposent sur des estimations très incertaines** (énergie par token, consommation par utilisateur) : elles sont toujours en `confidence: "low"` et affichées avec le préfixe `≈` et en plus clair, comme les indicateurs pays.
- Ne jamais inventer une valeur : une entrée manquante rend l'étage absent du graphique, pas estimé à zéro.
- `layer_value.py` écrit `data/layer_value.json` avec, par étage : `price_usd_mwh`, `value_added_usd_mwh`, `multiplier`, `value_added_negative`, `confidence`, et l'historique des calculs précédents (pour suivre l'évolution mois par mois).
- Workflow `AI Radar - Layer Value` : le 1er de chaque mois, comme `country_indicators.yml`. Il envoie un court message Telegram : l'étage qui crée le plus de valeur ajoutée par MWh ce mois-ci, sa variation par rapport au mois précédent, et les étages en valeur ajoutée négative. Les relances `next_review` s'ajoutent à `send_data_reminders`.
- Tests unitaires : à partir d'un jeu d'entrées fictives, vérifier que le PUE est appliqué en P2 et en P3, et qu'un étage avec `P_k < P_(k-1)` produit bien `value_added_negative: true` et une valeur ajoutée négative (pas une valeur tronquée à zéro).
- v2 : vérification **top-down** (revenus IA annuels de la couche / électricité annuelle consommée par l'IA), pour vérifier la cohérence des ordres de grandeur.

## 7. Restitution

### 7.1 Dashboard — section « 🎂 AI layer cake » (domaine `ia`, en tête du panneau)

Dans l'ordre d'affichage :

1. **Bandeau « Mouvements majeurs »** : les événements majeurs des 7 derniers jours (et les migrations de goulot confirmé), du plus récent au plus ancien. À droite du bandeau, l'état du goulot : `Goulot : Énergie (confirmé)`, `Goulot : Puces (signalé)` ou `Pas de goulot net`. Format court : `⬆️ EDF · Énergie → Modèles · annonce · 7,4`. Masqué s'il n'y en a pas.

2. **Vue gâteau** (SVG généré en Python, nouvelle fonction `_cake_svg(...)`), avec un sélecteur de période 7 j / 30 j / 90 j (bascule JS côté export, comme le sélecteur de domaine) :
   - cinq bandes empilées, énergie en bas ;
   - pour chaque bande : nom, nombre de **signaux forts** de la période (§7.2), jauge de rareté (z-score ramené à 0–1, pointillée si la normale est en reconstruction), sparkline du panier et variation en %. Pour la couche Modèles : sparkline du panier et de l'indice de prix des tokens. La bande du goulot porte une étiquette « goulot signalé » ou « goulot confirmé » ;
   - à droite, une flèche verticale par événement inter-couches de la période, de la couche d'origine à la couche cible : **trait plein pour une descente, pointillé pour une montée** (la couleur ne doit pas être le seul indice), épaisseur proportionnelle au score, étiquette courte (A, B, C…) reliée à la liste des fiches ;
   - un clic sur une flèche ouvre la fiche, un clic sur une bande filtre la liste des signaux forts sur cette couche.

3. **Fiches événements** : direction, distance, engagement (avec la frise annonce → contrat → mise en service et les dates de passage), score et son détail (D, E, W, N), poids de l'acteur et ses quatre composantes, `what`, couches enjambées, `bypassed` / `winners` / `losers`, articles liés (liens sources systématiques). Si l'événement appartient à une histoire sémantique (`story_id` d'un de ses articles), lien vers sa timeline dans « 📖 Suivi d'histoires ».

4. **Matrice origine → cible** (heatmap Plotly 5×5, diagonale vide) : nombre de mouvements (ou somme des scores, bascule) sur 90 jours. Bascule « évolution » : différence avec les 90 jours précédents. C'est la vue qui montre les shifts structurels (par exemple, les descentes vers l'énergie qui passent devant les montées vers les applications).

5. **Escalier de la valeur par MWh** (barres Plotly, une par étage, de l'énergie aux applications, échelle log) : prix par MWh au compteur, valeur ajoutée et multiplicateur affichés au-dessus de chaque barre, étages `confidence: "low"` en plus clair avec `≈`. Un étage en valeur ajoutée négative a sa valeur ajoutée en rouge, précédée de `▼`, et une info-bulle « vend sous le coût des couches inférieures ». Bascule « évolution » : la valeur ajoutée de chaque étage mois par mois, pour voir laquelle crée le plus de valeur au fil du temps. Pied de graphique avec date de mise à jour et sources, comme `_indicator_footer_html`.

6. **Courbes de couche** : les cinq paniers en base 100 sur la période, et en dessous le z-score du solde de rareté par couche (même axe de temps), avec une ligne horizontale au seuil de 1,5 et les périodes de goulot confirmé surlignées. On lit ainsi ensemble les prix et les tensions.

### 7.2 Définition d'un « signal fort »

- **Signal fort d'une couche** : une histoire sémantique *hot* (règles existantes de `_apply_story_flags`) dont au moins la moitié des articles ont cette couche dans `layers`, **ou** un événement de score ≥ 5 dont la couche est l'origine ou la cible.
- **Signal à cheval sur plusieurs couches** : tout événement inter-couches (quel que soit son score), et toute histoire hot dont les articles couvrent au moins deux couches de manière équilibrée (chaque couche ≥ 30 % des articles).

### 7.3 Telegram

- **Message par événement majeur**, envoyé avant le digest `ia` :

```
🎂 Mouvement majeur — 07/11/2026
⬆️ EDF : Énergie → Modèles (3 couches)
📌 EDF publie un modèle de fondation entraîné en France
🔖 Annonce · Score 7,4 (nouveau franchissement)
📉 Perdants : labos de modèles américains sur le marché français
📈 Gagnants : fournisseurs de GPU, hébergeurs souverains
📊 Voir le gâteau (lien dashboard ancre #cake)
```

- Progression d'un événement majeur (ex. annonce → contrat) : même format, titre `🎂 Mouvement majeur — progression`.
- Migration de goulot **confirmé** uniquement : `🎂 Le goulot passe de Puces à Énergie (confirmé par : prix de l'électricité +14 % sur 30 j)`. Les goulots seulement signalés ne sont pas notifiés.
- Dans le digest `ia` existant, une ligne : `🎂 Cake : 3 mouvements (1 majeur) · goulot : Énergie (confirmé)`.
- Le 1er du mois, le rapport de valeur par MWh (§6.5).

## 8. Modèle de données (migrations SQL à exécuter une fois en Supabase)

```sql
-- Phase A
ALTER TABLE articles ADD COLUMN IF NOT EXISTS layers TEXT[] DEFAULT '{}';
ALTER TABLE articles ADD COLUMN IF NOT EXISTS cross_layer BOOLEAN DEFAULT FALSE;
ALTER TABLE articles ADD COLUMN IF NOT EXISTS scarcity TEXT DEFAULT 'aucune';     -- 'tension' | 'detente' | 'aucune'
ALTER TABLE articles ADD COLUMN IF NOT EXISTS sublayers TEXT[] DEFAULT '{}';  -- v2, vide en v1

-- Phase B
CREATE TABLE IF NOT EXISTS actors (
    id BIGSERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    aliases TEXT[] DEFAULT '{}',
    home_layers TEXT[] NOT NULL DEFAULT '{}',
    home_sublayers TEXT[] DEFAULT '{}',          -- v2
    parent TEXT,                                 -- groupe parent éventuel (longévité)
    founded_year INT,
    ticker TEXT,                                 -- yfinance, si coté
    market_cap_slug TEXT,                        -- companiesmarketcap.com, si coté
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
    origin_sublayer TEXT,                             -- v2
    target_sublayer TEXT,                             -- v2
    direction TEXT,                                   -- 'montee' | 'descente'
    distance INT,
    engagement TEXT,                                  -- 'annonce' | 'contrat' | 'mise_en_service'
    engagement_history JSONB DEFAULT '[]',            -- [{"engagement": "...", "date": "..."}]
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
    relation TEXT NOT NULL,          -- 'fournit' | 'utilise' | 'partenaire' | 'investit'
    to_actor TEXT NOT NULL,
    article_url TEXT,
    seen_at DATE NOT NULL
);
```

```sql
-- Phase C
CREATE TABLE IF NOT EXISTS layer_daily (
    day DATE NOT NULL,
    layer TEXT NOT NULL,
    article_count INT,
    tension_count INT,
    detente_count INT,
    scarcity_balance NUMERIC,        -- solde lissé 7 j
    scarcity_z NUMERIC,
    sources_fingerprint TEXT,        -- hash des sources actives de la couche
    baseline_rebuilding BOOLEAN DEFAULT FALSE,
    bottleneck_status TEXT,          -- NULL | 'signale' | 'confirme'
    PRIMARY KEY (day, layer)
);
```

`market_data` existe déjà (voir `CLAUDE.md`) : aucune migration. Le code reste rétrocompatible : si une colonne ou une table manque, la section cake est masquée et un avertissement est logué, sans casser le reste du dashboard ni le pipeline.

`data/actors_manual.json` : générer la liste initiale à partir de l'**Annexe A** (dans la version amendée par Jérémie), plus les entreprises de `company_valuations` dans `country_indicators_manual.json` qui n'y seraient pas. Pour chacun : `name`, `aliases`, `home_layers`, `home_sublayers` (la sous-couche de l'annexe, utile dès la v2), `founded_year`, `ticker` / `market_cap_slug` si coté, et pour les non cotés un bloc `revenue` sourcé. Marquer `needs_review` tant que Jérémie n'a pas validé. Ne pas inventer de chiffre : un champ inconnu reste vide.

## 9. Feuille de route

Chaque phase est livrable seule. Ne pas passer à la suivante avant d'avoir vérifié la précédente sur au moins 3 runs réels.

1. **Phase A — Étiquetage.** `data/cake_layers.json`, migration des colonnes `articles`, champ `cake_layer` dans `sources.json` et filtre de pertinence élargi, nouvelles sources infrastructure (URLs vérifiées), clés `layers` / `cross_layer` / `scarcity` dans `_classify_one`. Dashboard : un compteur d'articles par couche dans la section cake (version minimale). **Critères** : consommation Groq mesurée et documentée ; sur un échantillon de 50 articles relus, `layers` juste dans au moins 80 % des cas ; aucun article énergie non-IA pertinent perdu par le filtre sur les sources `cake_layer`.
2. **Phase B — Événements et acteurs.** Tables `actors`, `layer_events`, `actor_relations`, fichier `actors_manual.json`, `extract_layer_events`, calcul origine / direction / distance, fusion et progression, poids de l'acteur avec les composantes disponibles (taille, longévité, implantation v1 par co-mention ; influence en démarrage à froid), score, alertes Telegram, bandeau, vue gâteau (sans les données de marché), fiches. **Critères** : les cas de test du §4.4 passent ; pas de doublon d'événement pour un même mouvement ; un événement majeur produit exactement un message Telegram.
3. **Phase C — Marché et rareté.** `yfinance`, **validation de chaque ticker de l'Annexe A** (un symbole qui ne renvoie pas de cours est logué et retiré du panier, pas deviné), paniers et historique d'un an, indice de prix des tokens, influence complète (§5.2), table `layer_daily`, solde de rareté, goulots signalés et confirmés, empreinte des sources, matrice origine → cible, courbes de couche, jauges et sparklines dans la vue gâteau. **Critères** : le workflow quotidien reste sous son timeout de 20 minutes ; un échec de yfinance n'empêche ni la collecte ni l'export ; sur un échantillon relu, un article « la pénurie se résorbe » est bien classé `detente`.
4. **Phase D — Valeur par MWh.** `layer_value.py`, `layer_value_manual.json` rempli avec des sources, workflow mensuel et son message Telegram, escalier de la valeur avec valeur ajoutée négative, relances `next_review`. **Critères** : chaque nombre affiché a une URL de source ou est marqué comme estimation ; les tests unitaires du §6.5 passent.
5. **Phase E — v1.1.** Relations dirigées dans l'implantation, API EIA pour l'électricité, rattrapage `layers` sur les 90 derniers jours (sur le modèle de `--backfill-topics`, plafonné pour respecter le quota Groq).
6. **v2 — Sous-couches** (§10).

## 10. v2 — Sous-couches

La nuance « la valeur se déplace » montre qu'il y a des couches dans chaque couche : quand OpenAI descend vers l'énergie, les producteurs perdent mais les équipementiers gagnent. La v2 applique la même lecture au niveau des sous-couches.

Proposition de taxonomie (à valider), chaque liste étant ordonnée du bas vers le haut, ce qui permet de calculer une direction et une distance à l'intérieur d'une couche :

| Couche | Sous-couches (bas → haut) |
|---|---|
| Énergie | combustible et matières (uranium, gaz) · équipement (turbines, transformateurs) · production · réseau et raccordement |
| Puces | équipement de fabrication (lithographie) · fonderie · mémoire (HBM) · packaging avancé · conception (GPU, ASIC) · interconnexion et réseau |
| Infrastructure | foncier et construction · alimentation et refroidissement · systèmes et réseau du datacenter · cloud hyperscaler · neocloud et location de GPU |
| Modèles | données · outillage d'entraînement et d'inférence · modèles ouverts · modèles frontière · modèles spécialisés (bio, robotique, physique) |
| Applications | outils développeurs et agents de code · applications d'entreprise horizontales · verticales métier · grand public · IA physique (robots, véhicules) |

Ce que la v1 prépare déjà : colonnes `sublayers`, `home_sublayers`, `origin_sublayer`, `target_sublayer`, et la clé `sublayers` de `cake_layers.json`. En v2 :
- un mouvement à l'intérieur d'une couche (distance 0 au niveau couche) devient un événement si sa distance en sous-couches est ≥ 1 ;
- la vue gâteau permet de « zoomer » sur une couche pour afficher ses sous-couches avec les mêmes flèches ;
- `winners` / `losers` sont rattachés à des sous-couches, pour visualiser où la valeur se déplace ;
- l'escalier de la valeur par MWh peut se décomposer par sous-couche quand les données le permettent.

## 11. Risques et points ouverts

- **Quota Groq** (200 000 tokens/jour, partagés avec la collecte et le rattrapage des sujets) : mesurer en Phase A avant d'aller plus loin. Si besoin, réduire les exemples du prompt ou réserver `cross_layer` au modèle complet pour les seuls articles hot.
- **Faux positifs `cross_layer`** : principal risque qualitatif. La vérification `target_layer ∉ home_layers` côté Python et la relecture des acteurs `needs_review` sont les garde-fous. Loguer les rejets pour améliorer le prompt.
- **Qualité de `home_layers`** : elle conditionne tout le calcul origine / distance. La liste initiale doit être relue à la main.
- **Tickers des paniers** et **coefficients du score et du poids** : propositions de départ, à valider et ajuster après quelques semaines de données. Les symboles de l'Annexe A n'ont pas pu être testés sur Yahoo Finance au moment de la rédaction : la validation automatique de la Phase C est obligatoire.
- **Panier Modèles étroit** : deux titres chinois cotés à Hong Kong, sensibles à des facteurs propres à la Chine (régulation, restrictions d'export, sentiment du marché de Hong Kong). Ne pas sur-interpréter ce panier seul ; l'indice de prix des tokens est le signal principal de la couche tant que des labos occidentaux ne sont pas cotés.
- **Confirmation des goulots par les prix** : un panier peut surperformer pour des raisons sans rapport avec une pénurie (résultats, rotation sectorielle). C'est pour cela qu'on exige d'abord le signal presse, et que la confirmation par le prix seul ne suffit jamais.
- **Prix du GPU-heure** : pas de source publique gratuite identifiée à ce stade. Relevé manuel en attendant.
- **Énergie par token et par utilisateur** : estimations très incertaines, toujours affichées comme telles.
- **yfinance** : non officiel, peut casser sans préavis (déjà noté dans `SPECS_MULTIDOMAINE.md`). Fallback : garder les dernières valeurs et le signaler dans le bandeau « données à revoir ».
- **Le dashboard montre des signaux, il ne prédit pas.** Ne pas afficher de recommandation d'achat ou de vente.

## Annexe A — Liste proposée des acteurs et des paniers (à amender par Jérémie)

**Mode d'emploi.** Barrer ou supprimer une ligne pour la retirer, en ajouter librement, changer « Panier » en oui/non. Claude Code génère `data/actors_manual.json` et les paniers de `cake_layers.json` à partir de cette annexe **après** relecture.

- **Ticker** : au format yfinance (suffixes `.PA` Paris, `.DE` Xetra, `.AS` Amsterdam, `.MI` Milan, `.L` Londres, `.SW` Zurich, `.HK` Hong Kong, `.KS` Séoul, `.T` Tokyo, `.TW` Taïwan, `.SS` Shanghai). `—` = non coté. Les symboles n'ont pas pu être testés sur Yahoo Finance au moment de la rédaction : Claude Code doit les valider en Phase C.
- **Panier** : oui = entre dans le panier équipondéré de la couche. Non = suivi comme acteur seulement (non coté, conglomérat, ou activité IA trop marginale dans le cours). « À discuter » = mon avis est partagé, c'est à toi de trancher.
- **Sous-couche** : celle du §10, pour préparer la v2.

### A.1 Énergie

| Acteur | Ticker | Sous-couche | Panier | Note |
|---|---|---|---|---|
| Cameco | CCJ | Combustible et matières | oui | Uranium |
| Centrus Energy | LEU | Combustible et matières | oui | Uranium enrichi (HALEU) pour les petits réacteurs |
| GE Vernova | GEV | Équipement | oui | Turbines à gaz, réseau |
| Siemens Energy | ENR.DE | Équipement | oui | Turbines, transformateurs |
| Bloom Energy | BE | Équipement | oui | Piles à combustible installées sur site des datacenters |
| Mitsubishi Heavy Industries | 7011.T | Équipement | non | Turbines, mais conglomérat (défense, aéronautique) |
| Constellation Energy | CEG | Production | oui | Premier exploitant nucléaire américain, contrats avec des hyperscalers |
| Vistra | VST | Production | oui | |
| Talen Energy | TLN | Production | oui | Centrale nucléaire adossée à un campus de datacenters |
| NRG Energy | NRG | Production | oui | |
| NextEra Energy | NEE | Production | à discuter | Gros renouvelable, mais en partie utility régulée : signal IA dilué |
| Engie | ENGI.PA | Production | à discuter | Signal IA dilué ; utile pour une lecture européenne |
| Oklo | OKLO | Production | oui | Petits réacteurs, pré-revenus : très volatil |
| NuScale Power | SMR | Production | oui | Petits réacteurs |
| EDF | — | Production | non | Non cotée (État) ; acteur clé pour la souveraineté française |
| Quanta Services | PWR | Réseau et raccordement | oui | Construction de lignes et de raccordements |
| Prysmian | PRY.MI | Réseau et raccordement | oui | Câbles haute tension |
| Nexans | NEX.PA | Réseau et raccordement | oui | Câbles haute tension |
| National Grid | NG.L | Réseau et raccordement | non | Utility régulée, peu sensible à l'IA en bourse |

### A.2 Puces

| Acteur | Ticker | Sous-couche | Panier | Note |
|---|---|---|---|---|
| ASML | ASML | Équipement de fabrication | oui | Lithographie EUV ; aussi `ASML.AS` |
| Applied Materials | AMAT | Équipement de fabrication | oui | |
| Lam Research | LRCX | Équipement de fabrication | oui | |
| KLA | KLAC | Équipement de fabrication | oui | |
| Tokyo Electron | 8035.T | Équipement de fabrication | oui | |
| ASM International | ASM.AS | Équipement de fabrication | oui | |
| TSMC | TSM | Fonderie | oui | ADR ; aussi `2330.TW` |
| Intel | INTC | Fonderie | à discuter | Fonderie et conception : signal mixte |
| Samsung Electronics | 005930.KS | Mémoire | non | Mémoire, fonderie et smartphones : conglomérat |
| SK Hynix | 000660.KS | Mémoire | oui | Leader de la HBM |
| Micron | MU | Mémoire | oui | |
| ASE Technology | ASX | Packaging avancé | oui | |
| Amkor | AMKR | Packaging avancé | oui | |
| BE Semiconductor | BESI.AS | Packaging avancé | oui | Équipement de collage hybride |
| Nvidia | NVDA | Conception | oui | Home layers à discuter : puces + infrastructure (DGX Cloud) ? + modèles (Nemotron) ? |
| AMD | AMD | Conception | oui | |
| Broadcom | AVGO | Conception | oui | ASIC sur mesure (TPU, puce OpenAI) et réseau |
| Marvell | MRVL | Conception | oui | ASIC sur mesure, optique |
| Arm | ARM | Conception | oui | |
| Cerebras | CBRS | Conception | oui | Coté sur le Nasdaq depuis mai 2026 ; propose aussi du cloud (home layers : puces + infrastructure ?) |
| Cambricon | 688256.SS | Conception | oui | Puces IA chinoises |
| Qualcomm | QCOM | Conception | non | Surtout mobile |
| Astera Labs | ALAB | Interconnexion et réseau | oui | |
| Credo | CRDO | Interconnexion et réseau | oui | |
| Coherent | COHR | Interconnexion et réseau | oui | Optique |
| Lumentum | LITE | Interconnexion et réseau | oui | Optique |

### A.3 Infrastructure

| Acteur | Ticker | Sous-couche | Panier | Note |
|---|---|---|---|---|
| Equinix | EQIX | Foncier et construction | oui | Colocation |
| Digital Realty | DLR | Foncier et construction | oui | Colocation |
| GDS Holdings | GDS | Foncier et construction | à discuter | Datacenters en Chine |
| Vertiv | VRT | Alimentation et refroidissement | oui | |
| Schneider Electric | SU.PA | Alimentation et refroidissement | oui | Aussi présent sur le réseau (énergie) |
| Legrand | LR.PA | Alimentation et refroidissement | oui | |
| Eaton | ETN | Alimentation et refroidissement | oui | |
| Modine | MOD | Alimentation et refroidissement | à discuter | Refroidissement, mais activité IA minoritaire |
| Super Micro | SMCI | Systèmes et réseau du datacenter | oui | Serveurs |
| Dell | DELL | Systèmes et réseau du datacenter | à discuter | Serveurs IA, mais aussi PC |
| HPE | HPE | Systèmes et réseau du datacenter | à discuter | |
| Celestica | CLS | Systèmes et réseau du datacenter | oui | |
| Hon Hai (Foxconn) | 2317.TW | Systèmes et réseau du datacenter | à discuter | Serveurs IA, mais aussi électronique grand public |
| Wiwynn | 6669.TW | Systèmes et réseau du datacenter | oui | Serveurs pour hyperscalers |
| Arista Networks | ANET | Systèmes et réseau du datacenter | oui | |
| Oracle | ORCL | Cloud hyperscaler | oui | Cours aujourd'hui tiré par le cloud IA ; home layers : infrastructure + applications |
| CoreWeave | CRWV | Neocloud et location de GPU | oui | |
| Nebius | NBIS | Neocloud et location de GPU | oui | |
| IREN | IREN | Neocloud et location de GPU | oui | Ancien mineur de bitcoin devenu neocloud |
| Applied Digital | APLD | Neocloud et location de GPU | oui | |
| Nscale | — | Neocloud et location de GPU | non | IPO déposée à New York (ticker proposé `NSCL`, à vérifier) : à ajouter dès la cotation |
| Lambda | — | Neocloud et location de GPU | non | Non coté, IPO envisagée |
| Crusoe | — | Neocloud et location de GPU | non | Non coté |

### A.4 Modèles

| Acteur | Ticker | Sous-couche | Panier | Note |
|---|---|---|---|---|
| Zhipu AI (Knowledge Atlas) | 2513.HK | Modèles frontière | oui | Coté depuis janvier 2026 |
| MiniMax | 0100.HK | Modèles frontière | oui | Coté depuis janvier 2026 |
| Anthropic | — | Modèles frontière | non | S-1 public, ticker proposé `ANT` : passer à « oui » dès la première cotation. Home layers : modèles + applications |
| OpenAI | — | Modèles frontière | non | Home layers : modèles + applications ; Stargate = mouvement en cours vers l'infrastructure |
| Mistral AI | — | Modèles frontière | non | Acteur clé pour la souveraineté européenne |
| DeepSeek | — | Modèles ouverts | non | |
| Moonshot AI | — | Modèles frontière | non | |
| Cohere | — | Modèles frontière | non | Orienté entreprise |
| Hugging Face | — | Outillage | non | Plateforme de modèles ouverts |
| Databricks | — | Données | non | |
| Scale AI | — | Données | non | |

### A.5 Applications

| Acteur | Ticker | Sous-couche | Panier | Note |
|---|---|---|---|---|
| iShares Expanded Tech-Software ETF | IGV | Applications d'entreprise horizontales | oui | ETF logiciel : approximation de la couche |
| Palantir | PLTR | Applications d'entreprise horizontales | oui | |
| ServiceNow | NOW | Applications d'entreprise horizontales | oui | |
| Salesforce | CRM | Applications d'entreprise horizontales | oui | |
| SAP | SAP.DE | Applications d'entreprise horizontales | à discuter | Lecture européenne |
| Adobe | ADBE | Applications d'entreprise horizontales | à discuter | Souvent vu comme menacé par l'IA générative : signal inverse possible |
| Tempus AI | TEM | Verticales métier | oui | Santé |
| Recursion | RXRX | Verticales métier | oui | Découverte de médicaments (entraîne aussi ses propres modèles) |
| Duolingo | DUOL | Grand public | oui | |
| AppLovin | APP | Grand public | oui | Publicité pilotée par l'IA |
| Symbotic | SYM | IA physique | oui | Robotique d'entrepôt |
| Cursor (Anysphere) | — | Outils développeurs et agents de code | non | |
| Perplexity | — | Grand public | non | |
| Harvey | — | Verticales métier | non | Juridique |
| Figure AI | — | IA physique | non | Robots humanoïdes |

### A.6 Acteurs multi-couches (jamais dans les paniers)

Pour eux, l'important est la liste `home_layers`, qui conditionne le calcul de l'origine et de la distance (§4.3). **C'est la partie à relire avec le plus d'attention.**

| Acteur | Ticker | Home layers proposés | Note |
|---|---|---|---|
| Microsoft | MSFT | infrastructure, modèles, applications | Modèles maison en plus du partenariat OpenAI |
| Alphabet | GOOGL | puces, infrastructure, modèles, applications | TPU, Google Cloud, Gemini |
| Amazon | AMZN | puces, infrastructure, modèles, applications | Trainium, AWS, Nova |
| Meta | META | infrastructure, modèles, applications | Puces MTIA à évaluer : home layer ou mouvement en cours ? |
| Apple | AAPL | puces, applications | |
| Tesla | TSLA | énergie, puces, applications | Megapack et solaire ; puces de conduite autonome ; véhicules et robots. Projet de fabrication de puces avec SpaceX à suivre comme mouvement |
| SpaceX (xAI) | SPCX | infrastructure, modèles, applications | Coté depuis juin 2026 ; xAI absorbé en février 2026 (Colossus, Grok, X) |
| Alibaba | BABA | puces, infrastructure, modèles, applications | T-Head, Alibaba Cloud, Qwen ; aussi `9988.HK` |
| Tencent | 0700.HK | infrastructure, modèles, applications | |
| Baidu | BIDU | puces, infrastructure, modèles, applications | Kunlunxin, Ernie |
| ByteDance | — | infrastructure, modèles, applications | Non coté |
