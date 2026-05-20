# Veille-IA / AI Radar

**Un agrégateur de veille IA entièrement automatisé**, de la collecte à la diffusion.

## Ce que ça fait

Chaque matin à 8h (CET), un pipeline GitHub Actions :

1. **Collecte** les actualités IA depuis ~100+ sources : flux RSS (blogs, médias tech, institutions), Bluesky (13 requêtes thématiques), et anciennement Twitter
2. **Classe** chaque article dans 6 catégories (Innovation/Tech, Politique/Régulation, Business, Société/Éthique, Recherche Académique, Drama/Controverses) via Groq (LLM rapide)
3. **Détecte les sujets chauds** en croisant 4 signaux : Google Trends, débats Hacker News, repos GitHub trending, et auto-bootstrap depuis la DB
4. **Priorise** les articles : `hot_topic` → `supa_hot` (badge 🌋) si fortement cross-mentionné
5. **Génère un résumé** en 1 phrase par article (Groq), **analyse le sentiment**, et **envoie un digest sur Telegram**
6. **Persiste** tout dans Supabase (PostgreSQL)

## Stack technique

| Composant | Technologie |
|-----------|-------------|
| Pipeline | Python 3.12, `asyncio`, `feedparser`, `aiohttp` |
| LLM | Groq (classification, résumés, expansion de requêtes) |
| DB | Supabase (Postgres) |
| Dashboard | Streamlit (`dashboard.py`, 1600+ lignes) avec globe D3 canvas |
| Recherche | Sémantique avec expansion de requêtes Groq + scoring multi-champ |
| Automatisation | GitHub Actions (3 workflows : collecte quotidienne, dashboard, digest hebdo) |
| Social scraping | Bluesky API (`bluesky_scraper.py`) |

## Structure (plate, comme voulu)

```
main.py              — pipeline principal (860 lignes)
dashboard.py         — UI Streamlit (1600 lignes)
bluesky_scraper.py   — scraper Bluesky
sources.json         — toutes les sources (~100+ entrées avec catégorie, pays)
.github/workflows/   — 3 workflows CI/CD
weekly_digest/       — génération du digest hebdomadaire
_globe_component/    — composant globe D3 (halftone canvas)
```

## Workflow Détaillé du Script Python
Cron Job : GitHub Actions lance le script main.py à 08h00 CET tous les jours.
Initialisation : Chargement du fichier sources.json.
Fetch : Boucle asynchrone pour télécharger les RSS et requêter l'API Hacker News.
Dédoublonnage : Comparaison des URL récupérées avec les URL déjà présentes dans Notion (ou conservation d'un fichier cache/historique sur GitHub) pour éviter les doublons.
Classification Groq : Envoi des nouveaux articles par lots (batchs) ou requêtes asynchrones à l'API Groq pour assignation Catégorie/Sentiment.
Push Notion : Insertion des lignes qualifiées dans la base de données Notion.
Notification : Calcul des statistiques du jour et envoi du message via l'API Telegram.
