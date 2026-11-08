"""
"ℹ️ Infos" button next to each dashboard chart: the method behind the chart and
how to read its scores. Shared by the Streamlit app and the static export.

Thresholds quoted here mirror data/cake_layers.json;
tests/test_chart_info.py fails if they drift apart.
"""

from html import escape

_LABELS = {"fr": ("Méthode", "Comment lire"), "en": ("Method", "How to read")}

INFO: dict[str, dict] = {
    # -- AI layer cake ---------------------------------------------------------
    "cake": {
        "title": "Le gâteau de l'IA",
        "method": [
            "Cinq couches, de bas en haut : énergie, puces, infrastructure, modèles, applications. Groq attribue à chaque article de 0 à 3 couches.",
            "Une flèche est un mouvement inter-couches : un acteur entre dans une couche où il n'était pas encore actif. Son origine est la couche d'attache de l'acteur la plus proche de la cible. Seuls les 12 meilleurs scores de la période sont dessinés.",
            "Jauge rouge : z-score de rareté de la couche, ramené entre 0 (z ≤ 0) et 1 (z ≥ 3). Le z-score compare le solde « tension − détente » des articles sur 7 jours aux 90 jours précédents de la même couche.",
            "Courbe bleue et pourcentage : panier boursier équipondéré de la couche sur la période. Courbe jaune (Modèles) : indice du prix des tokens en sortie (médiane des 20 modèles payants les plus utilisés sur OpenRouter). Chaque courbe a sa propre échelle.",
            "Signaux forts : histoires chaudes concentrées sur une couche, et mouvements dont le score atteint 5.",
        ],
        "reading": [
            "Flèche pleine = descente : l'acteur sécurise un approvisionnement en amont. Flèche pointillée = montée : il conquiert un débouché en aval.",
            "Plus la flèche est épaisse, plus le score est élevé. Le détail du score est dans les fiches événements.",
            "Jauge remplie à moitié = z de 1,5, le seuil à partir duquel la tension est inhabituelle pour cette couche. Une jauge en pointillés indique une normale en reconstruction : les sources de la couche ont changé, la lecture est moins fiable pendant 30 jours.",
            "« Goulot signalé » = couche au z le plus haut, au-dessus de 1,5 pendant 3 jours. « Goulot confirmé » = un signal de prix de la même couche va dans le même sens.",
            "Le nombre d'articles mesure l'attention médiatique, pas l'importance réelle d'une couche.",
        ],
    },
    "cake_counts": {
        "title": "Articles par couche",
        "method": [
            "Groq attribue à chaque article de 0 à 3 couches : énergie, puces, infrastructure, modèles, applications.",
            "La barre est proportionnelle au nombre d'articles de la couche, sur la période sélectionnée.",
        ],
        "reading": [
            "Un article peut compter dans plusieurs couches : le total dépasse donc le nombre d'articles.",
            "C'est une mesure d'attention médiatique, qui dépend aussi du nombre de sources suivies pour chaque couche.",
        ],
    },
    "cake_events": {
        "title": "Score d'un mouvement",
        "method": [
            "Score = 10 × (0,35·D + 0,30·E + 0,35·W) × N, plafonné à 10.",
            "D, distance entre origine et cible : 1 couche = 0,40 · 2 = 0,60 · 3 = 0,85 · 4 = 1.",
            "E, engagement : annonce = 0,40 · contrat = 0,70 · mise en service = 1.",
            "W, poids de l'acteur entre 0 et 1 : taille 30 % (chiffre d'affaires, échelle log), influence 30 % (réaction boursière médiane autour de ses actualités), longévité 15 % (ancienneté, plafonnée à 15 ans), implantation 25 % (PageRank des co-mentions et des relations entre acteurs).",
            "N, nouveauté : ×1,2 si aucun mouvement n'a franchi ce couple origine → cible depuis 180 jours, sinon ×1.",
            "Un même mouvement est suivi sur 90 jours : son score monte quand il progresse d'annonce à contrat, puis à mise en service.",
        ],
        "reading": [
            "Score ≥ 7 : mouvement majeur (badge, alerte Telegram). Score ≥ 5 : signal fort. En dessous : à surveiller.",
            "« ≈ » devant le poids = estimation peu fiable : une composante manque ou l'influence n'a pas encore 3 observations. Le score est alors à prendre avec prudence.",
            "« Acteur à relire » : ses couches d'attache n'ont pas été vérifiées à la main, or elles déterminent l'origine et la distance.",
            "Gagnants, perdants et acteurs court-circuités sont extraits des articles par Groq : c'est une lecture de presse, pas une analyse financière.",
        ],
    },
    "matrix": {
        "title": "Mouvements origine → cible",
        "method": [
            "Chaque case compte les mouvements inter-couches apparus sur les 90 derniers jours : la ligne est la couche d'origine, la colonne la couche cible.",
            "Les boutons affichent le nombre de mouvements, la somme de leurs scores, ou l'écart de chacun avec les 90 jours précédents.",
            "La diagonale est vide : un mouvement change de couche par définition.",
        ],
        "reading": [
            "Les couches vont d'Énergie à Applications, en partant du coin bas-gauche. Les cases en bas à droite de la diagonale sont des montées (conquête d'un débouché), celles en haut à gauche des descentes (sécurisation d'un approvisionnement).",
            "Beaucoup de mouvements mais une somme de scores modeste : de petits acteurs ou de simples annonces. Peu de mouvements et une somme élevée : quelques opérations lourdes.",
            "En vue évolution, orange = plus que sur les 90 jours précédents, bleu = moins. Les premiers mois, l'historique est court et l'écart surestime la hausse.",
        ],
    },
    "value_staircase": {
        "title": "Valeur d'un MWh à chaque couche",
        "method": [
            "Chaque étage convertit un prix observable en dollars par MWh d'électricité au compteur, refroidissement et pertes compris (PUE).",
            "Énergie : prix de gros de l'électricité aux États-Unis.",
            "Puces : électricité + prix de l'accélérateur de référence (H100) réparti sur l'énergie consommée pendant sa durée de vie.",
            "Infrastructure : prix du GPU-heure ÷ énergie consommée en une heure.",
            "Modèles : prix du million de tokens ÷ énergie d'un million de tokens.",
            "Applications : revenu mensuel par abonné ÷ énergie consommée par abonné.",
            "Valeur ajoutée = prix de l'étage − prix de l'étage inférieur. Multiplicateur = prix de l'étage ÷ prix de l'étage inférieur.",
        ],
        "reading": [
            "L'échelle est logarithmique : une marche de même hauteur correspond au même multiplicateur.",
            "▼ en rouge = valeur ajoutée négative : la couche vend sous le coût des couches inférieures. C'est un signal de marge comprimée ou de subvention, pas une erreur.",
            "« ≈ » et barre claire = estimation. Les étages Modèles et Applications sont toujours des estimations très incertaines : avec l'énergie par token d'un modèle plus petit, la couche Modèles changerait nettement de niveau. Retiens l'ordre de grandeur, pas le chiffre.",
            "La vue évolution suit la valeur ajoutée de chaque étage, mois par mois.",
        ],
    },
    "layer_curves": {
        "title": "Paniers par couche et tensions",
        "method": [
            "Haut : panier boursier équipondéré de chaque couche, ramené à 100 au début de la courbe. QQQ (Nasdaq 100) sert de référence.",
            "Bas : z-score de rareté. Pour chaque couche, solde (articles en tension − articles en détente) ÷ articles sur 7 jours, comparé aux 90 jours précédents de cette même couche. Il faut au moins 30 jours d'historique.",
            "La tension et la détente de chaque article sont évaluées par Groq.",
        ],
        "reading": [
            "Au-dessus de 100, le panier a progressé depuis le début de la courbe. Au-dessus de QQQ, il fait mieux que le marché technologique.",
            "z = 0 : tension normale pour cette couche. Ligne rouge (z = 1,5) : tension inhabituelle. Le z-score se compare entre couches, car chacune a sa propre référence.",
            "Zone orange : goulot confirmé, c'est-à-dire z > 1,5 pendant 3 jours pour la couche la plus tendue, avec un signal de prix sur 30 jours (panier à +5 points contre QQQ, électricité à +10 %, hausse du GPU-heure ou des tokens selon la couche). La couche Applications n'a pas de signal de prix fiable : elle peut être signalée, pas confirmée.",
            "Un z élevé traduit un discours de pénurie dans la presse. Il ne mesure pas directement l'offre et la demande, d'où la confirmation par les prix.",
            "Le panier Modèles ne contient que deux titres cotés (la plupart des laboratoires ne sont pas en bourse) : il représente mal la couche.",
        ],
    },
    # -- Generic charts ---------------------------------------------------------
    "globe": {
        "title": "Couverture mondiale",
        "method": [
            "Chaque pays est coloré selon le nombre d'articles qui lui sont rattachés sur la période sélectionnée.",
            "Le pays d'un article est celui identifié par Groq à partir de son contenu. Les articles sans pays précis (« Global ») n'apparaissent pas.",
        ],
        "reading": [
            "L'intensité de la couleur suit le nombre d'articles rattachés au pays.",
            "Cela reflète aussi le choix des sources suivies : un pays peu coloré n'est pas forcément peu actif.",
            "Un clic sur un pays filtre les articles affichés plus bas.",
        ],
    },
    "categories": {
        "title": "Répartition par catégorie",
        "method": [
            "Groq classe chaque article dans une seule catégorie.",
            "Les barres donnent le nombre d'articles par catégorie sur la période sélectionnée, et leur part du total.",
        ],
        "reading": [
            "La catégorie dominante indique ce dont parlent le plus les sources suivies, pas ce qui compte le plus.",
        ],
    },
    "story_cloud": {
        "title": "Nuage d'histoires",
        "method": [
            "Une histoire regroupe les articles des 30 derniers jours qui parlent d'un même sujet (événement, acteur ou problématique), identifié par Groq.",
            "Il faut au moins 3 articles de 2 sources différentes. Seules les histoires qui ont de l'actualité aujourd'hui apparaissent dans le nuage.",
        ],
        "reading": [
            "Taille et couleur suivent le nombre d'articles du jour, indiqué en exposant. La plus active est au centre.",
            "L'échelle est relative aux histoires du jour : un mot très gros un jour calme peut représenter peu d'articles.",
            "🌋 = au moins 5 articles aujourd'hui sur cette histoire.",
        ],
    },
    # -- "Who really invests in AI?" (English, like the tabs) -------------------
    "company_valuations": {
        "lang": "en",
        "title": "Most valuable AI model makers",
        "method": [
            "Companies that develop their own frontier AI models. Nvidia, TSMC and Apple are excluded.",
            "Listed companies: stock market capitalisation (companiesmarketcap.com). Private companies: post-money valuation of the latest closed funding round, each with its own source.",
        ],
        "reading": [
            "The figure is the value of the whole company, including non-AI businesses: it overstates the AI share of the tech giants.",
            "Market caps move daily, while a private valuation is frozen at its last round, so the two are not strictly comparable.",
            "Hatched bars and “≈” mark low-confidence figures.",
        ],
    },
    "company_revenues": {
        "lang": "en",
        "title": "Revenue of the AI model makers",
        "method": [
            "Revenue over the last 12 months, for the same companies as the Valuations tab. Companies with no known figure are left out.",
            "Listed companies: companiesmarketcap.com. Private companies: audited filings when available, otherwise press reports. Annualised run-rates are never used.",
        ],
        "reading": [
            "Whole-company revenue: for the tech giants, most of it does not come from AI.",
            "Compare with the Valuations tab to see how much future growth the market is pricing in.",
            "Hatched bars and “≈” mark press-reported, unaudited figures.",
        ],
    },
    "hardware_imports": {
        "lang": "en",
        "title": "Net imports of servers & GPU parts",
        "method": [
            "Imports minus exports of HS codes 847150 (servers and computing units) and 847330 (parts, including GPU cards), from UN Comtrade, divided by GDP.",
            "Uses the most recent year with enough reporting countries. Economies below $50bn of GDP and net exporters are left out.",
        ],
        "reading": [
            "A higher percentage means the country spends more on computing hardware relative to the size of its economy. This is money actually spent, not an announcement.",
            "These codes include ordinary servers, so this is a proxy for AI hardware, not a direct measure.",
            "Transit and assembly hubs can be distorted. Countries that do not report customs data, such as the UAE, are missing.",
        ],
    },
    "public_compute": {
        "lang": "en",
        "title": "Public AI supercomputers",
        "method": [
            "Sum of the Rmax performance (HPL benchmark, PFlop/s) of GPU-accelerated TOP500 systems at research, academic and government sites, per $1tn of GDP.",
            "Refreshed with each TOP500 list, in June and November.",
        ],
        "reading": [
            "A higher value means more public computing capacity relative to the size of the economy.",
            "Private clusters are rarely submitted to TOP500, so this measures the public effort only, not a country's total AI compute.",
            "HPL measures double-precision scientific computing, which is only an approximation of AI training capacity.",
        ],
    },
    "announcements": {
        "lang": "en",
        "title": "Announced public AI budgets",
        "method": [
            "Public AI budgets announced in official documents, each with its own source, spread evenly over the length of the plan and divided by GDP.",
            "Press-reported targets with no official budget, such as Project Transcendence or MGX, are excluded.",
        ],
        "reading": [
            "A higher percentage means a bigger stated commitment per year relative to the economy.",
            "These are announcements, not money spent: compare with the Hardware and Supercomputers tabs to see whether a country follows through.",
            "Perimeters differ between national plans. Hatched bars and “≈” mark low-confidence figures.",
        ],
    },
}

INFO_CSS = """
<style>
  .chart-info { font-family:'DM Mono', ui-monospace, Menlo, Consolas, monospace; margin:6px 8px; }
  .chart-info summary { display:inline-block; cursor:pointer; list-style:none; font-size:11px; color:#c8d4ec;
    background:#0c1222; border:1px solid #1a2540; border-radius:6px; padding:3px 9px; }
  .chart-info summary::-webkit-details-marker { display:none; }
  .chart-info summary:hover, .chart-info[open] summary { border-color:#4488ff; color:#e8eef8; }
  .chart-info summary:focus-visible { outline:2px solid #4488ff; outline-offset:2px; }
  .chart-info__body { margin-top:6px; padding:10px 14px; background:#0c1222; border:1px solid #1a2540;
    border-radius:8px; font-size:12px; line-height:1.6; color:#c8d4ec; }
  .chart-info__body h4 { margin:8px 0 2px; font-size:12px; font-weight:600; color:#e8eef8; }
  .chart-info__body h4:first-child { margin-top:0; }
  .chart-info__body ul { margin:0; padding-left:18px; }
</style>"""


def _sections(key: str) -> tuple[str, str, list[str], str, list[str]]:
    entry = INFO[key]
    method_label, reading_label = _LABELS[entry.get("lang", "fr")]
    return entry["title"], method_label, entry["method"], reading_label, entry["reading"]


def info_html(key: str) -> str:
    """Collapsible button and panel, without JS (static export and HTML blocks).
    The page must include INFO_CSS once."""
    _, method_label, method, reading_label, reading = _sections(key)

    def items(lines: list[str]) -> str:
        return "".join(f"<li>{escape(line)}</li>" for line in lines)

    return (f'<details class="chart-info"><summary>ℹ️ Infos</summary><div class="chart-info__body">'
            f"<h4>{method_label}</h4><ul>{items(method)}</ul>"
            f"<h4>{reading_label}</h4><ul>{items(reading)}</ul></div></details>")


def info_markdown(key: str) -> str:
    """Panel content as Streamlit markdown. `$` is escaped: Streamlit reads it as LaTeX."""
    title, method_label, method, reading_label, reading = _sections(key)

    def items(lines: list[str]) -> str:
        return "\n".join(f"- {line}" for line in lines)

    text = f"##### {title}\n\n**{method_label}**\n\n{items(method)}\n\n**{reading_label}**\n\n{items(reading)}"
    return text.replace("$", r"\$")


def st_info(key: str) -> None:
    """The same button in Streamlit, as a native popover."""
    import streamlit as st
    with st.popover("ℹ️ Infos"):
        st.markdown(info_markdown(key))
