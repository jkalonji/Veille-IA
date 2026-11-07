"""
AI layer cake — reading the AI news layer by layer (energy → chips →
infrastructure → models → applications). See SPECS_LAYER_CAKE.md.

This module holds the config loader and the article-level tagging helpers
used by main.py (classification prompt, output validation, relevance filter).
"""

import json
import re
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).parent
CONFIG_PATH = ROOT / "data" / "cake_layers.json"

SCARCITY_VALUES = ("tension", "detente", "aucune")


@lru_cache(maxsize=1)
def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def layer_codes() -> list[str]:
    """Layer codes, bottom to top."""
    return [l["code"] for l in sorted(load_config()["layers"], key=lambda l: l["rank"])]


def layer_meta() -> dict[str, dict]:
    return {l["code"]: l for l in load_config()["layers"]}


# ---------------------------------------------------------------------------
# Article tagging (Phase A)
# ---------------------------------------------------------------------------

# Appended to the Groq classification prompt of the "ia" domain.
CAKE_PROMPT = """

- "layers": liste de 0 a 3 couches dont parle l'article, parmi ["energie", "puces", "infrastructure", "modeles", "applications"] :
  energie = production et reseau electriques, nucleaire, gaz, uranium, raccordement des data centers ;
  puces = conception et fabrication de semi-conducteurs, GPU, memoire HBM, fonderies, lithographie ;
  infrastructure = data centers, refroidissement, serveurs, cloud, location de GPU ;
  modeles = entrainement et publication de modeles d'IA, laboratoires d'IA ;
  applications = produits et services construits sur les modeles (assistants, agents, logiciels, robots).
  Liste vide si aucune ne s'applique (ex: debat ethique general).

- "cross_layer": true si l'article decrit un acteur qui ENTRE dans une couche ou il n'est pas encore actif (il construit, achete, concoit, produit, investit ou signe pour y operer) ; false pour une activite dans sa couche habituelle.
  Exemples true : "OpenAI construit sa propre centrale electrique", "Un energeticien publie un modele d'IA".
  Exemples false : "Nvidia presente son nouveau GPU", "OpenAI lance un nouveau modele".

- "scarcity": une valeur parmi ["tension", "detente", "aucune"]. "tension" si l'article decrit une penurie, un rationnement, un allongement des delais ou une capacite saturee dans une des couches de "layers" ; "detente" si une telle tension se resorbe (ex: "la penurie de HBM s'attenue") ; "aucune" sinon."""


def parse_cake_fields(result: dict) -> tuple[list[str], bool, str]:
    """Validate the cake keys of a Groq classification: unknown layer codes
    are dropped, at most 3 layers are kept, defaults are [] / False / 'aucune'."""
    codes = set(layer_codes())
    raw = result.get("layers")
    layers: list[str] = []
    if isinstance(raw, list):
        for code in raw:
            if isinstance(code, str) and code.strip().lower() in codes and code.strip().lower() not in layers:
                layers.append(code.strip().lower())
    layers = layers[:3]
    cross = result.get("cross_layer") is True or str(result.get("cross_layer")).lower() == "true"
    scarcity = str(result.get("scarcity") or "").strip().lower().replace("é", "e")
    if scarcity not in SCARCITY_VALUES:
        scarcity = "aucune"
    # A move or a shortage needs at least one layer to make sense
    if not layers:
        cross, scarcity = False, "aucune"
    return layers, cross, scarcity


@lru_cache(maxsize=None)
def _keyword_re(layer: str) -> re.Pattern | None:
    kws = load_config()["relevance_keywords"].get(layer) or []
    if not kws:
        return None
    # Matched at the start of a word, so plurals ("data centers") still match
    return re.compile(r"(?<!\w)(?:" + "|".join(re.escape(k.lower()) for k in kws) + ")")


def matches_layer_keywords(text: str, layer: str) -> bool:
    """True if `text` mentions one of the relevance keywords of `layer`."""
    pattern = _keyword_re(layer)
    return bool(pattern and pattern.search(text.lower()))
