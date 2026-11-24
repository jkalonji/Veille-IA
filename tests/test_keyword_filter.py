import pytest

import main


@pytest.mark.parametrize("text", [
    "Il s'agit d'une réforme des retraites",   # "agi" in "s'agit"
    "storage costs average 3% more",           # "rag" in "average"
    "Output of the steel plant",               # "tpu" in "output"
    "Mais il a dit jamais, said the minister",  # "ai" in mais/jamais/said
])
def test_ai_filter_rejects_substring_matches(text):
    assert not main._is_relevant_ai_article(text)


@pytest.mark.parametrize("text", [
    "OpenAI ships a RAG toolkit",
    "Why LLMs hallucinate",                    # plural of a strong keyword
    "L'intelligence artificielle au travail",
    "Nvidia unveils a new AI chip",            # weak keywords only, 3 of them
])
def test_ai_filter_keeps_ai_articles(text):
    assert main._is_relevant_ai_article(text)


def test_gdelt_filter_whole_words():
    assert not main._is_relevant_gdelt_article("New software moves forward with a coupon")
    assert main._is_relevant_gdelt_article("Protests spread after the coup")
