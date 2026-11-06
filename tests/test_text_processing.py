
import pytest

from src.projet.text_processing import nettoyer_texte


@pytest.mark.parametrize(
    "texte, attendu",
    [
        (
            "COVID-19: New VACCINE!",
            "covid 19 new vaccine"
        ),
        (
            "Coronavirus\nResearch",
            "coronavirus research"
        ),
        (
            "COVID_19\t&\nVACCINE!!!",
            "covid 19 vaccine"
        ),
        (
            "The virus, the VIRUS.",
            "the virus the virus"
        ),
        (
            "ÉTUDES, Virus.",
            "études virus"
        ),
        (
            "",
            ""
        ),
        (
            "    ",
            ""
        ),
    ],
)
def test_nettoyer_texte(texte: str, attendu: str) -> None:
    assert nettoyer_texte(texte) == attendu

def test_nettoyage_idempotent() -> None:
    texte = "COVID-19: New VACCINE!"
    resultat = nettoyer_texte(texte)
    assert nettoyer_texte(resultat) == resultat