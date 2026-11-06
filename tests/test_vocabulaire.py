from datetime import datetime, timezone
import pytest
from projet.corpus import Corpus
from projet.document import RedditDocument

@pytest.fixture
def corpus_vierge(monkeypatch):
    monkeypatch.setattr(Corpus, "_Corpus__instance", None)
    return Corpus.get_instance("Corpus de test")

def ajouter_document(corpus: Corpus, texte: str):
    document = RedditDocument(
        title="Document de test",
        author="test_user",
        publication_date=datetime(
            2026, 1, 1, tzinfo=timezone.utc
        ),
        url="https://example.org",
        text=texte,
        comment_count=0
    )
    corpus.add_document(document)
    return document

def test_vocabulaire_sans_doublons(corpus_vierge):
    document1 = ajouter_document(
        corpus_vierge,
        "COVID-19, virus virus."
    )
    ajouter_document(
        corpus_vierge,
        "VACCINE virus!"
    )
    vocabulaire = corpus_vierge.construire_vocabulaire()
    assert vocabulaire == {
        "19": 0,
        "covid": 1,
        "vaccine": 2,
        "virus": 3
    }
    assert document1.text == "COVID-19, virus virus."

def test_vocabulaire_vide(corpus_vierge):
    vocabulaire = corpus_vierge.construire_vocabulaire()
    assert vocabulaire == {}

def test_vocabulaire_stable(corpus_vierge):
    ajouter_document(
        corpus_vierge,
        "Virus COVID vaccine"
    )
    premier = corpus_vierge.construire_vocabulaire()
    deuxieme = corpus_vierge.construire_vocabulaire()
    assert premier == deuxieme
