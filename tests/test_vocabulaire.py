from datetime import datetime, timezone
import pytest
from projet.corpus import Corpus
from projet.document import RedditDocument
from projet.search_engine import SearchEngine
from scipy.sparse import isspmatrix_csr

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
    
def test_calculer_frequences(corpus_vierge):
    ajouter_document(
        corpus_vierge,
        "COVID-19, virus virus."
    )
    ajouter_document(
        corpus_vierge,
        "VACCINE virus!"
    )
    freq = corpus_vierge.calculer_frequences()
    frequences = freq.set_index("mot")[
        "occurrences"
    ].to_dict()
    assert frequences == {
        "19": 1,
        "covid": 1,
        "vaccine": 1,
        "virus": 3
    }
    assert freq["occurrences"].sum() == 6

def test_frequences_identifiants(corpus_vierge):
    ajouter_document(
        corpus_vierge,
        "Virus COVID vaccine"
    )
    vocabulaire = corpus_vierge.construire_vocabulaire()
    freq = corpus_vierge.calculer_frequences()
    identifiants = freq.set_index("mot")[
        "identifiant"
    ].to_dict()
    assert identifiants == vocabulaire

def test_frequences_corpus_vide(corpus_vierge):
    freq = corpus_vierge.calculer_frequences()
    assert freq.empty
    assert list(freq.columns) == [
        "mot",
        "identifiant",
        "occurrences",
        "document_frequency"
    ]

def test_document_frequency(corpus_vierge):
    ajouter_document(
        corpus_vierge,
        "covid virus virus"
    )
    ajouter_document(
        corpus_vierge,
        "covid vaccine"
    )
    ajouter_document(
        corpus_vierge,
        "virus treatment"
    )
    freq = corpus_vierge.calculer_frequences()
    stats = freq.set_index("mot")
    assert stats.loc["virus", "occurrences"] == 3
    assert stats.loc["virus", "document_frequency"] == 2
    assert stats.loc["covid", "document_frequency"] == 2
    assert stats.loc["vaccine", "document_frequency"] == 1
    assert stats.loc["treatment", "document_frequency"] == 1

def test_document_frequency_repetitions(corpus_vierge):
    ajouter_document(
        corpus_vierge,
        "virus virus virus virus"
    )
    freq = corpus_vierge.calculer_frequences()
    stats = freq.set_index("mot")
    assert stats.loc["virus", "occurrences"] == 4
    assert stats.loc["virus", "document_frequency"] == 1

def test_document_frequency_coherence(corpus_vierge):
    ajouter_document(
        corpus_vierge,
        "covid covid vaccine"
    )
    ajouter_document(
        corpus_vierge,
        "virus vaccine"
    )
    freq = corpus_vierge.calculer_frequences()
    nombre_documents = corpus_vierge.get_document_count()
    assert (freq["document_frequency"] <= nombre_documents).all()
    assert (freq["document_frequency"] <= freq["occurrences"]).all()
    assert len(freq) == len(corpus_vierge.construire_vocabulaire())
    
def test_vocab_structure(corpus_vierge):
    ajouter_document(
        corpus_vierge,
        "covid virus virus"
    )
    ajouter_document(
        corpus_vierge,
        "covid vaccine"
    )
    ajouter_document(
        corpus_vierge,
        "virus treatment"
    )
    moteur = SearchEngine(corpus_vierge)
    assert moteur.vocab == {
        "covid": {
            "id": 0,
            "total_occurrences": 2,
            "document_frequency": 2
        },
        "treatment": {
            "id": 1,
            "total_occurrences": 1,
            "document_frequency": 1
        },
        "vaccine": {
            "id": 2,
            "total_occurrences": 1,
            "document_frequency": 1
        },
        "virus": {
            "id": 3,
            "total_occurrences": 3,
            "document_frequency": 2
        }
    }

def test_vocab_identifiants(corpus_vierge):
    ajouter_document(
        corpus_vierge,
        "Virus COVID vaccine"
    )
    moteur = SearchEngine(corpus_vierge)
    identifiants = {
        mot: infos["id"]
        for mot, infos in moteur.vocab.items()
    }
    assert identifiants == (
        corpus_vierge.construire_vocabulaire()
    )

def test_vocab_corpus_vide(corpus_vierge):
    moteur = SearchEngine(corpus_vierge)
    assert moteur.vocab == {}

def test_matrice_tf_valeurs(corpus_vierge):
    ajouter_document(
        corpus_vierge,
        "covid virus virus"
    )
    ajouter_document(
        corpus_vierge,
        "covid vaccine"
    )
    ajouter_document(
        corpus_vierge,
        "virus treatment"
    )
    moteur = SearchEngine(corpus_vierge)
    assert moteur.document_ids == [1, 2, 3]
    assert moteur.mat_TF.shape == (3, 4)
    assert isspmatrix_csr(moteur.mat_TF)
    assert moteur.mat_TF.toarray().tolist() == [
        [1, 0, 0, 2],
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]

def test_matrice_tf_repetitions(corpus_vierge):
    document = ajouter_document(
        corpus_vierge,
        "COVID covid virus"
    )
    moteur = SearchEngine(corpus_vierge)
    colonne_covid = moteur.vocab["covid"]["id"]
    colonne_virus = moteur.vocab["virus"]["id"]
    assert moteur.mat_TF[0, colonne_covid] == 2
    assert moteur.mat_TF[0, colonne_virus] == 1
    assert document.text == "COVID covid virus"

def test_matrice_tf_corpus_vide(corpus_vierge):
    moteur = SearchEngine(corpus_vierge)
    assert moteur.vocab == {}
    assert moteur.document_ids == []
    assert moteur.mat_TF.shape == (0, 0)
    assert moteur.mat_TF.nnz == 0
    ajouter_document(corpus_vierge, "")
    moteur = SearchEngine(corpus_vierge)
    assert moteur.mat_TF.shape == (1, 0)

def test_matrice_tf_identifiants_documents(corpus_vierge):
    ajouter_document(corpus_vierge, "a")
    ajouter_document(corpus_vierge, "virus virus")
    ajouter_document(corpus_vierge, "covid")
    corpus_vierge.remove_short_documents(
        minimum_length=3
    )
    moteur = SearchEngine(corpus_vierge)
    assert moteur.document_ids == [2, 3]
    assert moteur.mat_TF.toarray().tolist() == [
        [0, 2],
        [1, 0]
    ]