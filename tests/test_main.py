from datetime import datetime, timezone
import pytest
import main
from projet.corpus import Corpus
from projet.document import RedditDocument
from projet.search_engine import SearchEngine

@pytest.fixture
def corpus_exemple(monkeypatch):
    monkeypatch.setattr(Corpus, "_Corpus__instance", None)
    corpus = Corpus.get_instance("Corpus de test")
    document = RedditDocument(
        title="Article sur le vaccin",
        author="Alice",
        publication_date=datetime(
            2026, 1, 1, tzinfo=timezone.utc
        ),
        url="https://example.org/article",
        text="covid vaccine treatment",
        comment_count=0
    )
    corpus.add_document(document)
    return corpus

def test_interface_recherches_successives(corpus_exemple, monkeypatch, capsys):
    moteur = SearchEngine(corpus_exemple)
    saisies = iter([
        "covid",
        "1",
        "vaccine",
        "",
        "q"
    ])
    monkeypatch.setattr(
        "builtins.input",
        lambda _: next(saisies)
    )
    main.run_search_interface(moteur)
    sortie = capsys.readouterr().out
    assert sortie.count(
        "document(s) trouvé(s)"
    ) == 2
    assert "Article sur le vaccin" in sortie
    assert "Au revoir !" in sortie

def test_interface_saisies_invalides(corpus_exemple, monkeypatch, capsys):
    moteur = SearchEngine(corpus_exemple)
    saisies = iter([
        "",
        "covid",
        "abc",
        "covid",
        "-2",
        "covid",
        "1",
        "q"
    ])
    monkeypatch.setattr(
        "builtins.input",
        lambda _: next(saisies)
    )
    main.run_search_interface(moteur)
    sortie = capsys.readouterr().out
    assert "Veuillez saisir au moins un mot-clé." in sortie
    assert "Veuillez saisir un nombre entier valide." in sortie
    assert "Le nombre doit être supérieur à zéro." in sortie
    assert "Article sur le vaccin" in sortie

def test_application_complete_avec_csv(corpus_exemple, monkeypatch, tmp_path, capsys):
    chemin_csv = tmp_path / "corpus.csv"
    corpus_exemple.save(chemin_csv)
    monkeypatch.setattr(
        main,
        "CORPUS_FILE_PATH",
        chemin_csv
    )
    def api_interdite(corpus):
        pytest.fail(
            "Les API ne doivent pas être appelées "
            "si le CSV existe."
        )
    monkeypatch.setattr(
        main,
        "build_corpus_from_apis",
        api_interdite
    )
    saisies = iter(["covid", "1", "q"])
    monkeypatch.setattr(
        "builtins.input",
        lambda _: next(saisies)
    )
    main.run_project()
    sortie = capsys.readouterr().out
    assert "Indexation terminée." in sortie
    assert "Article sur le vaccin" in sortie
    assert "Au revoir !" in sortie