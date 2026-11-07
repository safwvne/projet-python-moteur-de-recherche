from __future__ import annotations

from datetime import UTC, datetime
import os
from pathlib import Path
from typing import Any
import urllib.error
import urllib.parse
import urllib.request
from xml.parsers.expat import ExpatError

import praw
from prawcore.exceptions import PrawcoreException
import xmltodict

from projet.author import Author
from projet.corpus import Corpus
from projet.document import (
    ARXIV_TYPE,
    REDDIT_TYPE,
    Document,
    DocumentFactory,
)
from projet.search_engine import SearchEngine

CORPUS_NAME: str = "Coronavirus"
CORPUS_FILE_PATH: Path = Path("data") / "corpus.csv"
DOCUMENT_LIMIT_PER_SOURCE: int = 1000
MINIMUM_DOCUMENT_LENGTH: int = 100
DISPLAY_DOCUMENT_COUNT: int = 5
ARXIV_API_URL: str = "https://export.arxiv.org/api/query"
UNKNOWN_AUTHOR: str = "Unknown"
REQUEST_TIMEOUT_SECONDS: int = 30


def get_reddit_documents(theme: str, limit: int) -> list[Document]:
    client_id: str | None = os.getenv("REDDIT_CLIENT_ID")
    client_secret: str | None = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent: str | None = os.getenv("REDDIT_USER_AGENT")
    if not client_id or not client_secret or not user_agent:
        raise ValueError("Reddit environment variables are not defined.")
    reddit: praw.Reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )
    posts: list[Any] = list(reddit.subreddit(theme).hot(limit=limit))
    documents: list[Document] = []
    for post in posts:
        title: str = _clean_text(post.title)
        body: str = _clean_text(post.selftext)
        text: str = _clean_text(f"{title}. {body}")
        author: str = str(post.author) if post.author else UNKNOWN_AUTHOR
        document: Document = DocumentFactory.create_document(
            document_type=REDDIT_TYPE,
            title=title,
            author=author,
            publication_date=datetime.fromtimestamp(post.created_utc, tz=UTC),
            url=str(post.url),
            text=text,
            comment_count=int(post.num_comments),
        )
        documents.append(document)
    return documents


def get_arxiv_documents(theme: str, limit: int) -> list[Document]:
    query_parameters: dict[str, str | int] = {
        "search_query": f"all:{theme}",
        "start": 0,
        "max_results": limit,
    }
    query: str = urllib.parse.urlencode(query_parameters)
    request: urllib.request.Request = urllib.request.Request(
        f"{ARXIV_API_URL}?{query}",
        headers={"User-Agent": "M1-search-engine-project"},
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        xml_content: bytes = response.read()
    parsed_content: dict[str, Any] = xmltodict.parse(xml_content)
    entries: list[dict[str, Any]] = _get_arxiv_entries(parsed_content)
    documents: list[Document] = []
    for entry in entries:
        documents.append(_create_arxiv_document(entry))
    return documents


def _get_arxiv_entries(parsed_content: dict[str, Any]) -> list[dict[str, Any]]:
    feed: dict[str, Any] = parsed_content.get("feed", {})
    entry_data: object = feed.get("entry", [])
    entries: list[dict[str, Any]] = []
    if isinstance(entry_data, dict):
        entries = [entry_data]
    elif isinstance(entry_data, list):
        entries = entry_data
    return entries


def _create_arxiv_document(entry: dict[str, Any]) -> Document:
    author_names: list[str] = _get_arxiv_author_names(entry)
    main_author: str = UNKNOWN_AUTHOR
    co_authors: list[str] = []
    if author_names:
        main_author = author_names[0]
        co_authors = author_names[1:]
    publication_date: datetime = datetime.fromisoformat(
        str(entry["published"]).replace("Z", "+00:00")
    )
    return DocumentFactory.create_document(
        document_type=ARXIV_TYPE,
        title=_clean_text(entry.get("title", "")),
        author=main_author,
        publication_date=publication_date,
        url=str(entry.get("id", "")),
        text=_clean_text(entry.get("summary", "")),
        co_authors=co_authors,
    )


def _get_arxiv_author_names(entry: dict[str, Any]) -> list[str]:
    author_data: object = entry.get("author", [])
    authors: list[dict[str, Any]] = []
    if isinstance(author_data, dict):
        authors = [author_data]
    elif isinstance(author_data, list):
        authors = author_data
    names: list[str] = []
    for author in authors:
        name: str = _clean_text(author.get("name", ""))
        if name:
            names.append(name)
    return names


def _clean_text(value: object) -> str:
    return " ".join(str(value).split())


def build_corpus_from_apis(corpus: Corpus) -> None:
    reddit_documents: list[Document] = get_reddit_documents(
        CORPUS_NAME, DOCUMENT_LIMIT_PER_SOURCE
    )
    arxiv_documents: list[Document] = get_arxiv_documents(
        CORPUS_NAME, DOCUMENT_LIMIT_PER_SOURCE
    )
    documents: list[Document] = reddit_documents + arxiv_documents
    for document in documents:
        corpus.add_document(document)
    print(f"Reddit documents acquired: {len(reddit_documents)}")
    print(f"Arxiv documents acquired: {len(arxiv_documents)}")


def display_document_statistics(corpus: Corpus) -> None:
    statistics: dict[int, dict[str, int]] = corpus.get_document_statistics()
    for document_id, counts in statistics.items():
        print(
            f"Document {document_id}: {counts['word_count']} words, "
            f"{counts['sentence_count']} sentences"
        )


def display_sorted_documents(corpus: Corpus) -> None:
    print("\nNewest documents:")
    by_date: list[Document] = corpus.get_documents_by_date(DISPLAY_DOCUMENT_COUNT)
    for document in by_date:
        print(f"- {document}")
    print("\nDocuments sorted by title:")
    by_title: list[Document] = corpus.get_documents_by_title(DISPLAY_DOCUMENT_COUNT)
    for document in by_title:
        print(f"- {document}")


def display_author_statistics(corpus: Corpus) -> None:
    author_name: str = input("\nEnter an author name: ").strip()
    authors: dict[str, Author] = corpus.authors
    if author_name in authors:
        author: Author = authors[author_name]
        average_length: float = author.get_average_document_length()
        print(f"Documents: {author.document_count}")
        print(f"Average text length: {average_length:.2f} characters")
    else:
        print(f"Author '{author_name}' was not found.")


def run_search_interface(moteur: SearchEngine) -> None:
    print("\n" + "=" * 45)
    print("           MOTEUR DE RECHERCHE")
    print("=" * 45)
    print(f"Documents : {len(moteur.document_ids)}")
    print(f"Mots distincts : {len(moteur.vocab)}")
    print("\nTapez 'q' pour quitter.")
    while True:
        try:
            mots_cles = input(
                "\nMots-clés (q pour quitter) : "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nFermeture du moteur de recherche.")
            break
        if mots_cles.lower() in ("q", "quit", "exit"):
            print("Au revoir !")
            break
        if not mots_cles:
            print("Veuillez saisir au moins un mot-clé.")
            continue
        try:
            saisie = input(
                "Nombre de résultats [5] : "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nFermeture du moteur de recherche.")
            break
        try:
            nombre_resultats = (
                int(saisie) if saisie else 5
            )
        except ValueError:
            print("Veuillez saisir un nombre entier valide.")
            continue
        if nombre_resultats <= 0:
            print("Le nombre doit être supérieur à zéro.")
            continue
        resultats = moteur.search(
            mots_cles=mots_cles,
            n=nombre_resultats
        )
        if resultats.empty:
            print("Aucun document pertinent trouvé.")
            continue
        print(
            f"\n{len(resultats)} document(s) trouvé(s) :\n"
        )
        print(
            resultats.to_string(
                index=False,
                formatters={
                    "score": lambda valeur: f"{valeur:.4f}"
                }
            )
        )

def run_project() -> None:
    corpus: Corpus = Corpus.get_instance(CORPUS_NAME)
    if CORPUS_FILE_PATH.is_file():
        corpus.load(CORPUS_FILE_PATH)
        print(f"Corpus chargé depuis {CORPUS_FILE_PATH}")
    else:
        build_corpus_from_apis(corpus)
        display_document_statistics(corpus)
        removed_count: int = corpus.remove_short_documents(
            MINIMUM_DOCUMENT_LENGTH
        )
        print(f"Documents de moins de {MINIMUM_DOCUMENT_LENGTH} caractères supprimés : {removed_count}")
        corpus.save(CORPUS_FILE_PATH)
        print(f"Corpus sauvegardé dans {CORPUS_FILE_PATH}")
    print(f"\n{corpus!r}")
    corpus_text: str = corpus.get_concatenated_text()
    print(
        f"Longueur du texte concaténé : "
        f"{len(corpus_text)} caractères"
    )
    display_sorted_documents(corpus)
    if corpus.get_document_count() == 0:
        print("Le corpus ne contient aucun document.")
        return
    print("\nConstruction du moteur de recherche...")
    moteur: SearchEngine = SearchEngine(corpus)
    print("Indexation terminée.")
    print(f"Matrice TF : {moteur.mat_TF.shape}")
    print(
        f"Matrice TF-IDF : "
        f"{moteur.mat_TFxIDF.shape}"
    )
    run_search_interface(moteur)

def main() -> None:
    try:
        run_project()
    except (
        PrawcoreException,
        urllib.error.URLError,
        ExpatError,
        OSError,
        ValueError,
    ) as error:
        print(f"Unable to run the project: {error}")

if __name__ == "__main__":
    main()
