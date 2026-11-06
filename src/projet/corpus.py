from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from .author import Author
from .document import ArxivDocument, Document, DocumentFactory, RedditDocument
import re
from .text_processing import nettoyer_texte
from collections import Counter

DATAFRAME_COLUMNS: list[str] = [
    "identifier",
    "text",
    "origin",
    "title",
    "author",
    "publication_date",
    "url",
    "comment_count",
    "co_authors",
]


class CorpusInstantiationError(RuntimeError):
    pass


class Corpus:
    __instance: Corpus | None = None
    __CREATION_KEY: object = object()

    def __init__(self, name: str, creation_key: object) -> None:
        if creation_key is not Corpus.__CREATION_KEY:
            raise CorpusInstantiationError(
                "Corpus is a singleton, use Corpus.get_instance()."
            )
        self.__name: str = name
        self.__documents: dict[int, Document] = {}
        self.__document_id: int = 0
        self.__authors: dict[str, Author] = {}
        self.__concatenated_text_cache: str | None = None

    @classmethod
    def get_instance(cls, name: str = "Search corpus") -> Corpus:
        if cls.__instance is None:
            cls.__instance = cls(name, cls.__CREATION_KEY)
        return cls.__instance

    @property
    def name(self) -> str:
        return self.__name

    @property
    def documents(self) -> dict[int, Document]:
        return self.__documents.copy()

    @property
    def document_id(self) -> int:
        return self.__document_id

    @property
    def authors(self) -> dict[str, Author]:
        return self.__authors.copy()

    def add_document(self, document: Document) -> int:
        self.__document_id += 1
        document_id: int = self.__document_id
        self.__documents[document_id] = document
        self.__add_document_to_authors(document_id, document)
        self.__concatenated_text_cache = None
        return document_id

    def get_document_count(self) -> int:
        return len(self.__documents)

    def get_document_statistics(self) -> dict[int, dict[str, int]]:
        statistics: dict[int, dict[str, int]] = {}
        for document_id, document in self.__documents.items():
            words: list[str] = document.text.split()
            sentences: list[str] = [
                sentence for sentence in document.text.split(".") if sentence.strip()
            ]
            statistics[document_id] = {
                "word_count": len(words),
                "sentence_count": len(sentences),
            }
        return statistics

    def remove_short_documents(self, minimum_length: int) -> int:
        identifiers: list[int] = [
            document_id
            for document_id, document in self.__documents.items()
            if len(document.text) < minimum_length
        ]
        for document_id in identifiers:
            del self.__documents[document_id]
        self.__concatenated_text_cache = None
        self.__rebuild_authors()
        return len(identifiers)

    def get_concatenated_text(self) -> str:
        if self.__concatenated_text_cache is None:
            self.__concatenated_text_cache = " ".join(
                doc.text for doc in self.__documents.values()
            )
        return self.__concatenated_text_cache

    def get_documents_by_date(self, count: int) -> list[Document]:
        if count < 0:
            raise ValueError("The document count cannot be negative.")
        documents: list[Document] = sorted(
            self.__documents.values(),
            key=lambda document: document.publication_date,
            reverse=True,
        )
        return documents[:count]

    def get_documents_by_title(self, count: int) -> list[Document]:
        if count < 0:
            raise ValueError("The document count cannot be negative.")
        documents: list[Document] = sorted(
            self.__documents.values(),
            key=lambda document: document.title.lower(),
        )
        return documents[:count]

    def to_dataframe(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for document_id, document in self.__documents.items():
            row: dict[str, object] = self.__create_dataframe_row(
                document_id, document
            )
            rows.append(row)
        return pd.DataFrame(rows, columns=DATAFRAME_COLUMNS)

    def save(self, file_path: Path) -> None:
        dataframe: pd.DataFrame = self.to_dataframe()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(file_path, sep="\t", index=False)

    def load(self, file_path: Path) -> None:
        dataframe: pd.DataFrame = pd.read_csv(file_path, sep="\t")
        missing_columns: set[str] = set(DATAFRAME_COLUMNS) - set(dataframe.columns)
        if missing_columns:
            raise ValueError(f"Missing CSV columns: {sorted(missing_columns)}")
        self.__documents = {}
        self.__authors = {}
        self.__document_id = 0
        self.__concatenated_text_cache = None
        for _, row in dataframe.iterrows():
            document: Document = self.__create_document_from_row(row)
            document_id: int = int(row["identifier"])
            self.__documents[document_id] = document
            self.__document_id = max(self.__document_id, document_id)
            self.__add_document_to_authors(document_id, document)

    def __add_document_to_authors(
        self, document_id: int, document: Document
    ) -> None:
        author_names: list[str] = [document.author]
        if isinstance(document, ArxivDocument):
            author_names.extend(document.co_authors)
        unique_names: list[str] = []
        for author_name in author_names:
            if author_name not in unique_names:
                unique_names.append(author_name)
        for author_name in unique_names:
            if author_name not in self.__authors:
                self.__authors[author_name] = Author(author_name)
            self.__authors[author_name].add(document_id, document)

    def __rebuild_authors(self) -> None:
        self.__authors = {}
        for document_id, document in self.__documents.items():
            self.__add_document_to_authors(document_id, document)

    @staticmethod
    def __create_dataframe_row(
        document_id: int, document: Document
    ) -> dict[str, object]:
        comment_count: int = 0
        co_authors: list[str] = []
        if isinstance(document, RedditDocument):
            comment_count = document.comment_count
        elif isinstance(document, ArxivDocument):
            co_authors = document.co_authors
        return {
            "identifier": document_id,
            "text": document.text,
            "origin": document.document_type,
            "title": document.title,
            "author": document.author,
            "publication_date": document.publication_date.isoformat(),
            "url": document.url,
            "comment_count": comment_count,
            "co_authors": json.dumps(co_authors, ensure_ascii=False),
        }

    @staticmethod
    def __create_document_from_row(row: pd.Series) -> Document:
        document_type: str = str(row["origin"])
        co_authors_value: object = row["co_authors"]
        co_authors: list[str] = []
        if pd.notna(co_authors_value):
            co_authors = list(json.loads(str(co_authors_value)))
        comment_value: object = row["comment_count"]
        comment_count: int = 0
        if pd.notna(comment_value):
            comment_count = int(comment_value)
        return DocumentFactory.create_document(
            document_type=document_type,
            title=str(row["title"]),
            author=str(row["author"]),
            publication_date=pd.to_datetime(row["publication_date"]).to_pydatetime(),
            url=str(row["url"]),
            text=str(row["text"]),
            comment_count=comment_count,
            co_authors=co_authors,
        )

    def __repr__(self) -> str:
        return (
            f"Corpus(name={self.name!r}, documents={len(self.__documents)}, "
            f"authors={len(self.__authors)})"
        )

    def search(self, keyword: str, context_size: int = 60) -> list[str]:
        keyword = keyword.strip()
        if not keyword:
            return []
        text = self.get_concatenated_text()
        pattern = rf"\b{re.escape(keyword)}\b"
        return [
            text[ max(0, match.start() - context_size): match.end() + context_size]
            for match in re.finditer(
                pattern,
                text,
                flags=re.IGNORECASE
            )
        ]


    def concorde(self, expression: str, context_size: int = 30) -> pd.DataFrame:
        columns: list[str] = [
            "contexte gauche",
            "motif trouvé",
            "contexte droit"
        ]
        if context_size < 0:
            raise ValueError(
                "La taille du contexte ne peut pas être négative."
            )
        expression = expression.strip()
        if not expression:
            return pd.DataFrame(columns=columns)
        text: str = self.get_concatenated_text()
        pattern: str = rf"(?<!\w){re.escape(expression)}(?!\w)"
        results: list[dict[str, str]] = []
        for match in re.finditer(
            pattern,
            text,
            flags=re.IGNORECASE
        ):
            start: int = match.start()
            end: int = match.end()
            left_context: str = text[
                max(0, start - context_size):start
            ]
            right_context: str = text[
                end:min(len(text), end + context_size)
            ]
            results.append({
                "contexte gauche": left_context,
                "motif trouvé": match.group(0),
                "contexte droit": right_context
            })
        return pd.DataFrame(results, columns=columns)
    
    def construire_vocabulaire(self) -> dict[str, int]:
        mots_uniques: set[str] = set()
        for document in self.__documents.values():
            texte = nettoyer_texte(document.text)
            mots = texte.split()
            mots_uniques.update(mots)
        mots_tries = sorted(mots_uniques)
        vocabulaire = {
            mot: identifiant
            for identifiant, mot in enumerate(mots_tries)
        }
        return vocabulaire
    
    def calculer_frequences(self) -> pd.DataFrame:
        occurrences: Counter[str] = Counter()
        document_frequency: Counter[str] = Counter()
        for document in self.__documents.values():
            texte: str = nettoyer_texte(document.text)
            mots: list[str] = texte.split()
            occurrences.update(mots)
            document_frequency.update(set(mots))
        freq: pd.DataFrame = pd.DataFrame(
            [
                {
                    "mot": mot,
                    "identifiant": identifiant,
                    "occurrences": occurrences[mot],
                    "document_frequency": document_frequency[mot]
                }
                for identifiant, mot in enumerate(
                    sorted(occurrences)
                )
            ],
            columns=[
                "mot",
                "identifiant",
                "occurrences",
                "document_frequency"
            ]
        )
        return freq