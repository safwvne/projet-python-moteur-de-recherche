from __future__ import annotations

from datetime import datetime

REDDIT_TYPE: str = "reddit"
ARXIV_TYPE: str = "arxiv"


class Document:
    def __init__(
        self,
        title: str,
        author: str,
        publication_date: datetime,
        url: str,
        text: str,
        document_type: str,
    ) -> None:
        self.__title: str = title
        self.__author: str = author
        self.__publication_date: datetime = publication_date
        self.__url: str = url
        self.__text: str = text
        self.__document_type: str = document_type

    @property
    def title(self) -> str:
        return self.__title

    @property
    def author(self) -> str:
        return self.__author

    @property
    def publication_date(self) -> datetime:
        return self.__publication_date

    @property
    def url(self) -> str:
        return self.__url

    @property
    def text(self) -> str:
        return self.__text

    @property
    def document_type(self) -> str:
        return self.__document_type

    def display_information(self) -> None:
        print(f"Title: {self.title}")
        print(f"Author: {self.author}")
        print(f"Date: {self.publication_date.isoformat()}")
        print(f"URL: {self.url}")
        print(f"Type: {self.document_type}")
        print(f"Text: {self.text}")

    def __str__(self) -> str:
        return self.title


class RedditDocument(Document):
    def __init__(
        self,
        title: str,
        author: str,
        publication_date: datetime,
        url: str,
        text: str,
        comment_count: int,
    ) -> None:
        super().__init__(
            title,
            author,
            publication_date,
            url,
            text,
            REDDIT_TYPE,
        )
        self.comment_count = comment_count

    @property
    def comment_count(self) -> int:
        return self.__comment_count

    @comment_count.setter
    def comment_count(self, value: int) -> None:
        if value < 0:
            raise ValueError("The comment count cannot be negative.")
        self.__comment_count: int = value

    def get_type(self) -> str:
        return REDDIT_TYPE

    def __str__(self) -> str:
        return f"[Reddit] {self.title} ({self.comment_count} comments)"


class ArxivDocument(Document):
    def __init__(
        self,
        title: str,
        author: str,
        publication_date: datetime,
        url: str,
        text: str,
        co_authors: list[str],
    ) -> None:
        super().__init__(
            title,
            author,
            publication_date,
            url,
            text,
            ARXIV_TYPE,
        )
        self.co_authors = co_authors

    @property
    def co_authors(self) -> list[str]:
        return self.__co_authors.copy()

    @co_authors.setter
    def co_authors(self, value: list[str]) -> None:
        self.__co_authors: list[str] = value.copy()

    def get_type(self) -> str:
        return ARXIV_TYPE

    def __str__(self) -> str:
        author_count: int = 1 + len(self.co_authors)
        return f"[Arxiv] {self.title} ({author_count} authors)"


class DocumentFactory:
    @staticmethod
    def create_document(
        document_type: str,
        title: str,
        author: str,
        publication_date: datetime,
        url: str,
        text: str,
        comment_count: int = 0,
        co_authors: list[str] | None = None,
    ) -> Document:
        source: str = document_type.lower()
        document: Document
        if source == REDDIT_TYPE:
            document = RedditDocument(
                title,
                author,
                publication_date,
                url,
                text,
                comment_count,
            )
        elif source == ARXIV_TYPE:
            authors: list[str] = [] if co_authors is None else co_authors
            document = ArxivDocument(
                title,
                author,
                publication_date,
                url,
                text,
                authors,
            )
        else:
            raise ValueError(f"Unknown document type: {document_type}")
        return document
