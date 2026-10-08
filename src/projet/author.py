from __future__ import annotations

from .document import Document


class Author:
    def __init__(self, name: str) -> None:
        self.__name: str = name
        self.__document_count: int = 0
        self.__production: dict[int, Document] = {}

    @property
    def name(self) -> str:
        return self.__name

    @property
    def document_count(self) -> int:
        return self.__document_count

    @property
    def production(self) -> dict[int, Document]:
        return self.__production.copy()

    def add(self, document_id: int, document: Document) -> None:
        if document_id not in self.__production:
            self.__production[document_id] = document
            self.__document_count += 1

    def get_average_document_length(self) -> float:
        average_length: float = 0.0
        if self.__document_count > 0:
            total_length: int = sum(
                len(document.text) for document in self.__production.values()
            )
            average_length = total_length / self.__document_count
        return average_length

    def __str__(self) -> str:
        return f"{self.name}: {self.document_count} document(s)"
