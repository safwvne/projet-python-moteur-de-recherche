from __future__ import annotations
from .corpus import Corpus

class SearchEngine:
    def __init__(self, corpus: Corpus) -> None:
        self.corpus: Corpus = corpus
        self.vocab: dict[str, dict[str, int]] = (
            self._construire_vocabulaire()
        )

    def _construire_vocabulaire(self) -> dict[str, dict[str, int]]:
        freq = self.corpus.calculer_frequences()
        vocabulaire: dict[str, dict[str, int]] = {}
        for ligne in freq.itertuples(index=False):
            mot: str = str(ligne.mot)
            vocabulaire[mot] = {
                "id": int(ligne.identifiant),
                "total_occurrences": int(ligne.occurrences),
                "document_frequency": int(
                    ligne.document_frequency
                )
            }
        return vocabulaire