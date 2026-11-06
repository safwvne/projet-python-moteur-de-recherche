from __future__ import annotations
from .corpus import Corpus
from collections import Counter
from scipy.sparse import csr_matrix
from .text_processing import nettoyer_texte

class SearchEngine:
    def __init__(self, corpus: Corpus) -> None:
        self.corpus: Corpus = corpus
        self.vocab: dict[str, dict[str, int]] = (
            self._construire_vocabulaire()
        )
        self.document_ids: list[int] = sorted(
            self.corpus.documents.keys()
        )
        self.mat_TF: csr_matrix = (
            self._construire_matrice_tf()
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
    
    def _construire_matrice_tf(self) -> csr_matrix:
        documents = self.corpus.documents
        lignes: list[int] = []
        colonnes: list[int] = []
        valeurs: list[int] = []
        for indice_ligne, document_id in enumerate(
            self.document_ids
        ):
            document = documents[document_id]
            texte: str = nettoyer_texte(document.text)
            mots: list[str] = texte.split()
            occurrences: Counter[str] = Counter(mots)
            for mot, tf in occurrences.items():
                indice_colonne: int = self.vocab[mot]["id"]
                lignes.append(indice_ligne)
                colonnes.append(indice_colonne)
                valeurs.append(tf)
        nombre_documents: int = len(self.document_ids)
        nombre_mots: int = len(self.vocab)
        mat_TF: csr_matrix = csr_matrix(
            (valeurs, (lignes, colonnes)),
            shape=(nombre_documents, nombre_mots),
            dtype=int
        )
        return mat_TF