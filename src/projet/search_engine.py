from __future__ import annotations
from collections import Counter
from scipy.sparse import csr_matrix
from .corpus import Corpus
from .text_processing import nettoyer_texte
from math import log

class SearchEngine:
    def __init__(self, corpus: Corpus) -> None:
        self.corpus: Corpus = corpus
        documents = self.corpus.documents
        self.document_ids: list[int] = sorted(documents)
        compteurs_documents: list[Counter[str]] = []
        mots_uniques: set[str] = set()
        for document_id in self.document_ids:
            document = documents[document_id]
            texte = nettoyer_texte(document.text)
            mots = texte.split()
            compteur = Counter(mots)
            compteurs_documents.append(compteur)
            mots_uniques.update(compteur)
        self.vocab: dict[str, dict[str, int]] = (
            self._construire_vocabulaire(mots_uniques)
        )
        self.mat_TF: csr_matrix = (
            self._construire_matrice_tf(
                compteurs_documents
            )
        )
        self._mettre_a_jour_statistiques()
        self.mat_TFxIDF: csr_matrix = (
            self._construire_matrice_tfidf()
        )

    def _construire_vocabulaire(
        self,
        mots_uniques: set[str]
    ) -> dict[str, dict[str, int]]:
        vocabulaire: dict[str, dict[str, int]] = {
            mot: {
                "id": identifiant,
                "total_occurrences": 0,
                "document_frequency": 0
            }
            for identifiant, mot in enumerate(
                sorted(mots_uniques)
            )
        }
        return vocabulaire

    def _construire_matrice_tf(
        self,
        compteurs_documents: list[Counter[str]]
    ) -> csr_matrix:
        lignes: list[int] = []
        colonnes: list[int] = []
        valeurs: list[int] = []
        for indice_ligne, compteur in enumerate(
            compteurs_documents
        ):
            for mot, tf in compteur.items():
                indice_colonne = self.vocab[mot]["id"]
                lignes.append(indice_ligne)
                colonnes.append(indice_colonne)
                valeurs.append(tf)
        nombre_documents = len(self.document_ids)
        nombre_mots = len(self.vocab)
        return csr_matrix(
            (valeurs, (lignes, colonnes)),
            shape=(nombre_documents, nombre_mots),
            dtype=int
        )

    def _mettre_a_jour_statistiques(self) -> None:
        tf_totaux = self.mat_TF.sum(axis=0).A1
        df_totaux = (
            (self.mat_TF > 0).sum(axis=0).A1
        )
        for mot, informations in self.vocab.items():
            colonne = informations["id"]
            informations["total_occurrences"] = int(
                tf_totaux[colonne]
            )
            informations["document_frequency"] = int(
                df_totaux[colonne]
            )
                  
    def _construire_matrice_tfidf(self) -> csr_matrix:
        nombre_documents, nombre_mots = self.mat_TF.shape
        self.idf: list[float] = [0.0] * nombre_mots
        for informations in self.vocab.values():
            colonne: int = informations["id"]
            df: int = informations[
                "document_frequency"
            ]
            poids_idf: float = log(
                (nombre_documents + 1) / (df + 1)
            ) + 1
            self.idf[colonne] = poids_idf
        if nombre_mots == 0:
            return csr_matrix(
                (nombre_documents, 0),
                dtype=float
            )
        mat_TFxIDF: csr_matrix = (
            self.mat_TF.multiply(self.idf).tocsr()
        )
        return mat_TFxIDF