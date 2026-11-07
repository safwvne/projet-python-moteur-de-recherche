from __future__ import annotations
from collections import Counter
from scipy.sparse import csr_matrix
from .corpus import Corpus
from .text_processing import nettoyer_texte
from math import log, sqrt
import pandas as pd
from tqdm.auto import tqdm

class SearchEngine:
    def __init__(self, corpus: Corpus, show_progress: bool = False) -> None:
        self.corpus: Corpus = corpus
        documents = self.corpus.documents
        self.document_ids: list[int] = sorted(documents)
        compteurs_documents: list[Counter[str]] = []
        mots_uniques: set[str] = set()
        for document_id in tqdm(
            self.document_ids,
            desc="Préparation de l'index",
            unit="document",
            disable=not show_progress
        ):
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
        self.normes_documents: list[float] = [
            sqrt(float(total))
            for total in (
                self.mat_TFxIDF
                .multiply(self.mat_TFxIDF)
                .sum(axis=1)
                .A1
            )
        ]

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
    
    def _vectoriser_requete(self, mots_cles: str) -> csr_matrix:
        texte: str = nettoyer_texte(mots_cles)
        mots: list[str] = texte.split()
        occurrences: Counter[str] = Counter(mots)
        colonnes: list[int] = []
        valeurs: list[float] = []
        for mot, frequence in occurrences.items():
            if mot not in self.vocab:
                continue
            colonne: int = self.vocab[mot]["id"]
            poids: float = frequence * self.idf[colonne]
            colonnes.append(colonne)
            valeurs.append(poids)
        lignes: list[int] = [0] * len(colonnes)
        return csr_matrix(
            (valeurs, (lignes, colonnes)),
            shape=(1, len(self.vocab)),
            dtype=float
        )
    
    def _calculer_similarites(self, vecteur_requete: csr_matrix) -> list[float]:
        norme_requete: float = sqrt(
            float(
                vecteur_requete
                .multiply(vecteur_requete)
                .sum()
            )
        )
        if norme_requete == 0:
            return [0.0] * len(self.document_ids)
        produits_scalaires = (
            self.mat_TFxIDF @ vecteur_requete.T
        ).toarray().ravel()
        scores: list[float] = []
        for indice_ligne, produit in enumerate(
            produits_scalaires
        ):
            norme_document = self.normes_documents[
                indice_ligne
            ]
            if norme_document == 0:
                scores.append(0.0)
                continue
            score = float(produit) / (
                norme_document * norme_requete
            )
            scores.append(score)
        return scores
    
    def search(self, mots_cles: str, n: int = 5) -> pd.DataFrame:
        colonnes = [
            "document_id",
            "title",
            "author",
            "source",
            "score",
            "url"
        ]
        if n < 0:
            raise ValueError(
                "Le nombre de résultats ne peut pas être négatif."
            )
        if n == 0:
            return pd.DataFrame(columns=colonnes)
        vecteur_requete = self._vectoriser_requete(
            mots_cles
        )
        scores = self._calculer_similarites(
            vecteur_requete
        )
        classement = [
            (score, self.document_ids[indice])
            for indice, score in enumerate(scores)
            if score > 0.0
        ]
        classement.sort(
            key=lambda element: (-element[0], element[1])
        )
        meilleurs = classement[:n]
        documents = self.corpus.documents
        resultats: list[dict[str, object]] = []
        for score, document_id in meilleurs:
            document = documents[document_id]
            resultats.append({
                "document_id": document_id,
                "title": document.title,
                "author": document.author,
                "source": document.document_type,
                "score": float(score),
                "url": document.url
            })
        return pd.DataFrame(
            resultats,
            columns=colonnes
        )