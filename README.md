# Moteur de recherche documentaire — Coronavirus

**Master 1 Informatique — Université Lumière Lyon 2**  
**Programmation de spécialité (Python) — Année universitaire 2026–2027**

## Présentation

Ce projet consiste à développer un moteur de recherche documentaire en Python, sans utiliser de moteur de recherche ni de vectoriseur prêt à l'emploi. Il porte sur la thématique **Coronavirus** et exploite un corpus de documents provenant de **Reddit** et d'**Arxiv**.

Le programme permet de récupérer et sauvegarder les documents, de les organiser en objets Python, d'analyser leur contenu textuel et de retrouver les documents les plus pertinents à partir de mots-clés à l'aide de la pondération **TF-IDF** et de la **similarité cosinus**.

## Technologies et dépendances

- **Python 3.13 :** version utilisée lors du développement et prévue pour l'évaluation. Le fichier `pyproject.toml` indique `requires-python = ">=3.11"`.
- **pandas :** tableaux de données, lecture/écriture du corpus et présentation des résultats.
- **SciPy (`scipy.sparse`) :** représentation creuse CSR des matrices documents × termes.
- **PRAW :** récupération des publications Reddit.
- **`urllib` et `xmltodict` :** interrogation de l'API Arxiv et lecture de ses réponses XML.
- **`re` :** expressions régulières pour le nettoyage du texte, la recherche et le concordancier.
- **`tqdm` :** progression pendant la préparation de l'index documentaire.
- **`ipywidgets`, `IPython` :** interface interactive du notebook.
- **JupyterLab et `ipykernel` :** exécution du notebook (dépendances de développement).
- **pytest :** tests unitaires et tests d'intégration.
- **uv :** gestion de l'environnement et des dépendances.

Le vocabulaire, les fréquences, la matrice TF-IDF et la similarité cosinus sont construits par le code du projet, **sans `scikit-learn` ni `nltk` pour le cœur du moteur**.

Les dépendances figurent dans `pyproject.toml` et leurs versions résolues sont consignées dans `uv.lock`.

## Installation

### Prérequis

- Git ;
- [uv](https://docs.astral.sh/uv/getting-started/installation/) ;
- Python 3.13 (installable et gérable avec `uv`).

Cloner le dépôt et sélectionner la branche contenant la troisième partie :

```bash
git clone https://github.com/safwvne/projet-python-moteur-de-recherche.git
cd projet-python-moteur-de-recherche
```

Installer les dépendances et préparer l'environnement Python :

```bash
uv sync --python 3.13
```

Lancer les commandes ci-dessous **depuis la racine du projet**. Le corpus est fourni dans `data/corpus.csv` ; il n'est donc pas nécessaire de contacter les API pour effectuer des recherches.

## Exécution

### 1. Interface de recherche dans le terminal

```bash
uv run python main.py
```

Au démarrage, le programme :

1. Charge `data/corpus.csv` si le fichier existe ;
2. Sinon, tente de récupérer des documents depuis Reddit et Arxiv, élimine les documents trop courts, puis sauvegarde le corpus ;
3. Prépare le vocabulaire, la matrice TF et la matrice TF-IDF ;
4. Affiche une interface textuelle permettant d'effectuer plusieurs recherches.

L'utilisateur saisit ses mots-clés et le nombre maximal de résultats souhaité (5 par défaut). Les résultats sont affichés par pertinence décroissante, avec le titre, l'auteur, la source, le score et l'URL des documents. Il est possible d'enchaîner les requêtes et de quitter avec `q`, `quit` ou `exit`.

### 2. Interface graphique dans Jupyter — v3

Lancer JupyterLab :

```bash
uv run jupyter lab
```

Ouvrir **`interface_recherche.ipynb`** à la racine du projet, sélectionner le noyau Python de l'environnement `uv`, puis exécuter les cellules dans l'ordre.

Le notebook :

1. Importe les classes existantes depuis `projet` ;
2. Charge le corpus sauvegardé, sans solliciter les API ;
3. Instancie `SearchEngine` avec `show_progress=True`, ce qui active une barre `tqdm` pendant la préparation des documents ;
4. Vérifie le moteur sur trois requêtes d'exemple ;
5. Affiche une interface basée sur `ipywidgets`.

L'interface comprend un champ de mots-clés, un curseur pour choisir le nombre de résultats, un bouton **Rechercher** et une zone de résultats (`Output`). Les résultats précédents sont effacés à chaque nouvelle recherche.

### Configuration des API (uniquement pour une nouvelle acquisition)

Les variables d'environnement requises pour Reddit sont :

```dotenv
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=
REDDIT_USER_AGENT=
```

Un exemple sans valeurs confidentielles est fourni dans `.env.example`.

Le programme lit ces informations avec `os.getenv()`. **Il ne charge pas automatiquement le fichier `.env`** : les valeurs doivent être exportées dans l'environnement d'exécution avant une nouvelle acquisition.

**Arxiv ne demande pas d'identifiants** pour les requêtes utilisées dans ce projet : le programme interroge directement son API publique.

Lorsque `data/corpus.csv` existe, les deux API sont évitées. Le fichier porte l'extension `.csv`, mais les données sont enregistrées avec une **tabulation** comme séparateur (`sep="\t"`).

## Organisation du projet

```text
projet-python-moteur-de-recherche/
├── .env.example
├── .gitignore
├── .python-version
├── data/
│   └── corpus.csv
├── src/
│   └── projet/
│       ├── __init__.py
│       ├── author.py
│       ├── corpus.py
│       ├── document.py
│       ├── search_engine.py
│       └── text_processing.py
├── tests/
│   ├── test_main.py
│   ├── test_text_processing.py
│   └── test_vocabulaire.py
├── interface_recherche.ipynb
├── main.py
├── pyproject.toml
├── uv.lock
└── README.md
```

- **`document.py` :** classe de base `Document`, spécialisations Reddit/Arxiv et fabrique `DocumentFactory`.
- **`author.py` :** regroupement des productions et statistiques par auteur.
- **`corpus.py` :** Singleton, gestion et sauvegarde des documents, recherche textuelle, concordancier, vocabulaire et fréquences.
- **`text_processing.py` :** normalisation des textes par expressions régulières.
- **`search_engine.py` :** construction des matrices, TF-IDF, vectorisation des requêtes, similarité cosinus, classement et filtrage par source.
- **`main.py` :** acquisition/chargement du corpus et interface de recherche en terminal.
- **`interface_recherche.ipynb` :** interface graphique Jupyter et bonus interactifs. Le moteur lui-même reste dans les modules `.py`.
- **`tests/` :** tests unitaires du traitement de texte et du moteur, ainsi que tests d'intégration de l'interface terminal.

## Fonctionnement du moteur de recherche

### 1. Prétraitement des textes

Le nettoyage consiste à convertir le texte en minuscules, remplacer les retours à la ligne et la ponctuation par des espaces, supprimer les espaces superflus, puis découper le texte en mots. La même fonction `nettoyer_texte()` est utilisée pour les documents et les requêtes.

### 2. Vocabulaire et fréquences

Chaque mot distinct du corpus reçoit un identifiant de colonne stable, attribué après tri alphabétique. Le dictionnaire `vocab` mémorise :

- **`id` :** identifiant du terme dans les matrices ;
- **`total_occurrences` :** nombre total d'occurrences du terme ;
- **`document_frequency` (DF) :** nombre de documents contenant le terme au moins une fois.

### 3. Matrice TF

La matrice `mat_TF` possède une ligne par document et une colonne par terme. Chaque cellule `TF(d,t)` contient le nombre d'occurrences du terme `t` dans le document `d`.

La correspondance entre les lignes des matrices et les identifiants d'origine est conservée dans `document_ids`. Cette précaution est nécessaire car les identifiants peuvent ne plus être consécutifs après suppression de documents.

### 4. Matrice TF-IDF

Pour chaque terme `t`, le moteur calcule un poids IDF lissé :

$$
\operatorname{IDF}(t) = \ln\left(\frac{N+1}{\operatorname{DF}(t)+1}\right)+1
$$

Puis, pour chaque document `d` :

$$
\operatorname{TF-IDF}(d,t) = \operatorname{TF}(d,t)\times\operatorname{IDF}(t)
$$

`N` est le nombre de documents. Le lissage permet notamment de gérer les termes fréquents sans division par zéro. Les matrices `mat_TF` et `mat_TFxIDF` utilisent le format creux **CSR** de SciPy afin de ne pas stocker tous les zéros.

### 5. Vectorisation et similarité cosinus

Les mots-clés sont normalisés puis convertis en vecteur TF-IDF sur **le même vocabulaire** que les documents. Les mots inconnus sont ignorés.

La similarité entre un document `d` et la requête `q` est calculée par :

$$
\operatorname{sim}(d,q) = \frac{d\cdot q}{\lVert d\rVert\,\lVert q\rVert}
$$

Les documents sont classés par score décroissant, puis par identifiant en cas d'égalité. Les documents de score nul sont exclus. Les requêtes vides ou ne contenant aucun mot connu renvoient un tableau vide. Les normes nulles sont gérées pour éviter les divisions par zéro.

### Utilisation directe en Python

```python
from pathlib import Path
from projet.corpus import Corpus
from projet.search_engine import SearchEngine

corpus = Corpus.get_instance("Coronavirus")
corpus.load(Path("data/corpus.csv"))

moteur = SearchEngine(corpus)

# Recherche dans toutes les sources
resultats = moteur.search("covid vaccine", n=5)
print(resultats.to_string(index=False))

# Recherche limitée aux documents Arxiv
resultats_arxiv = moteur.search(
    "covid vaccine", n=5, source="arxiv"
)
print(resultats_arxiv.to_string(index=False))
```

`search()` renvoie un `pandas.DataFrame` avec les colonnes `document_id`, `title`, `author`, `source`, `score` et `url`. Le paramètre `source` accepte `None` (toutes les sources), `"reddit"` ou `"arxiv"`.

### Exemple de résultats dans le notebook

Lors d'une exécution enregistrée du notebook avec le corpus fourni, le chargement a indiqué **1 240 documents**, et la construction du moteur **11 631 mots distincts** (matrices de taille **1 240 × 11 631**). Trois requêtes de vérification sont présentes : `covid vaccine`, `coronavirus treatment` et `vaccine research`.

Ces chiffres décrivent cette exécution et peuvent changer si le corpus est remplacé ou reconstruit.

## Tests et validation

Lancer tous les tests disponibles :

```bash
uv run python -m pytest tests/ -v
```

Lancer les fichiers séparément :

```bash
uv run python -m pytest tests/test_text_processing.py -v
uv run python -m pytest tests/test_vocabulaire.py -v
uv run python -m pytest tests/test_main.py -v
```

## Utilisation de l'intelligence artificielle

**Outil utilisé : ChatGPT.**

L'assistant a été utilisé comme aide technique et pédagogique pour des sujets précis du projet :

- **Matrices et TF-IDF :** compréhension et mise en œuvre des matrices creuses `scipy.sparse.csr_matrix`, des fréquences TF et DF, de la formule IDF, de la vectorisation des requêtes et de la similarité cosinus.
- **Expressions régulières :** explication des motifs employés dans le nettoyage, la recherche et le concordancier, notamment les frontières de mots et l'échappement des caractères spéciaux.
- **Découverte de Pytest :** accompagnement pour la première mise en place de tests unitaires en Python, les fixtures, `monkeypatch` et `pytest.approx()`.
- **Tests techniques complexes :** assistance pour les tests `test_matrice_tf_identifiants_documents`, `test_optimisation_un_nettoyage_par_document`, `test_matrice_tfidf_valeurs` et `test_similarite_cosinus_et_classement`.
- **Débogage :** aide à la résolution d'un import manquant entraînant `NameError: name 'isspmatrix_csr' is not defined`, à l'uniformisation des imports `src.projet` / `projet`, et à l'analyse des cas limites (matrices vides, normes nulles et division par zéro).
- **Intégration des interfaces :** conseils pour relier `SearchEngine` à `main.py`, organiser les cellules du notebook, utiliser `tqdm` et `ipywidgets`.