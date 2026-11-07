from __future__ import annotations
import re

def nettoyer_texte(texte: str) -> str:
    texte = texte.lower().replace("\n", " ")
    texte = re.sub(r"[^\w\s]|_", " ", texte)
    texte = " ".join(texte.split())
    return texte