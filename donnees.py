import pandas as pd
import config as cfg
import exceptions as exc

def charger_csv(chemin: str) :
    if not os.path.exists(chemin): 
        raise FileNotFoundError(f"Fichier introuvable : {chemin}")
    with open(chemin, encoding="utf-8") as f : 
        df = pd.read_csv(f)
    return df

def verifier_colonnes(df, colonnes):
    colonnes_absentes = [col for col in colonnes if col not in df.columns]
    if colonnes_absentes:
        raise exc.ColonneManquanteError(colonnes_absentes)
    return None

def garder_colonnes(df, colonnes) :
    return df[colonnes].copy()

def convertir_numeriques(df, colonnes) : 
    for c in colonnes : 
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def nettoyer_categories(df, colonnes) : 
    for c in colonnes : 
        df[c] = df[c].fillna("Unknown").astype(str).str.strip()
    return df 

def supprimer_lignes_invalides(df, colonnes) : 
    df = df.dropna(subset=colonnes)
    df = df.drop_dupicates()
    return df 

def nettoyer(df, colonnes_num, colonnes_cat, cible):
    df = garder_colonnes(df, colonnes_num + colonnes_cat + [cible])
    df = convertir_numeriques(df, colonnes_num + [cible])
    df = nettoyer_categories(df, colonnes_cat)
    df = supprimer_lignes_invalides(df, colonnes_num + [cible])
    return df