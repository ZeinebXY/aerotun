# -*- coding: utf-8 -*-
"""
AEROTUN — Application Streamlit : sobriété, efficacité énergétique et photovoltaïque.

Lancer :   streamlit run app.py
Installer: pip install streamlit pandas numpy plotly statsmodels scipy openpyxl
"""
import io
import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.formula.api as smf
import streamlit as st
from scipy.optimize import brentq

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------
DATA_PATH = os.environ.get("AEROTUN_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "Aerotun.xlsx"))
SHEET = "Feuil1"


def _version_streamlit():
    try:
        return tuple(int(x) for x in st.__version__.split(".")[:2])
    except Exception:
        return (1, 0)


# Largeur pleine : paramètre « width » (Streamlit >= 1.50) ou « use_container_width » (anciennes versions)
STRETCH = {"width": "stretch"} if _version_streamlit() >= (1, 50) else {"use_container_width": True}

st.set_page_config(page_title="AEROTUN — Transition énergétique", page_icon="⚡", layout="wide")

st.markdown(
    """
    <style>
    /* ---------- Animations pro AEROTUN ---------- */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(18px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to   { opacity: 1; }
    }
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-16px); }
        to   { opacity: 1; transform: translateX(0); }
    }
    @keyframes shimmer {
        0%   { background-position: -200% 0; }
        100% { background-position:  200% 0; }
    }
    @keyframes pulseSoft {
        0%, 100% { box-shadow: 0 4px 20px rgba(11, 61, 145, 0.18); }
        50%      { box-shadow: 0 6px 28px rgba(46, 196, 182, 0.28); }
    }
    @keyframes gradientShift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Contenu principal : apparition fluide */
    .main .block-container {
        animation: fadeInUp 0.45s ease-out both;
    }

    /* Hero banner */
    .hero {
        background: linear-gradient(120deg, #0b3d91 0%, #1b6ca8 45%, #2ec4b6 100%);
        background-size: 200% 200%;
        animation: gradientShift 12s ease infinite, fadeInUp 0.55s ease-out both, pulseSoft 4s ease-in-out infinite;
        padding: 28px 32px;
        border-radius: 14px;
        color: white;
        margin-bottom: 18px;
        position: relative;
        overflow: hidden;
    }
    .hero::after {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
        background-size: 200% 100%;
        animation: shimmer 4.5s ease-in-out infinite;
        pointer-events: none;
    }
    .hero h1 {
        margin: 0 0 6px 0;
        font-size: 2.0rem;
        color: white;
        position: relative;
        z-index: 1;
        animation: slideInLeft 0.5s ease-out 0.1s both;
    }
    .hero p {
        margin: 0;
        font-size: 1.05rem;
        opacity: .95;
        position: relative;
        z-index: 1;
        animation: fadeIn 0.55s ease-out 0.25s both;
    }

    /* Pills */
    .pill {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        background: #e8f1fb;
        color: #0b3d91;
        font-size: .82rem;
        margin-right: 6px;
        margin-bottom: 4px;
        transition: transform 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
        animation: fadeInUp 0.4s ease-out both;
    }
    .pill:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(11, 61, 145, 0.15);
        background: #d6e8f8;
    }

    /* Metrics Streamlit */
    [data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 12px 16px;
        transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
        animation: fadeInUp 0.4s ease-out both;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(11, 61, 145, 0.12);
        border-color: #2ec4b6;
    }
    [data-testid="stMetricValue"] {
        transition: color 0.2s ease;
    }
    [data-testid="stMetric"]:hover [data-testid="stMetricValue"] {
        color: #0b3d91;
    }

    /* Boutons sidebar */
    section[data-testid="stSidebar"] .stButton > button {
        transition: transform 0.18s ease, box-shadow 0.18s ease, background 0.18s ease;
        border-radius: 10px !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        transform: translateX(4px);
        box-shadow: 0 4px 14px rgba(11, 61, 145, 0.18);
    }
    section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
        box-shadow: 0 2px 10px rgba(11, 61, 145, 0.25);
    }

    /* Boutons généraux */
    .stButton > button {
        transition: all 0.2s ease;
        border-radius: 8px !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }
    .stButton > button:active {
        transform: translateY(0);
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        transition: color 0.2s ease, border-color 0.2s ease !important;
    }
    button[data-baseweb="tab"]:hover {
        color: #0b3d91 !important;
    }

    /* Dataframes / tables */
    [data-testid="stDataFrame"], .stTable {
        animation: fadeIn 0.45s ease-out both;
        border-radius: 10px;
        overflow: hidden;
    }

    /* Expanders */
    [data-testid="stExpander"] {
        transition: box-shadow 0.2s ease;
        border-radius: 10px !important;
    }
    [data-testid="stExpander"]:hover {
        box-shadow: 0 4px 16px rgba(11, 61, 145, 0.08);
    }

    /* Info / success / warning boxes */
    [data-testid="stAlert"] {
        animation: fadeInUp 0.4s ease-out both;
        border-radius: 10px !important;
    }

    /* Plotly charts container */
    .js-plotly-plot, [data-testid="stPlotlyChart"] {
        animation: fadeIn 0.5s ease-out both;
        transition: transform 0.25s ease;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        animation: slideInLeft 0.4s ease-out both;
    }

    /* Radio / toggle */
    [data-testid="stRadio"] label, [data-testid="stCheckbox"] label {
        transition: color 0.15s ease;
    }

    /* Réduction motion pour accessibilité */
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

COLONNES_NUM = [
    "Heures-machine CNC", "Cycles autoclave", "Conso. électrique (kWh)", "Taux fuites air comprimé (%)",
    "Temp. salle blanche (°C)", "Humidité salle blanche (%)", "Taux rebut qualité (%)",
    "Part heures pleines (%)", "Coût énergie (TND)", "IPE élec. (kWh/h-machine)",
]
RENOMMAGE = {
    "Conso. électrique (kWh)": "kwh", "Coût énergie (TND)": "cout", "Heures-machine CNC": "cnc",
    "Cycles autoclave": "autoclave", "Taux fuites air comprimé (%)": "fuites",
    "Temp. salle blanche (°C)": "temp", "Humidité salle blanche (%)": "hum",
    "Taux rebut qualité (%)": "rebut", "Part heures pleines (%)": "hp",
    "IPE élec. (kWh/h-machine)": "ipe", "Anomalie climatisation": "anom", "Arrêt annuel": "arret",
}
JOURS_NON_OUVRES = [5, 6]
SEUIL_P = 0.05

# ----------------------------------------------------------------------------
# PARAMÈTRES PAR DÉFAUT (modifiables dans la page « Paramètres »)
# ----------------------------------------------------------------------------
# Aligné sur le notebook Colab :
# TAUX_ACTUALISATION=0.10, HAUSSE_PRIX_ELEC=0.03, INFLATION_OPEX=0.055,
# PV_OPEX_KWC=62 TND/kWc/an (GIZ 2013), PV_PART_REMPLACEMENT=0.10 en année 12.
# Le facteur d'émission n'est plus ici : il est piloté par les scénarios (voir FE_SCEN).
P_DEFAUT = dict(
    taux=0.10, g=0.03, infl=0.055, prix_carbone=50.0, usd=3.0,
    pv_kwc=150.0, pv_prod=1650.0, pv_capex_kwc=2500.0, pv_opex_kwc=62.0,
    pv_remp_an=12, pv_remp_part=0.10,
    tarif_vente=0.12, pv_surplus_max=0.30, pv_deg=0.005,
    capex_sob=5000.0, opex_sob=0.0, capex_eff=85000.0, opex_eff=2000.0,
    vie_sob=5, vie_eff=10, vie_pv=20,
    ancre_auto=0.16, ancre_cnc=0.12, ancre_veille=0.60, part_veille=0.15,
    tr_prudent=1 / 3, tr_central=2 / 3, tr_ambitieux=1.0,
    fte=False, fte_pv=0.20, fte_pv_plafond=200000.0, fte_eff=0.50,
)

# ----------------------------------------------------------------------------
# PARAMÈTRES BENCHMARKÉS : trois valeurs (Prudent / Central / Ambitieux)
# Par défaut, chaque paramètre est sur « Central » (le plus raisonnable des trois).
# ----------------------------------------------------------------------------
SCENARIOS = ["Prudent", "Central", "Ambitieux"]
PERSO = "Personnalisé"            # scénario construit paramètre par paramètre (page Paramètres)
ANOMALIES_EVITEES = {"Prudent": 0.50, "Central": 0.75, "Ambitieux": 1.00}
WEEKEND_ARRET = {"Prudent": 0.15, "Central": 0.30, "Ambitieux": 0.50}
FUITES_CIBLE = {"Prudent": 11.0, "Central": 10.0, "Ambitieux": 8.0}
# Facteur d'émission du réseau tunisien (kgCO2/kWh)
#  Prudent   : fiche Safran Soliman 2022 (site ≈ 2,8× plus gros qu'AEROTUN, non audité : borne haute)
#  Central   : mix 2025 (année la plus proche de la période étudiée)
#  Ambitieux : STEG, rapport annuel 2024 (officiel, le plus bas, intensité des centrales hors pertes de réseau)
FE_SCEN = {"Prudent": 0.587, "Central": 0.48, "Ambitieux": 0.4189}

CLES_BENCH = ["fe", "veille", "autoclave", "cnc", "anomalies", "weekend", "fuites"]
CHOIX_DEFAUT = {k: "Central" for k in CLES_BENCH}
LIBELLES = {
    "fe": "Facteur d'émission du réseau (kgCO₂/kWh)",
    "veille": "Veille évitable (jours ouvrés, part du talon)",
    "autoclave": "Gain autoclave (ancre OPTO-CLAVE −16 %)",
    "cnc": "Gain usinage CNC (ancre −12 %)",
    "anomalies": "Anomalies de climatisation évitées",
    "weekend": "Week-ends en mode arrêt (part de l'écart)",
    "fuites": "Fuites d'air comprimé cible (%)",
}
SOURCES_BENCH = {
    "fe": "Prudent : Safran Soliman 2022 (site ≈ 2,8× plus gros, fiche non auditée). Central : mix 2025. "
          "Ambitieux : STEG 2024 (officiel, hors pertes de réseau). À remplacer par le facteur ANME.",
    "veille": "Ancre veille −60 % (littérature CNC, fiabilité faible) × part de veille dans le talon × transférabilité",
    "autoclave": "Projet UE OPTO-CLAVE (−16 % d'énergie) × transférabilité",
    "cnc": "Littérature CNC (borne haute de −9 à −12 %) × transférabilité",
    "anomalies": "Hypothèse : part des dérives de climatisation évitées (pas d'ancre publiée)",
    "weekend": "Hypothèse : part de l'écart base week-end / arrêt annuel réellement obtenue",
    "fuites": "Objectif ANME : diviser le taux de fuite par deux ; ordre de grandeur Johnson Controls ; hypothèse",
}
PROJETS = ["Sobriété", "Efficacité", "Photovoltaïque"]
COULEURS = {"Sobriété": "#2ec4b6", "Efficacité": "#e71d36", "Photovoltaïque": "#ff9f1c"}


def p_defaut():
    p = dict(P_DEFAUT)
    p["choix"] = dict(CHOIX_DEFAUT)
    return p


def choix_effectifs(P):
    return {**CHOIX_DEFAUT, **P.get("choix", {})}


def valeurs_bench(P):
    """Valeurs Prudent / Central / Ambitieux de chaque paramètre benchmarké."""
    tr = {"Prudent": P["tr_prudent"], "Central": P["tr_central"], "Ambitieux": P["tr_ambitieux"]}
    return {
        "fe": dict(FE_SCEN),
        "veille": {sc: P["ancre_veille"] * P["part_veille"] * t for sc, t in tr.items()},
        "autoclave": {sc: P["ancre_auto"] * t for sc, t in tr.items()},
        "cnc": {sc: P["ancre_cnc"] * t for sc, t in tr.items()},
        "anomalies": dict(ANOMALIES_EVITEES),
        "weekend": dict(WEEKEND_ARRET),
        "fuites": dict(FUITES_CIBLE),
    }


def params_scenario(P, sc):
    """Valeurs retenues pour un scénario pur, ou pour « Personnalisé » (choix de la page Paramètres)."""
    vb = valeurs_bench(P)
    ch = choix_effectifs(P) if sc == PERSO else {k: sc for k in CLES_BENCH}
    v = {k: vb[k][ch[k]] for k in CLES_BENCH}
    return {"talon_evitable": v["veille"], "anomalies_evitees": v["anomalies"], "weekend_arret": v["weekend"],
            "fuites_cible": v["fuites"], "gain_autoclave": v["autoclave"], "gain_cnc": v["cnc"],
            "fe": v["fe"], "choix": ch}


def scenarios_params(P):
    return {sc: params_scenario(P, sc) for sc in SCENARIOS + [PERSO]}


def fe_choisi(P):
    """Facteur d'émission retenu dans la page Paramètres (Central par défaut)."""
    return valeurs_bench(P)["fe"][choix_effectifs(P)["fe"]]


def fmt_bench(cle, v):
    if cle == "fe":
        return f"{v:g}".replace(".", ",")
    if cle == "fuites":
        return f"{v:g} %".replace(".", ",")
    return f"{round(v * 100, 1):g} %".replace(".", ",")


def table_scenarios(P, choix=None):
    """Tableau Prudent / Central / Ambitieux + valeur retenue, pour chaque paramètre benchmarké."""
    vb = valeurs_bench(P)
    ch = choix_effectifs(P) if choix is None else choix
    lignes = []
    for k in CLES_BENCH:
        lignes.append({"Paramètre": LIBELLES[k], **{sc: fmt_bench(k, vb[k][sc]) for sc in SCENARIOS},
                       "Scénario retenu": ch[k], "Valeur retenue": fmt_bench(k, vb[k][ch[k]])})
    return pd.DataFrame(lignes)


# ----------------------------------------------------------------------------
# DONNÉES DE L'ENTREPRISE (dictionnaire et indicateurs fournis)
# ----------------------------------------------------------------------------
DICTIONNAIRE = pd.DataFrame(
    [
        ["Date", "Date du relevé journalier", "jj/mm/aaaa"],
        ["Jour de la semaine", "Jour calendaire", "texte"],
        ["Mois / Année", "Composantes calendaires de la date", "entier"],
        ["Heures-machine CNC", "Total des heures de fonctionnement des centres d'usinage CNC dans la journée", "heures/jour"],
        ["Cycles autoclave", "Nombre de cycles de cuisson de pièces composites réalisés dans la journée", "cycles/jour"],
        ["Conso. électrique (kWh)", "Consommation électrique totale du site (usinage, autoclave, HVAC, air comprimé, fixe)", "kWh/jour"],
        ["Taux fuites air comprimé (%)", "Part estimée de l'air comprimé produit perdue en fuites sur le réseau", "%"],
        ["Temp. salle blanche (°C)", "Température de consigne mesurée dans la salle blanche de drapage composites (cible 21°C)", "°C"],
        ["Humidité salle blanche (%)", "Humidité relative mesurée dans la salle blanche (cible ~45 %)", "%"],
        ["Taux rebut qualité (%)", "Part des pièces rejetées au contrôle qualité dans la journée", "%"],
        ["Part heures pleines (%)", "Part de la consommation électrique facturée en heures pleines", "%"],
        ["Coût énergie (TND)", "Coût énergétique journalier estimé (100 % électrique), tarifs indicatifs", "TND/jour"],
        ["IPE élec. (kWh/h-machine)", "Indicateur de performance énergétique électrique, rapporté à l'heure-machine CNC", "kWh/heure-machine"],
        ["Anomalie climatisation", "1 = dérive avérée de la climatisation de la salle blanche précédant une intervention ; 0 = normal", "binaire"],
        ["Arrêt annuel", "1 = congé industriel d'été (~10 jours ouvrés en août) ; à exclure des moyennes de fonctionnement normal", "binaire"],
    ],
    columns=["Variable (colonne)", "Description", "Unité"],
)
AGREGATS = pd.DataFrame(
    [
        ["Nombre de jours couverts (total)", 730.0, "Sur la période 2023-2024"],
        ["Nombre de jours d'arrêt annuel", 24.0, "Congé industriel d'été, 2 occurrences"],
        ["Heures-machine CNC totales hors arrêt", 93995.3, ""],
        ["Cycles autoclave totaux", 862.0, ""],
        ["Consommation électrique totale (kWh)", 6276711.0, "Site 100 % électrique"],
        ["Coût énergétique total (TND)", 2019095.1, "Tarifs indicatifs"],
        ["Taux de fuites moyen — 1ère moitié de période", 22.1, "Avant la campagne de réparation"],
        ["Taux de fuites moyen — 2nde moitié de période", 12.8, "Après la campagne de réparation"],
        ["Taux de rebut qualité moyen (%) hors arrêt", 1.8, "Sensible aux dérives de climatisation"],
        ["Nombre de jours en anomalie climatisation", 51.0, "Répartis sur 3 épisodes de dérive"],
    ],
    columns=["Indicateur", "Valeur", "Commentaire"],
)

# ----------------------------------------------------------------------------
# CHARGEMENT ET NETTOYAGE
# ----------------------------------------------------------------------------
def _num(s: pd.Series) -> pd.Series:
    s = s.astype(str)
    for esp in (" ", "\u00a0", "\u202f", "\u2009", "\t"):   # espaces normaux, insécables, fines (pandas 3 / pyarrow ne les gère pas avec \s)
        s = s.str.replace(esp, "", regex=False)
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def nettoyer(brut: pd.DataFrame) -> pd.DataFrame:
    df = brut.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for c in COLONNES_NUM:
        if c in df.columns:
            df[c] = _num(df[c])
    for c in ["Anomalie climatisation", "Arrêt annuel"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    def _date(v):
        if isinstance(v, str):
            return pd.to_datetime(v, dayfirst=True, errors="coerce")
        return pd.Timestamp(v) if pd.notna(v) else pd.NaT

    dt = df["Date"].map(_date)
    if "Mois" in df.columns and dt.notna().any():
        mois = pd.to_numeric(df["Mois"], errors="coerce")
        inv = pd.to_datetime(dict(year=dt.dt.year, month=dt.dt.day, day=dt.dt.month), errors="coerce")
        # Excel lit certains jj/mm en mm/jj : on inverse seulement les lignes dont le mois ne colle pas à la colonne « Mois »
        swap = (dt.dt.month != mois) & (inv.dt.month == mois)
        dt = dt.where(~swap, inv)
    df["Date"] = dt
    df = df.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
    return df


def preparer(df: pd.DataFrame) -> pd.DataFrame:
    d = df.rename(columns=RENOMMAGE).copy()
    requis = ["kwh", "cout", "cnc", "autoclave", "fuites", "temp", "hp", "anom", "arret"]
    d = d.dropna(subset=[c for c in requis if c in d.columns]).reset_index(drop=True)
    if "ipe" in d.columns:
        d["ipe"] = d["ipe"].fillna(d["ipe"].median())
    d["dow"] = d["Date"].dt.dayofweek
    d["type_jour"] = np.where(d["arret"] == 1, "Arrêt annuel", np.where(d["dow"].isin(JOURS_NON_OUVRES), "Week-end", "Ouvré"))
    d["est_weekend"] = d["dow"].isin(JOURS_NON_OUVRES).astype(int)
    d["est_arret_annuel"] = d["arret"].astype(int)
    d["mois"] = d["Date"].dt.to_period("M").astype(str)
    return d


@st.cache_data(show_spinner="Lecture du fichier Excel…")
def lire_chemin(path: str, mtime: float) -> pd.DataFrame:
    return nettoyer(pd.read_excel(path, sheet_name=SHEET))


@st.cache_data(show_spinner="Lecture du fichier Excel…")
def lire_octets(contenu: bytes) -> pd.DataFrame:
    return nettoyer(pd.read_excel(io.BytesIO(contenu), sheet_name=SHEET))


def charger_donnees():
    if os.path.exists(DATA_PATH):
        return lire_chemin(DATA_PATH, os.path.getmtime(DATA_PATH)), DATA_PATH
    st.sidebar.warning(f"Fichier introuvable :\n{DATA_PATH}")
    fichier = st.sidebar.file_uploader("Charger Aerotun.xlsx", type=["xlsx"])
    if fichier is not None:
        return lire_octets(fichier.getvalue()), "fichier chargé"
    return None, None


# ----------------------------------------------------------------------------
# ANALYSE (modèles, gisements, finance)
# ----------------------------------------------------------------------------
def modele_pareto(d):
    variables = ["cnc", "autoclave", "fuites"]
    while True:
        m = smf.ols("kwh ~ " + " + ".join(variables), data=d).fit()
        p, c = m.pvalues.drop("Intercept"), m.params.drop("Intercept")
        mauvaises = p[(p > SEUIL_P) | (c < 0)]
        if mauvaises.empty or len(variables) == 1:
            return m, variables
        variables.remove(mauvaises.idxmax() if (p > SEUIL_P).any() else c.idxmin())


def comparaison_modeles(d):
    formules = {
        "M1 : CNC + autoclave": "kwh ~ cnc + autoclave",
        "M2 : M1 + température": "kwh ~ cnc + autoclave + temp",
        "M5 : M2 + week-end": "kwh ~ cnc + autoclave + temp + est_weekend",
        "M6 : M2 + week-end + arrêt annuel": "kwh ~ cnc + autoclave + temp + est_weekend + est_arret_annuel",
    }
    lignes, objets = [], {}
    for nom, f in formules.items():
        if "est_arret_annuel" in f and d["est_arret_annuel"].nunique() < 2:
            continue
        m = d.pipe(lambda x: smf.ols(f, data=x).fit())
        n, k = len(d), len(m.params)
        rmse = np.sqrt((m.resid ** 2).sum() / (n - k))
        lignes.append({"Modèle": nom, "R² ajusté": m.rsquared_adj, "CV(RMSE) %": rmse / d["kwh"].mean() * 100,
                       "AIC": m.aic, "p max": m.pvalues.drop("Intercept").max()})
        objets[nom] = m
    return pd.DataFrame(lignes).set_index("Modèle"), objets


def cout_tarif(d):
    return d["cout"].sum() / d["kwh"].sum()


def autoconso_pv(d, P, kwc, facteur_residuel=1.0):
    prod_an = kwc * P["pv_prod"]
    prod_j = prod_an / 365.0
    auto_j = np.minimum(prod_j, d["kwh"] * facteur_residuel * d["hp"] / 100.0)
    taux = auto_j.sum() / (prod_j * len(d))
    return taux, taux * prod_an, (1 - taux) * prod_an, auto_j


def flux_projet(sp, P, tarif, g=None):
    g = P["g"] if g is None else g
    f = [-sp["capex"]]
    for t in range(1, sp["duree"] + 1):
        fp = (1 - sp["deg"]) ** (t - 1)
        eco = sp["kwh"] * fp * tarif * (1 + g) ** (t - 1)
        vente = sp["surplus"] * fp * P["tarif_vente"]
        opex = sp["opex"] * (1 + P["infl"]) ** (t - 1)
        f.append(eco + vente - opex - sp["remp"].get(t, 0.0))
    return np.array(f)


def van(flux, a):
    return sum(x / (1 + a) ** t for t, x in enumerate(flux))


def tri(flux):
    if not (flux[0] < 0 and flux[1:].sum() > 0):
        return np.nan
    try:
        return brentq(lambda r: van(flux, r), -0.99, 1000.0)
    except ValueError:
        return np.nan


def payback(flux, a=0.0):
    cum = flux[0]
    for t in range(1, len(flux)):
        v = flux[t] / (1 + a) ** t
        if cum < 0 <= cum + v:
            return t - 1 + (-cum) / v
        cum += v
    return np.nan


def evaluer(sp, P, tarif, a=None, g=None):
    """P doit contenir « fe » (facteur du scénario évalué) ; à défaut, le facteur Central."""
    a = P["taux"] if a is None else a
    n = sp["duree"]
    fl = flux_projet(sp, P, tarif, g)
    v = van(fl, a)
    fa = a * (1 + a) ** n / ((1 + a) ** n - 1)
    kwh_t = np.array([sp["kwh"] * (1 - sp["deg"]) ** (t - 1) for t in range(1, n + 1)])
    opex_t = np.array([sp["opex"] * (1 + P["infl"]) ** (t - 1) + sp["remp"].get(t, 0.0) for t in range(1, n + 1)])
    actu = np.array([(1 + a) ** -t for t in range(1, n + 1)])
    cga = sp["capex"] + (opex_t * actu).sum()
    co2_t = kwh_t * P.get("fe", FE_SCEN["Central"]) / 1000.0
    ae = v * fa
    t_ = tri(fl)
    return {
        "Projet": sp["nom"], "Durée (ans)": n, "CAPEX (TND)": sp["capex"], "Économie (kWh/an, an 1)": sp["kwh"],
        "Économie nette an 1 (TND)": fl[1], "VAN (TND)": v, "VAN/I": v / sp["capex"],
        "TRI (%)": t_ * 100 if pd.notna(t_) else np.nan,
        "Payback simple (ans)": sp["capex"] / fl[1] if fl[1] > 0 else np.nan,
        "Payback actualisé (ans)": payback(fl, a), "AE (TND/an)": ae,
        "LCOE / coût de l'énergie (TND/kWh)": cga / (kwh_t * actu).sum(),
        "CO2 évité an 1 (t)": co2_t[0], "CO2 évité moy. (t/an)": co2_t.mean(),
        "Coût d'abattement (TND/tCO2)": -ae / co2_t.mean() if co2_t.mean() > 0 else np.nan,
        "VAN avec valeur carbone (TND)": v + (co2_t * actu).sum() * P["prix_carbone"],
        "flux": fl,
    }


def spec(nom, kwh, P, surplus=0.0, capex=None, opex=None, remp=None):
    cle = {"Sobriété": "sob", "Efficacité": "eff", "Photovoltaïque": "pv"}[nom]
    return dict(nom=nom, kwh=float(kwh), surplus=float(surplus),
                capex=float(P.get("capex_" + cle, 0.0) if capex is None else capex),
                opex=float(P.get("opex_" + cle, 0.0) if opex is None else opex),
                duree=int(P["vie_" + cle]), deg=P["pv_deg"] if nom == "Photovoltaïque" else 0.0, remp=remp or {})


@st.cache_data(show_spinner="Calcul des modèles et des gisements…")
def analyser(d: pd.DataFrame, P_json: str):
    P = json.loads(P_json)
    n = len(d)
    fact_an = 365.0 / n
    R = {"n": n, "fact_an": fact_an}
    R["conso_an"] = d["kwh"].sum() * fact_an
    R["cout_an"] = d["cout"].sum() * fact_an
    R["tarif"] = cout_tarif(d)

    # --- Pareto des usages (modèle CNC + autoclave + fuites)
    m_par, variables = modele_pareto(d)
    noms = {"cnc": "Usinage CNC", "autoclave": "Autoclave", "fuites": "Fuites d'air comprimé", "Intercept": "Consommation de base (talon)"}
    contrib = {noms["Intercept"]: m_par.params["Intercept"] * n}
    for v in variables:
        contrib[noms[v]] = m_par.params[v] * d[v].sum()
    usage = pd.Series(contrib).sort_values(ascending=False) * fact_an
    R["usage_an"] = usage
    R["pareto"] = pd.DataFrame({"kWh/an": usage, "% kWh": usage / usage.sum() * 100,
                                "% cumulé": usage.cumsum() / usage.sum() * 100})
    R["talon_jour"] = float(m_par.params["Intercept"])
    R["c_fuites"] = float(m_par.params.get("fuites", np.nan))

    # --- Comparaison des modèles M1, M2, M6
    fin_base = d["Date"].min() + pd.DateOffset(months=12) - pd.Timedelta(days=1)
    d_ref = d[d["Date"] <= fin_base]          # période de référence 12 mois, comme le notebook
    R["comp_modeles"], objets = comparaison_modeles(d_ref)

    # --- Effet de l'arrêt annuel (M6) et anomalies HVAC
    m6 = objets.get("M6 : M2 + week-end + arrêt annuel")
    R["coef_arret"] = float(m6.params["est_arret_annuel"]) if m6 is not None else 0.0
    R["p_arret"] = float(m6.pvalues["est_arret_annuel"]) if m6 is not None else np.nan
    R["coef_we"] = float(m6.params["est_weekend"]) if m6 is not None else 0.0
    R["p_we"] = float(m6.pvalues["est_weekend"]) if m6 is not None else np.nan
    R["arret_ok"] = (R["coef_arret"] < 0) and (R["p_arret"] <= SEUIL_P)
    R["coef_anom"], R["p_anom"] = 0.0, np.nan
    if d["anom"].nunique() > 1:
        m_a = smf.ols("kwh ~ cnc + autoclave + fuites + anom", data=d).fit()
        R["coef_anom"], R["p_anom"] = float(m_a.params["anom"]), float(m_a.pvalues["anom"])
    R["anom_ok"] = (R["coef_anom"] > 0) and (R["p_anom"] <= SEUIL_P)

    R["jours_we"] = (d["type_jour"] == "Week-end").sum() * fact_an
    R["jours_ouvres"] = (d["type_jour"] == "Ouvré").sum() * fact_an
    R["jours_anom"] = d["anom"].sum() * fact_an
    R["talon_ouvres_an"] = R["talon_jour"] * R["jours_ouvres"]
    R["mesure_arret"] = float(d.loc[d["arret"] == 1, "kwh"].mean()) if (d["arret"] == 1).any() else np.nan

    # --- Fuites résiduelles (période de suivi = 2e année)
    d_post = d[d["Date"] > d["Date"].min() + pd.Timedelta(days=365)]
    SC = scenarios_params(P)      # Prudent, Central, Ambitieux et Personnalisé (choix de la page Paramètres)
    R["SC"] = SC

    def gain_fuites(cible):
        if not (R["c_fuites"] > 0) or len(d_post) == 0:
            return 0.0
        return R["c_fuites"] * (d_post["fuites"] - cible).clip(lower=0).sum() * 365.0 / len(d_post)

    sav = {}
    for sc, s in SC.items():
        veille = R["talon_ouvres_an"] * s["talon_evitable"]
        weekend = abs(R["coef_arret"]) * R["jours_we"] * s["weekend_arret"] if R["arret_ok"] else 0.0
        anom = R["coef_anom"] * R["jours_anom"] * s["anomalies_evitees"] if R["anom_ok"] else 0.0
        fuites = gain_fuites(s["fuites_cible"])
        auto = float(usage.get(noms["autoclave"], 0.0)) * s["gain_autoclave"]
        cnc = float(usage.get(noms["cnc"], 0.0)) * s["gain_cnc"]
        sav[sc] = {"Veille jours ouvrés": veille, "Week-ends en mode arrêt": weekend, "Anomalies HVAC": anom,
                   "Fuites résiduelles": fuites, "Autoclave": auto, "Usinage CNC": cnc}
    R["savings"] = pd.DataFrame(sav).T

    # --- Projets, par scénario (le facteur d'émission suit le scénario)
    res, specs_sc, taux_pv = {}, {}, {}
    for sc in SC:
        s_ = R["savings"].loc[sc]
        eco_sob = s_[["Veille jours ouvrés", "Week-ends en mode arrêt", "Anomalies HVAC"]].sum()
        eco_eff = s_[["Fuites résiduelles", "Autoclave", "Usinage CNC"]].sum()
        fr = 1 - (eco_sob + eco_eff) / R["conso_an"]
        taux, auto, surplus, _ = autoconso_pv(d, P, P["pv_kwc"], fr)
        surplus = min(surplus, P["pv_surplus_max"] * (auto + surplus))
        capex_pv = P["pv_kwc"] * P["pv_capex_kwc"]
        # OPEX PV = 62 TND/kWc/an (Colab / GIZ 2013), hors remplacement onduleur
        sp_pv = spec("Photovoltaïque", auto, P, surplus, capex=capex_pv,
                     opex=P["pv_opex_kwc"] * P["pv_kwc"],
                     remp={int(P["pv_remp_an"]): P["pv_remp_part"] * capex_pv})
        sps = [spec("Sobriété", eco_sob, P), spec("Efficacité", eco_eff, P), sp_pv]
        if P["fte"]:
            sps[1]["capex"] -= min(P["fte_eff"] * sps[1]["capex"], 1e18)
            sps[2]["capex"] -= min(P["fte_pv"] * sps[2]["capex"], P["fte_pv_plafond"])
        specs_sc[sc], taux_pv[sc] = sps, taux
        Psc = dict(P, fe=s["fe"])
        res[sc] = {sp["nom"]: evaluer(sp, Psc, R["tarif"]) for sp in sps}
    R["specs"], R["taux_pv"], R["res"] = specs_sc, taux_pv, res
    return R


# ----------------------------------------------------------------------------
# COMPOSANTS D'INTERFACE
# ----------------------------------------------------------------------------
def hero(titre, sous_titre):
    st.markdown(f'<div class="hero"><h1>{titre}</h1><p>{sous_titre}</p></div>', unsafe_allow_html=True)


def fmt(x, nd=0, suffixe=""):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n.d."
    return f"{x:,.{nd}f}".replace(",", " ") + suffixe


def ressources(liste):
    st.markdown("**Ressources**")
    for nom, lien in liste:
        st.markdown(f"- [{nom}]({lien})" if lien else f"- {nom}")


# ----------------------------------------------------------------------------
# PAGES
# ----------------------------------------------------------------------------
def page_overview(d, R):
    hero("AEROTUN — Stratégie énergétique", "Sobriété, efficacité énergétique et photovoltaïque : que faire, dans quel ordre, et pour quel retour ?")
    st.markdown('<span class="pill">Site 100 % électrique</span><span class="pill">CNC + autoclave + salle blanche</span>'
                '<span class="pill">Tunisie</span>', unsafe_allow_html=True)
    st.subheader("Problématique")
    chiffres = (f" Le site consomme environ **{fmt(R['conso_an'] / 1e6, 2)} GWh/an** pour **{fmt(R['cout_an'] / 1e3, 0)} kTND/an** "
                f"(tarif moyen {R['tarif']:.3f} TND/kWh)." if R else "")
    st.info(
        "AEROTUN est un site aéronautique tunisien dont l'activité repose sur des centres d'usinage CNC, des autoclaves de cuisson "
        "de composites et une salle blanche climatisée." + chiffres +
        "\n\n**Comment réduire cette facture et les émissions associées (scope 2) sans compromettre la qualité de la salle blanche, "s
        "et quel portefeuille d'investissements retenir entre sobriété, efficacité énergétique et photovoltaïque ?**"
    )
    st.subheader("Objectif")
    st.markdown(
        "1. **Comprendre** la consommation : normaliser par l'activité (heures CNC, cycles d'autoclave) et distinguer jours ouvrés, week-ends et arrêt annuel.\n"
        "2. **Chiffrer les gisements de sobriété et d’efficacité énergétique à partir des données de l’entreprise et de benchmarks (aéronautique, Tunisie), selon trois scénarios (Prudent, Central, Ambitieux), et évaluer le potentiel d’une centrale photovoltaïque en autoconsommation : **Prudent, Central, Ambitieux**.\n"
        "3. **Évaluer** chaque projet : VAN, TRI, temps de retour, VAN/I, annuité équivalente, LCOE, CO₂ évité, coût d'abattement.\n"
        "4. **Choisir** par optimisation multi-objectifs (énergie, économie, environnement)."
    )
    c1, c2, c3 = st.columns(3)
    c1.markdown("##### 🟢 Sobriété\nSupprimer les consommations inutiles : veille, week-ends, dérives de climatisation.")
    c2.markdown("##### 🔴 Efficacité\nAméliorer les équipements : fuites d'air comprimé, cycle d'autoclave, usinage CNC.")
    c3.markdown("##### 🟠 Photovoltaïque\nProduire sur place et autoconsommer, avec injection plafonnée du surplus.")
    if R:
        P = st.session_state["P"]
        st.subheader("Le site en chiffres")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Consommation annuelle", fmt(R["conso_an"] / 1e6, 2, " GWh"))
        k2.metric("Coût énergétique annuel", fmt(R["cout_an"] / 1e3, 0, " kTND"))
        k3.metric("Jours analysés", fmt(R["n"]))
        k4.metric("Émissions annuelles (scope 2)", fmt(R["conso_an"] * fe_choisi(P) / 1000, 0, " tCO₂"),
                  help=f"Facteur d'émission retenu : {fe_choisi(P):g} kgCO₂/kWh (scénario {choix_effectifs(P)['fe']}, page Paramètres).")
    st.subheader("Comment naviguer")
    st.table(pd.DataFrame({
        "Page": ["Distribution de la consommation", "Données de l'entreprise", "Paramètres", "Benchmarking", "Dashboard"],
        "Contenu": ["Histoire de la consommation : corrélations, modèles, formules, ressources",
                    "Dictionnaire de données, indicateurs agrégés, contrôle de cohérence",
                    "Hypothèses chiffrées, sources, scénario Prudent / Central / Ambitieux de chaque paramètre benchmarké",
                    "Cas d'entreprises utilisés (aéronautique, Tunisie, industrie) et paramètres qu'ils calibrent",
                    "Les trois piliers : maths, données, ressources, choix final"],
    }))


def page_distribution_consommation(d, R):
    hero("Distribution de la consommation — Comprendre la consommation", "Avant d'économiser, il faut savoir ce qui consomme, et de combien.")
    P = st.session_state["P"]

    st.header("Scène 1 · Un site, trois gros mangeurs d'énergie")
    corr_cnc = d["kwh"].corr(d["cnc"])
    corr_auto = d["kwh"].corr(d["autoclave"])
    st.markdown(
        f"Chaque jour, le site relève sa consommation électrique, l'activité des centres d'usinage et le nombre de cycles d'autoclave. "
        f"Le premier constat est net : la consommation suit l'activité. Sa corrélation avec les heures-machine CNC est de **{corr_cnc:.2f}** "
        f"et avec les cycles d'autoclave de **{corr_auto:.2f}**. Mais une partie ne dépend d'aucune machine : le **talon** "
        f"(climatisation de la salle blanche, air comprimé, éclairage, auxiliaires)."
    )

    st.header("Scène 2 · Ce que disent les corrélations")
    variables = {"cnc": "Heures CNC", "autoclave": "Cycles autoclave", "kwh": "Conso. (kWh)", "fuites": "Fuites (%)",
                 "temp": "Température", "hum": "Humidité", "rebut": "Rebut (%)", "hp": "Heures pleines (%)", "cout": "Coût (TND)", "ipe": "IPE (kWh/h-machine)"}
    cols = [c for c in variables if c in d.columns]
    dd = d[cols].copy()
    dd["Émissions (tCO₂)"] = d["kwh"] * fe_choisi(P) / 1000
    corr = dd.rename(columns=variables).corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
                    title="Matrice de corrélation — production, énergie, coût et émissions")
    fig.update_layout(height=560)
    st.plotly_chart(fig, **STRETCH)
    st.caption("Lecture : le coût et les émissions sont des dérivés de la consommation (corrélation 1) ; le taux de rebut et la température "
               f"rappellent le lien entre dérives de climatisation et qualité. Émissions calculées avec FE = {fe_choisi(P):g} kgCO₂/kWh "
               f"(scénario {choix_effectifs(P)['fe']}).")
    st.latex(r"r_{xy}=\frac{\operatorname{cov}(x,y)}{\sigma_x\,\sigma_y}\qquad \text{tCO}_2=\frac{E_{\text{kWh}}\times FE}{1000}")

    st.header("Scène 4 · Jours ouvrés, week-ends et congés")
    par_type = d.groupby("type_jour").agg(jours=("kwh", "size"), kwh_jour=("kwh", "mean"), cnc=("cnc", "mean"), kwh_total=("kwh", "sum"))
    par_type["Part de la conso. (%)"] = par_type["kwh_total"] / par_type["kwh_total"].sum() * 100
    fig2 = px.bar(par_type.reset_index(), x="type_jour", y="kwh_jour", color="type_jour", text_auto=".0f",
                  labels={"kwh_jour": "kWh par jour", "type_jour": ""}, title="Consommation moyenne par type de jour")
    fig2.update_layout(showlegend=False, height=380)
    st.plotly_chart(fig2, **STRETCH)
    st.markdown(f"Même en congé, le site consomme **{fmt(R['mesure_arret'])} kWh/jour** : climatisation de la salle blanche, fuites d'air comprimé "
                f"et équipements en veille.")

    st.header("Scène 5 · Où va l'énergie ? (Pareto)")
    fig3 = go.Figure()
    par = R["pareto"]
    fig3.add_bar(x=par.index, y=par["kWh/an"], name="kWh/an", marker_color="#1b6ca8")
    fig3.add_scatter(x=par.index, y=par["% cumulé"], name="% cumulé", yaxis="y2", mode="lines+markers", line=dict(color="#e71d36"))
    fig3.update_layout(yaxis2=dict(overlaying="y", side="right", range=[0, 105]), height=380, title="Pareto des usages")
    st.plotly_chart(fig3, **STRETCH)
    st.latex(r"E_{\text{usage}}=\beta_{\text{usage}}\times\sum_j x_{j}\times\frac{365}{n}")

    ressources([
        ("ISO 50001 — systèmes de management de l'énergie", "https://www.iso.org/iso-50001-energy-management.html"),
        ("ANME — Agence nationale pour la maîtrise de l'énergie", "https://www.anme.tn"),
        ("STEG — tarifs moyenne tension", "https://www.steg.com.tn/fr/page/tarifs-moyenne-tension"),
        ("Facteur d'émission du réseau tunisien (U4E, 2019)", "https://united4efficiency.org/wp-content/uploads/2019/11/FR_TUN_U4E-Country-Saving-Assessment_Cooling.pdf"),
        ("Protocole IPMVP (mesure et vérification) — EVO", ""),
        ("ASHRAE Guideline 14 (calibrage des modèles énergétiques)", ""),
    ])


def page_donnees(d, R):
    hero("Données de l'entreprise", "Dictionnaire des variables, indicateurs agrégés fournis et contrôle de cohérence avec le fichier Excel.")
    st.subheader("Variables (colonnes)")
    st.dataframe(DICTIONNAIRE, **STRETCH, hide_index=True)
    st.subheader("Indicateurs agrégés — calculés hors jours d'arrêt annuel")
    st.dataframe(AGREGATS.style.format({"Valeur": "{:,.1f}"}), **STRETCH, hide_index=True)
    if d is None:
        st.warning("Fichier Excel non chargé : le contrôle de cohérence n'est pas disponible.")
        return
    st.subheader("Contrôle de cohérence : fichier Excel vs indicateurs de l'entreprise")
    hors = d[d["arret"] == 0]
    d_post_limite = d["Date"].min() + pd.Timedelta(days=365)
    calcul = [
        len(d), int(d["arret"].sum()), hors["cnc"].sum(), d["autoclave"].sum(), d["kwh"].sum(), d["cout"].sum(),
        d.loc[d["Date"] <= d_post_limite - pd.Timedelta(days=1), "fuites"].mean(), d.loc[d["Date"] > d_post_limite - pd.Timedelta(days=1), "fuites"].mean(),
        hors["rebut"].mean() if "rebut" in hors else np.nan, int(d["anom"].sum()),
    ]
    ctrl = AGREGATS[["Indicateur", "Valeur"]].copy()
    ctrl["Fichier Excel"] = calcul
    ctrl["Écart (%)"] = (ctrl["Fichier Excel"] - ctrl["Valeur"]) / ctrl["Valeur"] * 100
    tol = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 0.1]
    ctrl["Statut"] = ["OK" if abs(e) <= t else "À vérifier" for e, t in zip(ctrl["Écart (%)"], tol)]
    st.dataframe(ctrl.style.format({"Valeur": "{:,.1f}", "Fichier Excel": "{:,.1f}", "Écart (%)": "{:+.2f}"}), **STRETCH, hide_index=True)
    conso_hors = hors["kwh"].sum()
    if abs(d["kwh"].sum() - 6276711) < abs(conso_hors - 6276711):
        st.warning("Le total de 6 276 711 kWh correspond à la somme des jours **arrêts inclus** : l'intitulé « hors arrêt » est à confirmer avec l'entreprise.")
    ouv = d[d["type_jour"] == "Ouvré"]
    st.caption(f"Jours ouvrés hors congés et week-ends : {len(ouv)} jours, {fmt(ouv['kwh'].sum())} kWh "
               f"({ouv['kwh'].sum() / d['kwh'].sum() * 100:.1f} % de la consommation totale).")
    with st.expander("Aperçu des données journalières"):
        st.dataframe(d.drop(columns=["dow", "mois"], errors="ignore").head(60), **STRETCH)
        st.download_button("Télécharger les données nettoyées (CSV)", d.to_csv(index=False).encode("utf-8"), "aerotun_nettoye.csv", "text/csv")


# Lignes fixes du tableau des paramètres (alignées sur le notebook Colab : mêmes valeurs et mêmes sources)
PARAM_AVANT = [
    ("Taux d'actualisation", "10 %", "Notebook Colab : TAUX_ACTUALISATION = 0,10 (taux directeur BCT 7 % + prime de risque ; test 7-14 %)", "Aligné Colab"),
    ("Hausse du prix de l'électricité", "3 %/an", "Notebook Colab : HAUSSE_PRIX_ELEC = 0,03 ; barème MT STEG inchangé depuis 01/10/2022", "Aligné Colab"),
    ("Inflation de la maintenance", "5,5 %/an", "Notebook Colab : INFLATION_OPEX = 0,055 (INS/BCT sept. 2026)", "Aligné Colab"),
    ("Prix du carbone", "50 TND/tCO₂", "Notebook Colab : PRIX_CARBONE_REF = 50 (illustratif)", "Aligné Colab"),
    ("Taux de change", "3,0 TND/USD", "BCT, 08/10/2026", "Sourcé"),
]
PARAM_APRES = [
    ("Productible PV", "1 650 kWh/kWc/an", "Guide ANME d'autoproduction (identique Colab)", "Sourcé"),
    ("CAPEX PV", "2 500 TND/kWc", "Notebook Colab : PV_CAPEX_KWC = 2500 ; marché 2 000-2 900 ; Safran Soliman 2 765", "Aligné Colab"),
    ("OPEX PV (hors onduleur)", "62 TND/kWc/an", "Notebook Colab : PV_OPEX_KWC = 62 (GIZ 2013) — plus 2 % du CAPEX", "Aligné Colab"),
    ("Remplacement onduleur", "10 % du CAPEX en année 12", "Notebook Colab : PV_ANNEE_REMPLACEMENT = 12 ; PV_PART_REMPLACEMENT = 0,10", "Aligné Colab"),
    ("Durée de vie et dégradation PV", "20 ans ; -0,5 %/an", "Guide ANME ; notebook Colab", "Sourcé"),
    ("Tarif d'achat du surplus", "0,12 TND/kWh", "Notebook Colab / STEG barème autoproduction", "Aligné Colab"),
    ("Plafond de vente du surplus", "30 % de la production", "Guide ENR Tunisie 2019", "Sourcé"),
    ("Tarif moyen implicite", "calculé", "Coût total / consommation totale des données", "Calculé"),
    ("CAPEX / OPEX sobriété", "5 000 TND / 0", "Notebook Colab (hypothèse à remplacer par devis)", "Aligné Colab — devis prioritaires"),
    ("CAPEX / OPEX efficacité", "85 000 TND / 2 000", "Notebook Colab (hypothèse à remplacer par devis)", "Aligné Colab — devis prioritaires"),
    ("Durées de vie sobriété / efficacité", "5 / 10 ans", "Notebook Colab : DUREE_VIE", "Aligné Colab"),
    ("Subventions FTE", "20 % PV (plafond 200 000) ; jusqu'à 50 % efficacité", "ANME (guide ER 2024)", "Sourcé"),
]
COL_SC = "Prudent / Central / Ambitieux"


def param_table(P):
    """Tableau des paramètres : les paramètres benchmarkés ont leur colonne Prudent / Central / Ambitieux."""
    vb = valeurs_bench(P)
    ch = choix_effectifs(P)
    lignes = [[a, b, "—", "—", c, e] for a, b, c, e in PARAM_AVANT]
    for k in CLES_BENCH:
        lignes.append([LIBELLES[k], fmt_bench(k, vb[k]["Central"]), " / ".join(fmt_bench(k, vb[k][sc]) for sc in SCENARIOS),
                       ch[k], SOURCES_BENCH[k], "Benchmark / hypothèse — défaut : Central"])
    lignes.append(["Transférabilité du benchmark (pilote veille, autoclave, CNC)", "⅔", "⅓ / ⅔ / 1", "suit les trois paramètres ci-dessus",
                   "Pratique courante : on n'applique pas un gain publié tel quel à un autre site", "Hypothèse"])
    lignes += [[a, b, "—", "—", c, e] for a, b, c, e in PARAM_APRES]
    return pd.DataFrame(lignes, columns=["Paramètre", "Valeur par défaut", COL_SC, "Scénario retenu", "Source", "Statut"])


def page_parametres(d, R):
    hero("Paramètres", "Toutes les hypothèses, leur source et leur statut. Les paramètres benchmarkés existent en trois valeurs ; modifie-les ci-dessous : le dashboard se recalcule.")
    P = st.session_state["P"]
    st.markdown("Les paramètres issus de benchmarks ne sont pas transférables tels quels : ils ont une valeur **Prudent**, **Central** et **Ambitieux**. "
                "Le **Central** (le plus raisonnable) est retenu par défaut. Les autres paramètres n'ont qu'une valeur.")
    st.dataframe(param_table(P), **STRETCH, hide_index=True)
    st.caption("Facteur d'émission : Prudent = valeur la plus haute (empreinte de référence non sous-estimée) ; Ambitieux = valeur la plus basse "
               "(réseau qui se décarbone). Les gains en % sont exprimés sur l'usage concerné (veille : part du talon).")

    st.subheader("Scénario retenu par paramètre")
    b1, b2, b3 = st.columns(3)
    for col, sc in zip((b1, b2, b3), SCENARIOS):
        if col.button(f"Tout en {sc}", **STRETCH, key=f"tout_{sc}"):
            st.session_state["P"] = dict(P, choix={k: sc for k in CLES_BENCH})
            st.rerun()

    st.subheader("Modifier les paramètres")
    vb = valeurs_bench(P)
    ch = choix_effectifs(P)
    with st.form("form_params"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Finance**")
            taux = st.number_input("Taux d'actualisation", 0.0, 0.5, float(P["taux"]), 0.005, format="%.3f")
            g = st.number_input("Hausse du prix de l'électricité", 0.0, 0.3, float(P["g"]), 0.005, format="%.3f")
            infl = st.number_input("Inflation de la maintenance", 0.0, 0.3, float(P["infl"]), 0.005, format="%.3f")
            prix_c = st.number_input("Prix du carbone (TND/tCO₂)", 0.0, 1000.0, float(P["prix_carbone"]), 5.0)
            st.caption("Facteur d'émission : voir les scénarios ci-dessous.")
        with c2:
            st.markdown("**Photovoltaïque**")
            kwc = st.number_input("Puissance (kWc)", 10.0, 2000.0, float(P["pv_kwc"]), 10.0)
            prod = st.number_input("Productible (kWh/kWc/an)", 1000.0, 2200.0, float(P["pv_prod"]), 10.0)
            capex_kwc = st.number_input("CAPEX (TND/kWc)", 1000.0, 5000.0, float(P["pv_capex_kwc"]), 50.0)
            opex_kwc = st.number_input("OPEX PV (TND/kWc/an, hors onduleur)", 0.0, 200.0, float(P.get("pv_opex_kwc", 62.0)), 1.0)
            t_vente = st.number_input("Tarif de vente du surplus (TND/kWh)", 0.0, 0.5, float(P["tarif_vente"]), 0.01)
        with c3:
            st.markdown("**Sobriété et efficacité**")
            capex_sob = st.number_input("CAPEX sobriété (TND)", 0.0, 500000.0, float(P["capex_sob"]), 1000.0)
            opex_sob = st.number_input("OPEX sobriété (TND/an)", 0.0, 100000.0, float(P["opex_sob"]), 500.0)
            capex_eff = st.number_input("CAPEX efficacité (TND)", 0.0, 2000000.0, float(P["capex_eff"]), 5000.0)
            opex_eff = st.number_input("OPEX efficacité (TND/an)", 0.0, 200000.0, float(P["opex_eff"]), 500.0)
            part_veille = st.number_input("Part de veille dans le talon", 0.0, 1.0, float(P["part_veille"]), 0.01)
        st.markdown("**Scénario de chaque paramètre benchmarké** (Ambitieux par défaut)")
        nouveau_choix = {}
        grille = st.columns(4)
        for i, k in enumerate(CLES_BENCH):
            with grille[i % 4]:
                nouveau_choix[k] = st.selectbox(
                    LIBELLES[k], SCENARIOS, index=SCENARIOS.index(ch[k]),
                    format_func=lambda sc, k=k: f"{sc} ({fmt_bench(k, vb[k][sc])})")
        ok = st.form_submit_button("Appliquer les paramètres", type="primary")
    if ok:
        nouveau = dict(P)
        nouveau.update(taux=taux, g=g, infl=infl, prix_carbone=prix_c, pv_kwc=kwc, pv_prod=prod, pv_capex_kwc=capex_kwc,
                       pv_opex_kwc=opex_kwc, tarif_vente=t_vente, capex_sob=capex_sob, opex_sob=opex_sob, capex_eff=capex_eff,
                       opex_eff=opex_eff, part_veille=part_veille, choix=nouveau_choix)
        st.session_state["P"] = nouveau
        st.success("Paramètres appliqués. Ouvre le Dashboard (scénario « Personnalisé ») pour voir les résultats.")
    if st.button("Revenir aux valeurs par défaut"):
        st.session_state["P"] = p_defaut()
        st.rerun()


BENCH = pd.DataFrame(
    [
        ["Aéronautique", "OPTO-CLAVE (projet UE)", "Autoclave", "Optimisation du cycle de cuisson d'un panneau composite", "-16 % d'énergie ; -12 % de durée", 16.0, "Élevée (un cas)", "Gain autoclave (ancre −16 %)"],
        ["Aéronautique", "GKN Aerospace / US Air Force", "Usinage CNC", "Outils, lubrification, programmation, paramètres de coupe", "jusqu'à -73 % (cellule)", 73.0, "Moyenne (borne très haute)", "Justifie le levier CNC (non utilisé directement)"],
        ["Aéronautique", "Politecnico di Milano", "Usinage CNC", "Mise en veille par composants", "gains significatifs (non chiffrés)", np.nan, "Moyenne", "Veille évitable (qualitatif)"],
        ["Aéronautique", "Fabricant aéronautique (Kaeser)", "Air comprimé", "Régulation et dimensionnement des compresseurs", "-7 % de puissance spécifique", 7.0, "Moyenne", "Fuites cible (contexte)"],
        ["Aéronautique", "Safran (sites européens)", "Sobriété", "Plan de sobriété du groupe", "-10 % d'énergie (2024 vs 2019)", 10.0, "Moyenne", "Contrôle de cohérence"],
        ["Aéronautique", "Plymouth « cool-clave »", "Autoclave", "Pressurisation à l'air froid", "≈ -35 % (estimation)", 35.0, "Faible", "Non retenu"],
        ["Industrie générale", "General Dynamics (munitions)", "Sobriété", "Arrêts le week-end, consignes de refroidissement", "+11,9 % de performance (3 ans)", 11.9, "Moyenne", "Week-ends en mode arrêt (contexte)"],
        ["Industrie générale", "Volvo CE", "Sobriété", "Habitudes opérateurs, arrêt des machines à vide", "-14 % d'électricité totale", 14.0, "Moyenne", "Veille évitable (contexte)"],
        ["Industrie générale", "Cummins", "Sobriété", "Management de l'énergie, changements à faible coût", "+12,6 % de performance", 12.6, "Moyenne", "Contrôle de cohérence"],
        ["Industrie générale", "Graham Packaging", "Sobriété", "Arrêt automatique de charges à vide", "-3 % d'électricité totale", 3.0, "Moyenne", "Veille évitable (contexte)"],
        ["Industrie générale", "Johnson Controls", "Air comprimé", "Gestion des fuites, suivi de pression", "-39 % d'électricité d'air comprimé", 39.0, "Moyenne", "Fuites cible (contexte)"],
        ["Industrie générale", "Programme DOE ISO 50001 / SEP", "Efficacité", "Diagnostic, procédés, modernisation", "5,6 % à 30,6 % sur 3 ans", 30.6, "Moyenne", "Contrôle de cohérence"],
        ["Tunisie", "ANME — audit énergétique obligatoire", "Industrie (grands consommateurs)", "Audit et cogénération (thermique inclus)", "-30 % de facture ; -35 % d'énergie primaire", 30.0, "Moyenne (presse)", "Contrôle de cohérence"],
        ["Tunisie", "ANME — cimenterie (récupération de chaleur)", "Électricité", "Récupération de chaleur", "-30 % d'électricité ; retour < 2 ans", 30.0, "Moyenne (presse)", "Contrôle de cohérence"],
        ["Tunisie", "ANME / GIZ — diagnostic de l'air comprimé", "Air comprimé", "Cahier des charges des diagnostics (2021)", "rendement de l'air comprimé souvent limité à 10-15 %", np.nan, "Moyenne (presse)", "Fuites cible (objectif ANME : taux divisé par deux)"],
        ["Tunisie", "ANME — contrats-programmes", "Industrie", "Investissements d'efficacité", "retour ≈ 2 ans ; tep économisée ≈ 152 DT", np.nan, "Moyenne (ancien)", "Contrôle de cohérence"],
        ["Tunisie", "Safran Soliman", "Photovoltaïque", "PV 974,7 kWc ; couverture ≈ 16,8 %", "retour annoncé 6-7 ans", np.nan, "Moyenne (fiche non auditée)", "Facteur d'émission (Prudent, 0,587) ; CAPEX PV"],
    ],
    columns=["Domaine", "Entreprise / source", "Cible", "Levier", "Résultat publié", "Gain (%)", "Fiabilité", "Paramètre calibré"],
)


def page_benchmarking(d, R):
    hero("Benchmarking", "Les modèles d'entreprises utilisés pour calibrer les gisements et contrôler les résultats.")
    P = st.session_state["P"]
    st.markdown("Les gains machine par machine (autoclave, CNC) viennent de cas **aéronautiques internationaux**, faute d'équivalent tunisien publié. "
                "Les références **tunisiennes** (ANME, STEG, GIZ, Safran Soliman) servent aux coûts, tarifs, subventions et contrôles de plausibilité. "
                "Les résultats ne sont pas transférables tels quels : chaque paramètre calibré a une valeur **Prudent / Central / Ambitieux**.")
    domaines = st.multiselect("Domaines", sorted(BENCH["Domaine"].unique()), default=sorted(BENCH["Domaine"].unique()))
    b = BENCH[BENCH["Domaine"].isin(domaines)]
    st.dataframe(b.drop(columns=["Gain (%)"]), **STRETCH, hide_index=True)
    bb = b.dropna(subset=["Gain (%)"])
    if not bb.empty:
        fig = px.bar(bb, x="Gain (%)", y="Entreprise / source", color="Domaine", orientation="h", text="Gain (%)",
                     title="Gains publiés (ordre de grandeur, périmètres différents)")
        fig.update_layout(height=520, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, **STRETCH)

    st.subheader("Paramètres calibrés par ces benchmarks")
    st.dataframe(table_scenarios(P), **STRETCH, hide_index=True)
    st.caption("Le scénario retenu se choisit dans la page Paramètres (Central par défaut).")

    if R is not None:
        st.subheader("Safran Soliman vs AEROTUN (photovoltaïque)")
        saf = dict(kwc=974.7, prod=1_500_000.0, capex=2_695_000.0, gain=410_000.0, conso=8_935_000.0)
        sp = R["specs"][PERSO][2]
        res = R["res"][PERSO]["Photovoltaïque"]
        prod_a = sp["kwh"] + sp["surplus"]
        tab = pd.DataFrame({
            "Safran Soliman (fiche)": [saf["kwc"], saf["prod"], saf["prod"] / saf["kwc"], saf["capex"] / saf["kwc"],
                                       saf["prod"] / saf["conso"] * 100, saf["gain"] / saf["prod"], saf["capex"] / saf["gain"]],
            "AEROTUN (paramètres actuels)": [P["pv_kwc"], prod_a, P["pv_prod"], sp["capex"] / P["pv_kwc"],
                                             prod_a / R["conso_an"] * 100, res["Économie nette an 1 (TND)"] / prod_a, res["Payback simple (ans)"]],
        }, index=["Puissance (kWc)", "Production (kWh/an)", "kWh/kWc", "CAPEX (TND/kWc)", "Couverture de la conso. (%)",
                  "Gain (TND/kWh produit)", "Payback simple (ans)"])
        st.dataframe(tab.style.format("{:,.2f}"), **STRETCH)
        st.caption("Chiffres Safran : fiche projet reprise dans le document fourni (non auditée). Safran consomme environ "
                   f"{saf['conso'] / R['conso_an']:.1f} fois plus qu'AEROTUN : son facteur d'émission (0,587 kgCO₂/kWh) est donc placé en scénario "
                   f"Prudent ; AEROTUN utilise {fe_choisi(P):g} (scénario {choix_effectifs(P)['fe']}).")


def page_dashboard(d, R):
    hero("Dashboard", "Les trois piliers à suivre : les maths, les données de l'entreprise, les ressources — et le choix final.")
    P = st.session_state["P"]
    c1, c2 = st.columns([2, 1])
    scenario = c1.radio("Scénario de gisement", SCENARIOS + [PERSO], index=3, horizontal=True, key="scenario_dash",
                        help="Prudent / Central / Ambitieux : tous les paramètres benchmarkés dans le même scénario. "
                             "Personnalisé : le scénario choisi paramètre par paramètre dans la page Paramètres (Central par défaut).")
    fte = c2.toggle("Avec subventions FTE (borne haute)", value=P["fte"])
    if fte != P["fte"]:
        st.session_state["P"] = dict(P, fte=fte)
        st.rerun()
    res = R["res"][scenario]
    specs = R["specs"][scenario]
    sav = R["savings"].loc[scenario]
    tarif = R["tarif"]
    Pv = dict(P, fe=R["SC"][scenario]["fe"])      # paramètres + facteur d'émission du scénario affiché

    with st.expander(f"Paramètres benchmarkés appliqués — scénario {scenario}"):
        st.dataframe(table_scenarios(P, R["SC"][scenario]["choix"]), **STRETCH, hide_index=True)

    t_sob, t_eff, t_pv, t_cmp = st.tabs(["🟢 Sobriété", "🔴 Efficacité", "🟠 Photovoltaïque", "⚖️ Comparaison et choix"])

    def kpis(nom):
        r = res[nom]
        a, b, c, e = st.columns(4)
        a.metric("Économie (kWh/an)", fmt(r["Économie (kWh/an, an 1)"]))
        b.metric("Économie 1re année (TND)", fmt(r["Économie nette an 1 (TND)"]))
        c.metric("VAN (TND)", fmt(r["VAN (TND)"]))
        e.metric("TRI", fmt(r["TRI (%)"], 0, " %"))
        a, b, c, e = st.columns(4)
        a.metric("Payback actualisé", fmt(r["Payback actualisé (ans)"], 1, " ans"))
        b.metric("VAN/I", fmt(r["VAN/I"], 2))
        c.metric("CO₂ évité (t/an, an 1)", fmt(r["CO2 évité an 1 (t)"], 0), help=f"Facteur d'émission du scénario {scenario} : {Pv['fe']:g} kgCO₂/kWh")
        e.metric("Coût d'abattement (TND/tCO₂)", fmt(r["Coût d'abattement (TND/tCO2)"], 0))

    # ---------------- Sobriété
    with t_sob:
        st.subheader("Sobriété énergétique : supprimer l'inutile")
        kpis("Sobriété")
        m, dn, rs = st.tabs(["Les maths", "Les données de l'entreprise", "Les ressources"])
        with m:
            st.latex(r"E_{sob}=E_{talon,ouvrés}\,\tau_{veille}+|\beta_{arrêt}|\,J_{we}\,\tau_{we}+\beta_{anom}\,J_{anom}\,\tau_{anom}")
            st.latex(r"\tau_{veille}=\tau^{bench}_{veille}\times part_{veille}\times transfert")
            comp = R["savings"][["Veille jours ouvrés", "Week-ends en mode arrêt", "Anomalies HVAC"]]
            fig = px.bar(comp.reset_index().melt("index"), x="index", y="value", color="variable", labels={"index": "Scénario", "value": "kWh/an", "variable": "Levier"},
                         title="Gisement de sobriété par levier et par scénario")
            st.plotly_chart(fig, **STRETCH)
        with dn:
            st.markdown(f"- Talon du modèle (Pareto) : **{fmt(R['talon_jour'])} kWh/jour** ; consommation mesurée pendant l'arrêt annuel : **{fmt(R['mesure_arret'])} kWh/jour**.")
            st.markdown(f"- Effet de l'arrêt annuel (M6) : **{R['coef_arret']:+,.0f} kWh/jour** (p = {R['p_arret']:.3f}) ; week-end : {R['coef_we']:+.0f} kWh/j (p = {R['p_we']:.2f}).")
            st.markdown(f"- Anomalies de climatisation : {R['jours_anom']:.1f} jours/an, surconsommation {R['coef_anom']:+,.0f} kWh/jour (p = {R['p_anom']:.3f}) → "
                        f"{'retenue' if R['anom_ok'] else 'non retenue (non significative)'}.")
            st.markdown(f"- Jours par an : {R['jours_ouvres']:.0f} ouvrés, {R['jours_we']:.0f} week-ends hors arrêt.")
        with rs:
            ressources([("ANME — audit énergétique", "http://www.anme.tn/fr/project/energy-audit-programme"),
                        ("La Presse — économies d'énergie dans l'industrie (18/10/2026)", "https://www.lapresse.tn/2026/10/18/industrie-tunisienne-les-economies-denergie-un-levier-de-competitivite-encore-sous-exploite-2/"),
                        ("Volvo CE, General Dynamics, Cummins, Graham Packaging : voir Benchmarking", "")])

    # ---------------- Efficacité
    with t_eff:
        st.subheader("Efficacité énergétique : améliorer les équipements")
        kpis("Efficacité")
        m, dn, rs = st.tabs(["Les maths", "Les données de l'entreprise", "Les ressources"])
        with m:
            st.latex(r"E_{eff}=\beta_{fuites}\sum_j(F_j-F_{cible})^{+}\tfrac{365}{n}+E_{autoclave}\,g_{auto}+E_{CNC}\,g_{cnc}")
            comp = R["savings"][["Fuites résiduelles", "Autoclave", "Usinage CNC"]]
            fig = px.bar(comp.reset_index().melt("index"), x="index", y="value", color="variable", labels={"index": "Scénario", "value": "kWh/an", "variable": "Levier"},
                         title="Gisement d'efficacité par levier et par scénario")
            st.plotly_chart(fig, **STRETCH)
        with dn:
            par = R["pareto"]
            fig = go.Figure()
            fig.add_bar(x=par.index, y=par["kWh/an"], marker_color="#1b6ca8", name="kWh/an")
            fig.add_scatter(x=par.index, y=par["% cumulé"], yaxis="y2", name="% cumulé", line=dict(color="#e71d36"))
            fig.update_layout(yaxis2=dict(overlaying="y", side="right", range=[0, 105]), title="Pareto des usages (kWh/an)", height=360)
            st.plotly_chart(fig, **STRETCH)
            st.markdown(f"Les fuites d'air comprimé ont déjà été réduites (campagne de réparation) : seul le potentiel **résiduel** de la période de suivi est valorisé "
                        f"(coefficient : {R['c_fuites']:+.1f} kWh/jour par point de fuite).")
        with rs:
            ressources([("Projet UE OPTO-CLAVE (CORDIS)", "https://cordis.europa.eu/project/id/323520/reporting"),
                        ("GKN Aerospace — Modern Machine Shop", "https://www.mmsonline.com/articles/savings-from-sustainability"),
                        ("ANME — cahier des charges de l'audit énergétique industriel", "https://www.anme.tn/sites/default/files/2025-01/cahier_des_charges_audit_secteur_industriel_2011-07-07.pdf")])

    # ---------------- PV
    with t_pv:
        st.subheader("Photovoltaïque : produire et autoconsommer")
        kpis("Photovoltaïque")
        m, dn, rs = st.tabs(["Les maths", "Les données de l'entreprise", "Les ressources"])
        sp_pv = specs[2]
        with m:
            st.latex(r"Prod_{PV}=P_{kWc}\times Prod_{sp\acute e}\qquad \tau_{auto}=\frac{\sum_j \min(Prod_j,\,E_j^{r\acute es}\,part_{HP,j})}{\sum_j Prod_j}")
            st.latex(r"LCOE=\frac{I+\sum_t (OPEX_t+Remp_t)/(1+a)^t}{\sum_t E_t/(1+a)^t}")
            r_pv = res["Photovoltaïque"]
            lcoe_pv = r_pv["LCOE / coût de l'énergie (TND/kWh)"]
            st.markdown(f"LCOE = **{lcoe_pv:.3f} TND/kWh** contre un tarif réseau moyen de **{tarif:.3f} TND/kWh**.")
            tailles = [50, 100, 150, 250, 400, 600]
            lignes = []
            fr = 1 - (sav[["Veille jours ouvrés", "Week-ends en mode arrêt", "Anomalies HVAC", "Fuites résiduelles", "Autoclave", "Usinage CNC"]].sum()) / R["conso_an"]
            for k in tailles:
                tx, au, su, _ = autoconso_pv(d, P, k, fr)
                su = min(su, P["pv_surplus_max"] * (au + su))
                capex = k * P["pv_capex_kwc"]
                s_ = spec("Photovoltaïque", au, P, su, capex=capex,
                          opex=P["pv_opex_kwc"] * k,
                          remp={int(P["pv_remp_an"]): P["pv_remp_part"] * capex})
                r = evaluer(s_, Pv, tarif)
                lignes.append({"kWc": k, "Autoconsommation (%)": tx * 100, "VAN (TND)": r["VAN (TND)"], "TRI (%)": r["TRI (%)"]})
            sweep = pd.DataFrame(lignes)
            fig = px.line(sweep, x="kWc", y="VAN (TND)", markers=True, title="VAN selon la puissance installée (CAPEX/kWc constant)")
            st.plotly_chart(fig, **STRETCH)
            st.dataframe(sweep.round(1), hide_index=True, **STRETCH)
        with dn:
            fr_pv = R["taux_pv"][scenario]
            st.markdown(f"- Production : **{fmt(P['pv_kwc'] * P['pv_prod'])} kWh/an** = {P['pv_kwc'] * P['pv_prod'] / R['conso_an'] * 100:.1f} % de la consommation ; "
                        f"facteur de charge {P['pv_prod'] / 8760 * 100:.1f} %.")
            st.markdown(f"- Autoconsommation estimée : **{fr_pv * 100:.1f} %** (borne haute : données journalières, sans saisonnalité) ; "
                        f"surplus vendu : {fmt(sp_pv['surplus'])} kWh/an (plafond {P['pv_surplus_max'] * 100:.0f} %).")
            _, _, _, auto_j = autoconso_pv(d, P, P["pv_kwc"])
            dm = pd.DataFrame({"mois": d["mois"], "Production": P["pv_kwc"] * P["pv_prod"] / 365.0, "Autoconsommée": auto_j,
                               "Conso. diurne": d["kwh"] * d["hp"] / 100.0}).groupby("mois").sum()
            fig = go.Figure()
            fig.add_bar(x=dm.index, y=dm["Conso. diurne"], name="Conso. diurne (kWh)", marker_color="#b0c4de")
            fig.add_bar(x=dm.index, y=dm["Production"], name="Production PV (kWh)", marker_color="#ff9f1c")
            fig.update_layout(barmode="group", title="Production PV vs consommation diurne (heures pleines)", height=380)
            st.plotly_chart(fig, **STRETCH)
        with rs:
            ressources([("STEG — tarifs moyenne tension", "https://www.steg.com.tn/fr/page/tarifs-moyenne-tension"),
                        ("STEG — tarifs d'autoproduction (surplus)", "https://www.steg.com.tn/fr/page/tarifs-de-l%E2%80%99%C3%A9lectricit%C3%A9-produite-%C3%A0-partir-d%E2%80%99autres-types-d%E2%80%99autoproduction"),
                        ("Guide des projets ENR en Tunisie (2019)", "https://www.tunisieindustrie.gov.tn/upload/ENR/Guide_detaille_ENR_tunisie_mai2019.pdf"),
                        ("Guide ANME d'autoproduction PV", "https://www.anme.tn/sites/default/files/2025-02/Guide%20pour%20l%E2%80%99%C3%A9laboration%20des%20%C3%A9tudes%20de%20faisabilit%C3%A9%20et%20l%E2%80%99assistance%20technique%20des%20projets%20d%E2%80%99autoproduction%20d%E2%80%99%C3%A9lectricit%C3%A9%20par%20l%E2%80%99%C3%A9nergie%20solaire%20photovolta%C3%AFque.pdf"),
                        ("Marché photovoltaïque en Tunisie (GIZ, 2013)", "https://energypedia.info/images/6/64/March%C3%A9PhotovoltaiqueTunisie_GIZ_082013.pdf"),
                        ("Formation ANME/GIZ : aspects économiques des projets PV (module 3)", "https://energypedia.info/images/8/8e/Module_3_Formation_BE.pdf")])

    # ---------------- Comparaison
    with t_cmp:
        st.subheader("Comparaison des trois piliers et choix")
        table = pd.DataFrame({n: {k: v for k, v in res[n].items() if k not in ("flux", "Projet")} for n in PROJETS}).T
        st.dataframe(table.style.format("{:,.2f}"), **STRETCH)
        a, b, c = st.columns(3)
        figv = px.bar(table.reset_index(), x="index", y="VAN (TND)", color="index", color_discrete_map=COULEURS, text_auto=",.0f",
                      labels={"index": ""}, title="VAN (durée de vie propre à chaque projet)")
        figv.update_layout(showlegend=False, height=340)
        a.plotly_chart(figv, **STRETCH)
        figi = px.bar(table.reset_index(), x="index", y="VAN/I", color="index", color_discrete_map=COULEURS, text_auto=".2f",
                      labels={"index": ""}, title="VAN/I (taux d'enrichissement)")
        figi.update_layout(showlegend=False, height=340)
        b.plotly_chart(figi, **STRETCH)
        figt = px.bar(table.reset_index(), x="index", y="AE (TND/an)", color="index", color_discrete_map=COULEURS, text_auto=",.0f",
                      labels={"index": ""}, title="Annuité équivalente (TND/an)")
        figt.update_layout(showlegend=False, height=340)
        c.plotly_chart(figt, **STRETCH)
        st.caption("La VAN absolue dépend du capital engagé : le PV immobilise un CAPEX élevé, l'efficacité un CAPEX faible. "
                   "Les projets ne sont pas exclusifs ; compare VAN/I, TRI et AE.")

        fig = go.Figure()
        for nom in PROJETS:
            fl = res[nom]["flux"]
            fig.add_scatter(x=list(range(len(fl))), y=np.cumsum([f / (1 + P["taux"]) ** t for t, f in enumerate(fl)]),
                            mode="lines+markers", name=nom, line=dict(color=COULEURS[nom]))
        fig.add_hline(y=0, line_color="black")
        fig.update_layout(title="Flux de trésorerie cumulés actualisés (TND)", xaxis_title="Année", height=380)
        st.plotly_chart(fig, **STRETCH)

        st.markdown("#### Les trois scénarios côte à côte")
        lignes_sc = []
        for sc in SCENARIOS + [PERSO]:
            for nom in PROJETS:
                r = R["res"][sc][nom]
                lignes_sc.append({"Scénario": sc, "Projet": nom, "Économie (kWh/an)": r["Économie (kWh/an, an 1)"], "VAN (TND)": r["VAN (TND)"],
                                  "TRI (%)": r["TRI (%)"], "FE (kgCO₂/kWh)": R["SC"][sc]["fe"], "CO₂ évité an 1 (t)": r["CO2 évité an 1 (t)"]})
        st.dataframe(pd.DataFrame(lignes_sc).set_index(["Projet", "Scénario"]).sort_index().style.format(
            {"Économie (kWh/an)": "{:,.0f}", "VAN (TND)": "{:,.0f}", "TRI (%)": "{:.1f}", "FE (kgCO₂/kWh)": "{:g}", "CO₂ évité an 1 (t)": "{:,.1f}"}), **STRETCH)

        st.markdown("#### Choix multicritère (programmation par objectif)")
        st.caption("Renseignez indépendamment chaque poids entre 0 et 1, puis cliquez sur **Simuler**. La somme des poids doit être inférieure ou égale à 1.")
        w1, w2, w3 = st.columns(3)
        with w1:
            w_e = st.number_input("Poids énergie", min_value=0.0, max_value=1.0, value=1 / 3, step=0.05, format="%.2f", key="mc_w_e")
        with w2:
            w_a = st.number_input("Poids économie (AE)", min_value=0.0, max_value=1.0, value=1 / 3, step=0.05, format="%.2f", key="mc_w_a")
        with w3:
            w_c = st.number_input("Poids environnement (CO₂)", min_value=0.0, max_value=1.0, value=1 / 3, step=0.05, format="%.2f", key="mc_w_c")
        somme_poids = w_e + w_a + w_c
        st.caption(f"Somme des poids : **{somme_poids:.2f}**")
        simuler = st.button("Simuler", type="primary", key="mc_simuler")
        if simuler:
            if somme_poids > 1.0 + 1e-9:
                st.error(f"Erreur : la somme des poids ({somme_poids:.2f}) est supérieure à 1. Veuillez ajuster les valeurs.")
            else:
                crit = table[["Économie (kWh/an, an 1)", "AE (TND/an)", "CO2 évité moy. (t/an)"]].copy()
                crit.columns = ["Énergie", "Économie", "Environnement"]
                ecart = ((crit.max() - crit) / (crit.max() - crit.min()).replace(0, np.nan)).fillna(0.0)
                score = ecart["Énergie"] * w_e + ecart["Économie"] * w_a + ecart["Environnement"] * w_c
                st.latex(
                    r"S_i=\sum_k w_k\,\dfrac{x_k^{\max}-x_{i,k}}{x_k^{\max}-x_k^{\min}}"
                    r"\quad\text{(à minimiser)}"
                )
                if abs(somme_poids - 1.0) > 1e-6:
                    st.info(f"Somme des poids = {somme_poids:.2f} (≤ 1). Les poids sont utilisés tels quels (sans renormalisation).")
                st.bar_chart(score.rename("Score (plus bas = meilleur)"))
                st.success(f"Projet retenu avec ces poids (scénario {scenario}) : **{score.idxmin()}**")
                dom = [i for i in PROJETS if any(j != i and (crit.loc[j] >= crit.loc[i]).all() and (crit.loc[j] > crit.loc[i]).any() for j in PROJETS)]
                st.caption("Projets dominés au sens de Pareto : " + (", ".join(dom) if dom else "aucun (les trois sont Pareto-optimaux)"))
        else:
            st.info("Ajustez les poids puis cliquez sur **Simuler** pour calculer le score multicritère.")

        st.markdown("#### Sensibilité de la VAN")
        proj = st.selectbox("Projet", PROJETS, index=2)
        sp_sel = specs[PROJETS.index(proj)]
        taux_grid = [0.07, 0.08, 0.10, 0.12, 0.14]
        g_grid = [0.0, 0.03, 0.05, 0.06]
        mat = pd.DataFrame({f"g = {g:.0%}": [evaluer(sp_sel, Pv, tarif, a=a, g=g)["VAN (TND)"] for a in taux_grid] for g in g_grid},
                           index=[f"a = {a:.0%}" for a in taux_grid])
        st.plotly_chart(px.imshow(mat, text_auto=",.0f", color_continuous_scale="RdYlGn", aspect="auto",
                                  title=f"VAN de {proj} (scénario {scenario}) : taux d'actualisation × hausse du prix de l'électricité"), **STRETCH)
        ressources([("Formation ANME/GIZ : taux d'actualisation, croissance des tarifs (module 3)", "https://energypedia.info/images/8/8e/Module_3_Formation_BE.pdf"),
                    ("Prix indicatif du carbone proposé en Tunisie (La Presse/allAfrica, 2026)", "https://fr.allafrica.com/stories/202603240305.html"),
                    ("IRENA — Renewable Cost Database", ""), ("BCT — taux directeur et inflation", "https://fr.allafrica.com/stories/202610080466.html"),
                    ("Support de cours : Choix d'investissements pour les systèmes énergétiques", "")])


# ----------------------------------------------------------------------------
# NAVIGATION
# ----------------------------------------------------------------------------
PAGES = ["Overview", "Distribution de la consommation", "Données de l'entreprise", "Paramètres", "Benchmarking", "Dashboard"]
ICONES = {"Overview": "🏠", "Distribution de la consommation": "📖", "Données de l'entreprise": "🗂️", "Paramètres": "⚙️", "Benchmarking": "🏭", "Dashboard": "📊"}

if "page" not in st.session_state:
    st.session_state["page"] = "Overview"
if "P" not in st.session_state:
    st.session_state["P"] = p_defaut()
else:   # compatibilité avec une session ouverte avant l'ajout des scénarios
    st.session_state["P"] = {**p_defaut(), **st.session_state["P"]}

st.sidebar.title("⚡ AEROTUN")
for p in PAGES:
    if st.sidebar.button(f"{ICONES[p]}  {p}", **STRETCH,
                         type="primary" if st.session_state["page"] == p else "secondary", key=f"btn_{p}"):
        st.session_state["page"] = p
        st.rerun()
st.sidebar.divider()

donnees, source = charger_donnees()
d = preparer(donnees) if donnees is not None else None
R = analyser(d, json.dumps(st.session_state["P"], sort_keys=True)) if d is not None else None
if source:
    st.sidebar.caption(f"Données : {source}\n{len(d)} jours")

page = st.session_state["page"]
if d is None and page in ("Overview", "Distribution de la consommation", "Dashboard", "Benchmarking"):
    if page == "Benchmarking":
        page_benchmarking(None, None)
    elif page == "Overview":
        page_overview(None, None)
    else:
        st.error("Le fichier Excel est introuvable : vérifie le chemin DATA_PATH en haut de app.py ou charge le fichier dans la barre latérale.")
else:
    {"Overview": page_overview, "Distribution de la consommation": page_distribution_consommation, "Données de l'entreprise": page_donnees,
     "Paramètres": page_parametres, "Benchmarking": page_benchmarking, "Dashboard": page_dashboard}[page](d, R)