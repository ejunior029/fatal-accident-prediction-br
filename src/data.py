"""Carregamento e preparacao dos dados de acidentes da PRF (dataset de classificacao)."""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CAMINHO_PADRAO = Path(__file__).resolve().parent.parent / "data" / "datatran2024.csv"

ALVO = "com_vitima_fatal"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Colunas que vazam a resposta: sao derivadas do mesmo desfecho do acidente
# que definimos como alvo (classificacao_acidente vira com_vitima_fatal).
COLUNAS_VAZAMENTO = [
    "classificacao_acidente",
    "mortos",
    "feridos_leves",
    "feridos_graves",
    "ilesos",
    "ignorados",
    "feridos",
]

# Identificadores e colunas sem valor preditivo para o baseline (alta
# cardinalidade ou redundantes com outras features).
COLUNAS_IDENTIFICADORAS = [
    "id",
    "data_inversa",
    "municipio",
    "regional",
    "delegacia",
    "uop",
    "km",
    "horario",
]

COLUNAS_CATEGORICAS = [
    "dia_semana",
    "uf",
    "br",
    "causa_acidente",
    "tipo_acidente",
    "fase_dia",
    "sentido_via",
    "condicao_metereologica",
    "tipo_pista",
    "tracado_via",
    "uso_solo",
]

COLUNAS_NUMERICAS = ["hora", "pessoas", "veiculos", "latitude", "longitude"]


def carregar_dados(caminho: Path = CAMINHO_PADRAO) -> pd.DataFrame:
    """Le o CSV da PRF e cria a coluna alvo binaria com_vitima_fatal."""
    df = pd.read_csv(caminho, sep=";", encoding="latin1")
    df[ALVO] = (df["classificacao_acidente"] == "Com Vítimas Fatais").astype(int)
    df["hora"] = pd.to_datetime(df["horario"], format="%H:%M:%S").dt.hour
    return df


def separar_x_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Remove colunas de vazamento/identificadoras e separa features do alvo."""
    colunas_remover = [c for c in COLUNAS_VAZAMENTO + COLUNAS_IDENTIFICADORAS if c in df.columns]
    X = df.drop(columns=colunas_remover + [ALVO])
    y = df[ALVO]
    return X, y


def construir_preprocessador() -> ColumnTransformer:
    """ColumnTransformer: one-hot para categoricas, padronizacao para numericas."""
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), COLUNAS_CATEGORICAS),
        ("num", StandardScaler(), COLUNAS_NUMERICAS),
    ])
