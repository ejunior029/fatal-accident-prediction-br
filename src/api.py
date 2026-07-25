"""API minima de predicao: risco de vitima fatal em acidentes de transito (PRF)."""
import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

RAIZ = Path(__file__).resolve().parent.parent
MODELO_PATH = RAIZ / "models" / "modelo_final_optuna.joblib"
LIMIAR_PATH = RAIZ / "models" / "limiar_otimo.json"

app = FastAPI(
    title="Fatal Accident Prediction API",
    description="Prevê se um acidente em rodovia federal (PRF) tem risco de vítima fatal.",
    version="1.0.0",
)

_pipeline = joblib.load(MODELO_PATH)
_limiar = 0.5
if LIMIAR_PATH.exists():
    _limiar = json.loads(LIMIAR_PATH.read_text(encoding="utf-8"))["limiar"]


class AcidenteInput(BaseModel):
    dia_semana: str = Field(..., examples=["segunda-feira"])
    uf: str = Field(..., examples=["SP"])
    br: int = Field(..., examples=[101])
    causa_acidente: str = Field(..., examples=["Velocidade Incompatível"])
    tipo_acidente: str = Field(..., examples=["Colisão traseira"])
    fase_dia: str = Field(..., examples=["Plena Noite"])
    sentido_via: str = Field(..., examples=["Crescente"])
    condicao_metereologica: str = Field(..., examples=["Céu Claro"])
    tipo_pista: str = Field(..., examples=["Simples"])
    tracado_via: str = Field(..., examples=["Reta"])
    uso_solo: str = Field(..., examples=["Rural"])
    pessoas: int = Field(..., ge=1, examples=[2])
    veiculos: int = Field(..., ge=1, examples=[2])
    latitude: float = Field(..., examples=[-15.7801])
    longitude: float = Field(..., examples=[-47.9292])
    hora: int = Field(..., ge=0, le=23, examples=[14])


class Predicao(BaseModel):
    previsao: int
    probabilidade: float
    limiar_usado: float


@app.get("/")
def raiz():
    return {
        "nome": "Fatal Accident Prediction API",
        "docs": "/docs",
        "endpoints": ["/health", "/predict"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=Predicao)
def prever(entrada: AcidenteInput):
    linha = pd.DataFrame([entrada.model_dump()])
    probabilidade = float(_pipeline.predict_proba(linha)[:, 1][0])
    previsao = int(probabilidade >= _limiar)
    return Predicao(previsao=previsao, probabilidade=round(probabilidade, 4), limiar_usado=_limiar)
