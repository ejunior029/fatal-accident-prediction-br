"""API minima de predicao: risco de vitima fatal em acidentes de transito (PRF)."""
import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Path(__file__) = caminho deste arquivo (src/api.py).
# .parent sobe uma pasta (para src/), .parent de novo sobe outra (para a raiz do projeto).
# Assim os caminhos abaixo funcionam independente de onde o comando "uvicorn" for executado.
RAIZ = Path(__file__).resolve().parent.parent
MODELO_PATH = RAIZ / "models" / "modelo_final_optuna.joblib"
LIMIAR_PATH = RAIZ / "models" / "limiar_otimo.json"

# FastAPI() cria a aplicacao web em si. title/description/version aparecem
# automaticamente na pagina de documentacao (/docs) gerada pelo FastAPI.
app = FastAPI(
    title="Fatal Accident Prediction API",
    description="Prevê se um acidente em rodovia federal (PRF) tem risco de vítima fatal.",
    version="1.0.0",
)

# Carrega o modelo (pipeline: pre-processador + XGBoost) UMA UNICA VEZ, quando
# o servidor sobe — nao a cada pedido. Por isso essas linhas ficam fora de
# qualquer funcao/endpoint: rodam so na inicializacao.
_pipeline = joblib.load(MODELO_PATH)

# O limiar de decisao (ex.: 0.5854, achado no notebook 04) fica salvo em um
# JSON separado do modelo. Se por algum motivo esse arquivo nao existir ainda
# (ex.: notebook 04 nunca rodou ate o fim), a API cai para o padrao 0.5 em vez
# de quebrar.
_limiar = 0.5
if LIMIAR_PATH.exists():
    _limiar = json.loads(LIMIAR_PATH.read_text(encoding="utf-8"))["limiar"]


# BaseModel (do Pydantic) descreve o "formato" que um pedido de previsao
# precisa ter. O FastAPI usa essa classe para validar automaticamente cada
# requisicao antes de chegar na funcao "prever" la embaixo: se faltar um
# campo, ou vier um tipo errado (ex.: texto no lugar de numero), o pedido eh
# rejeitado com uma mensagem de erro clara, sem precisar escrever if/else
# manualmente para cada campo.
#
# Os nomes dos campos abaixo sao EXATAMENTE os mesmos nomes de coluna que o
# pipeline espera (ver src/data.py: COLUNAS_CATEGORICAS + COLUNAS_NUMERICAS) —
# isso garante que o DataFrame montado em "prever()" seja compativel com o
# ColumnTransformer treinado.
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
    pessoas: int = Field(..., ge=1, examples=[2])  # ge=1 -> Pydantic rejeita valores < 1
    veiculos: int = Field(..., ge=1, examples=[2])
    latitude: float = Field(..., examples=[-15.7801])
    longitude: float = Field(..., examples=[-47.9292])
    hora: int = Field(..., ge=0, le=23, examples=[14])  # entre 0 e 23


# Outra classe Pydantic, mas para descrever a SAIDA da API (o que o usuario
# recebe de volta). Declarar isso em response_model (no @app.post abaixo) faz
# o FastAPI validar e documentar tambem o formato da resposta.
class Predicao(BaseModel):
    previsao: int          # 0 = sem vitima fatal previsto | 1 = com vitima fatal previsto
    probabilidade: float   # probabilidade bruta do modelo (0.0 a 1.0), antes de aplicar o limiar
    limiar_usado: float    # qual limiar foi usado para decidir 0 ou 1


# Cada "@app.<metodo>(<caminho>)" registra um endpoint: uma combinacao de URL
# + verbo HTTP que o servidor vai responder. A funcao logo abaixo do decorator
# eh o que roda quando alguem acessa aquele endpoint.

@app.get("/")
def raiz():
    # Endpoint de "boas-vindas": so para confirmar que a API existe e apontar
    # para onde estao a documentacao e os outros endpoints.
    return {
        "nome": "Fatal Accident Prediction API",
        "docs": "/docs",
        "endpoints": ["/health", "/predict"],
    }


@app.get("/health")
def health():
    # Endpoint simples de "esta vivo?" — muito comum em APIs reais, usado por
    # ferramentas de monitoramento para checar se o servico esta no ar.
    return {"status": "ok"}


@app.post("/predict", response_model=Predicao)
def prever(entrada: AcidenteInput):
    # Quando o FastAPI chama esta funcao, "entrada" ja chegou validada e
    # convertida para um objeto AcidenteInput (o FastAPI cuida disso sozinho,
    # usando a classe definida acima).

    # entrada.model_dump() transforma o objeto Pydantic em um dicionario comum
    # (ex.: {"dia_semana": "segunda-feira", "uf": "SP", ...}).
    # pd.DataFrame([...]) cria uma tabela de 1 linha só, no mesmo formato que
    # o pipeline espera (o pipeline foi treinado recebendo um DataFrame).
    linha = pd.DataFrame([entrada.model_dump()])

    # O pipeline inteiro (pre-processamento + modelo) roda aqui dentro:
    # primeiro os dados passam pelo ColumnTransformer (one-hot + padronizacao),
    # depois pelo XGBoost. predict_proba devolve duas colunas [prob classe 0,
    # prob classe 1]; pegamos so a da classe 1 (fatal) com [:, 1], e [0] pega
    # o unico valor (ja que so tem 1 linha).
    probabilidade = float(_pipeline.predict_proba(linha)[:, 1][0])

    # Aqui esta o ajuste feito no notebook 04: em vez de usar o limiar padrao
    # de 0.5, comparamos com o limiar otimizado (_limiar). Se a probabilidade
    # for maior ou igual a ele, previsao = 1 (risco de fatalidade).
    previsao = int(probabilidade >= _limiar)

    # Devolve um objeto Predicao (validado contra a classe definida acima).
    # O FastAPI converte isso automaticamente para JSON na resposta HTTP.
    return Predicao(previsao=previsao, probabilidade=round(probabilidade, 4), limiar_usado=_limiar)
