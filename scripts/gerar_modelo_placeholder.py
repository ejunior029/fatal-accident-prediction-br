"""Gera um modelo "de mentira" em models/, so para o CI conseguir testar o
`docker build` sem precisar do modelo real (que nao e versionado no git).

NAO USE ISSO PARA SERVIR PREVISOES DE VERDADE — e treinado em 10 linhas de
exemplo (tests/fixtures/amostra_datatran.csv), so serve para provar que o
Dockerfile e o requirements-api.txt continuam funcionando juntos. O modelo
real vem de rodar os notebooks 02 a 04.
"""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import joblib  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from src.data import carregar_dados, construir_preprocessador, separar_x_y  # noqa: E402

CAMINHO_FIXTURE = RAIZ / "tests" / "fixtures" / "amostra_datatran.csv"
PASTA_MODELOS = RAIZ / "models"

df = carregar_dados(CAMINHO_FIXTURE)
X, y = separar_x_y(df)

pipeline = Pipeline([
    ("preprocessador", construir_preprocessador()),
    ("modelo", LogisticRegression(max_iter=1000)),
])
pipeline.fit(X, y)

PASTA_MODELOS.mkdir(exist_ok=True)
joblib.dump(pipeline, PASTA_MODELOS / "modelo_final_optuna.joblib")
(PASTA_MODELOS / "limiar_otimo.json").write_text('{"limiar": 0.5}', encoding="utf-8")

print("Modelo placeholder gerado em models/ (uso exclusivo do CI, nao e o modelo real).")
