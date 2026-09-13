"""Fixtures compartilhadas pelos testes.

Nada aqui depende do dataset real (data/datatran2024.csv) nem do modelo
treinado de verdade (models/modelo_final_optuna.joblib) — os dois ficam
fora do git, entao os testes nao teriam como rodar em uma maquina nova
(ou no CI) se dependessem deles. Em vez disso, usamos uma amostra pequena
(tests/fixtures/amostra_datatran.csv) e um modelo minusculo treinado na
hora, so para exercitar o codigo.
"""
import os
from pathlib import Path

import joblib
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

CAMINHO_FIXTURE_CSV = Path(__file__).resolve().parent / "fixtures" / "amostra_datatran.csv"


@pytest.fixture
def df_amostra():
    """DataFrame pequeno (10 linhas), ja com a coluna alvo, para testar src/data.py."""
    from src.data import carregar_dados

    return carregar_dados(CAMINHO_FIXTURE_CSV)


@pytest.fixture(scope="session")
def cliente_api(tmp_path_factory):
    """Cliente de teste da API, usando um modelo de mentira treinado na amostra.

    Define as variaveis de ambiente MODELO_PATH/LIMIAR_PATH ANTES de importar
    src.api, para a API carregar esse modelo de teste em vez de procurar o
    modelo real (que pode nem existir na maquina rodando os testes).
    tmp_path_factory eh uma fixture pronta do pytest: cria uma pasta temporaria
    e cuida de limpar depois, sem depender de nenhum caminho fixo do sistema
    operacional (funciona igual no Windows, Linux ou no CI).
    """
    from src.data import carregar_dados, construir_preprocessador, separar_x_y

    df = carregar_dados(CAMINHO_FIXTURE_CSV)
    X, y = separar_x_y(df)

    pipeline_teste = Pipeline([
        ("preprocessador", construir_preprocessador()),
        ("modelo", LogisticRegression(max_iter=1000)),
    ])
    pipeline_teste.fit(X, y)

    diretorio_tmp = tmp_path_factory.mktemp("modelo_teste")
    caminho_modelo = diretorio_tmp / "modelo_teste.joblib"
    caminho_limiar = diretorio_tmp / "limiar_teste.json"

    joblib.dump(pipeline_teste, caminho_modelo)
    caminho_limiar.write_text('{"limiar": 0.5}', encoding="utf-8")

    os.environ["MODELO_PATH"] = str(caminho_modelo)
    os.environ["LIMIAR_PATH"] = str(caminho_limiar)

    from fastapi.testclient import TestClient
    from src.api import app  # importado so agora, depois das variaveis de ambiente

    return TestClient(app)
