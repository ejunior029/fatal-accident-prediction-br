"""Testes para o carregamento e preparacao dos dados (src/data.py)."""
from src.data import (
    ALVO,
    COLUNAS_CATEGORICAS,
    COLUNAS_NUMERICAS,
    construir_preprocessador,
    separar_x_y,
)

# Fixas de proposito (nao importadas de src.data.COLUNAS_VAZAMENTO): se
# alguem remover uma dessas colunas da lista no codigo-fonte por engano, o
# teste ainda precisa acusar o vazamento. Importar a mesma constante que o
# codigo usa faria o teste "confiar cegamente" na fonte que deveria auditar.
COLUNAS_QUE_NUNCA_PODEM_VAZAR = [
    "classificacao_acidente",
    "mortos",
    "feridos_leves",
    "feridos_graves",
    "ilesos",
    "ignorados",
    "feridos",
]


def test_carregar_dados_cria_alvo_binario_correto(df_amostra):
    # A fixture (tests/fixtures/amostra_datatran.csv) tem 3 linhas
    # "Com Vítimas Fatais" entre as 10.
    assert df_amostra[ALVO].sum() == 3
    assert set(df_amostra[ALVO].unique()) <= {0, 1}


def test_carregar_dados_cria_coluna_hora_a_partir_do_horario(df_amostra):
    # A primeira linha da fixture tem horario "03:10:00".
    assert df_amostra.loc[0, "hora"] == 3
    assert df_amostra["hora"].between(0, 23).all()


def test_separar_x_y_remove_colunas_de_vazamento(df_amostra):
    """Nenhuma coluna que entrega a resposta pode sobrar em X."""
    X, _ = separar_x_y(df_amostra)

    for coluna in COLUNAS_QUE_NUNCA_PODEM_VAZAR:
        assert coluna not in X.columns, f"{coluna} vaza a resposta e nao deveria estar em X"

    assert ALVO not in X.columns


def test_separar_x_y_preserva_as_features_uteis(df_amostra):
    X, y = separar_x_y(df_amostra)

    for coluna in COLUNAS_CATEGORICAS + COLUNAS_NUMERICAS:
        assert coluna in X.columns

    assert y.tolist() == df_amostra[ALVO].tolist()


def test_construir_preprocessador_gera_uma_coluna_por_categoria_e_numerica(df_amostra):
    X, _ = separar_x_y(df_amostra)
    preprocessador = construir_preprocessador()

    X_transformado = preprocessador.fit_transform(X)

    n_colunas_categoricas = sum(X[coluna].nunique() for coluna in COLUNAS_CATEGORICAS)
    n_colunas_esperado = n_colunas_categoricas + len(COLUNAS_NUMERICAS)

    assert X_transformado.shape == (len(X), n_colunas_esperado)
