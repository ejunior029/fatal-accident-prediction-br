"""Testes para a API de predicao (src/api.py).

Usa o cliente da fixture "cliente_api" (tests/conftest.py), que roda por
cima de um modelo de teste minusculo — nao importamos "src.api" direto
aqui em cima do arquivo de proposito, senao ele carregaria antes das
variaveis de ambiente serem definidas.
"""

PAYLOAD_VALIDO = {
    "dia_semana": "segunda-feira", "uf": "SP", "br": 101,
    "causa_acidente": "Velocidade Incompatível", "tipo_acidente": "Colisão frontal",
    "fase_dia": "Plena Noite", "sentido_via": "Crescente", "condicao_metereologica": "Céu Claro",
    "tipo_pista": "Simples", "tracado_via": "Reta", "uso_solo": "Rural",
    "pessoas": 4, "veiculos": 2, "latitude": -15.7801, "longitude": -47.9292, "hora": 3,
}


def test_health_responde_ok(cliente_api):
    resposta = cliente_api.get("/health")

    assert resposta.status_code == 200
    assert resposta.json() == {"status": "ok"}


def test_raiz_responde_com_endpoints_disponiveis(cliente_api):
    resposta = cliente_api.get("/")

    assert resposta.status_code == 200
    assert "/predict" in resposta.json()["endpoints"]


def test_predict_com_payload_valido_devolve_previsao_coerente(cliente_api):
    resposta = cliente_api.post("/predict", json=PAYLOAD_VALIDO)

    assert resposta.status_code == 200
    corpo = resposta.json()
    assert corpo["previsao"] in (0, 1)
    assert 0.0 <= corpo["probabilidade"] <= 1.0
    # limiar fixado em 0.5 no modelo de teste (tests/conftest.py)
    assert corpo["limiar_usado"] == 0.5


def test_predict_com_campo_obrigatorio_faltando_devolve_422(cliente_api):
    payload_incompleto = {k: v for k, v in PAYLOAD_VALIDO.items() if k != "pessoas"}

    resposta = cliente_api.post("/predict", json=payload_incompleto)

    assert resposta.status_code == 422


def test_predict_com_tipo_de_dado_errado_devolve_422(cliente_api):
    payload_invalido = {**PAYLOAD_VALIDO, "pessoas": "muitas"}

    resposta = cliente_api.post("/predict", json=payload_invalido)

    assert resposta.status_code == 422


def test_predict_recusa_hora_fora_do_intervalo_valido(cliente_api):
    payload_invalido = {**PAYLOAD_VALIDO, "hora": 25}  # valido eh 0-23

    resposta = cliente_api.post("/predict", json=payload_invalido)

    assert resposta.status_code == 422
