# Imagem-base: Python 3.11 já instalado, versão "slim" (Linux mínimo, sem
# extras que a gente não precisa). É o "sistema operacional" de dentro do
# container.
FROM python:3.11-slim

# A partir daqui, todo comando roda dentro dessa pasta (ela é criada se não
# existir). É o equivalente ao "cd" dentro do container.
WORKDIR /app

# Copiamos SÓ o arquivo de dependências primeiro (não o código ainda).
# Motivo: o Docker guarda cada instrução em camadas (cache). Se o código
# mudar mas as dependências não, o Docker reaproveita essa camada de
# instalação em vez de reinstalar tudo de novo — builds seguintes ficam bem
# mais rápidos.
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Só agora copiamos o código e o modelo treinado — o que muda com mais
# frequência fica nas últimas camadas.
COPY src/ src/
COPY models/ models/

# Isso é documentação/sinalização: avisa que o container vai escutar na
# porta 8000. Não abre a porta sozinho (isso é feito no "docker run").
EXPOSE 8000

# Comando que roda quando o container liga. Note "--host 0.0.0.0" em vez de
# "127.0.0.1" (localhost): dentro do container, 0.0.0.0 significa "aceite
# conexões vindas de fora do container", não só de dentro dele.
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
