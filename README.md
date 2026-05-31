# wine-mlops — Solução de Referência

Gabarito **executável** da Atividade Avaliativa do módulo *MLOps — Modelos de IA em produção* (Pós-graduação em IA, iCEV, 2026.1).

Implementa o fluxo MLOps completo sobre o dataset Wine (sklearn): coleta → treino → validação → teste → registro → A/B automático → deploy condicional.

---

## Pré-requisitos

- Python 3.11 instalado (ou deixe o `uv` instalar).
- `[uv](https://docs.astral.sh/uv/)` instalado:
  ```powershell
  # Windows
  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
  ```bash
  # macOS / Linux
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- Docker Desktop (opcional, para os passos com contêineres).
- `[just](https://github.com/casey/just)` (opcional — facilita os atalhos do `justfile`).

---

## Setup em 30 segundos

> **Windows / PowerShell:** defina `PYTHONIOENCODING=utf-8` antes de executar qualquer comando — o MLflow 3.x imprime emojis no log de fim de run que crasham no `cp1252` padrão.
>
> ```powershell
> $env:PYTHONIOENCODING = "utf-8"
> ```

```bash
# 1. Instalar dependências
uv sync

# 2. Subir o MLflow server local (deixe rodando em um terminal)
uv run mlflow server --host 127.0.0.1 --port 5000 \
    --backend-store-uri sqlite:///mlflow.db

# 3. Em outro terminal, criar o champion baseline (uma única vez)
uv run python -m scripts.bootstrap_champion
```

A partir daqui o ambiente está pronto e o modelo `wine-classifier@champion` está registrado.

---

## Fluxo guiado, tarefa por tarefa

### Tarefa 1 — Pipeline de dados

```bash
uv run python -m src.data
# saída esperada:
# train=106 val=36 test=36
```

### Tarefa 2 — Treinar challenger

```bash
uv run python -m src.train --n_estimators 200 --max_depth 8
```

Abra o MLflow UI em [http://127.0.0.1:5000](http://127.0.0.1:5000):

- novo run em **Experiments → wine-classifier** com 4 métricas e 2 parâmetros;
- nova versão em **Models → wine-classifier**.

Anote o `run_id` impresso no console.

### Tarefa 3 — Avaliar no conjunto de teste

```bash
uv run python -m src.evaluate --run-id dbd443793b8142cd96b87643745123ec
```

Verifique:

- métricas `test_*` agregadas ao mesmo run no MLflow;
- arquivo `models/challenger.pkl` criado.

### Tarefa 4 — Teste A/B automático

```bash
uv run python -m src.promote
```

Saída (exemplo, com challenger vencendo o baseline fraco):

```json
{
  "champion_version": 1,
  "challenger_version": 2,
  "f1_champion": 0.6437,
  "f1_challenger": 0.9722,
  "threshold": 0.01,
  "promoted": true
}
```

Confirme no MLflow UI que o alias `@champion` agora aponta para a Version 2 e o run `promotion-decision` foi criado com a tag `promoted=true`.

### Tarefa 5 — Subir a API

```bash
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Em outro terminal:

```bash
curl -s http://localhost:8000/health
# {"status":"ok","champion_version":2}

curl -s -X POST http://localhost:8000/predict \
  -H "content-type: application/json" \
  -d '{
        "alcohol":13.2,"malic_acid":1.78,"ash":2.14,"alcalinity_of_ash":11.2,
        "magnesium":100,"total_phenols":2.65,"flavanoids":2.76,
        "nonflavanoid_phenols":0.26,"proanthocyanins":1.28,"color_intensity":4.38,
        "hue":1.05,"od280/od315_of_diluted_wines":3.40,"proline":1050
      }'
# {"class":0,"proba":0.94,"all_probs":[0.94,0.05,0.01],"champion_version":2}
```

Documentação automática em [http://localhost:8000/docs](http://localhost:8000/docs).

### Tarefa 6 — Deploy automático

O workflow `.github/workflows/deploy.yml` executa:

1. lint + testes;
2. sobe MLflow no runner;
3. bootstrap champion;
4. treina challenger;
5. avalia no teste;
6. executa promoção A/B e exporta `promoted` como `output`;
7. **somente se `promoted=True`**, faz build e push da imagem para o GHCR e executa o passo de deploy (rolling update simulado).

Para acionar localmente sem GitHub, simule o output com:

```bash
uv run python -m src.promote > promote.json
cat promote.json
```

---

## Pipeline completo em um comando

```bash
# Com just
just pipeline

# Sem just
uv run python -m src.train --n_estimators 200 --max_depth 8
RUN_ID=$(uv run python -c "import mlflow; mlflow.set_tracking_uri('http://127.0.0.1:5000'); \
    print(mlflow.search_runs(experiment_names=['wine-classifier'], \
          order_by=['attributes.start_time DESC']).iloc[0]['run_id'])")
uv run python -m src.evaluate --run-id $RUN_ID
uv run python -m src.promote
```

---

## Execução conteinerizada

```bash
# 1. Subir MLflow + API em containers
docker compose up -d

# 2. Bootstrap champion (do host, apontando para o MLflow do compose)
MLFLOW_TRACKING_URI=http://127.0.0.1:5000 uv run python -m scripts.bootstrap_champion

# 3. Health-check
curl http://localhost:8000/health
```

---

## Testes

```bash
# Apenas testes de dados (não exigem MLflow)
uv run pytest tests/test_data.py

# Suíte completa (a API exige MLflow + champion registrados)
uv run pytest
```

Para pular o teste de API quando o MLflow não está disponível:

```bash
MLFLOW_SKIP=1 uv run pytest
```

---

## Estrutura

```
wine-mlops-solution/
├── pyproject.toml           # dependências + config (ruff, pytest)
├── .python-version          # 3.11
├── Dockerfile               # imagem multi-stage da API
├── docker-compose.yml       # MLflow + API
├── justfile                 # atalhos de execução
├── scripts/
│   └── bootstrap_champion.py
├── src/
│   ├── data.py              # Tarefa 1
│   ├── train.py             # Tarefa 2
│   ├── evaluate.py          # Tarefa 3
│   ├── promote.py           # Tarefa 4
│   └── api/main.py          # Tarefa 5
├── tests/
│   ├── test_data.py
│   └── test_api.py
└── .github/workflows/
    └── deploy.yml           # Tarefa 6
```

---

## Variáveis de ambiente


| Variável              | Default                 | O quê                                               |
| --------------------- | ----------------------- | --------------------------------------------------- |
| `MLFLOW_TRACKING_URI` | `http://127.0.0.1:5000` | Endereço do MLflow tracking server                  |
| `PROMOTION_THRESHOLD` | `0.01`                  | Ganho mínimo de F1 macro para promover o challenger |
| `MLFLOW_SKIP`         | (vazio)                 | `1` pula o teste de API no pytest                   |


---

## Resetar o ambiente (limpar MLflow local)

```bash
# Mata o MLflow server primeiro
rm -f mlflow.db mlflow.db-journal
rm -rf mlruns/ models/*.pkl

# Suba o MLflow de novo e refaça bootstrap_champion
```

