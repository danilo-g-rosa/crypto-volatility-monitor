# ⚡ Crypto Volatility Monitor

Um painel analítico de alta performance para monitoramento de volatilidade de criptoativos e detecção de anomalias baseado em **Z-Score** e **Bandas de Bollinger**, construído com Python, **Streamlit** e **Plotly**.

A aplicação segue fielmente o princípio de **Separação de Conceitos (SoC)**, dividindo responsabilidades entre camadas de dados (Data Layer), processamento matemático puro (Engine) e interface visual (UI).

---

## 🎨 Design Estético

O design foi concebido com uma estética **futurista/cyberpunk**, utilizando:
* **Tema Dark:** Fundo escuro com gradientes radiais.
* **Cores de Destaque Neon:**
  * Ciano Neon (`#00ffcc`) para métricas estáveis, limites superiores de Bollinger e tendências positivas.
  * Rosa Neon (`#ff0055`) para sinalizar anomalias críticas, limites inferiores de Bollinger e alertas.
* **Efeitos Visuais:** Fontes customizadas (`Orbitron` para cabeçalhos e `Rajdhani` para dados/textos), efeito *glassmorphism* nos cards de KPIs e sombreamento brilhante (*glowing*).

---

## 🏗️ Estrutura de Diretórios (SoC)

A organização modular do projeto garante reusabilidade em outros projetos de BI:

```text
Estudos python autonomo/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── extractor.py      # Extrator resiliente da API REST da Binance (Data Layer)
│   ├── engine/
│   │   ├── __init__.py
│   │   └── analytics.py      # Cálculos estatísticos puros: Bollinger e Z-Score (Engine)
│   └── ui/
│       ├── __init__.py
│       └── components.py     # Elementos visuais puros e CSS customizado (UI Layer)
├── app.py                     # Orquestrador de fluxo e gerenciador de estado da sessão
├── requirements.txt           # Dependências do projeto
└── README.md                  # Documentação de arquitetura e execução (Este arquivo)
```

---

## ⚙️ Arquitetura dos Módulos

### 1. Data Layer (`src/data/extractor.py`)
Contém a interface abstrata `BaseExtractor` e a implementação concreta `BinanceExtractor`. É encarregado de:
* Realizar chamadas HTTP tipadas e seguras à API de *Klines* da Binance.
* Fazer a limpeza estrutural e sanitização do JSON, gerando um `pd.DataFrame` padronizado.
* Tratar exceções de rede como timeouts, erros de conexão (HTTPError) ou retornos vazios.

### 2. Analytics Engine (`src/engine/analytics.py`)
A classe `FinanceEngine` realiza o processamento matemático e estatístico de forma puramente desconectada da API da Binance.
* Recebe um DataFrame genérico contendo pelo menos as colunas `close` (fechamento) e `volume`.
* Calcula as **Bandas de Bollinger** (Média Móvel de 20 períodos e bandas com desvio de 2.0x por padrão).
* Calcula o **Z-Score** do fechamento atual sobre a média móvel, indicando o número de desvios padrão de afastamento.
* Define a flag de **anomalia** (`anomaly`) para desvios extremos absolutos superiores ao limiar configurável.

### 3. UI Layer (`src/ui/components.py`)
Módulo responsável por construir os elementos visuais da interface:
* `apply_custom_css()`: Injeta CSS puro para redesenhar a identidade visual do Streamlit com tons neon cyberpunk.
* `render_metrics_cards(df)`: Gera 4 cards responsivos baseados em HTML/CSS para monitorar Preço, Volume, Z-Score e Status de Anomalia.
* `render_candlestick_chart(df)`: Desenha o gráfico de velas integrado ao gráfico de desvio do Z-Score via Plotly com realces neon e marcação visual de anomalias detectadas.

### 4. Entry Point & Orchestrator (`app.py`)
Gerencia o fluxo de controle e estado do Streamlit (`st.session_state`):
* Implementa cache de estado inteligente: ajustes em parâmetros analíticos (como períodos de cálculo ou limiares de Z-score) disparam apenas o recálculo estatístico local no DataFrame em memória, sem refazer chamadas de rede lentas para a Binance.
* Desenha os painéis laterais de configuração interativos.

---

## 🚀 Como Executar o Projeto Localmente (Recomendado via venv)

### Pré-requisitos
* Python 3.9 ou superior instalado.
* Acesso à internet para consultar os dados históricos na Binance API.

### Passo 1: Configurar o Ambiente Virtual (venv)
Abra o terminal (PowerShell ou CMD) na raiz do projeto e crie o ambiente virtual:

```bash
python -m venv venv
```

### Passo 2: Ativar o Ambiente Virtual
Dependendo do terminal que estiver utilizando no Windows, execute o comando correspondente:

* **No PowerShell:**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
  *(Nota: Se receber um erro de permissão no PowerShell, execute antes: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process`)*

* **No Prompt de Comando (CMD):**
  ```cmd
  .\venv\Scripts\activate.bat
  ```

Após a ativação, você verá a indicação `(venv)` no início da linha de comando do terminal.

### Passo 3: Instalar as Dependências
Com o ambiente virtual ativo, instale as dependências listadas no `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Passo 4: Executar a Aplicação
Inicie o servidor do Streamlit:

```bash
streamlit run app.py
```


A aplicação abrirá automaticamente no seu navegador padrão (geralmente no endereço `http://localhost:8501`).
