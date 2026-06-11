"""
Motor de Modelo de Ising para Análise de Sentimento de Mercado.

Implementa um modelo de Ising 2D adaptado para capturar a dinâmica de
sentimento coletivo no mercado de criptomoedas. Cada agente numa grade NxN
possui spin +1 (compra) ou -1 (venda), e a evolução do sistema via
Monte Carlo Metropolis-Hastings revela o grau de consenso ou desordem
no mercado.

Fundamento Matemático:
    Hamiltoniano: H = -J·Σ(s_i·s_j) - h·Σ(s_i)
    onde:
        J = constante de acoplamento (força de manada), calibrada pela
            autocorrelação dos retornos
        h = campo externo, calibrado pela tendência recente dos retornos
        T = temperatura, calibrada pela volatilidade realizada

    Magnetização: M = <s> = média de todos os spins
        M ≈ +1  → consenso altista forte
        M ≈ -1  → consenso baixista forte
        M ≈  0  → indecisão / transição de fase

    Temperatura Crítica (2D): Tc = 2J / ln(1 + √2) ≈ 2.269J
        T < Tc → fase ordenada (mercado com tendência definida)
        T > Tc → fase desordenada (mercado caótico)
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class IsingMarketEngine:
    """
    Motor de simulação do Modelo de Ising 2D para sentimento de mercado.

    Utiliza a dinâmica de Metropolis-Hastings para simular a interação
    entre agentes de mercado numa rede quadrada, onde cada agente decide
    comprar (+1) ou vender (-1) com base na influência dos vizinhos,
    no campo externo (tendência) e na temperatura (volatilidade).
    """

    def __init__(self) -> None:
        """Inicializa o motor com semente aleatória para reprodutibilidade."""
        np.random.seed(42)
        logger.info("IsingMarketEngine inicializado.")

    def _calibrate_coupling_J(self, returns: np.ndarray) -> float:
        """
        Calibra a constante de acoplamento J a partir da autocorrelação
        dos retornos.

        A autocorrelação de lag-1 dos retornos mede o quanto os participantes
        do mercado seguem uns aos outros (efeito manada). Quanto maior a
        autocorrelação, maior J.

        Parâmetros:
            returns: Array de retornos logarítmicos.

        Retorna:
            J calibrado, limitado ao intervalo [0.1, 2.0].
        """
        if len(returns) < 3:
            logger.warning("Retornos insuficientes para autocorrelação. Usando J padrão.")
            return 1.0

        mean_r = np.mean(returns)
        var_r = np.var(returns)

        if var_r < 1e-12:
            logger.warning("Variância dos retornos nula. Usando J padrão.")
            return 1.0

        # Autocorrelação de lag-1
        n = len(returns)
        autocorr = np.sum((returns[:-1] - mean_r) * (returns[1:] - mean_r)) / (n * var_r)
        autocorr = np.clip(autocorr, -1.0, 1.0)

        # Mapear autocorrelação para J: autocorr alta → J alto (manada forte)
        # J = 0.5 + 1.5 * |autocorr| garante J ∈ [0.5, 2.0]
        J = 0.5 + 1.5 * abs(autocorr)
        J = float(np.clip(J, 0.1, 2.0))

        logger.debug(f"Autocorrelação lag-1: {autocorr:.4f}, J calibrado: {J:.4f}")
        return J

    def _calibrate_field_h(self, returns: np.ndarray, window: int = 20) -> float:
        """
        Calibra o campo externo h a partir da tendência recente dos retornos.

        Tendência positiva → h > 0 (favorece spins +1, compra)
        Tendência negativa → h < 0 (favorece spins -1, venda)

        Parâmetros:
            returns: Array de retornos logarítmicos.
            window: Janela de retornos recentes para calcular tendência.

        Retorna:
            h calibrado, limitado ao intervalo [-1.0, 1.0].
        """
        recent = returns[-window:] if len(returns) >= window else returns
        trend = np.mean(recent)

        # Normalizar pela volatilidade para obter uma escala adimensional
        vol = np.std(returns)
        if vol < 1e-12:
            h = 0.0
        else:
            # Sinal de tendência normalizado, escalado para [-1, 1]
            h = float(np.clip(trend / vol, -1.0, 1.0))

        logger.debug(f"Tendência recente: {trend:.6f}, h calibrado: {h:.4f}")
        return h

    def _calibrate_temperature(self, returns: np.ndarray, J: float) -> float:
        """
        Calibra a temperatura T a partir da volatilidade realizada.

        Alta volatilidade → alta temperatura → desordem (spins aleatórios)
        Baixa volatilidade → baixa temperatura → ordem (consenso)

        A temperatura é escalada relativamente à temperatura crítica Tc
        do modelo de Ising 2D: Tc = 2J / ln(1 + √2).

        Parâmetros:
            returns: Array de retornos logarítmicos.
            J: Constante de acoplamento calibrada.

        Retorna:
            T calibrado, com mínimo de 0.01 para evitar divisão por zero.
        """
        Tc = 2.0 * J / np.log(1.0 + np.sqrt(2.0))

        vol = np.std(returns)
        if vol < 1e-12:
            logger.warning("Volatilidade nula. Usando T = 0.5 * Tc.")
            return 0.5 * Tc

        # Quantil da volatilidade: mapear vol para fração de Tc
        # Usar uma escala onde volatilidade média → T ≈ Tc
        # Percentil da volatilidade atual em relação à distribuição de vol rolling
        rolling_vols = np.array([
            np.std(returns[max(0, i - 20):i + 1])
            for i in range(len(returns))
        ])
        rolling_vols = rolling_vols[rolling_vols > 1e-12]

        if len(rolling_vols) < 2:
            T = Tc
        else:
            # Rank percentil da volatilidade atual
            current_vol = np.std(returns[-20:]) if len(returns) >= 20 else vol
            percentile = np.searchsorted(np.sort(rolling_vols), current_vol) / len(rolling_vols)
            # Mapear percentil [0, 1] para T ∈ [0.3·Tc, 2.5·Tc]
            T = Tc * (0.3 + 2.2 * percentile)

        T = float(max(T, 0.01))
        logger.debug(f"Volatilidade: {vol:.6f}, Tc: {Tc:.4f}, T calibrado: {T:.4f}")
        return T

    def _initialize_grid(self, grid_size: int, h: float) -> np.ndarray:
        """
        Inicializa a grade de spins com viés baseado no campo externo.

        Parâmetros:
            grid_size: Tamanho da grade (N para NxN).
            h: Campo externo calibrado.

        Retorna:
            Matriz NxN de spins (+1 ou -1).
        """
        # Probabilidade de spin +1 baseada no campo h
        prob_up = 0.5 + 0.3 * h  # h=+1 → 80% up, h=-1 → 20% up
        prob_up = np.clip(prob_up, 0.1, 0.9)

        grid = np.where(
            np.random.random((grid_size, grid_size)) < prob_up, 1, -1
        )
        return grid

    def _compute_energy(self, grid: np.ndarray, J: float, h: float) -> float:
        """
        Calcula a energia total do sistema (Hamiltoniano).

        H = -J·Σ(s_i·s_j) - h·Σ(s_i)

        A soma sobre vizinhos considera condições de contorno periódicas
        (grade toroidal) para eliminar efeitos de borda.

        Parâmetros:
            grid: Matriz NxN de spins.
            J: Constante de acoplamento.
            h: Campo externo.

        Retorna:
            Energia total do sistema.
        """
        # Interação com vizinhos (condições de contorno periódicas)
        neighbor_sum = (
            np.roll(grid, 1, axis=0) +  # vizinho acima
            np.roll(grid, -1, axis=0) +  # vizinho abaixo
            np.roll(grid, 1, axis=1) +  # vizinho à esquerda
            np.roll(grid, -1, axis=1)    # vizinho à direita
        )
        interaction_energy = -J * np.sum(grid * neighbor_sum) / 2.0  # /2 evita dupla contagem
        field_energy = -h * np.sum(grid)

        return float(interaction_energy + field_energy)

    def _metropolis_step(
        self, grid: np.ndarray, J: float, h: float, T: float
    ) -> np.ndarray:
        """
        Executa um passo completo de Monte Carlo via Metropolis-Hastings.

        Para cada spin na grade (varrido aleatoriamente), calcula-se a
        variação de energia ΔE caso o spin seja invertido. O spin é aceito
        com probabilidade:
            p = min(1, exp(-ΔE/T))

        Parâmetros:
            grid: Matriz NxN de spins (modificada in-place).
            J: Constante de acoplamento.
            h: Campo externo.
            T: Temperatura.

        Retorna:
            Grade atualizada após N² tentativas de flip.
        """
        N = grid.shape[0]
        beta = 1.0 / T  # Inverso da temperatura

        for _ in range(N * N):
            # Escolher spin aleatório
            i = np.random.randint(0, N)
            j = np.random.randint(0, N)

            spin = grid[i, j]

            # Soma dos vizinhos (condições de contorno periódicas)
            neighbors = (
                grid[(i + 1) % N, j] +
                grid[(i - 1) % N, j] +
                grid[i, (j + 1) % N] +
                grid[i, (j - 1) % N]
            )

            # Variação de energia para inverter o spin
            # ΔE = 2·J·s_i·Σ(s_j) + 2·h·s_i
            delta_E = 2.0 * J * spin * neighbors + 2.0 * h * spin

            # Critério de Metropolis
            if delta_E <= 0 or np.random.random() < np.exp(-beta * delta_E):
                grid[i, j] = -spin

        return grid

    def run_simulation(
        self,
        df: pd.DataFrame,
        grid_size: int = 20,
        mc_steps: int = 100
    ) -> Dict[str, Any]:
        """
        Executa a simulação do Modelo de Ising 2D para sentimento de mercado.

        O modelo mapeia a dinâmica de mercado para um sistema de spins:
        1. Calibra J (acoplamento) pela autocorrelação dos retornos
        2. Calibra h (campo externo) pela tendência recente
        3. Calibra T (temperatura) pela volatilidade realizada
        4. Inicializa grade NxN e executa MCMC via Metropolis-Hastings
        5. Calcula magnetização final como indicador de sentimento

        Parâmetros:
            df: DataFrame com coluna 'close' (preços de fechamento).
            grid_size: Tamanho da grade (padrão: 20 para grade 20x20).
            mc_steps: Número de passos de Monte Carlo (padrão: 100).

        Retorna:
            Dicionário com magnetização, sentimento, parâmetros calibrados,
            históricos e grade final.

        Levanta:
            ValueError: Se os dados forem insuficientes (< 10 registros).
        """
        # --- Validação de entrada ---
        if df is None or df.empty or len(df) < 10:
            raise ValueError(
                "Dados insuficientes para simulação de Ising. "
                "São necessários pelo menos 10 registros com coluna 'close'."
            )

        if 'close' not in df.columns:
            raise ValueError("DataFrame deve conter coluna 'close'.")

        logger.info(
            f"Iniciando simulação de Ising: grade {grid_size}x{grid_size}, "
            f"{mc_steps} passos MC, {len(df)} candles."
        )

        # --- Calcular retornos logarítmicos ---
        close = df['close'].dropna().values.astype(float)
        if len(close) < 10:
            raise ValueError("Menos de 10 preços válidos após remover NaN.")

        # Evitar log(0) com clip
        close = np.clip(close, 1e-10, None)
        returns = np.diff(np.log(close))

        if len(returns) < 2:
            raise ValueError("Retornos insuficientes para calibração.")

        # --- Calibração dos parâmetros ---
        J = self._calibrate_coupling_J(returns)
        h = self._calibrate_field_h(returns)
        T = self._calibrate_temperature(returns, J)

        # Temperatura crítica teórica do Ising 2D
        critical_temp = 2.0 * J / np.log(1.0 + np.sqrt(2.0))
        phase = "ORDERED" if T < critical_temp else "DISORDERED"

        logger.info(
            f"Parâmetros calibrados: J={J:.4f}, h={h:.4f}, T={T:.4f}, "
            f"Tc={critical_temp:.4f}, fase={phase}"
        )

        # --- Inicializar grade ---
        np.random.seed(42)  # Reprodutibilidade da simulação
        grid = self._initialize_grid(grid_size, h)

        # --- Executar MCMC ---
        magnetization_history: List[float] = []
        energy_history: List[float] = []

        for step in range(mc_steps):
            grid = self._metropolis_step(grid, J, h, T)

            mag = float(np.mean(grid))
            energy = self._compute_energy(grid, J, h)

            magnetization_history.append(mag)
            energy_history.append(energy)

            if (step + 1) % max(1, mc_steps // 5) == 0:
                logger.debug(
                    f"Passo MC {step + 1}/{mc_steps}: M={mag:.4f}, E={energy:.2f}"
                )

        # --- Resultado final ---
        # Magnetização final: média dos últimos 20% dos passos (equilíbrio)
        equilibrium_start = max(1, int(mc_steps * 0.8))
        final_magnetization = float(np.mean(magnetization_history[equilibrium_start:]))

        # Classificar sentimento
        if final_magnetization > 0.3:
            sentiment = "BULLISH"
        elif final_magnetization < -0.3:
            sentiment = "BEARISH"
        else:
            sentiment = "INDECISO"

        result = {
            "magnetization": final_magnetization,
            "sentiment": sentiment,
            "temperature": T,
            "coupling_J": J,
            "field_h": h,
            "magnetization_history": magnetization_history,
            "grid_final": grid.tolist(),
            "energy_history": energy_history,
            "critical_temp": critical_temp,
            "phase": phase,
        }

        logger.info(
            f"Simulação de Ising concluída. Sentimento: {sentiment}, "
            f"Magnetização: {final_magnetization:.4f}, Fase: {phase}"
        )

        return result
