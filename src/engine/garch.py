import pandas as pd
import numpy as np
import logging
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

class GarchEngine:
    """
    Motor analítico para estimar e prever a volatilidade condicional usando o modelo GARCH(1,1).
    A calibração dos coeficientes é realizada via otimização numérica SLSQP (Máxima Verossimilhança).
    """

    def estimate_garch(self, df: pd.DataFrame, steps: int = 15) -> dict:
        """
        Estima os parâmetros do GARCH(1,1) e projeta a volatilidade futura.
        
        Args:
            df (pd.DataFrame): DataFrame contendo a coluna ['close'].
            steps (int): Quantidade de passos temporais futuros para projetar a volatilidade.
            
        Returns:
            dict: Parâmetros estimados do modelo (omega, alpha, beta, persistência) e curvas de volatilidade.
        """
        if df is None or df.empty or len(df) < 10:
            raise ValueError("Dados históricos insuficientes para calibrar o modelo GARCH.")

        prices = df["close"].values
        
        # 1. Calcular os retornos logarítmicos
        returns = np.log(prices[1:] / prices[:-1])
        
        # Desvia os retornos em relação à média (resíduos)
        mean_return = returns.mean()
        eps = returns - mean_return
        
        # Variância amostral inicial como aproximação
        init_var = np.var(returns)
        if init_var == 0:
            init_var = 1e-6

        # 2. Definição da Função de Log-Verossimilhança Negativa (NLL)
        # Queremos minimizar esta função para encontrar os estimadores de máxima verossimilhança
        def negative_log_likelihood(params):
            omega, alpha, beta = params
            n = len(eps)
            sigma2 = np.zeros(n)
            sigma2[0] = init_var
            
            # Recorrência do GARCH(1,1): sigma2_t = omega + alpha * eps^2_{t-1} + beta * sigma2_{t-1}
            for t in range(1, n):
                sigma2[t] = omega + alpha * (eps[t - 1] ** 2) + beta * sigma2[t - 1]
            
            # Evita divisões por zero ou valores não positivos
            sigma2 = np.clip(sigma2, 1e-10, None)
            
            # NLL formula (ignoring constant factor 1/2*log(2*pi))
            nll_val = 0.5 * np.sum(np.log(sigma2) + (eps ** 2) / sigma2)
            return nll_val

        # 3. Otimização SLSQP
        # Chutes iniciais razoáveis (GARCH típico possui alta persistência beta)
        x0 = np.array([0.05 * init_var, 0.09, 0.85])
        
        # Limites físicos dos parâmetros: omega > 0, alpha >= 0, beta >= 0
        bounds = [(1e-10, 1.0), (0.0001, 0.999), (0.0001, 0.999)]
        
        # Restrição de estabilidade: alpha + beta < 1
        def stability_constraint(params):
            omega, alpha, beta = params
            return 0.999 - (alpha + beta) # Retorna positivo se a restrição for satisfeita

        constraints = {"type": "ineq", "fun": stability_constraint}

        try:
            res = minimize(
                negative_log_likelihood, 
                x0, 
                method="SLSQP", 
                bounds=bounds, 
                constraints=constraints,
                options={"maxiter": 100, "ftol": 1e-6}
            )
            
            if res.success:
                omega, alpha, beta = res.x
                success = True
            else:
                logger.warning(f"Otimizador GARCH não convergiu perfeitamente: {res.message}. Usando coeficientes padrão.")
                omega, alpha, beta = x0
                success = False
        except Exception as e:
            logger.error(f"Erro ao otimizar GARCH: {e}. Usando coeficientes de fallback.")
            omega, alpha, beta = x0
            success = False

        # 4. Reconstrução da Variância Histórica
        n = len(eps)
        sigma2_history = np.zeros(n)
        sigma2_history[0] = init_var
        for t in range(1, n):
            sigma2_history[t] = omega + alpha * (eps[t - 1] ** 2) + beta * sigma2_history[t - 1]

        # 5. Projeção Futura da Variância (k passos à frente)
        # Passo 1 usa o último resíduo e a última variância real calculada
        last_sigma2 = sigma2_history[-1]
        last_eps = eps[-1]
        
        forecast_var = np.zeros(steps)
        forecast_var[0] = omega + alpha * (last_eps ** 2) + beta * last_sigma2
        
        # Passos subsequentes convergem para a média incondicional (reversão à média)
        # E_t[sigma2_{t+k}] = omega + (alpha + beta) * sigma2_{t+k-1}
        persistence = alpha + beta
        for k in range(1, steps):
            forecast_var[k] = omega + persistence * forecast_var[k - 1]

        # 6. Variância de Longo Prazo (Uncondicional)
        if persistence < 1.0:
            long_term_var = omega / (1.0 - persistence)
        else:
            long_term_var = init_var

        # 7. Conversão da Variância para Volatilidade (%) por candle/período
        # Multiplicamos por 100 para expressar em formato percentual
        vol_history = np.sqrt(sigma2_history) * 100.0
        vol_forecast = np.sqrt(forecast_var) * 100.0
        vol_long_term = np.sqrt(long_term_var) * 100.0
        
        # Geração dos Timestamps Futuros
        if len(df) > 1:
            time_delta = df["timestamp"].diff().mean()
        else:
            time_delta = pd.Timedelta(days=1)
            
        last_timestamp = df["timestamp"].iloc[-1]
        future_timestamps = [last_timestamp + ((i + 1) * time_delta) for i in range(steps)]

        return {
            "omega": float(omega),
            "alpha": float(alpha),
            "beta": float(beta),
            "persistence": float(persistence),
            "success": success,
            "current_vol": float(vol_history[-1]),
            "vol_long_term": float(vol_long_term),
            "history_vol": vol_history.tolist(),
            "forecast_vol": vol_forecast.tolist(),
            "timestamps": future_timestamps
        }
