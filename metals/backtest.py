"""Backtesting simple sobre el holdout: invertir el mes siguiente sólo si el modelo predice Alcista.

Usa los retornos entre promedios mensuales (los mismos de la etiqueta), no precios ejecutables:
es una aproximación académica, sin impuestos, deslizamiento ni apalancamiento.
"""
import pandas as pd

COMMISSION = 0.001  # 0,1 % por cada entrada o salida de la posición


def max_drawdown(equity):
    return float((equity / equity.cummax() - 1).min())


def backtest(holdout, commission=COMMISSION):
    """holdout: filas mensuales con date, target_return y prediction (2 = Alcista)."""
    frame = holdout.sort_values("date")
    invested = (frame.prediction == 2).astype(int).reset_index(drop=True)
    returns = frame.target_return.reset_index(drop=True)
    trades = invested.diff().abs().fillna(invested.iloc[0])
    strategies = {"modelo": (invested * returns - trades * commission, int(trades.sum())),
                  "comprar_y_mantener": (returns - pd.Series([commission] + [0.0] * (len(returns) - 1)), 1)}
    result = {"meses": len(frame), "meses_invertido": int(invested.sum()), "comision": commission}
    for name, (monthly, operations) in strategies.items():
        equity = pd.concat([pd.Series([1.0]), (1 + monthly).cumprod()], ignore_index=True)
        result[name] = {"retorno_acumulado": float(equity.iloc[-1] - 1), "caida_maxima": max_drawdown(equity),
                        "operaciones": operations}
    return result
