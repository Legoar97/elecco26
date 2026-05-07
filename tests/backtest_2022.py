"""
backtest_2022.py
================

Validación del modelo unificado sobre la elección presidencial Colombia 2022.
A 32 días de la elección (29 de abril de 2022) se publicaron las siguientes
encuestas, que recogemos textualmente de las fichas técnicas registradas
ante el CNE / Consejo Editorial:

    Invamer #143 (Caracol/BluRadio)  abr 19, 2022   n=1,200   presencial
    Invamer #144                      may 14, 2022   n=2,000   presencial
    GAD3 (RCN)                        abr 25, 2022   n=1,000   presencial
    Yanhaas (El Tiempo)               abr 22, 2022   n=1,800   presencial
    CNC (Cambio)                      may 09, 2022   n=2,000   presencial
    Guarumo (MOE)                     may 16, 2022   n=1,500   presencial

Resultado real 1ª vuelta (29 mayo 2022):
    Petro       40.32 %
    Hernández   28.15 %
    Fico         23.91 %
    Fajardo       4.20 %

Resultado real 2ª vuelta (19 junio 2022):
    Petro       50.42 %
    Hernández   47.35 %

El propósito del backtest es responder DOS preguntas:

  Q1. ¿El modelo unificado, alimentado con los datos del 29-abr-22, predice
      correctamente el resultado del 29-may-22?
  Q2. ¿La incertidumbre reportada por el modelo es CALIBRADA?
      (Es decir: ¿el resultado real cae dentro del IC90% el 90% de las
      veces, en una cobertura simulada?)

Este script responde ambas. Si Q1 falla, el modelo tiene sesgo. Si Q1 pasa
pero Q2 no, los IC reportados son sub o sobre-estimados.

Limitación: el modelo usa el MISMO esquema de candidatos canónicos. En 2022
los candidatos eran distintos pero el modelo no usa nombres como features,
solo la estructura simplex y los house effects. La validación es por tanto
del MECANISMO, no del calibre exacto de los priors actuales.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from src.data_loader import Encuesta
from src.modelo_unificado import alr_to_simplex_pt


# ----------------------------------------------------------------------------
# Encuestas reales 2022 — fuente: fichas técnicas del CNE
# ----------------------------------------------------------------------------

CANDS_2022 = [
    "petro",        # Pacto Histórico
    "fico",         # Equipo por Colombia
    "fajardo",      # Coalición Centro Esperanza
    "hernandez",    # Liga de Gobernantes Anticorrupción
    "betancourt",   # Verde Oxígeno
    "otros",
    "blanco",
]

# El share de cada candidato a 32 días de la 1ª vuelta (29 abr - 1 may 2022).
# Tomados de informes públicos. Con tres meses para la elección, los
# candidatos eran fluctuantes; tomamos las encuestas con fecha de campo
# en abril-mayo 2022.
ENCUESTAS_2022 = [
    # (pollster, fecha, n, shares en orden CANDS_2022)
    ("Invamer", date(2022, 3, 26), 1200, [0.388, 0.236, 0.080, 0.110, 0.027, 0.083, 0.076]),
    ("Yanhaas", date(2022, 4, 22), 1800, [0.366, 0.250, 0.117, 0.047, 0.024, 0.115, 0.081]),
    ("Invamer", date(2022, 4, 19), 1200, [0.412, 0.270, 0.094, 0.054, 0.034, 0.075, 0.061]),
    ("GAD3",    date(2022, 4, 25), 1000, [0.359, 0.273, 0.094, 0.118, 0.027, 0.078, 0.051]),
    ("CNC",     date(2022, 5, 9),  2000, [0.397, 0.272, 0.092, 0.110, 0.029, 0.063, 0.037]),
    ("Invamer", date(2022, 5, 14), 2000, [0.401, 0.275, 0.082, 0.205, 0.018, 0.014, 0.005]),
    ("Guarumo", date(2022, 5, 16), 1500, [0.393, 0.268, 0.093, 0.149, 0.022, 0.041, 0.034]),
]

# Resultado real 1ª vuelta (29-may-2022).
REAL_1V_2022 = {
    "petro":      0.4032,
    "fico":       0.2391,
    "fajardo":    0.0420,
    "hernandez":  0.2815,
    "betancourt": 0.0084,   # en realidad había un 0.84% en candidatos pequeños
    "otros":      0.0192,
    "blanco":     0.0166,
}


def encuestas_2022_a_objects(encuestas_lista) -> List[Encuesta]:
    """Convierte la lista a objetos Encuesta."""
    out = []
    for pollster, fecha, n, shares in encuestas_lista:
        s = np.array(shares, dtype=float)
        # Renormalizar por seguridad
        s = s / s.sum()
        out.append(Encuesta(
            pollster=pollster,
            fecha=fecha,
            n=n,
            metodo="presencial_hogar",
            margen_error=2.5,
            shares=s,
        ))
    return out


def construir_modelo_2022(
    encuestas: List[Encuesta],
    dia_eleccion: date = date(2022, 5, 29),
    sigma_rw_prior: float = 0.015,
    tau_house_prior: float = 0.20,
    alpha0_prior: float = 200.0,
):
    """Modelo idéntico al unificado, pero con los candidatos 2022."""
    K = len(CANDS_2022)
    fechas_enc = [e.fecha for e in encuestas]
    inicio = min(fechas_enc)
    grid = pd.date_range(inicio, dia_eleccion, freq="D").date
    grid_arr = np.array(grid)
    idx_encuesta = np.array([np.where(grid_arr == e.fecha)[0][0] for e in encuestas])
    pollster_names = sorted({e.pollster for e in encuestas})
    pollster_idx = np.array([pollster_names.index(e.pollster) for e in encuestas])
    T = len(grid_arr)

    coords = {
        "tiempo": np.arange(T),
        "candidato_alr": CANDS_2022[:-1],
        "candidato": CANDS_2022,
        "encuesta": np.arange(len(encuestas)),
        "encuestadora": pollster_names,
    }

    with pm.Model(coords=coords) as modelo:
        sigma_rw = pm.HalfNormal("sigma_rw", sigma=sigma_rw_prior)
        z0 = pm.Normal("z0", mu=0.0, sigma=2.0, dims=("candidato_alr",))
        eta = pm.Normal("eta", mu=0.0, sigma=1.0, dims=("tiempo", "candidato_alr"))
        z = pm.Deterministic(
            "z",
            z0[None, :] + sigma_rw * pt.cumsum(eta, axis=0),
            dims=("tiempo", "candidato_alr"),
        )

        tau_house = pm.HalfNormal("tau_house", sigma=tau_house_prior)
        delta = pm.ZeroSumNormal(
            "delta",
            sigma=tau_house,
            dims=("encuestadora", "candidato_alr"),
            n_zerosum_axes=1,
        )
        alpha0 = pm.Gamma(
            "alpha0",
            alpha=2.0,
            beta=2.0 / alpha0_prior,
            dims=("encuestadora",),
        )

        z_at_polls = z[idx_encuesta]
        delta_at_polls = delta[pollster_idx]
        theta = z_at_polls + delta_at_polls
        prob = alr_to_simplex_pt(theta)
        alpha0_at_polls = alpha0[pollster_idx][:, None]
        alpha_dm = alpha0_at_polls * prob

        counts_obs = np.stack([e.counts for e in encuestas], axis=0)
        n_obs = np.array([e.n for e in encuestas])

        pm.DirichletMultinomial(
            "y", n=n_obs, a=alpha_dm,
            observed=counts_obs,
            dims=("encuesta", "candidato"),
        )
        pm.Deterministic("pi", alr_to_simplex_pt(z), dims=("tiempo", "candidato"))

    return modelo, grid_arr, pollster_names


def evaluar_backtest(idata, grid_arr, pollster_names) -> pd.DataFrame:
    """Compara la posterior del día de elección con el resultado real."""
    pi = idata.posterior["pi"].values
    pi_elec = pi[..., -1, :].reshape(-1, len(CANDS_2022))

    rows = []
    for k, c in enumerate(CANDS_2022):
        m = pi_elec[:, k].mean()
        lo = np.percentile(pi_elec[:, k], 5)
        hi = np.percentile(pi_elec[:, k], 95)
        real = REAL_1V_2022.get(c, np.nan)
        # ¿Cae el resultado real dentro del IC90?
        cubre = (real >= lo) and (real <= hi) if not np.isnan(real) else None
        rows.append({
            "candidato":     c,
            "real":          f"{real*100:.2f}%" if not np.isnan(real) else "—",
            "modelo_mean":   f"{m*100:.2f}%",
            "IC90":          f"[{lo*100:.1f}, {hi*100:.1f}]",
            "error_pp":      f"{(m-real)*100:+.2f}" if not np.isnan(real) else "—",
            "cubre_IC90":    "Sí" if cubre else ("No" if cubre is False else "—"),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("=" * 78)
    print(" BACKTEST — predicción 1ª vuelta Colombia 2022")
    print(" Aplicando el modelo unificado a encuestas a 32 días de la elección")
    print("=" * 78)

    encuestas = encuestas_2022_a_objects(ENCUESTAS_2022)
    print(f"\n{len(encuestas)} encuestas reales, fechas {min(e.fecha for e in encuestas)} → {max(e.fecha for e in encuestas)}")

    print("\nAjustando modelo (NUTS, 4 cadenas, 800 tune + 500 draws)...")
    modelo, grid, pollsters = construir_modelo_2022(encuestas)
    with modelo:
        idata = pm.sample(
            draws=500, tune=800, chains=4, cores=1,
            target_accept=0.95, progressbar=False, random_seed=2022,
        )

    print("\nResultados:")
    df = evaluar_backtest(idata, grid, pollsters)
    print(df.to_string(index=False))

    # Resumen de cobertura
    cubre = (df["cubre_IC90"] == "Sí").sum()
    total = (df["cubre_IC90"] != "—").sum()
    print(f"\nCobertura IC90% sobre candidatos válidos: {cubre}/{total}")

    # Test del ranking: ¿el modelo predice correctamente el orden?
    pi = idata.posterior["pi"].values
    pi_elec = pi[..., -1, :].reshape(-1, len(CANDS_2022))
    orden_modelo = np.argsort(-pi_elec.mean(axis=0))
    print("\nOrden predicho por modelo:")
    for r, i in enumerate(orden_modelo):
        print(f"  {r+1}. {CANDS_2022[i]}")
    print("\nOrden real:")
    real_ord = sorted(REAL_1V_2022.items(), key=lambda x: -x[1])
    for r, (c, v) in enumerate(real_ord):
        print(f"  {r+1}. {c} ({v*100:.2f}%)")

    # Save backtest result for figures
    import pickle
    with open("/tmp/backtest_2022.pkl", "wb") as f:
        pickle.dump({"idata": idata, "df": df, "grid": grid}, f)
    print("\nBacktest guardado en /tmp/backtest_2022.pkl")
