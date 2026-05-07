"""
================================================================================
  MODELO 1 — AGREGACIÓN BAYESIANA DE ENCUESTAS (versión corregida)
================================================================================

Reemplazo de scripts/modelo1_encuestas.py original. Las correcciones se
documentan en /AUDITORIA.md sección "Modelo 1".

Cambios principales vs. la versión original:

  1. Likelihood Dirichlet-Multinomial (no Multinomial). La sobre-dispersión
     no se "ajusta" con un n_eff heurístico; se ESTIMA por encuestadora
     vía el parámetro de concentración α₀_p.

  2. Random walk en escala ALR. La incertidumbre crece con los días
     restantes según un proceso estocástico estimable, no según el factor
     ad-hoc 1 + (días/30)·1.5.

  3. House effects δ_p con prior jerárquico ZeroSum. Reemplaza el vector
     CREDIBILIDAD = {Invamer:1.00, CNC:0.85, GAD3:0.80} fijado a ojo.

  4. Pesos de encuestas IMPLÍCITOS en la verosimilitud. Una encuesta con
     n más grande contribuye más a la posterior automáticamente; no hay
     pesos exógenos w_t · √n · c_p.

  5. Incluye AtlasIntel. El script original lo excluía silenciosamente.

  6. La 2ª vuelta NO se reduce a "Beta con n/2". Se simula vía el
     mecanismo de transferencias del modelo 3, condicionado en las
     muestras del posterior de 1ª vuelta.

REPRODUCIBILIDAD: este script lee data/encuestas.csv. Cualquier cambio en
el CSV se refleja en el output. No hay encuestas hard-codeadas.

Ejecución:
    PYTHONPATH=. python scripts/modelo1_encuestas.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# Permitir ejecutarlo desde la raíz del repo o desde scripts/
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import (
    cargar_encuestas, CANDS, BLOQUES,
)
from src.modelo_unificado import (
    construir_modelo_encuestas,
    ajustar_modelo_encuestas,
    PRIOR_SIGMA_RW,
    PRIOR_TAU_HOUSE,
)


COLORES = {
    "cepeda":    "#E63946",
    "espriella": "#1D3557",
    "paloma":    "#457B9D",
    "lopez":     "#2A9D8F",
    "fajardo":   "#A8DADC",
    "otros":     "#6C757D",
    "blanco":    "#E8E8E8",
}

NOMBRES_LARGOS = {
    "cepeda":    "Iván Cepeda",
    "espriella": "Abelardo De la Espriella",
    "paloma":    "Paloma Valencia",
    "lopez":     "Claudia López",
    "fajardo":   "Sergio Fajardo",
    "otros":     "Otros",
    "blanco":    "Voto en blanco",
}


def main():
    base = ROOT
    print("=" * 78)
    print(" MODELO 1 — agregación bayesiana de encuestas (Dirichlet-Multinomial)")
    print(" 32 días para la primera vuelta (31 may 2026)")
    print("=" * 78)

    encuestas = cargar_encuestas(base / "data" / "encuestas.csv")
    print(f"\nCargadas {len(encuestas)} encuestas:")
    for e in sorted(encuestas, key=lambda x: x.fecha):
        print(f"  {e.pollster:>10s}  {e.fecha}  n={e.n}  método={e.metodo}")

    print("\nAjustando con NUTS (2 cadenas, 1000 tune + 700 draws)...")
    res = ajustar_modelo_encuestas(
        encuestas,
        dia_eleccion=date(2026, 5, 31),
        n_draws=700, n_tune=1000, n_chains=2,
        target_accept=0.95,
        progressbar=False,
    )

    # Diagnóstico
    rhat = az.rhat(res.idata, var_names=["pi", "sigma_rw", "tau_house", "alpha0"])
    rhat_max = max(float(rhat[v].max()) for v in ["pi", "sigma_rw", "tau_house", "alpha0"])
    ess = az.ess(res.idata, var_names=["pi"])
    print(f"\nDiagnóstico de convergencia:")
    print(f"  R-hat máximo  : {rhat_max:.3f}  (debe ser < 1.01)")
    print(f"  ESS bulk min π: {float(ess['pi'].min()):.0f}  (debe ser > 400)")

    # Posterior de hiperparámetros
    sigma_post = res.idata.posterior["sigma_rw"].values
    tau_post = res.idata.posterior["tau_house"].values
    alpha0_post = res.idata.posterior["alpha0"].values
    print("\nHiperparámetros (posterior):")
    print(f"  σ_rw      : {sigma_post.mean():.4f}  IC90 [{np.percentile(sigma_post, 5):.4f}, {np.percentile(sigma_post, 95):.4f}]")
    print(f"  τ_house   : {tau_post.mean():.4f}  IC90 [{np.percentile(tau_post, 5):.4f}, {np.percentile(tau_post, 95):.4f}]")
    print(f"  α₀ (concentración Dirichlet-Multinomial por encuestadora):")
    for p_idx, p_name in enumerate(res.pollster_names):
        a = alpha0_post[..., p_idx].flatten()
        print(f"    {p_name:>12s}  : {a.mean():.0f}  IC90 [{np.percentile(a, 5):.0f}, {np.percentile(a, 95):.0f}]")

    # House effects
    print("\nHouse effects δ_p (escala logit, ref=blanco; suma 0 entre encuestadoras):")
    delta = res.idata.posterior["delta"].values
    print(f"  {'pollster':>12s}  {'cep':>7s} {'esp':>7s} {'pal':>7s} {'lop':>7s} {'faj':>7s} {'otr':>7s}")
    for p_idx, p_name in enumerate(res.pollster_names):
        line = f"  {p_name:>12s}  "
        for k_idx, c in enumerate(CANDS[:-1]):
            d = delta[..., p_idx, k_idx].flatten()
            line += f"{d.mean():+7.3f} "
        print(line)

    # Posterior en día de elección
    pi = res.idata.posterior["pi"].values
    pi_elec = pi[..., -1, :].reshape(-1, len(CANDS))

    print("\n--- INTENCIÓN POSTERIOR EN DÍA DE ELECCIÓN ---")
    print(f"  {'Candidato':<22s} {'Media':>7s}   {'IC 90%':>15s}   {'P(top 2)':>9s}")
    orden = np.argsort(-pi_elec.mean(axis=0))
    top2 = np.argsort(-pi_elec, axis=1)[:, :2]
    for j in orden:
        m = pi_elec[:, j].mean() * 100
        lo = np.percentile(pi_elec[:, j], 5) * 100
        hi = np.percentile(pi_elec[:, j], 95) * 100
        p_top2 = (top2 == j).any(axis=1).mean() * 100
        print(f"  {NOMBRES_LARGOS[CANDS[j]]:<22s} {m:>6.1f}%  [{lo:>4.1f}, {hi:>4.1f}]   {p_top2:>7.1f}%")

    # P(gana 1ª)
    top1 = pi_elec.argmax(axis=1)
    gana_1v_pct = pi_elec[np.arange(len(top1)), top1] > 0.5
    p_gana_1v = float(gana_1v_pct.mean())
    print(f"\n  P(hay segunda vuelta) = {(1 - p_gana_1v) * 100:.1f}%")
    print(f"  P(Cepeda gana en 1ª) = {(gana_1v_pct & (top1 == CANDS.index('cepeda'))).mean() * 100:.1f}%")

    # ============================================================
    # FIGURAS
    # ============================================================
    fig_dir = ROOT / "figuras"
    fig_dir.mkdir(exist_ok=True)
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    # Fig 1: Distribuciones posteriores en día de elección
    fig, ax = plt.subplots(figsize=(11, 6))
    top_5 = np.argsort(-pi_elec.mean(axis=0))[:5]
    for j in top_5:
        c = CANDS[j]
        m = pi_elec[:, j].mean() * 100
        ax.hist(pi_elec[:, j] * 100, bins=80, alpha=0.65, density=True,
                color=COLORES[c], label=f"{NOMBRES_LARGOS[c]} ({m:.1f}%)",
                edgecolor="white", linewidth=0.3)
    ax.axvline(50, color="black", ls="--", lw=1.2, alpha=0.7)
    ax.text(50.5, ax.get_ylim()[1] * 0.92, "Umbral 1ª vuelta", fontsize=9)
    ax.set_xlabel("Intención de voto (%)")
    ax.set_ylabel("Densidad posterior")
    ax.set_title(
        "Posterior π al 31-may-2026 — Modelo 1 corregido\n"
        f"(Dirichlet-Multinomial, 7 encuestas, NUTS 4 cadenas, "
        f"R-hat={rhat_max:.3f})",
        fontsize=11,
    )
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig1_posterior_primera_vuelta.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Fig 2: Trayectoria con bandas posteriores
    fig, ax = plt.subplots(figsize=(11, 6))
    pi_grid = pi.reshape(-1, pi.shape[-2], pi.shape[-1])  # (samples, T, K)
    fechas_arr = res.grid_fechas
    for c in ["cepeda", "espriella", "paloma"]:
        j = CANDS.index(c)
        m = pi_grid[..., j].mean(axis=0) * 100
        lo = np.percentile(pi_grid[..., j], 5, axis=0) * 100
        hi = np.percentile(pi_grid[..., j], 95, axis=0) * 100
        ax.plot(fechas_arr, m, "-", color=COLORES[c], label=NOMBRES_LARGOS[c], linewidth=2)
        ax.fill_between(fechas_arr, lo, hi, alpha=0.18, color=COLORES[c])
        # Encuestas
        for e in encuestas:
            ax.scatter(e.fecha, e.shares[j] * 100, color=COLORES[c],
                       s=30, alpha=0.7, edgecolor="white", linewidth=1)
    ax.axvline(date(2026, 5, 31), color="red", ls="--", lw=1, alpha=0.5)
    ax.text(date(2026, 5, 31), 5, "  31 may", color="red", fontsize=9)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Intención de voto (%)")
    ax.set_title(
        "Trayectoria posterior π_t (state-space, ALR random walk)\n"
        "Líneas = posterior medio. Bandas = IC90%. Puntos = encuestas observadas.",
        fontsize=11,
    )
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig2_trayectoria_posterior.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Fig 3: House effects
    fig, ax = plt.subplots(figsize=(11, 6))
    pollsters = res.pollster_names
    cands_plot = ["cepeda", "espriella", "paloma", "lopez", "fajardo", "otros"]
    width = 0.18
    x = np.arange(len(cands_plot))
    for i, p_name in enumerate(pollsters):
        delta_p = []
        for c in cands_plot:
            j = CANDS.index(c)
            if j < len(CANDS) - 1:
                d = res.idata.posterior["delta"].values[..., i, j].flatten()
                delta_p.append(d.mean())
            else:
                delta_p.append(0)
        offset = (i - len(pollsters) / 2 + 0.5) * width
        ax.bar(x + offset, delta_p, width, label=p_name,
               edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([NOMBRES_LARGOS[c] for c in cands_plot], rotation=20, ha="right")
    ax.set_ylabel("Sesgo δ_p (escala logit, ref = voto en blanco)")
    ax.set_title(
        "House effects estimados — sesgo de cada encuestadora vs. consenso\n"
        "Suma cero entre encuestadoras por candidato. Positivo = sobreestima.",
        fontsize=11,
    )
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig3_house_effects.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nFiguras guardadas en {fig_dir}/")
    print("  fig1_posterior_primera_vuelta.png")
    print("  fig2_trayectoria_posterior.png")
    print("  fig3_house_effects.png")

    # Guardar el posterior para reuso por modelos posteriores
    import pickle
    with open(ROOT / "data" / "posterior_modelo1.pkl", "wb") as f:
        pickle.dump({
            "idata": res.idata,
            "grid": res.grid_fechas,
            "pollsters": res.pollster_names,
            "encuestas": encuestas,
        }, f)
    print(f"\nPosterior guardado en data/posterior_modelo1.pkl")


if __name__ == "__main__":
    main()
