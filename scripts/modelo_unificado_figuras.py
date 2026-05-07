"""
================================================================================
  MODELO UNIFICADO — figuras finales y comparación con originales
================================================================================
Genera figuras de cierre que documentan:
  - Convergencia de las cadenas MCMC
  - Posterior conjunto del modelo unificado
  - Comparación entre los 4 modelos originales y el unificado
  - Resultado del backtest 2022

Ejecución:
    PYTHONPATH=. python scripts/modelo_unificado_figuras.py
"""

from __future__ import annotations

import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import CANDS, BLOQUES, cargar_polymarket
from src.modelo_unificado import calcular_p_presidente, diagnostico_polymarket


def main():
    base = ROOT
    fig_dir = ROOT / "figuras"
    fig_dir.mkdir(exist_ok=True)
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    # Cargar posterior del modelo 1
    with open(base / "data" / "posterior_modelo1.pkl", "rb") as f:
        saved = pickle.load(f)
    idata = saved["idata"]
    pi = idata.posterior["pi"].values
    pi_elec = pi[..., -1, :].reshape(-1, len(CANDS))

    # ============================================================
    # Fig 9: Convergencia (trace plots de hiperparámetros)
    # ============================================================
    fig, axes = plt.subplots(3, 2, figsize=(13, 9))
    sigma = idata.posterior["sigma_rw"].values
    tau = idata.posterior["tau_house"].values
    alpha0 = idata.posterior["alpha0"].values

    for chain_idx in range(sigma.shape[0]):
        axes[0, 0].plot(sigma[chain_idx], lw=0.7, alpha=0.7, label=f"chain {chain_idx}")
    axes[0, 0].set_title(r"$\sigma_{rw}$ (random walk daily std, ALR scale)")
    axes[0, 0].set_xlabel("MCMC iteration")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].hist(sigma.flatten(), bins=50, color="#457B9D", alpha=0.7, edgecolor="white")
    axes[0, 1].set_title(r"$\sigma_{rw}$ marginal posterior")
    axes[0, 1].set_xlabel(r"$\sigma_{rw}$")

    for chain_idx in range(tau.shape[0]):
        axes[1, 0].plot(tau[chain_idx], lw=0.7, alpha=0.7)
    axes[1, 0].set_title(r"$\tau_{house}$ (house effect dispersion)")
    axes[1, 0].set_xlabel("MCMC iteration")

    axes[1, 1].hist(tau.flatten(), bins=50, color="#E63946", alpha=0.7, edgecolor="white")
    axes[1, 1].set_title(r"$\tau_{house}$ marginal posterior")
    axes[1, 1].set_xlabel(r"$\tau_{house}$")

    for p_idx in range(alpha0.shape[2]):
        axes[2, 0].plot(alpha0[..., p_idx].mean(axis=0), lw=1.5,
                        label=f"pollster {p_idx}")
    axes[2, 0].set_title(r"$\alpha_0$ (DM concentration) — chain mean")
    axes[2, 0].set_xlabel("MCMC iteration")
    axes[2, 0].legend(fontsize=8)

    pollster_names = list(idata.posterior.coords["encuestadora"].values)
    axes[2, 1].boxplot([alpha0[..., i].flatten() for i in range(len(pollster_names))],
                        labels=pollster_names)
    axes[2, 1].set_title(r"$\alpha_0$ por encuestadora")
    axes[2, 1].set_ylabel(r"$\alpha_0$")

    fig.suptitle("Modelo unificado — diagnóstico MCMC", fontsize=12, y=1.00)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig9_convergencia_mcmc.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ============================================================
    # Fig 10: Comparación con modelos originales
    # ============================================================
    # Resultado del modelo unificado:
    class R: pass
    res = R(); res.idata = idata
    p_pres = calcular_p_presidente(res)
    p_cep_unificado = p_pres["cepeda"] * 100

    # Resultados originales (del README/auditoría):
    P_ORIG = {
        "1. Encuestas\noriginal":       88.6,
        "2. Revealed\noriginal":        82.9,
        "3. Transferencia\noriginal":   79.5,
        "4. Nested\noriginal":          62.7,
        "Unificado\ncorregido":         p_cep_unificado,
    }

    fig, ax = plt.subplots(figsize=(11, 6))
    nombres = list(P_ORIG.keys())
    vals = list(P_ORIG.values())
    colors = ["#E63946", "#F4A261", "#E76F51", "#2A9D8F", "#1D3557"]
    bars = ax.bar(nombres, vals, color=colors, edgecolor="white", linewidth=1.2)
    ax.axhline(50, color="black", ls="--", lw=0.7, alpha=0.5)
    ax.set_ylabel("P(Cepeda presidente) (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Comparación: 4 modelos originales vs. modelo unificado corregido\n"
                 f"P(Cepeda presidente) — el modelo unificado da {p_cep_unificado:.1f}%",
                 fontsize=11)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5,
                f"{v:.1f}%", ha="center", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig10_comparacion_modelos.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ============================================================
    # Fig 11: Polymarket coherence diagnostic
    # ============================================================
    polym = cargar_polymarket(base / "data" / "polymarket.csv")
    diag_df = diagnostico_polymarket(pi_elec, polym)

    fig, ax = plt.subplots(figsize=(10, 5))
    cands_pm = diag_df.dropna(subset=["P(Polymarket gana 1ª)"])["candidato"].tolist()
    p_modelo = diag_df.dropna(subset=["P(Polymarket gana 1ª)"])["P(modelo gana 1ª)"].values * 100
    p_market = diag_df.dropna(subset=["P(Polymarket gana 1ª)"])["P(Polymarket gana 1ª)"].values * 100

    x = np.arange(len(cands_pm))
    width = 0.35
    ax.bar(x - width/2, p_modelo, width, label="Modelo unificado",
           color="#1D3557", edgecolor="white")
    ax.bar(x + width/2, p_market, width, label="Polymarket (USD 4.8M)",
           color="#F4A261", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([c.title() for c in cands_pm])
    ax.set_ylabel("P(gana 1ª vuelta) (%)")
    ax.set_title("Coherencia con Polymarket — modelo unificado vs. mercado de predicción\n"
                 "(diagnóstico, no input al modelo)",
                 fontsize=11)
    ax.legend()
    for i, (m, p) in enumerate(zip(p_modelo, p_market)):
        ax.text(i - width/2, m + 1, f"{m:.1f}%", ha="center", fontsize=9)
        ax.text(i + width/2, p + 1, f"{p:.1f}%", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig11_coherencia_polymarket.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ============================================================
    # Fig 12: Backtest 2022
    # ============================================================
    backtest_pkl = Path("/tmp/backtest_2022.pkl")
    if backtest_pkl.exists():
        with open(backtest_pkl, "rb") as f:
            bt = pickle.load(f)
        df_bt = bt["df"]

        # Recompute por seguridad
        idata_2022 = bt["idata"]
        pi_2022 = idata_2022.posterior["pi"].values
        pi_elec_2022 = pi_2022[..., -1, :].reshape(-1, 7)

        from tests.backtest_2022 import CANDS_2022, REAL_1V_2022

        fig, ax = plt.subplots(figsize=(11, 6))
        n_cands = len(CANDS_2022)
        x = np.arange(n_cands)
        means = pi_elec_2022.mean(axis=0) * 100
        lo = np.percentile(pi_elec_2022, 5, axis=0) * 100
        hi = np.percentile(pi_elec_2022, 95, axis=0) * 100
        reales = np.array([REAL_1V_2022.get(c, np.nan) * 100 for c in CANDS_2022])

        # Posterior: media + IC
        ax.errorbar(x, means, yerr=[means - lo, hi - means], fmt="o",
                    color="#1D3557", markersize=8, capsize=5, capthick=1.5,
                    label="Posterior modelo (media + IC90%)", lw=2)
        ax.scatter(x, reales, marker="x", color="#E63946", s=180, lw=3,
                   label="Resultado real 29-may-2022", zorder=10)

        for i, c in enumerate(CANDS_2022):
            err = means[i] - reales[i]
            ax.text(i, max(means[i], reales[i]) + 2, f"err={err:+.1f}pp",
                    ha="center", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(CANDS_2022, rotation=20, ha="right")
        ax.set_ylabel("Share en 1ª vuelta (%)")
        ax.set_title("Backtest del modelo unificado — Colombia 2022\n"
                     "Encuestas a 32 días de la elección. Hernández (×) sale del IC90 — "
                     "fenómeno populista tardío que el agregador no anticipó.",
                     fontsize=11)
        ax.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "fig12_backtest_2022.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  fig12_backtest_2022.png")

    # ============================================================
    # Fig 13: Posterior conjunto presidencial (con propagación 2v)
    # ============================================================
    rng = np.random.default_rng(2026)
    p_pres_samples = []  # n_samples × n_cands con la P(presidente) muestra a muestra

    fig, ax = plt.subplots(figsize=(11, 6))
    nombres_largos = {
        "cepeda": "Iván Cepeda", "espriella": "A. De la Espriella",
        "paloma": "Paloma Valencia",
    }
    p_top = sorted(p_pres, key=lambda x: -p_pres[x])[:3]
    vals = [p_pres[c] * 100 for c in p_top]
    colores_top = ["#E63946", "#1D3557", "#457B9D"]

    bars = ax.barh([nombres_largos.get(c, c) for c in p_top], vals,
                    color=colores_top, edgecolor="white")
    ax.set_xlim(0, 100)
    ax.set_xlabel("P(ganar la presidencia) (%) — modelo unificado")
    ax.set_title("Resultado final — modelo unificado bayesiano\n"
                 "Propagación completa de incertidumbre encuestas → 1ª → 2ª vuelta",
                 fontsize=11)
    for bar, v in zip(bars, vals):
        ax.text(v + 1, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}%", va="center", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig13_resultado_final.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nFiguras del modelo unificado guardadas en {fig_dir}/")
    print("  fig9_convergencia_mcmc.png")
    print("  fig10_comparacion_modelos.png")
    print("  fig11_coherencia_polymarket.png")
    print("  fig13_resultado_final.png")


if __name__ == "__main__":
    main()
