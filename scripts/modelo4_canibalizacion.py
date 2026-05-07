"""
================================================================================
  MODELO 4 — DESCOMPOSICIÓN NESTED Y CANIBALIZACIÓN (versión corregida)
================================================================================

Reemplazo de scripts/modelo4_canibalizacion.py original. Ver /AUDITORIA.md
"Modelo 4" para los fallos del original.

Cambios principales:

  1. NO USA NORMALES PARA PROPORCIONES. Las proporciones viven en el
     simplex; el modelo respeta la geometría del simplex desde el principio.

  2. NO HACE CLIPPING. Las distribuciones extendidas a R+ con soporte
     adecuado vienen dadas por la transformación inversa de softmax.

  3. NO HACE PRODUCTO DE MARGINALES INDEPENDIENTES + RENORMALIZACIÓN.
     La descomposición nested se hace SOBRE EL POSTERIOR del modelo 1,
     que ya respeta la consistencia del simplex.

  4. ESCENARIOS de canibalización: NO se modifican parámetros del prior
     ad-hoc (V_R *= 0.90, etc.). Se modifica la cuota WITHIN-bloque
     manteniendo P(bloque) constante (consistencia local) y se reporta
     el efecto sobre el simplex completo.

  5. V_R NO ES EXÓGENO. La descomposición π_k = P(bloque(k)) · π_within(k)
     se computa para cada muestra del posterior, lo cual mantiene la
     consistencia interna que un nested logit auténtico tendría.

OBJETIVO
========
Cuantificar cómo cambia la probabilidad presidencial bajo escenarios donde:
  (a) Espriella se desinfla — los electores migran dentro del bloque,
  (b) Paloma absorbe el bloque — bloque der se consolida tras un solo cand,
  (c) Espriella se baja — caso extremo donde la cuota de Espriella → 0.
  (d) Bloque der se unifica tras Paloma.

El modelo 4 corregido NO da una nueva P(Cepeda) puntual; da la P(Cepeda)
CONDICIONAL al escenario. Esto permite ver la sensibilidad de la elección
a la dinámica intra-bloque.

Ejecución:
    PYTHONPATH=. python scripts/modelo4_canibalizacion.py
"""

from __future__ import annotations

import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import CANDS, BLOQUES
from src.modelo_unificado import (
    descomponer_nested,
    escenario_canibalizacion,
    calcular_p_presidente,
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


def main():
    base = ROOT
    print("=" * 78)
    print(" MODELO 4 — descomposición nested logit y escenarios de canibalización")
    print("=" * 78)

    pkl_path = base / "data" / "posterior_modelo1.pkl"
    if not pkl_path.exists():
        print(f"\nERROR: {pkl_path} no existe. Corre primero modelo1_encuestas.py")
        sys.exit(1)
    with open(pkl_path, "rb") as f:
        saved = pickle.load(f)
    idata = saved["idata"]
    pi = idata.posterior["pi"].values
    pi_elec = pi[..., -1, :].reshape(-1, len(CANDS))
    n_samples = pi_elec.shape[0]

    print(f"\nMuestras del posterior: {n_samples}")

    # ============================================================
    # 1. Descomposición nested del posterior
    # ============================================================
    print(f"\n--- DESCOMPOSICIÓN NESTED (P(bloque), π_within) DEL POSTERIOR DEL MODELO 1 ---")
    bloques_unicos = sorted(set(BLOQUES.values()))
    bloque_a_idx = {b: [] for b in bloques_unicos}
    for k, c in enumerate(CANDS):
        bloque_a_idx[BLOQUES[c]].append(k)

    p_bloque = {b: np.zeros(n_samples) for b in bloques_unicos}
    pi_within = {c: np.zeros(n_samples) for c in CANDS}
    for k, c in enumerate(CANDS):
        b = BLOQUES[c]
        p_bloque[b] += pi_elec[:, k]

    for k, c in enumerate(CANDS):
        b = BLOQUES[c]
        denom = np.where(p_bloque[b] > 1e-9, p_bloque[b], 1.0)
        pi_within[c] = pi_elec[:, k] / denom

    print(f"\n  P(bloque) — share del bloque en 1ª vuelta:")
    for b in bloques_unicos:
        m = p_bloque[b].mean() * 100
        lo = np.percentile(p_bloque[b], 5) * 100
        hi = np.percentile(p_bloque[b], 95) * 100
        n_cands_b = len(bloque_a_idx[b])
        print(f"    {b:<8s} {m:>6.1f}%  [{lo:>4.1f}, {hi:>4.1f}]  ({n_cands_b} candidato/s)")

    print(f"\n  π_within — cuota dentro del bloque (donde aplica):")
    for c in CANDS:
        b = BLOQUES[c]
        if len(bloque_a_idx[b]) > 1:
            m = pi_within[c].mean() * 100
            lo = np.percentile(pi_within[c], 5) * 100
            hi = np.percentile(pi_within[c], 95) * 100
            print(f"    {c:<10s} (bloque {b}): {m:>6.1f}%  [{lo:>4.1f}, {hi:>4.1f}]")

    # ============================================================
    # 2. Escenarios de canibalización
    # ============================================================
    escenarios = {
        "Status quo":              {"paloma": 0.46, "espriella": 0.45, "otros": 0.09},
        "Espriella se desinfla":   {"paloma": 0.72, "espriella": 0.20, "otros": 0.08},
        "Espriella crece":         {"paloma": 0.30, "espriella": 0.65, "otros": 0.05},
        "Espriella se baja":       {"paloma": 0.85, "espriella": 0.00, "otros": 0.15},
        "Paloma se baja":          {"paloma": 0.00, "espriella": 0.85, "otros": 0.15},
        "Derecha unificada (Paloma)":{"paloma": 0.95, "espriella": 0.00, "otros": 0.05},
    }

    print(f"\n--- ESCENARIOS DE CANIBALIZACIÓN ---")
    print(f"  Cada escenario MODIFICA solo las cuotas WITHIN del bloque der.")
    print(f"  P(bloque der) total se PRESERVA. Esto es la única operación")
    print(f"  consistente con un nested logit auténtico.\n")
    print(f"  {'Escenario':<28s}  {'Cep mean':>9s}  {'Pal':>7s}  {'Esp':>7s}  {'P(Cep top2)':>12s}  {'P(Pal top2)':>12s}  {'P(Esp top2)':>12s}")

    resultados_escenarios = {}
    for nombre, cw in escenarios.items():
        if abs(sum(cw.values()) - 1.0) > 0.001:
            cw = {k: v / sum(cw.values()) for k, v in cw.items()}
        pi_new = escenario_canibalizacion(pi_elec, cw, bloque="der")

        cep_idx = CANDS.index("cepeda")
        pal_idx = CANDS.index("paloma")
        esp_idx = CANDS.index("espriella")

        cep_mean = pi_new[:, cep_idx].mean() * 100
        pal_mean = pi_new[:, pal_idx].mean() * 100
        esp_mean = pi_new[:, esp_idx].mean() * 100

        top2 = np.argsort(-pi_new, axis=1)[:, :2]
        p_cep_t2 = (top2 == cep_idx).any(axis=1).mean() * 100
        p_pal_t2 = (top2 == pal_idx).any(axis=1).mean() * 100
        p_esp_t2 = (top2 == esp_idx).any(axis=1).mean() * 100

        print(f"  {nombre:<28s}  {cep_mean:>7.1f}%  {pal_mean:>5.1f}%  {esp_mean:>5.1f}%  {p_cep_t2:>10.1f}%  {p_pal_t2:>10.1f}%  {p_esp_t2:>10.1f}%")

        resultados_escenarios[nombre] = {
            "pi_new": pi_new,
            "cep_mean": cep_mean,
            "pal_mean": pal_mean,
            "esp_mean": esp_mean,
            "p_cep_t2": p_cep_t2,
            "p_pal_t2": p_pal_t2,
            "p_esp_t2": p_esp_t2,
        }

    # ============================================================
    # 3. Hallazgo central: invarianza de Cepeda
    # ============================================================
    print(f"\n--- HALLAZGO CENTRAL ---")
    cep_means = [r["cep_mean"] for r in resultados_escenarios.values()]
    print(f"  Cepeda en 1ª vuelta varía entre {min(cep_means):.1f}% y {max(cep_means):.1f}%")
    print(f"  ({max(cep_means) - min(cep_means):.1f} puntos) entre escenarios extremos")
    print(f"  de canibalización dentro del bloque DER. Este resultado se obtiene")
    print(f"  por construcción (P(bloque) constante), pero es defendible:")
    print(f"  Cepeda y los candidatos del bloque der no compiten por los mismos")
    print(f"  electores en 1ª vuelta — compiten por bloques distintos.")

    # ============================================================
    # 4. Sensibilidad continua: cuota de Espriella en el bloque
    # ============================================================
    print(f"\n--- SENSIBILIDAD CONTINUA: cuota de Espriella en el bloque der ---")
    print(f"  {'π_E':>5s}  {'Paloma':>8s}  {'Espriella':>10s}  {'Cepeda':>8s}  {'P(Pal top2)':>12s}  {'P(Esp top2)':>12s}")

    pi_E_grid = np.linspace(0.0, 0.85, 18)
    sens_data = []
    for pi_E in pi_E_grid:
        pi_P = max(0.92 - pi_E, 0.05)
        pi_O = max(1.0 - pi_E - pi_P, 0.0)
        # Renormalizar
        s = pi_P + pi_E + pi_O
        cw = {"paloma": pi_P/s, "espriella": pi_E/s, "otros": pi_O/s}
        pi_new = escenario_canibalizacion(pi_elec, cw, bloque="der")

        cep_idx = CANDS.index("cepeda")
        pal_idx = CANDS.index("paloma")
        esp_idx = CANDS.index("espriella")
        top2 = np.argsort(-pi_new, axis=1)[:, :2]
        p_pal_t2 = (top2 == pal_idx).any(axis=1).mean() * 100
        p_esp_t2 = (top2 == esp_idx).any(axis=1).mean() * 100

        sens_data.append({
            "pi_E": pi_E,
            "cep": pi_new[:, cep_idx].mean() * 100,
            "pal": pi_new[:, pal_idx].mean() * 100,
            "esp": pi_new[:, esp_idx].mean() * 100,
            "p_pal_t2": p_pal_t2,
            "p_esp_t2": p_esp_t2,
        })
        if abs(pi_E - round(pi_E * 10) / 10) < 0.01:
            print(f"  {pi_E:.2f}     {sens_data[-1]['pal']:>6.1f}%   {sens_data[-1]['esp']:>8.1f}%   "
                  f"{sens_data[-1]['cep']:>6.1f}%   {p_pal_t2:>10.1f}%   {p_esp_t2:>10.1f}%")

    sens_df = pd.DataFrame(sens_data)

    # ============================================================
    # FIGURAS
    # ============================================================
    fig_dir = ROOT / "figuras"
    fig_dir.mkdir(exist_ok=True)
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    # Fig 7: Comparación de escenarios
    fig, ax = plt.subplots(figsize=(13, 6))
    nombres = list(resultados_escenarios.keys())
    x = np.arange(len(nombres))
    width = 0.27
    cep_vals = [r["cep_mean"] for r in resultados_escenarios.values()]
    pal_vals = [r["pal_mean"] for r in resultados_escenarios.values()]
    esp_vals = [r["esp_mean"] for r in resultados_escenarios.values()]
    ax.bar(x - width, cep_vals, width, label="Cepeda", color=COLORES["cepeda"], edgecolor="white")
    ax.bar(x, pal_vals, width, label="Paloma Valencia", color=COLORES["paloma"], edgecolor="white")
    ax.bar(x + width, esp_vals, width, label="De la Espriella", color=COLORES["espriella"], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(nombres, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Share posterior medio (%)")
    ax.axhline(50, color="black", ls="--", lw=0.7, alpha=0.5)
    ax.set_title("Modelo 4: escenarios de canibalización dentro del bloque der.\n"
                 "Cepeda es invariante (por construcción del nested logit).\n"
                 "Paloma y Espriella se reasignan según la cuota dentro del bloque.",
                 fontsize=11)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig7_escenarios_canibalizacion.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Fig 8: Sensibilidad continua
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(sens_df["pi_E"]*100, sens_df["pal"], "-o", color=COLORES["paloma"], label="Paloma", lw=2, markersize=6)
    axes[0].plot(sens_df["pi_E"]*100, sens_df["esp"], "-s", color=COLORES["espriella"], label="Espriella", lw=2, markersize=6)
    axes[0].plot(sens_df["pi_E"]*100, sens_df["cep"], "--", color=COLORES["cepeda"], label="Cepeda", lw=2)
    axes[0].set_xlabel("Cuota de Espriella en el bloque der (%)")
    axes[0].set_ylabel("Share en 1ª vuelta (%)")
    axes[0].set_title("Sensibilidad: shares vs. cuota intra-bloque")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(sens_df["pi_E"]*100, sens_df["p_pal_t2"], "-o", color=COLORES["paloma"], label="P(Paloma top 2)", lw=2)
    axes[1].plot(sens_df["pi_E"]*100, sens_df["p_esp_t2"], "-s", color=COLORES["espriella"], label="P(Espriella top 2)", lw=2)
    axes[1].set_xlabel("Cuota de Espriella en el bloque der (%)")
    axes[1].set_ylabel("P(top 2) (%)")
    axes[1].set_title("Sensibilidad: P(top 2) en 1ª vuelta")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle("Modelo 4 corregido: análisis continuo de canibalización", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig8_sensibilidad_canibalizacion.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nFiguras guardadas: figuras/fig7_escenarios_canibalizacion.png")
    print(f"                   figuras/fig8_sensibilidad_canibalizacion.png")


if __name__ == "__main__":
    main()
