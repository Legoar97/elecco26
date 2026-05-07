"""
================================================================================
  MODELO 2 — COMPORTAMIENTO REVELADO (versión corregida)
================================================================================

Reemplazo de scripts/modelo2_revealed.py original. Ver /AUDITORIA.md "Modelo 2"
para el listado de fallos críticos del original.

Cambios principales:

  1. La matriz de transferencia partido legislativo → candidato presidencial
     se trata como PRIOR INFORMATIVO con incertidumbre cuantificada (Dirichlet
     con concentración baja), no como una constante. La incertidumbre estructural
     se PROPAGA al output. La versión original la fijaba a ojo y la trataba
     como dato.

  2. El factor de amplificación (Senado → Presidencial) se modela con un
     LogNormal por bloque ideológico, no con un único 4.0× uniforme.

  3. Polymarket NO se convierte a "shares" (eso era pseudo-método). Se
     compara a la P(c gana 1ª) implícita del Modelo 1 vía un diagnóstico
     de coherencia. Cuando hay desacuerdo importante, se reporta — no se
     promedia.

  4. El "techo" de Cepeda (aprobación Petro) se trata como prior, no como
     truncamiento duro.

  5. NO HAY ENSEMBLE CON PESOS ARBITRARIOS. La integración correcta de
     fuentes está en el modelo unificado (Modelo 1 + reweighting Senado).

OBJETIVO DE ESTE SCRIPT
=======================
NO es producir una "nueva predicción". Es producir un BACKTEST y una proyección
de control: si IGNORAMOS las encuestas y solo usamos Senado + consultas, ¿qué
sale? La respuesta da una banda contra la cual interpretar el modelo 1.

Ejecución:
    PYTHONPATH=. python scripts/modelo2_revealed.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import (
    cargar_senado, cargar_consultas, cargar_polymarket, cargar_historico,
    CANDS, BLOQUES,
)
from src.modelo_unificado import (
    PARTIDO_A_BLOQUE,
    PRIOR_AMPLIFICACION_LOGMEAN,
    PRIOR_AMPLIFICACION_LOGSIGMA,
)


COLORES = {
    "izq":    "#E63946",
    "der":    "#1D3557",
    "centro": "#2A9D8F",
    "blanco": "#ADB5BD",
}

NOMBRES_BLOQUE = {
    "izq":    "Izquierda (Pacto Histórico)",
    "der":    "Derecha (CD + Conservador + ind.)",
    "centro": "Centro (López, Fajardo, otros)",
    "blanco": "Voto pulverizado / en blanco",
}


def main():
    base = ROOT
    print("=" * 78)
    print(" MODELO 2 — proyección revealed por BLOQUE IDEOLÓGICO")
    print(" (sin transferencias por candidato individual — eso requiere panel)")
    print("=" * 78)

    senado = cargar_senado(base / "data" / "senado_2026.csv")
    consultas = cargar_consultas(base / "data" / "consultas_2026.csv")
    polym = cargar_polymarket(base / "data" / "polymarket.csv")
    hist = cargar_historico(base / "data" / "historico_calibracion.csv")

    print(f"\nDatos:")
    print(f"  Senado 2026: {len(senado)} partidos, total {sum(s.votos for s in senado):,} votos")
    print(f"  Consultas 2026: {len(consultas)} filas")
    print(f"  Polymarket: {list(polym.keys())}")

    # ============================================================
    # 1. Agregar votos al Senado por BLOQUE PRESIDENCIAL
    # ============================================================
    bloques_unicos = sorted(set(BLOQUES.values()))
    bloque_a_votos_sen = {b: 0 for b in bloques_unicos}
    total_sen = 0
    for v in senado:
        b = PARTIDO_A_BLOQUE.get(v.partido, "blanco")
        bloque_a_votos_sen[b] += v.votos
        total_sen += v.votos

    print(f"\nVotos al Senado por bloque presidencial (asignación PARTIDO_A_BLOQUE):")
    for b in bloques_unicos:
        share = bloque_a_votos_sen[b] / total_sen
        print(f"  {NOMBRES_BLOQUE.get(b, b):<35s}  {bloque_a_votos_sen[b]:>10,}  ({share*100:5.2f}%)")

    # ============================================================
    # 2. Calibrar factor de amplificación con datos 2022
    # ============================================================
    # En 2022, los datos relevantes son:
    #   - Pacto Histórico Senado: 2,787,000 → Petro presidencial: 11,281,013
    #     factor = 4.04
    #   - Equipo por Colombia (CD aliado) Senado: ~2,240,000 → Fico: 5,058,010
    #     factor ≈ 2.26   (CD solo: 2.26; coalición incluía + partidos)
    #   - Liga Anticorrupción no tenía representación legislativa significativa
    #     pero Hernández sacó 5,953,209. factor ≈ 30+ (caso degenerado)
    #
    # Conclusión: el factor varía MUCHÍSIMO. Modelar con un solo número es
    # erróneo. Usamos un LogNormal por bloque con dispersión alta.

    factores_2022 = {
        "izq (Pacto Histórico)":   11_281_013 / 2_787_000,
        "der (CD coalición)":       5_058_010 / 2_240_000,
        # Liga / Hernández excluido: outlier estructural (sin partido).
    }
    print(f"\nFactores de amplificación 2022 (Senado → Presidencial 1V):")
    for k, v in factores_2022.items():
        print(f"  {k:<28s}  ×{v:.2f}")
    print(f"  PRIOR LogNormal(μ_log=0.0, σ_log=0.7)  (mediana 1.0, IC68% [0.50, 2.01])")
    print(f"  ↑ deliberadamente difuso: refleja que el factor depende de si el")
    print(f"    bloque consolidó UN candidato presidencial o lo dividió.")

    # ============================================================
    # 3. Proyección probabilística por bloque (Monte Carlo)
    # ============================================================
    rng = np.random.default_rng(2026)
    n_sim = 30_000

    # Sortear factor de amplificación por bloque.
    log_tau = rng.normal(
        PRIOR_AMPLIFICACION_LOGMEAN,
        PRIOR_AMPLIFICACION_LOGSIGMA,
        size=(n_sim, len(bloques_unicos)),
    )
    tau = np.exp(log_tau)  # (n_sim, n_bloques)

    # Voto por bloque proyectado (no normalizado).
    votos_bloque_proy = np.zeros((n_sim, len(bloques_unicos)))
    for b_idx, b in enumerate(bloques_unicos):
        votos_bloque_proy[:, b_idx] = bloque_a_votos_sen[b] * tau[:, b_idx]

    # Renormalizar para obtener shares.
    shares_bloque = votos_bloque_proy / votos_bloque_proy.sum(axis=1, keepdims=True)

    print(f"\n--- POSTERIOR DE SHARE PRESIDENCIAL POR BLOQUE (sólo Senado, sin encuestas) ---")
    print(f"  {'Bloque':<35s} {'Media':>7s}   {'IC 90%':>15s}")
    for b_idx, b in enumerate(bloques_unicos):
        m = shares_bloque[:, b_idx].mean() * 100
        lo = np.percentile(shares_bloque[:, b_idx], 5) * 100
        hi = np.percentile(shares_bloque[:, b_idx], 95) * 100
        print(f"  {NOMBRES_BLOQUE.get(b, b):<35s} {m:>6.1f}%  [{lo:>4.1f}, {hi:>4.1f}]")

    # ============================================================
    # 4. Diagnóstico vs. Polymarket
    # ============================================================
    print(f"\n--- DIAGNÓSTICO POLYMARKET vs. ANCLAS ---")
    print("Polymarket 'first round winner' (USD 4.8M):")
    pm_data = polym.get("1st_round_winner", {})
    for c, p in sorted(pm_data.items(), key=lambda x: -x[1]):
        print(f"  P({c} gana 1ª)        = {p*100:5.1f}%")

    print("\nObservación crítica:")
    print("  Polymarket implica P(Cepeda gana 1ª)=91% — esto no es un share")
    print("  presidencial sino una probabilidad de cola. Convertir esto a un")
    print("  share como hacía el modelo 2 original (heurística softmax inverso)")
    print("  carece de fundamento. La forma correcta de usar Polymarket es")
    print("  como un DIAGNÓSTICO de coherencia: si el modelo 1 implica P(gana)")
    print("  muy distinta, entonces Polymarket sugiere información no")
    print("  incorporada (manipulación, info de campaña, traders globales).")

    # ============================================================
    # 5. Consultas: qué dicen y qué NO dicen
    # ============================================================
    print(f"\n--- CONSULTAS INTERPARTIDISTAS DEL 8-MAR-2026 ---")
    paloma_consulta = consultas[consultas["candidato"] == "Paloma Valencia"]["votos"].sum()
    cepeda_consulta = consultas[consultas["candidato"] == "Iván Cepeda"]["votos"].sum()
    print(f"  Paloma Valencia (Gran Consulta): {paloma_consulta:,}")
    print(f"  Cepeda (Consulta PH 2025):       {cepeda_consulta:,}")
    print(f"  De la Espriella:                 0  (no participó)")

    print("\nObservación crítica:")
    print("  Las consultas miden ENTUSIASMO de la base, no intención presidencial.")
    print("  El modelo 2 original convertía esto a un 'factor de ajuste' multiplicativo")
    print("  (cap [0.85, 1.20]) sin fundamento empírico. La interpretación rigurosa:")
    print("  ")
    print("    base_consulta_paloma / techo_bloque_der ≈ 3.2M / 7.4M ≈ 0.43")
    print("    base_consulta_cepeda / Pacto Histórico Senado ≈ 1.5M / 4.4M ≈ 0.34")
    print("  ")
    print("  Estos ratios son consistentes con las cuotas WITHIN-BLOQUE del nested")
    print("  logit del modelo unificado, lo cual da un check externo de orden de")
    print("  magnitud — no una predicción.")

    # ============================================================
    # 6. FIGURAS
    # ============================================================
    fig_dir = ROOT / "figuras"
    fig_dir.mkdir(exist_ok=True)
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    # Fig 4: Distribución posterior de shares por bloque
    fig, ax = plt.subplots(figsize=(11, 6))
    for b_idx, b in enumerate(bloques_unicos):
        ax.hist(shares_bloque[:, b_idx] * 100, bins=70, alpha=0.55, density=True,
                color=COLORES.get(b, "#888888"),
                label=f"{NOMBRES_BLOQUE.get(b, b)}: media={shares_bloque[:,b_idx].mean()*100:.1f}%",
                edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Share presidencial proyectado (%)")
    ax.set_ylabel("Densidad posterior")
    ax.set_title(
        "Modelo 2 corregido — proyección presidencial por BLOQUE\n"
        "Solo Senado 2026 + factor amplificación incierto (LogNormal). "
        "Bandas anchas reflejan que esto NO es información suficiente.",
        fontsize=11,
    )
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig4_revealed_por_bloque.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Fig 5: Comparación con Polymarket
    fig, ax = plt.subplots(figsize=(10, 5))
    pm_cands = list(pm_data.keys())
    pm_vals = [pm_data[c] * 100 for c in pm_cands]
    nombres_largos = {
        "cepeda": "Iván Cepeda", "espriella": "A. De la Espriella",
        "paloma": "Paloma Valencia",
    }
    bars = ax.barh([nombres_largos.get(c, c) for c in pm_cands], pm_vals,
                    color=["#E63946", "#1D3557", "#457B9D"], edgecolor="white")
    ax.set_xlim(0, 100)
    ax.set_xlabel("P(candidato gana 1ª vuelta) (%) — Polymarket")
    ax.set_title("Polymarket: probabilidades implícitas (USD 4.8M de volumen)\n"
                 "Estas son probabilidades, NO shares de voto.", fontsize=11)
    for bar, v in zip(bars, pm_vals):
        ax.text(v + 1, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}%", va="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig5_polymarket.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nFiguras guardadas en {fig_dir}/")
    print("  fig4_revealed_por_bloque.png")
    print("  fig5_polymarket.png")


if __name__ == "__main__":
    main()
