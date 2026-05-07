"""
================================================================================
  MODELO 3 — TRANSFERENCIAS EN SEGUNDA VUELTA (versión corregida)
================================================================================

Reemplazo de scripts/modelo3_transferencia.py original. Ver /AUDITORIA.md
"Modelo 3" para los fallos del original.

Cambios principales:

  1. NO se hard-codea SHARE_1V ni P_PAR_ESPRIELLA. Ambos vienen del POSTERIOR
     conjunto del Modelo 1, vía data/posterior_modelo1.pkl. Esto preserva
     toda la incertidumbre.

  2. La matriz de transferencia se construye desde priors EXPLÍCITOS
     anclados en (a) bloque ideológico de origen, (b) 2ª preferencia
     Invamer agregada, (c) caso histórico Petro-Hernández 2022.
     Cada celda tiene un Dirichlet con concentración 100 (no 50 inventado).

  3. El cómputo de Cepeda en 2ª vuelta NO usa cociente de variables
     aleatorias. Trabaja directamente sobre conteos (Cepeda + rival) y
     reporta el share sobre votos VÁLIDOS de manera coherente.

  4. P(Cepeda presidente) se computa propagando incertidumbre del modelo
     1 hasta el final (sin truncar a medias puntuales).

  5. Se reconoce explícitamente que la matriz de transferencia NO ESTÁ
     IDENTIFICADA con los datos del repo. Se reporta SENSIBILIDAD a
     priors alternativos.

Ejecución:
    PYTHONPATH=. python scripts/modelo3_transferencia.py
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

from src.data_loader import (
    CANDS, BLOQUES,
    SEGUNDA_PREFERENCIA_INVAMER_AGREGADA,
    CARA_A_CARA_INVAMER_ABR2026,
)
from src.modelo_unificado import (
    simular_segunda_vuelta,
    construir_prior_transferencia,
    calcular_p_presidente,
)


def main():
    base = ROOT
    print("=" * 78)
    print(" MODELO 3 — segunda vuelta con propagación correcta de incertidumbre")
    print("=" * 78)

    # Cargar posterior del modelo 1
    pkl_path = base / "data" / "posterior_modelo1.pkl"
    if not pkl_path.exists():
        print(f"\nERROR: {pkl_path} no existe. Corre primero scripts/modelo1_encuestas.py")
        sys.exit(1)
    with open(pkl_path, "rb") as f:
        saved = pickle.load(f)
    idata = saved["idata"]
    pi = idata.posterior["pi"].values
    pi_elec = pi[..., -1, :].reshape(-1, len(CANDS))
    n_samples = pi_elec.shape[0]
    print(f"\nPosterior cargado: {n_samples} muestras del modelo 1.")

    # ============================================================
    # 1. Análisis de pares posibles en 2ª vuelta
    # ============================================================
    cepeda_idx = CANDS.index("cepeda")
    top1 = pi_elec.argmax(axis=1)
    top2 = np.argsort(-pi_elec, axis=1)[:, 1]
    gana_1v = pi_elec[np.arange(n_samples), top1] > 0.5
    sims_2v = ~gana_1v
    print(f"\n  P(2ª vuelta) = {sims_2v.mean()*100:.1f}%")

    # Identificar el rival cuando Cepeda está en el par
    cepeda_en_par = (top1 == cepeda_idx) | (top2 == cepeda_idx)
    rival = np.where(top1 == cepeda_idx, top2, top1)

    print(f"\nDistribución de rivales de Cepeda en 2ª vuelta:")
    for k, c in enumerate(CANDS):
        if c == "cepeda":
            continue
        p_par = float(((sims_2v & cepeda_en_par) & (rival == k)).mean())
        if p_par > 0.001:
            print(f"  vs {c:>10s}  P = {p_par*100:5.1f}%")

    # ============================================================
    # 2. Cara a cara directos vs. modelo
    # ============================================================
    print(f"\n--- COMPARACIÓN: cara a cara Invamer abr-2026 vs. modelo ---")
    print(f"  {'Rival':<12s}  {'Invamer (cep%/riv%)':<22s} {'Modelo (cep%/riv%, IC90%)':<35s}")
    rng = np.random.default_rng(2026)
    for rival_name, (cep_inv, riv_inv) in CARA_A_CARA_INVAMER_ABR2026.items():
        # Filtrar muestras donde el rival real coincide y simular 2v.
        mask_rival = sims_2v & cepeda_en_par & (rival == CANDS.index(rival_name))
        if mask_rival.sum() == 0:
            print(f"  {rival_name:<12s}  {cep_inv:.1f}% / {riv_inv:.1f}%        — par no observado en posterior —")
            continue
        cep_share = simular_segunda_vuelta(
            pi_elec[mask_rival], rival=rival_name, n_sim_per_post=20, rng=rng)
        m = cep_share.mean() * 100
        lo = np.percentile(cep_share, 5) * 100
        hi = np.percentile(cep_share, 95) * 100
        riv_m = 100 - m
        riv_lo = 100 - hi
        riv_hi = 100 - lo
        print(f"  {rival_name:<12s}  "
              f"{cep_inv:.1f}% / {riv_inv:.1f}%       "
              f"{m:.1f}% / {riv_m:.1f}%  [{lo:.1f}, {hi:.1f}] / [{riv_lo:.1f}, {riv_hi:.1f}]")

    print("\nObservación: la pregunta de cara a cara directo refleja LA OPINIÓN")
    print("DECLARADA HOY de un cara a cara HIPOTÉTICO. La transferencia REAL")
    print("(modelo 3 corregido) suma efectos de campaña entre 1ª y 2ª vuelta.")
    print("Si el modelo difiere mucho del cara a cara, sospechar de:")
    print("  - matriz de transferencia mal calibrada (revisar priors),")
    print("  - cara a cara con muestreo distinto al modelo 1 (sin re-pesar).")

    # ============================================================
    # 3. P(Cepeda presidente) con propagación completa
    # ============================================================
    class R: pass
    res = R(); res.idata = idata
    p_pres = calcular_p_presidente(res)
    print(f"\n--- PROBABILIDAD DE GANAR LA PRESIDENCIA (1ª + 2ª vuelta) ---")
    for c in sorted(p_pres, key=lambda x: -p_pres[x]):
        if p_pres[c] > 0.001:
            print(f"  {c:>10s}  {p_pres[c]*100:5.1f}%")

    # ============================================================
    # 4. Análisis de sensibilidad: ¿qué pasa si las transferencias son
    #    distintas?
    # ============================================================
    print(f"\n--- SENSIBILIDAD A LA MATRIZ DE TRANSFERENCIA (paloma → espriella en 2v vs Espriella) ---")
    print(f"  {'transfer P→E':<14s}  {'Cepeda 2v media':>18s}  {'P(Cep gana 2v)':>16s}")

    # Tomar muestras del par "Cepeda vs Espriella"
    mask_esp = sims_2v & cepeda_en_par & (rival == CANDS.index("espriella"))
    if mask_esp.sum() > 100:
        for transfer_p_to_e in [0.55, 0.65, 0.72, 0.85]:
            transfer_p_to_c = 0.05
            transfer_p_to_a = 1 - transfer_p_to_e - transfer_p_to_c
            # Override del prior solo para Paloma → (cep, esp, abst)
            from src.modelo_unificado import construir_prior_transferencia
            prior = construir_prior_transferencia(rival="espriella")
            prior.alpha["paloma"] = np.array([transfer_p_to_c, transfer_p_to_e, transfer_p_to_a]) * 100

            # Simular manualmente con el prior modificado
            from src.data_loader import CANDS as CANDS_LOC
            cep_idx = CANDS_LOC.index("cepeda")
            esp_idx = CANDS_LOC.index("espriella")
            n_per = 20
            n_total = mask_esp.sum() * n_per
            pi_rep = np.repeat(pi_elec[mask_esp], n_per, axis=0)
            L_cep = 1.0 / (1.0 + np.exp(-(np.log(0.94 / 0.06) + rng.normal(0, 0.3, n_total))))
            L_esp = 1.0 / (1.0 + np.exp(-(np.log(0.92 / 0.08) + rng.normal(0, 0.3, n_total))))

            cep_2v = L_cep * pi_rep[:, cep_idx]
            esp_2v = L_esp * pi_rep[:, esp_idx]

            transfer_samples = {}
            for origen, alpha in prior.alpha.items():
                transfer_samples[origen] = rng.dirichlet(alpha, size=n_total)
            for origen, tmat in transfer_samples.items():
                s_orig = pi_rep[:, CANDS_LOC.index(origen)]
                cep_2v += tmat[:, 0] * s_orig
                esp_2v += tmat[:, 1] * s_orig

            cep_share = cep_2v / (cep_2v + esp_2v) * 100
            print(f"  {transfer_p_to_e:.2f}            {cep_share.mean():>14.1f}%      {(cep_share>50).mean()*100:>13.1f}%")

    # ============================================================
    # FIGURAS
    # ============================================================
    fig_dir = ROOT / "figuras"
    fig_dir.mkdir(exist_ok=True)
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    # Fig 6: distribución 2v en escenarios
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for i, rival_name in enumerate(["espriella", "paloma"]):
        mask_r = sims_2v & cepeda_en_par & (rival == CANDS.index(rival_name))
        if mask_r.sum() > 50:
            cep_share = simular_segunda_vuelta(
                pi_elec[mask_r], rival=rival_name, n_sim_per_post=20, rng=rng)
            axes[i].hist(cep_share * 100, bins=80, color="#E63946", alpha=0.7,
                         density=True, edgecolor="white", linewidth=0.3)
            axes[i].axvline(50, color="black", ls="--", lw=1.2)
            m = cep_share.mean() * 100
            p_gana = (cep_share > 0.5).mean() * 100
            axes[i].set_xlabel("% de Cepeda en 2ª vuelta (votos válidos)")
            axes[i].set_ylabel("Densidad posterior")
            axes[i].set_title(f"Cepeda vs {rival_name.title()}\n"
                              f"media={m:.1f}%, P(Cepeda gana)={p_gana:.1f}%")
    fig.suptitle("Modelo 3 corregido: 2ª vuelta con propagación de incertidumbre\n"
                 "(condicional al par; las muestras del posterior del modelo 1 se filtran)",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig6_segunda_vuelta_correcta.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nFigura guardada: figuras/fig6_segunda_vuelta_correcta.png")


if __name__ == "__main__":
    main()
