# eleccol2026 — predicción presidencial Colombia 2026

**Iván Ramiro Pinzón**
Universidad Externado de Colombia · 30 de abril de 2026 (31 días para 1ª vuelta)

---

## Resumen

A 31 días de la primera vuelta presidencial estimo la probabilidad de que Iván Cepeda gane la presidencia integrando cuatro fuentes de información: encuestas, voto al Senado, mercado predictivo Polymarket y la lógica de canibalización dentro de bloques ideológicos. El resultado: **P(Cepeda presidente) ≈ 41 %**, contra el 89 % que sugiere leer la última encuesta de Invamer aisladamente.

| Candidato | P(presidente) |
|---|---:|
| Iván Cepeda | **41.1 %** |
| Abelardo De la Espriella | 37.7 % |
| Paloma Valencia | 21.0 % |

**Lectura cualitativa:** Cepeda es favorito individual y casi seguro pasa a 2ª vuelta (P = 98 %), pero pierde en media en ambas segundas vueltas posibles (47.5 % vs 52.5 % contra Espriella; 46.3 % vs 53.7 % contra Paloma). El bloque opositor en conjunto (Espriella + Paloma = 59 %) es favorito para la presidencia. La elección se decide en quién, dentro del bloque opositor, llega a 2ª vuelta.

El documento completo con metodología, resultados, validación y limitaciones está en [`documento/articulo_eleccol2026.pdf`](documento/articulo_eleccol2026.pdf).

---

## Datos

Doce encuestas balanceadas por casa encuestadora (2–3 mediciones por casa, así el modelo puede separar ruido muestral de sesgo metodológico):

| Casa | Mediciones | Período | Método |
|---|---:|---|---|
| Invamer | 2 | feb · abr | presencial |
| GAD3 | 3 | feb · mar · abr | telefónica |
| AtlasIntel | 3 | mar · abr · abr | digital RDR |
| CNC | 1 | mar | presencial |
| Guarumo | 3 | feb · mar · abr | presencial |

Más voto al Senado del 8-mar-2026 (19.6 M votos reales), consultas interpartidistas y precios de Polymarket (USD 4.8 M de volumen).

---

## Estructura del repositorio

```
eleccol2026/
├── README.md
├── requirements.txt
├── data/
│   ├── encuestas.csv               ← 12 encuestas balanceadas
│   ├── senado_2026.csv             ← voto al Senado por partido
│   ├── consultas_2026.csv          ← consultas interpartidistas 8-mar
│   ├── polymarket.csv              ← precios mercado predictivo
│   ├── historico_calibracion.csv   ← elección 2022 para backtest
│   └── posterior_modelo1.pkl       ← se genera al correr modelo1
├── src/
│   ├── data_loader.py              ← carga unificada con CANDS canónicos
│   └── modelo_unificado.py         ← especificación bayesiana en PyMC
├── scripts/
│   ├── modelo1_encuestas.py        ← capa de encuestas (state-space + DM)
│   ├── modelo2_revealed.py         ← capa de comportamiento revelado
│   ├── modelo3_transferencia.py    ← capa de segunda vuelta
│   ├── modelo4_canibalizacion.py   ← capa de descomposición nested
│   └── modelo_unificado_figuras.py ← figuras de síntesis
├── tests/
│   └── backtest_2022.py            ← validación sobre primera vuelta 2022
├── notebook/
│   └── modelos_eleccol2026.ipynb   ← ejecutable end-to-end
├── documento/
│   ├── articulo_eleccol2026.tex    ← paper completo
│   └── articulo_eleccol2026.pdf    ← versión compilada (9 páginas)
└── figuras/                         ← fig1–fig13 generadas por los scripts
```

---

## Modelo

Estructura bayesiana jerárquica con cuatro capas. Cada capa responde una pregunta distinta y todas comparten el mismo posterior sobre la intención latente $\pi$.

| Capa | Pregunta | Mecanismo |
|---|---|---|
| Modelo 1 | ¿Cuánto vale cada encuesta? | Random walk ALR + house effects ZeroSumNormal + Dirichlet-Multinomial con $\alpha_0$ por casa |
| Modelo 2 | ¿Qué dice el comportamiento revelado? | Mapeo Senado→bloques con Dirichlet por partido; Polymarket como diagnóstico de coherencia |
| Modelo 3 | ¿Cómo se transfiere el voto a 2ª vuelta? | Matriz $T$ con prior por bloque ideológico, calibrada con 2022 |
| Modelo 4 | ¿Cómo se canibalizan los candidatos? | Descomposición $\pi_k = P(b) \cdot \pi^{within}_{k\|b}$ aplicada al posterior |

**Diagnóstico:** $\hat{R} = 1.004$, ESS bulk min = 1{,}035 (NUTS, 2 cadenas, 1{,}000 tune + 700 draws en 68 s sin BLAS).

**Backtest 2022:** recupera correctamente a Petro (40.32 % real, 38.69 % modelo) y Fico (23.91 % real, 26.82 % modelo). Falla con Hernández (28.15 % real, 11.63 % modelo) por 16.5 pp — fenómeno populista tardío que ningún agregador a 32 días puede anticipar. La limitación se documenta honestamente en el paper.

---

## Reproducibilidad

```bash
pip install -r requirements.txt

# orden recomendado: modelo1 genera el .pkl que reusan los demás
PYTHONPATH=. python scripts/modelo1_encuestas.py
PYTHONPATH=. python scripts/modelo2_revealed.py
PYTHONPATH=. python scripts/modelo3_transferencia.py
PYTHONPATH=. python scripts/modelo4_canibalizacion.py
PYTHONPATH=. python scripts/modelo_unificado_figuras.py

# validación
PYTHONPATH=. python tests/backtest_2022.py
```

Las figuras se guardan en `figuras/`. El PDF del paper está pre-compilado en `documento/`.

---

## Licencia

MIT.
