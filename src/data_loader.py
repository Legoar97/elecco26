"""
data_loader.py
==============

Carga unificada de las fuentes de datos de la elección presidencial Colombia 2026.

Toda la información de entrada vive en archivos CSV/JSON en /data. Este módulo
se encarga de:

  - Leer y validar las encuestas, votación al Senado, consultas, Polymarket
    y datos históricos.
  - Convertir nombres de candidatos a un único alias canónico (`CANDS`).
  - Devolver matrices/diccionarios listos para entrar al modelo unificado.

Cualquier transformación de modelado (logit, suma-cero, simplex) se hace en
`src/modelo_unificado.py`. Aquí solo limpiamos.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


# Lista canónica y orden fijo de candidatos. Cualquier cambio requiere
# regenerar las matrices de transferencia. NO ALTERAR sin propagar.
CANDS: List[str] = [
    "cepeda",            # Iván Cepeda — Pacto Histórico (izquierda gobiernista)
    "espriella",         # Abelardo De la Espriella — independiente derecha
    "paloma",            # Paloma Valencia — Centro Democrático (derecha tradicional)
    "lopez",             # Claudia López — independiente centro
    "fajardo",           # Sergio Fajardo — Coalición Centro Esperanza
    "otros",             # Botero, Uribe Londoño, candidatos pequeños agregados
    "blanco",            # Voto en blanco
]

# Bloques ideológicos para el modelo nested logit. Cada candidato pertenece
# exactamente a un bloque. Los bloques no son una propiedad observable: son
# un supuesto del modelador, justificado por adhesiones de partidos
# legislativos y posicionamiento histórico.
BLOQUES: Dict[str, str] = {
    "cepeda":    "izq",
    "espriella": "der",
    "paloma":    "der",
    "lopez":     "centro",
    "fajardo":   "centro",
    "otros":     "der",      # Botero y M. Uribe Londoño son derecha
    "blanco":    "blanco",
}

# Mapeo de los nombres en los CSV originales a alias canónicos. Se usa
# `aliasar()` para todos los DataFrames.
NOMBRE_A_ALIAS: Dict[str, str] = {
    "Iván Cepeda":              "cepeda",
    "Cepeda":                   "cepeda",
    "Abelardo De la Espriella": "espriella",
    "De la Espriella":          "espriella",
    "Paloma Valencia":          "paloma",
    "Claudia López":            "lopez",
    "Sergio Fajardo":           "fajardo",
    "Santiago Botero":          "otros",
    "Miguel Uribe Londoño":     "otros",
    "Otros":                    "otros",
    "Voto en blanco":           "blanco",
}

# Encuestadoras conocidas. El nombre se usa como índice para los house effects.
ENCUESTADORAS: List[str] = ["Invamer", "GAD3", "CNC", "AtlasIntel"]


@dataclass
class Encuesta:
    """Encuesta individual: una observación multinomial en tiempo $t_i$."""
    pollster: str
    fecha: date
    n: int
    metodo: str
    margen_error: float
    # Vector de shares en orden CANDS (proporción, suma 1).
    shares: np.ndarray
    # Conteos enteros aproximados a partir de n y shares (para likelihood
    # multinomial).
    counts: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if abs(self.shares.sum() - 1.0) > 0.005:
            raise ValueError(f"shares no suman 1 para {self.pollster} {self.fecha}: "
                             f"suma = {self.shares.sum():.4f}")
        self.shares = self.shares / self.shares.sum()
        # Conteos: n × share, redondeado y ajustado para que sume n.
        cuentas = np.round(self.shares * self.n).astype(int)
        diff = self.n - cuentas.sum()
        if diff != 0:
            # Ajustar al candidato con mayor share residual.
            idx = int(np.argmax(np.abs(self.shares * self.n - cuentas)))
            cuentas[idx] += diff
        self.counts = cuentas


def aliasar(nombre: str) -> str:
    """Convierte cualquier variante de nombre al alias canónico."""
    if nombre in NOMBRE_A_ALIAS:
        return NOMBRE_A_ALIAS[nombre]
    raise KeyError(f"Nombre de candidato no reconocido: '{nombre}'")


def cargar_encuestas(ruta: Path | str) -> List[Encuesta]:
    """Carga las encuestas desde encuestas.csv.

    El CSV tiene columnas: pollster, fecha_inicio, fecha_fin, fecha_mediana,
    n, metodo, margen_error, y los shares por candidato en columnas
    independientes. Aquí agregamos `santiago_botero + miguel_uribe_londono`
    en `otros` para acoplar al esquema CANDS.
    """
    df = pd.read_csv(ruta, parse_dates=["fecha_mediana"])
    encuestas: List[Encuesta] = []

    col_a_alias = {
        "cepeda":              "cepeda",
        "de_la_espriella":     "espriella",
        "paloma_valencia":     "paloma",
        "claudia_lopez":       "lopez",
        "sergio_fajardo":      "fajardo",
        "santiago_botero":     "otros",  # se sumará
        "miguel_uribe_londono":"otros",  # se sumará
        "otros":               "otros",  # se sumará
        "voto_blanco":         "blanco",
    }

    for _, row in df.iterrows():
        agg: Dict[str, float] = {a: 0.0 for a in CANDS}
        for col, alias in col_a_alias.items():
            agg[alias] += float(row[col])
        # Pasar de % a proporción.
        total = sum(agg.values())
        shares = np.array([agg[a] for a in CANDS]) / total
        encuestas.append(Encuesta(
            pollster=str(row["pollster"]),
            fecha=row["fecha_mediana"].date(),
            n=int(row["n"]),
            metodo=str(row["metodo"]),
            margen_error=float(row["margen_error"]),
            shares=shares,
        ))
    return encuestas


@dataclass
class VotoSenado:
    """Una fila del DataFrame de Senado 2026."""
    partido: str
    votos: int
    porcentaje: float
    bloque: str  # 'izquierda', 'derecha_uribista', 'centro_pragmatico', etc.


def cargar_senado(ruta: Path | str) -> List[VotoSenado]:
    df = pd.read_csv(ruta)
    out: List[VotoSenado] = []
    for _, r in df.iterrows():
        out.append(VotoSenado(
            partido=str(r["partido"]),
            votos=int(r["votos"]),
            porcentaje=float(r["porcentaje"]),
            bloque=str(r["bloque_ideologico"]),
        ))
    return out


def cargar_polymarket(ruta: Path | str) -> Dict[str, Dict[str, float]]:
    """Devuelve {mercado: {alias_candidato: prob_implicita}}."""
    df = pd.read_csv(ruta)
    out: Dict[str, Dict[str, float]] = {}
    for mercado, grupo in df.groupby("mercado"):
        out[mercado] = {
            aliasar(r["candidato"]): float(r["probabilidad_implicita"])
            for _, r in grupo.iterrows()
        }
    return out


def cargar_consultas(ruta: Path | str) -> pd.DataFrame:
    return pd.read_csv(ruta, parse_dates=["fecha"])


def cargar_historico(ruta: Path | str) -> pd.DataFrame:
    return pd.read_csv(ruta)


# ----------------------------------------------------------------------------
# Cara a cara de 2ª vuelta y segunda preferencia (datos auxiliares)
# ----------------------------------------------------------------------------
# Estos datos no están en el repo original como CSV pero son usados por los
# scripts. Los expongo aquí como diccionarios fijos con la fuente declarada.

# Invamer #21 (15-24 abr 2026), pregunta de cara a cara:
#   "Si la 2ª vuelta fuera entre Cepeda y X, ¿por quién votaría?"
# Devuelve (% Cepeda, % rival).
CARA_A_CARA_INVAMER_ABR2026: Dict[str, tuple[float, float]] = {
    "paloma":    (51.2, 46.6),
    "espriella": (54.6, 42.6),
    "fajardo":   (59.8, 36.4),
    "lopez":     (62.6, 31.6),
}

# Invamer #21, pregunta "si su candidato no avanza, ¿cuál es su segunda
# opción?". Es una distribución agregada (no por candidato de origen).
SEGUNDA_PREFERENCIA_INVAMER_AGREGADA: Dict[str, float] = {
    "cepeda":    0.267,
    "paloma":    0.251,
    "espriella": 0.198,
    "lopez":     0.058,
    "fajardo":   0.044,
    "otros":     0.092,
    "blanco":    0.090,
}


def fechas_calendario(encuestas: List[Encuesta], dia_eleccion: date) -> tuple[np.ndarray, np.ndarray]:
    """Construye un grid diario desde la encuesta más antigua hasta el día
    de la elección. Devuelve (array_de_fechas, indice_de_cada_encuesta_en_el_grid).

    Esto es necesario para el random walk del estado latente.
    """
    fechas_encuestas = [e.fecha for e in encuestas]
    inicio = min(fechas_encuestas)
    grid = pd.date_range(inicio, dia_eleccion, freq="D").date
    grid_arr = np.array(grid)
    # Indice de cada encuesta dentro del grid.
    idx = np.array([np.where(grid_arr == e.fecha)[0][0] for e in encuestas])
    return grid_arr, idx


def resumen_basico(encuestas: List[Encuesta]) -> pd.DataFrame:
    """Tabla legible para inspección."""
    rows = []
    for e in encuestas:
        rows.append({
            "pollster": e.pollster,
            "fecha": e.fecha,
            "n": e.n,
            "metodo": e.metodo,
            **{c: f"{e.shares[i]*100:.1f}%" for i, c in enumerate(CANDS)},
        })
    return pd.DataFrame(rows).sort_values("fecha")


if __name__ == "__main__":
    base = Path(__file__).parent.parent / "data"
    enc = cargar_encuestas(base / "encuestas.csv")
    print(f"Cargadas {len(enc)} encuestas:")
    print(resumen_basico(enc).to_string(index=False))
    sen = cargar_senado(base / "senado_2026.csv")
    print(f"\n{len(sen)} partidos en Senado 2026.")
    pm = cargar_polymarket(base / "polymarket.csv")
    print(f"\nMercados Polymarket: {list(pm.keys())}")
