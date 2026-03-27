from typing import Any
import os

from duckdb import DuckDBPyConnection
import polars as pl
import numpy as np
import matplotlib.pyplot as plt

from ecoli.library.parquet_emitter import read_stacked_columns
from ecoli.analysis.single.voronoi_plot_function_copy import VoronoiMaster, COLORS




def plot(
    params: dict[str, Any],
    conn: DuckDBPyConnection,
    history_sql: str,
    config_sql: str,
    success_sql: str,
    sim_data_paths: dict[str, dict[int, str]],
    validation_data_paths: list[str],
    outdir: str,
    variant_metadata: dict[str, dict[int, Any]],
    variant_names: dict[str, str],
):
    voronoi_columns = [
        "listeners__mass__extracellular_mass",
        "listeners__mass__periplasm_mass",
        "listeners__mass__cytosol_mass",
        "listeners__mass__pilus_mass",
        "listeners__mass__outer_membrane_mass",
        "listeners__mass__projection_mass",
        "listeners__mass__membrane_mass",
        "listeners__mass__inner_membrane_mass",
        "listeners__mass__flagellum_mass",
    ]

    voronoi_data = pl.DataFrame(
        read_stacked_columns(history_sql, voronoi_columns, conn=conn)
    )

    def safe(x):
        x = float(x)
        return x if x > 1e-20 else 1e-20

    dictionaries = []
    for i in [0, -1]:
        compartments = {
            'extracellular': safe(voronoi_data["listeners__mass__extracellular_mass"][i]),
            'periplasm':     safe(voronoi_data["listeners__mass__periplasm_mass"][i]),
            'cytosol':       safe(voronoi_data["listeners__mass__cytosol_mass"][i]),
            'pilus':         safe(voronoi_data["listeners__mass__pilus_mass"][i]),
            'outer_membrane':safe(voronoi_data["listeners__mass__outer_membrane_mass"][i]),
            'projection':    safe(voronoi_data["listeners__mass__projection_mass"][i]),
            'membrane':      safe(voronoi_data["listeners__mass__membrane_mass"][i]),
            'inner_membrane':safe(voronoi_data["listeners__mass__inner_membrane_mass"][i]),
            'flagellum':     safe(voronoi_data["listeners__mass__flagellum_mass"][i]),
        }
        dictionaries.append(compartments)

    initial_dict = dictionaries[0]
    final_dict = dictionaries[1]

    # Set custom colors, one per compartment in dictionary order
    COLORS.clear()
    COLORS.extend([
        [1.0, 0.506, 0.016],  # extracellular  - orange
        [0.463, 0.361, 0.620],  # periplasm       - purple
        [0.498, 0.725, 0.357],  # cytosol         - green
        [1.0, 1.0, 0.329],  # pilus           - yellow
        [0.729, 0.459, 0.341],  # outer_membrane  - brown
        [0.937, 0.616, 0.851],  # projection      - pink/magenta
        [0.863, 0.255, 0.282],  # membrane        - yellow
        [0.314, 0.655, 0.769],  # inner_membrane  - red
        [0.20, 0.45, 0.85],  # flagellum       - blue (new)
    ])


    vm = VoronoiMaster()
    vm.plot(
        [[initial_dict, final_dict]],
        title=[["Initial compartment masses", "Final compartment masses"]],
        ax_shape=(1, 2),
        chained=False,
        font_size=3,
        verbose=True,
    )

    plotOutFileName = "compartment_masses_voronoi"
    full_path = os.path.join(outdir, f"{plotOutFileName}.png")
    plt.savefig(full_path, dpi=600)
    print(f"\nSaved plot to:\n {full_path}\n")
    plt.close()