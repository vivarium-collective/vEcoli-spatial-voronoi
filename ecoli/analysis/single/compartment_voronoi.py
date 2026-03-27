#This is Arnab's mass_fraction_voroni.py analysis just modified
from typing import Any
import os

from duckdb import DuckDBPyConnection
import polars as pl
import numpy as np
import matplotlib.pyplot as plt

from ecoli.library.parquet_emitter import read_stacked_columns, field_metadata
from ecoli.library.sim_data import LoadSimData


from wholecell.utils import units, voronoi_plot_main
from wholecell.analysis.analysis_tools import exportFigure
from wholecell.utils.voronoi_plot_main import VoronoiMaster, COLORS


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
    with open(os.path.join(outdir, "history_sql.txt"), "w") as f:
        f.write(history_sql)

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
        "bulk",
    ]

    voronoi_data = pl.DataFrame(
        read_stacked_columns(history_sql, voronoi_columns, conn=conn)
    )

    bulk_molecule_counts = np.stack(voronoi_data["bulk"])

    bulk_molecule_ids = field_metadata(conn, config_sql, "bulk")

    bulk_molecule_idx = {name: idx for idx, name in enumerate(bulk_molecule_ids)}

    exp_id = list(sim_data_paths.keys())[0]

    sim_data_path = list(sim_data_paths[exp_id].values())[0]

    sim_data = LoadSimData(sim_data_path).sim_data

    nAvogadro = sim_data.constants.n_avogadro

    # # # def find_mass_molecule_group(group_id):
    # # #     temp_ids = getattr(sim_data.molecule_groups, str(group_id))
    # # #     temp_indexes = np.array([bulk_molecule_idx[temp] for temp in temp_ids])
    # # #     temp_counts = bulk_molecule_counts[:, temp_indexes]
    # # #     temp_mw = sim_data.getter.get_masses(temp_ids)
    # # #     return (units.dot(temp_counts, temp_mw) / nAvogadro).asNumber(units.fg)
    # # #
    # # def find_mass_single_molecule(molecule_id):
    # #     temp_id = getattr(sim_data.molecule_ids, str(molecule_id))
    # #     temp_index = bulk_molecule_idx[temp_id]
    # #     temp_counts = bulk_molecule_counts[:, temp_index]
    # #     temp_mw = sim_data.getter.get_mass(temp_id)
    # #     return (units.multiply(temp_counts, temp_mw) / nAvogadro).asNumber(units.fg)

    extracellular = voronoi_data["listeners__mass__extracellular_mass"]
    periplasm = voronoi_data["listeners__mass__periplasm_mass"]
    cytosol = voronoi_data["listeners__mass__cytosol_mass"]
    pilus = voronoi_data["listeners__mass__pilus_mass"]
    outer_mem = voronoi_data["listeners__mass__outer_membrane_mass"]
    projection = voronoi_data["listeners__mass__projection_mass"]
    membrane = voronoi_data["listeners__mass__membrane_mass"]
    inner_membrane = voronoi_data["listeners__mass__inner_membrane_mass"]
    flagellum = voronoi_data["listeners__mass__flagellum_mass"]

    dic_initial = {
        "extracellular": float(extracellular[0]),
        "periplasm": float(periplasm[0]),
        "cytosol": float(cytosol[0]),
        "pilus": float(pilus[0]),
        "outer_membrane": float(outer_mem[0]),
        "projection": float(projection[0]),
        "membrane": float(membrane[0]),
        "inner_membrane": float(inner_membrane[0]),
        "flagellum": float(flagellum[0]),
    }
    dic_final = {
        "extracellular": float(extracellular[-1]),
        "periplasm": float(periplasm[-1]),
        "cytosol": float(cytosol[-1]),
        "pilus": float(pilus[-1]),
        "outer_membrane": float(outer_mem[-1]),
        "projection": float(projection[-1]),
        "membrane": float(membrane[-1]),
        "inner_membrane": float(inner_membrane[-1]),
        "flagellum": float(flagellum[-1]),
    }


    # Replace the entire COLORS list
    # Base colors (one per compartment) - this is to keep the colors for each compartment the same
    # # Base pastel colors (one per compartment)
    # base_colors = [
    #     np.array([255, 200, 120]) / 255,  # extracellular - light beige - light ornage now
    #     np.array([204, 187, 222]) / 255,  # periplasm - soft green - lavender now
    #     np.array([168, 216, 185]) / 255,  # cytosol - soft blue - soft green now
    #     np.array([248, 231, 165]) / 255,  # pilus - soft pink - pale yellow now
    #     np.array([181, 146, 104]) / 255,  # outer_membrane - pale yellow - light brown now
    #     np.array([244, 184, 208]) / 255,  # projection - powder blue - soft pink now
    #     np.array([255, 204, 153]) / 255,  # membrane - peach
    #     np.array([173, 197, 232]) / 255,  # inner_membrane - lavender - soft blue now
    #     np.array([242, 169, 159]) / 255,  # flagellum - muted coral
    # ]

    # # # Base pastel colors (one per compartment) - original colors
    # base_colors = [
    #     np.array([200, 210, 195]) / 255,  # extracellular - light beige
    #     np.array([168, 216, 185]) / 255,  # periplasm - soft green
    #     np.array([173, 197, 232]) / 255,  # cytosol - soft blue
    #     np.array([244, 184, 208]) / 255,  # pilus - soft pink
    #     np.array([248, 231, 165]) / 255,  # outer_membrane - pale yellow
    #     np.array([176, 224, 230]) / 255,  # projection - powder blue
    #     np.array([255, 204, 153]) / 255,  # membrane - peach
    #     np.array([204, 187, 222]) / 255,  # inner_membrane - lavender
    #     np.array([242, 169, 159]) / 255,  # flagellum - muted coral
    # ]


    base_colors = [
        [1, 0, 0],  # red
        [0, 1, 0],  # green
        [0, 0, 1],  # blue
        [1, 1, 0],  # yellow
        [1, 0, 1],  # magenta
        [0, 1, 1],  # cyan
        [0.5, 0, 0],  # dark red
        [0, 0.5, 0],  # dark green
        [0, 0, 0.5],  # dark blue
    ]
    COLORS[:] = base_colors * 2

    #haveing an issue with plotting --> the first color is going all over the plot


    # Repeat for initial + final so colors match
    #voronoi_plot_main.COLORS = base_colors * 2

    vm = VoronoiMaster()
    vm.plot(
        [[dic_initial, dic_final]],
        title=[["Initial biomass components", "Final biomass components"]],
        ax_shape=(1, 2),
        chained=False,
        font_size=4,
    )

    plotOutFileName = "compartments_mass_fractions_voronoi"

    # Save figure in main workspace (optional fallback)
    plt.savefig(f"{plotOutFileName}.png", dpi=200)

    # Save figure into analysis output directory
    full_path = os.path.join(outdir, f"{plotOutFileName}.png")
    plt.savefig(full_path, dpi=200)

    print(f"\nSaved Voronoi biomass plot to:\n {full_path}\n")

    plt.close()








