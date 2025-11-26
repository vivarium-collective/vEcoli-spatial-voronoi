#This is Arnab's mass_fraction_voroni.py analysis just modified
from typing import Any
import os

from duckdb import DuckDBPyConnection
import polars as pl
import numpy as np
import matplotlib.pyplot as plt

from ecoli.library.parquet_emitter import read_stacked_columns, field_metadata
from ecoli.library.sim_data import LoadSimData
from ecoli.processes.antibiotics.antibiotic_transport_steady_state import TEMPERATURE

from wholecell.utils import units
from wholecell.analysis.analysis_tools import exportFigure
from wholecell.utils.voronoi_plot_main import VoronoiMaster


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

    # # def find_mass_molecule_group(group_id):
    # #     temp_ids = getattr(sim_data.molecule_groups, str(group_id))
    # #     temp_indexes = np.array([bulk_molecule_idx[temp] for temp in temp_ids])
    # #     temp_counts = bulk_molecule_counts[:, temp_indexes]
    # #     temp_mw = sim_data.getter.get_masses(temp_ids)
    # #     return (units.dot(temp_counts, temp_mw) / nAvogadro).asNumber(units.fg)
    # #
    # def find_mass_single_molecule(molecule_id):
    #     temp_id = getattr(sim_data.molecule_ids, str(molecule_id))
    #     temp_index = bulk_molecule_idx[temp_id]
    #     temp_counts = bulk_molecule_counts[:, temp_index]
    #     temp_mw = sim_data.getter.get_mass(temp_id)
    #     return (units.multiply(temp_counts, temp_mw) / nAvogadro).asNumber(units.fg)

    extracellular = voronoi_data["listeners__mass__extracellular_mass"]
    periplasm = voronoi_data["listeners__mass__periplasm_mass"]
    cytosol = voronoi_data["listeners__mass__cytosol_mass"]
    pilus = voronoi_data["listeners__mass__pilus_mass"]
    outer_mem = voronoi_data["listeners__mass__outer_membrane_mass"]
    projection = voronoi_data["listeners__mass__projection_mass"]
    membrane = voronoi_data["listeners__mass__membrane_mass"]
    inner_mem = voronoi_data["listeners__mass__inner_membrane_mass"]

    dic_initial = {
            "extracellular": extracellular[0],
            "periplasm": periplasm[0],
            "cytosol": cytosol[0],
            "pilus": pilus[0],
            "outer_membrane": outer_mem[0],
            "projection": projection[0],
            "membrane": membrane[0],
            "inner_mem": inner_mem[0],
    }
    dic_final = {
        "extracellular": extracellular[-1],
        "periplasm": periplasm[-1],
        "cytosol": cytosol[-1],
        "pilus": pilus[-1],
        "outer_membrane": outer_mem[-1],
        "projection": projection[-1],
        "membrane": membrane[-1],
        "inner_mem": inner_mem[-1],
    }

    vm = VoronoiMaster()
    vm.plot(
        [[dic_initial, dic_final]],
        title=[["Initial biomass components", "Final biomass components"]],
        ax_shape=(1, 2),
        chained=True,
    )

    plotOutFileName = "compartments_mass_fractions_voronoi"

    # Save figure in main workspace (optional fallback)
    plt.savefig(f"{plotOutFileName}.png", dpi=200)

    # Save figure into analysis output directory
    full_path = os.path.join(outdir, f"{plotOutFileName}.png")
    plt.savefig(full_path, dpi=200)

    print(f"\nSaved Voronoi biomass plot to:\n {full_path}\n")

    plt.close()