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

from wholecell.utils.voronoi_plot_main import COLORS
import matplotlib.pyplot as plt
import numpy as np


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

    #TODO: make a function that is able to extract the ID and MW based off monomer ID
    def find_protein_mass(monomer_id):
        monomer_data = sim_data.process.translation.monomer_data
        monomer_weights = dict(zip(monomer_data["id"], monomer_data["mw"]))
        mw_monomer = monomer_weights[monomer_id]
        temp_index = bulk_molecule_idx[monomer_id]
        temp_counts = bulk_molecule_counts[:, temp_index]
        return (units.multiply(temp_counts, mw_monomer) / nAvogadro).asNumber(units.fg)

    def find_mass_group(monomer_ids):
        total = np.zeros(len(bulk_molecule_counts))
        for monomer in monomer_ids:
            total += find_protein_mass(monomer)
        return total

    flagella_monomers = [
        "EG10308-MONOMER[e]",
        "EG10309-MONOMER[c]",
        "EG10310-MONOMER[p]",
        "EG10311-MONOMER[o]",
        "EG10312-MONOMER[c]",
        "EG10313-MONOMER[l]",
        "EG10314-MONOMER[l]",
        "EG10315-MONOMER[l]",
     ]


    flagella_masses = find_mass_group(flagella_monomers)

    #Function because have such tiny numbers and voronoi cannot go 0 or under
        #Voronoi function cannot hangle zero or negative values
        #this function converts the value to a float, returns the original value unless extremely tiny
        #if less than 1e20, force it to be 1e20
    def safe(x):
        x = float(x)
        return x if x > 1e-20 else 1e-20

#listener/compartment masses
    extracellular = voronoi_data["listeners__mass__extracellular_mass"]
    periplasm = voronoi_data["listeners__mass__periplasm_mass"]
    cytosol = voronoi_data["listeners__mass__cytosol_mass"]
    pilus = voronoi_data["listeners__mass__pilus_mass"]
    outer_mem = voronoi_data["listeners__mass__outer_membrane_mass"]
    projection = voronoi_data["listeners__mass__projection_mass"]
    membrane = voronoi_data["listeners__mass__membrane_mass"]
    inner_mem = voronoi_data["listeners__mass__inner_membrane_mass"]


#fim Subunits
    FimA = find_protein_mass("EG10308-MONOMER[e]")
    FimB = find_protein_mass("EG10309-MONOMER[c]")
    FimC = find_protein_mass("EG10310-MONOMER[p]")
    FimD = find_protein_mass("EG10311-MONOMER[o]")
    FimE = find_protein_mass("EG10312-MONOMER[c]")
    FimF = find_protein_mass("EG10313-MONOMER[l]")
    FimG = find_protein_mass("EG10314-MONOMER[l]")
    FimH = find_protein_mass("EG10315-MONOMER[l]")


    dic_initial = {
            "extracell": {
                'extracellular': safe(extracellular[0]),
                'FimA': safe(FimA[0]),
            },
            "peri": {
               'periplasm':safe(periplasm[0]),
                'FimC': safe(FimC[0]),
                },
            "cyt": {
                'cytosol': safe(cytosol[0]),
                'FimB': safe(FimB[0]),
                'FimE':safe(FimE[0]),
            },
            "pil": {
               'pilus': safe(pilus[0]),
                'FimF':safe(FimF[0]),
                'FimG':safe(FimG[0]),
                'FimH': safe(FimH[0]),
            },
            "outer_mem": {
                'outer_membrane':safe(outer_mem[0]),
                'FimD':safe(FimD[0]),
            },
            'projection': safe(projection[0]),
            'membrane':safe(membrane[0]),
            'inner_membrane':safe(inner_mem[0]),
    }
    dic_final = {
        "extracell": {
            'extracellular': safe(extracellular[-1]),
            'FimA': safe(FimA[-1]),
        },
        "peri": {
            'periplasm': safe(periplasm[-1]),
            'FimC': safe(FimC[-1]),
        },
        "cyt": {
            'cytosol': safe(cytosol[-1]),
            'FimB': safe(FimB[-1]),
            'FimE': safe(FimE[-1]),
        },
        "pil": {
            'pilus': safe(pilus[-1]),
            'FimF': safe(FimF[-1]),
            'FimG': safe(FimG[-1]),
            'FimH': safe(FimH[-1]),
        },
        "outer_mem": {
            'outer_membrane': safe(outer_mem[-1]),
            'FimD': safe(FimD[-1]),
        },
        'projection': safe(projection[-1]),
        'membrane': safe(membrane[-1]),
        'inner_membrane': safe(inner_mem[-1]),
    }


    extra_colors = [list(plt.cm.tab20(i % 20))[:3] for i in range(200)] + \
                   [list(np.random.rand(3)) for _ in range(200)]

    COLORS.extend(extra_colors)

    vm = VoronoiMaster()
    vm.plot(
        [[dic_initial, dic_final]],
        title=[["Initial biomass components", "Final biomass components"]],
        ax_shape=(1, 2),
        chained=True,
        font_size=4,
    )

    plotOutFileName = "fim_compartment_mass_fractions_voronoi"

    # Save figure in main workspace (optional fallback)
    plt.savefig(f"{plotOutFileName}.png", dpi=200)

    # Save figure into analysis output directory
    full_path = os.path.join(outdir, f"{plotOutFileName}.png")
    plt.savefig(full_path, dpi=200)

    print(f"\nSaved Voronoi biomass plot to:\n {full_path}\n")

    plt.close()
