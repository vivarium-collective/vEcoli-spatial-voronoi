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
        "FLGB-FLAGELLAR-MOTOR-ROD-PROTEIN[j]",
        "FLGC-FLAGELLAR-MOTOR-ROD-PROTEIN[j]",
        "FLGF-FLAGELLAR-MOTOR-ROD-PROTEIN[j]",
        "FLGG-FLAGELLAR-MOTOR-ROD-PROTEIN[o]",
        "FLGH-FLAGELLAR-L-RING[j]",
        "FLGI-FLAGELLAR-P-RING[j]",
        "FLIF-FLAGELLAR-MS-RING[i]",
        "FLIG-FLAGELLAR-SWITCH-PROTEIN[i]",
        "FLIM-FLAGELLAR-C-RING-SWITCH[i]",
        "FLIN-FLAGELLAR-C-RING-SWITCH[m]",
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


#Flagella Subunits
    FLGG_ROD = find_protein_mass("FLGG-FLAGELLAR-MOTOR-ROD-PROTEIN[o]")

    FLGB_ROD = find_protein_mass("FLGB-FLAGELLAR-MOTOR-ROD-PROTEIN[j]")
    FLGC_ROD = find_protein_mass("FLGC-FLAGELLAR-MOTOR-ROD-PROTEIN[j]")
    FLGF_ROD = find_protein_mass("FLGF-FLAGELLAR-MOTOR-ROD-PROTEIN[j]")
    FLGH_RING = find_protein_mass("FLGH-FLAGELLAR-L-RING[j]")
    FLGI_RING = find_protein_mass("FLGI-FLAGELLAR-P-RING[j]")

    FLGF_RING = find_protein_mass("FLIF-FLAGELLAR-MS-RING[i]")
    FLIG_SWITCH = find_protein_mass("FLIG-FLAGELLAR-SWITCH-PROTEIN[i]")
    FLIM_SWITCH = find_protein_mass("FLIM-FLAGELLAR-C-RING-SWITCH[i]")

    FLIN_SWITCH = find_protein_mass("FLIN-FLAGELLAR-C-RING-SWITCH[m]")



    dic_initial = {
            "extracellular":safe(extracellular[0]),
            "periplasm":safe(periplasm[0]),
           # "cytosol":safe(cytosol[0]),
            "pilus":safe(pilus[0]),
            "outer_mem": {
                'outer_membrane':safe(outer_mem[0]),
                'FLGG_ROD_Protein':safe(FLGG_ROD[0]),
                'placeholder': safe(1e-50),
            },
            "proj": {
                'projection': safe(projection[0]),
                'FLGB_ROD':safe(FLGB_ROD[0]),
                'FLGC_ROD':safe(FLGC_ROD[0]),
                'FLGF_ROD':safe(FLGF_ROD[0]),
                'FLGH_RING':safe(FLGH_RING[0]),
                'FLGI_RING':safe(FLGI_RING[0]),
            },
            "mem": { #transmembrane/membrane embedded
                'membrane':safe(membrane[0]),
                'FLIN_SWITCH':safe(FLIN_SWITCH[0]),
                'placeholder': safe(1e-50),
            },
            "inner": {
                'inner_membrane':safe(inner_mem[0]),
                'FLGF_RING':safe(FLGF_RING[0]),
                'FLIG_SWITCH':safe(FLIG_SWITCH[0]),
                'FLIM_SWITCH':safe(FLIM_SWITCH[0]),
            }
    }
    dic_final = {
        "extracellular":safe(extracellular[-1]),
        "periplasm":safe(periplasm[-1]),
       # "cytosol":safe(cytosol[-1]),
        "pilus":safe(pilus[-1]),
        "outer_mem": {
            'outer_membrane':safe(outer_mem[-1]),
            'FLGG_ROD_Protein':safe(FLGG_ROD[-1]),
            'placeholder': safe(1e-50),
        },
        "proj": {
            'projection':safe(projection[-1]),
            'FLGB_ROD':safe(FLGB_ROD[-1]),
            'FLGC_ROD':safe(FLGC_ROD[-1]),
            'FLGF_ROD':safe(FLGF_ROD[-1]),
            'FLGH_RING':safe(FLGH_RING[-1]),
            'FLGI_RING':safe(FLGI_RING[-1]),
        },
        "mem": { #transmembrane/membrane embedded
            'membrane':safe(membrane[-1]),
            'FLIN_SWITCH':safe(FLIN_SWITCH[-1]),
            'placeholder': safe(1e-50),
            },
        "inner": {
            'inner_membrane':safe(inner_mem[-1]),
            'FLGF_RING':safe(FLGF_RING[-1]),
            'FLIG_SWITCH':safe(FLIG_SWITCH[-1]),
            'FLIM_SWITCH':safe(FLIM_SWITCH[-1]),
            }
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

    plotOutFileName = "fim_flagella_compartment_mass_fractions_voronoi"

    # Save figure in main workspace (optional fallback)
    plt.savefig(f"{plotOutFileName}.png", dpi=200)

    # Save figure into analysis output directory
    full_path = os.path.join(outdir, f"{plotOutFileName}.png")
    plt.savefig(full_path, dpi=200)

    print(f"\nSaved Voronoi biomass plot to:\n {full_path}\n")

    plt.close()
