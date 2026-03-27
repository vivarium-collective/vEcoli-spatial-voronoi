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
import matplotlib.pyplot as plt


# Replaced voronoi function because wanted to edit it without changing master
from ecoli.analysis.single.voronoi_plot_function_copy import VoronoiMaster
from ecoli.analysis.single.voronoi_plot_function_copy import COLORS


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

    def find_protein_mass(monomer_id):
        monomer_data = sim_data.process.translation.monomer_data
        monomer_weights = dict(zip(monomer_data["id"], monomer_data["mw"]))
        mw_monomer = monomer_weights[monomer_id]
        temp_index = bulk_molecule_idx[monomer_id]
        temp_counts = bulk_molecule_counts[:, temp_index]
        return (units.multiply(temp_counts, mw_monomer) / nAvogadro).asNumber(units.fg)
    #
    # # def find_mass_group(monomer_ids):
    # #     total = np.zeros(len(bulk_molecule_counts))
    # #     for monomer in monomer_ids:
    # #         total += find_protein_mass(monomer)
    # #     return total
    #
    # flagella_monomers = [
    #     "FLGB-FLAGELLAR-MOTOR-ROD-PROTEIN[j]",
    #     "FLGC-FLAGELLAR-MOTOR-ROD-PROTEIN[j]",
    #     "FLGF-FLAGELLAR-MOTOR-ROD-PROTEIN[j]",
    #     "FLGG-FLAGELLAR-MOTOR-ROD-PROTEIN[o]",
    #     "FLGH-FLAGELLAR-L-RING[j]",
    #     "FLGI-FLAGELLAR-P-RING[j]",
    #     "FLIF-FLAGELLAR-MS-RING[i]",
    #     "FLIG-FLAGELLAR-SWITCH-PROTEIN[i]",
    #     "FLIM-FLAGELLAR-C-RING-SWITCH[i]",
    #     "FLIN-FLAGELLAR-C-RING-SWITCH[m]",
    #     "G7028-MONOMER[i]",
    #     "G378-MONOMER[c]",
    #     "G377-MONOMER[c]",
    #     "G370-MONOMER[i]",
    #     "EG11977-MONOMER[i]",
    #     "EG11976-MONOMER[j]",
    #     "EG11975-MONOMER[i]",
    #     "EG11656-MONOMER[c]",
    #     "EG11224-MONOMER[j]",
    #    # "CPLX0-7451[j]",
    #     "MOTA-FLAGELLAR-MOTOR-STATOR-PROTEIN[i]",
    #     "MOTB-FLAGELLAR-MOTOR-STATOR-PROTEIN[i]",
    #     "EG11346-MONOMER[p]",
    #     "EG10322-MONOMER[j]",
    #     #"FLAGELLAR-MOTOR-COMPLEX[j]",
    #     "G361-MONOMER[c]",
    #     "EG11967-MONOMER[e]",
    #     "EG11545-MONOMER[e]",
    #     "EG10321-MONOMER[e]",
    #     "EG10841-MONOMER[e]",
    #     #"CPLX0-7452[j]" #flagellum
    # ]

    #flagella_masses = find_mass_group(flagella_monomers)


#Function because tiny numbers and Voronoi function cannot hangle zero or negative values
        #converts the value to a float, returns the original value unless extremely tiny
        #if less than 1e20, force it to be 1e20
    # def safe(x):
    #     x = float(x)
    #     return x if x > 1e-20 else 1e-20

        # Raise the safe() floor significantly - 1e-20 is dangerously small for Voronoi
    def safe(x):
        x = float(x)
        return x if x > 1e-6 else 1e-6  # or even 1e-3 depending on your mass scale

#listener/compartment masses
    extracellular = voronoi_data["listeners__mass__extracellular_mass"]
    periplasm = voronoi_data["listeners__mass__periplasm_mass"]
    cytosol = voronoi_data["listeners__mass__cytosol_mass"]
    pilus = voronoi_data["listeners__mass__pilus_mass"]
    outer_mem = voronoi_data["listeners__mass__outer_membrane_mass"]
    projection = voronoi_data["listeners__mass__projection_mass"]
    membrane = voronoi_data["listeners__mass__membrane_mass"]
    inner_mem = voronoi_data["listeners__mass__inner_membrane_mass"]
    flagellum = voronoi_data["listeners__mass__flagellum_mass"]


#Flagella Subunits
    FLGG = find_protein_mass("FLGG-FLAGELLAR-MOTOR-ROD-PROTEIN[o]")
    FLGB = find_protein_mass("FLGB-FLAGELLAR-MOTOR-ROD-PROTEIN[j]")
    FLGC = find_protein_mass("FLGC-FLAGELLAR-MOTOR-ROD-PROTEIN[j]")
    FLGF = find_protein_mass("FLGF-FLAGELLAR-MOTOR-ROD-PROTEIN[j]")
    FLGH_RING = find_protein_mass("FLGH-FLAGELLAR-L-RING[j]")
    FLGI_RING = find_protein_mass("FLGI-FLAGELLAR-P-RING[j]")

    FLGF_RING = find_protein_mass("FLIF-FLAGELLAR-MS-RING[i]")
    FLIG_SWITCH = find_protein_mass("FLIG-FLAGELLAR-SWITCH-PROTEIN[i]")
    FLIM_SWITCH = find_protein_mass("FLIM-FLAGELLAR-C-RING-SWITCH[i]")

    FLIN_SWITCH = find_protein_mass("FLIN-FLAGELLAR-C-RING-SWITCH[m]")
    FlhB = find_protein_mass("G7028-MONOMER[i]")

#cytosol flagella subunits
    FliJ = find_protein_mass("G378-MONOMER[c]")
    Flil = find_protein_mass("G377-MONOMER[c]")
    FliH = find_protein_mass("EG11656-MONOMER[c]")
    FlgE = find_protein_mass("G361-MONOMER[c]")

#inner membrane
    FlhA = find_protein_mass("G370-MONOMER[i]")
    FliR = find_protein_mass("EG11977-MONOMER[i]")
    FliP = find_protein_mass("EG11975-MONOMER[i]")
    MotA = find_protein_mass("MOTA-FLAGELLAR-MOTOR-STATOR-PROTEIN[i]")
    MotB = find_protein_mass("MOTB-FLAGELLAR-MOTOR-STATOR-PROTEIN[i]")

#projection
    FliQ = find_protein_mass("EG11976-MONOMER[j]")
    FliO = find_protein_mass("EG11224-MONOMER[j]")
   # Flg_Export_app = find_protein_mass("CPLX0-7451[j]")
    FliL = find_protein_mass("EG10322-MONOMER[j]")
  #  Flg_Motor = find_protein_mass("FLAGELLAR-MOTOR-COMPLEX[j]")
   # Flagellum = find_protein_mass("CPLX0-7452[j]")

#periplasm
    FliE = find_protein_mass("EG11346-MONOMER[p]")

#extracellular
    FlgK = find_protein_mass("EG11967-MONOMER[e]")
    FlgL = find_protein_mass("EG11545-MONOMER[e]")
    FliC = find_protein_mass("EG10321-MONOMER[e]")
    FliD = find_protein_mass("EG10841-MONOMER[e]")


    dictionaries = []
    for i in [0, -1]:
        compartments = {
            'extracellular': {
                'total': safe(extracellular[i]),
                'Flgk': safe(FlgK[i]),
                'FlgL': safe(FlgL[i]),
                'FliC': safe(FliC[i]),
                'FliD': safe(FliD[i]),
            },
            'flagellum': {
                'total': safe(flagellum[i]),
            },
            'periplasm': {
                'total': safe(periplasm[i]),
                'FliE': safe(FliE[i]),
            },
            'cytosol': {
                'total': safe(cytosol[i]),
                'FliJ': safe(FliJ[i]),
                'Flil': safe(Flil[i]),
                'FliH': safe(FliH[i]),
                'FlgE': safe(FlgE[i]),
            #    },
            # 'pilus': {
            #     'total': safe(pilus[i]), #NOTE: when this is not commented out, we get an error
            },
            'outer_membrane': {
                'total': safe(outer_mem[i]),
                'FLGG': safe(FLGG[i]),
            },
            'projection': {
                'total': safe(projection[i]),
                'FLGB': safe(FLGB[i]),
                'FLGC': safe(FLGC[i]),
                'FLGF': safe(FLGF[i]),
                'FLGH': safe(FLGH_RING[i]),
                'FLGI': safe(FLGI_RING[i]),
                'FliQ': safe(FliQ[i]),
                'FliO': safe(FliO[i]),
                #'Flg_export_app': safe(Flg_Export_app[i]),
                'FliL': safe(FliL[i]),
                #'Flg_Motor': safe(Flg_Motor[i]),
                #'Flagellum': safe(Flagellum[i]),
            },
            'membrane': {
                'total': safe(membrane[i]),
                'FLIN': safe(FLIN_SWITCH[i]),
            },
            'inner_membrane': {
                'total': safe(inner_mem[i]),
                'FLGF': safe(FLGF_RING[i]),
                'FLIG': safe(FLIG_SWITCH[i]),
                'FLIM': safe(FLIM_SWITCH[i]),
                'FlhB': safe(FlhB[i]),
                'FlhA': safe(FlhA[i]),
                'FliR': safe(FliR[i]),
                'FliP': safe(FliP[i]),
                'MotA': safe(MotA[i]),
                'MotB': safe(MotB[i]),
            }
        }

        # for compart_id, compart_dict in compartments.items():
        #     total = compart_dict.pop('total')
        #     used = sum(list(compart_dict.values()))
        #     remaining = total - used
        #     compart_dict[compart_id] = remaining



        # Guard against negative remainders when computing the "other" mass
        for compart_id, compart_dict in compartments.items():
            total = compart_dict.pop('total')
            used = sum(list(compart_dict.values()))
            remaining = total - used
            # If remaining is negative or near-zero, clamp it
            compart_dict[compart_id] = safe(remaining)  # use safe() here too!

        dictionaries.append(compartments)


    #indexing to get the first and second elements appended
    initial_dict = dictionaries[0]
    final_dict = dictionaries[1]

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

    # Extend with extra colors for sub-compartment polygons (layered voronoi)
    extra_colors = [list(plt.cm.tab20(i % 20))[:3] for i in range(200)] + \
                   [list(np.random.rand(3)) for _ in range(200)]
    COLORS.extend(extra_colors)


    vm = VoronoiMaster()
    vm.plot(
        [[initial_dict, final_dict]],
        title=[["Initial biomass components", "Final biomass components"]],
        ax_shape=(1, 2),
        chained=False,
        font_size=2,
    )

    plotOutFileName = "func_compartment_mass_fractions_voronoi"

    # Save figure in main workspace (optional fallback)
    plt.savefig(f"{plotOutFileName}.png", dpi=600)

    # Save figure into analysis output directory
    full_path = os.path.join(outdir, f"{plotOutFileName}.png")
    plt.savefig(full_path, dpi=600)

    print(f"\nSaved Voronoi biomass plot to:\n {full_path}\n")

    plt.close()

