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
from ecoli.analysis.single.voronoi_plot_function_copy import COMPARTMENT_COLOR_MAP


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


    # t=10 min (600 s) gives proteins time to be expressed; use last timepoint as final
    t10_idx = int((voronoi_data["time"] - 600.0).abs().arg_min())

    dictionaries = []
    for i in [t10_idx, -1]:
        raw = {
            'extracellular': (extracellular, {'Flgk': FlgK, 'FlgL': FlgL, 'FliC': FliC, 'FliD': FliD}),
            'periplasm':     (periplasm,     {'FliE': FliE}),
            'cytosol':       (cytosol,       {'FliJ': FliJ, 'Flil': Flil, 'FliH': FliH, 'FlgE': FlgE}),
            'outer_membrane':(outer_mem,     {'FLGG': FLGG}),
            'projection':    (projection,    {'FLGB': FLGB, 'FLGC': FLGC, 'FLGF': FLGF,
                                              'FLGH': FLGH_RING, 'FLGI': FLGI_RING,
                                              'FliQ': FliQ, 'FliO': FliO, 'FliL': FliL}),
            'membrane':      (membrane,      {'FLIN': FLIN_SWITCH}),
            'inner_membrane':(inner_mem,     {'FLGF': FLGF_RING, 'FLIG': FLIG_SWITCH,
                                              'FLIM': FLIM_SWITCH, 'FlhB': FlhB, 'FlhA': FlhA,
                                              'FliR': FliR, 'FliP': FliP,
                                              'MotA': MotA, 'MotB': MotB}),
            'flagellum':     (flagellum,     {}),
        }

        compartments = {}
        for compart_id, (total_series, proteins) in raw.items():
            total = float(total_series[i])
            sub = {k: float(v[i]) for k, v in proteins.items() if float(v[i]) > 0}
            remaining = total - sum(sub.values())
            if remaining > 0:
                sub[compart_id] = remaining
            if sub:
                compartments[compart_id] = sub

        dictionaries.append(compartments)


    #indexing to get the first and second elements appended
    initial_dict = dictionaries[0]
    final_dict = dictionaries[1]

    COMPARTMENT_COLOR_MAP.clear()
    COMPARTMENT_COLOR_MAP.update({
        'extracellular':  [1.0, 0.506, 0.016],   # orange
        'periplasm':      [0.463, 0.361, 0.620],  # purple
        'cytosol':        [0.498, 0.725, 0.357],  # green
        'outer_membrane': [0.729, 0.459, 0.341],  # brown
        'projection':     [0.937, 0.616, 0.851],  # pink
        'membrane':       [0.863, 0.255, 0.282],  # red
        'inner_membrane': [0.314, 0.655, 0.769],  # light blue
        'flagellum':      [0.20, 0.45, 0.85],     # dark blue
    })


    vm = VoronoiMaster()
    vm.plot(
        [[initial_dict, final_dict]],
        title=[["Biomass components (t=10 min)", "Final biomass components"]],
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

