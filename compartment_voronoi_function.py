#%%
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

    def find_protein_mass(monomer_id):
        monomer_data = sim_data.process.translation.monomer_data
        monomer_weights = dict(zip(monomer_data["id"], monomer_data["mw"]))
        mw_monomer = monomer_weights[monomer_id]
        temp_index = bulk_molecule_idx[monomer_id]
        temp_counts = bulk_molecule_counts[:, temp_index]
        return (units.multiply(temp_counts, mw_monomer) / nAvogadro).asNumber(units.fg)

    def find_molecule_mass(molecule_id):
        temp_id = getattr(sim_data.molecule_ids, str(molecule_id))
        temp_index = bulk_molecule_idx[temp_id]
        temp_counts = bulk_molecule_counts[:, temp_index]
        temp_mw = sim_data.getter.get_mass(temp_id)
        return (units.multiply(temp_counts, temp_mw) / nAvogadro).asNumber(units.fg)

#arnab mentioned getting mass from the bulk -
    def get_mass_from_bulk(sim_data):
        bulk_info = sim_data.internal_state.bulk_molecules.bulk_data
        return dict(zip(bulk_info["id"], bulk_info["mass"]))

#TODO: this function to get the mass of ids in bulk but not in monomer_data - arnab mentioned bulk has the mass
    # def find_mass_group(monomer_ids):
    #     temp_ids2 = getattr(sim_data.
    # #     total = np.zeros(len(bulk_molecule_counts))
    # #     for monomer in monomer_ids:
    # #         total += find_protein_mass(monomer)
    # #     return total

#TODO: there is a common names in sim data,
# make a function where you can type in the common name and it brings back the ID



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
    FLIF_RING = find_protein_mass("FLIF-FLAGELLAR-MS-RING[i]")
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

    # fim Subunits
    FimA = find_protein_mass("EG10308-MONOMER[e]")
    FimB = find_protein_mass("EG10309-MONOMER[c]")
    FimC = find_protein_mass("EG10310-MONOMER[p]")
    FimD = find_protein_mass("EG10311-MONOMER[o]")
    FimE = find_protein_mass("EG10312-MONOMER[c]")
    FimF = find_protein_mass("EG10313-MONOMER[l]")
    FimG = find_protein_mass("EG10314-MONOMER[l]")
    FimH = find_protein_mass("EG10315-MONOMER[l]")

# #LPS IDs
#     lps = find_molecule_mass("LPS")
#     #lps_A = find_protein_mass("CPD0-939[c]")
#     #lpxC = find_protein_mass("clpX[c]")
#     waaA = find_protein_mass("EG11351-MONOMER[c]") #protein involved in biosynthesis
#     #lps_c = find_protein_mass("CPD0-939[c]")
#     msbA = find_protein_mass("EG10613-MONOMER[m]") #transporter - flippase - C to P
#     waaC = find_protein_mass("EG11189-MONOMER[c]")
#     waaP = find_protein_mass("EG11340-MONOMER[c]")
#     waaF = find_protein_mass("EG12210-MONOMER[c]")
#     waaY = find_protein_mass("EG11425-MONOMER[i]")
#     waaG = find_protein_mass("EG11339-MONOMER[i]")
#     waaQ = find_protein_mass("EG11341-MONOMER[c]")
#     waaB = find_protein_mass("EG11351-MONOMER[c]")
#     waaO = find_protein_mass("EG11352-MONOMER[c]")
#     waaR = find_protein_mass("EG11353-MONOMER[i]") #also WaaJ as a synonym
#     waaU = find_protein_mass("EG11423-MONOMER[c]")
#
# #Curli Ids
#     #BAC OPERON TO MAKE THE FIBER
#     csgB = find_protein_mass("G6547-MONOMER[l]")
#     csgA = find_protein_mass("EG11489-MONOMER[e]")
#     csgC = find_protein_mass("G6548-MONOMER[p]")
#     #DEFG OPERON TO CONTROL AND SECRET SYSTEM
#     csgD = find_protein_mass("PD01379[i]") #has another id on ecocyc maybe it has been updated name
#     csgE = find_protein_mass("G6545-MONOMER[o]")
#     csgF = find_protein_mass("G6544-MONOMER[o]")
#     csgG = find_protein_mass("G6543-MONOMER[o]")
#
# #TODO: add in EPS components - PGA, cellulose, colanic acid

# PGA
#     pgaC = find_protein_mass("G6529-MONOMER[i]")
#     pgaD = find_protein_mass("G6528-MONOMER[i]")
#     pgaB = find_protein_mass("G6530-MONOMER[p]")
#     pgaA = find_protein_mass("G6531-MONOMER[o]")
#
# #cellulose
#     #bscA is the catalytic subunit in the bcs operon
#     celluloseA = find_protein_mass("EG12260-MONOMER[i]")
#     #bcsB is the cellulose synthase periplasmic subunit -- why is it in the inner membrane rn and not periplasm
#     # B is ESSENTIAL in strains able to form biofilms
#     celluloseB = find_protein_mass("EG12259-MONOMER[i]")
#     # bcsC is the outer membrane channel
#     celluloseC = find_protein_mass("EG12257-MONOMER[o]")
#     celluloseZ = find_protein_mass("EG12258-MONOMER[e]")
#
#     #bscEFG operon genes
#     # E identified as the c-di-GMP-binding protein
#     celluloseE = find_protein_mass("EG12263-MONOMER[c]")
#     celluloseF = find_protein_mass("EG12264-MONOMER[i]")
#     celluloseG = find_protein_mass("EG12265-MONOMER[i]")

# Colanic acid
#



    dictionaries = []
    for i in [0, -1]:
        compartments = {
            'extracellular': {
                'total': safe(extracellular[i]),
                'Flgk': safe(FlgK[i]),
                'FlgL': safe(FlgL[i]),
                'FliC': safe(FliC[i]),
                'FliD': safe(FliD[i]),
                'FimA': safe(FimA[i]),
                # 'CsgA': safe(csgA[i]),
                # 'BcsZ': safe(celluloseZ[i]),
            },
            'periplasm': {
                'total': safe(periplasm[i]),
                'FliE': safe(FliE[i]),
                'FimC': safe(FimC[i]),
                # 'CsgC': safe(csgC[i]),
                # 'PgaB': safe(pgaB[i]),
            },
            'cytosol': {
                'total': safe(cytosol[i]),
                'FliJ': safe(FliJ[i]),
                'Flil': safe(Flil[i]),
                'FliH': safe(FliH[i]),
                'FlgE': safe(FlgE[i]),
            #     'FimB': safe(FimB[i]),
            #     'FimE': safe(FimE[i]),
            #     'LPS': safe(lps[i]),
            #     'WaaA': safe(waaA[i]),
            #     'WaaC': safe(waaC[i]),
            #     'WaaP': safe(waaP[i]),
            #     'WaaF': safe(waaF[i]),
            #     'WaaQ': safe(waaQ[i]),
            #     'WaaB': safe(waaB[i]),
            #     'WaaO': safe(waaO[i]),
            #     'WaaU': safe(waaU[i]),
            #     'BcsE': safe(celluloseE[i]),
               },
            'pilus': {
                'total': safe(pilus[i]),
                'FimF': safe(FimF[i]),
                'FimG': safe(FimG[i]),
                'FimH': safe(FimH[i]),
                # 'CsgB': safe(csgB[i]),
            },
            'outer_membrane': {
                'total': safe(outer_mem[i]),
                'FLGG_ROD_Protein': safe(FLGG_ROD[i]),
                'FimD': safe(FimD[i]),
                # 'CsgE': safe(csgE[i]),
                # 'CsgF': safe(csgF[i]),
                # 'CsgG': safe(csgG[i]),
                # 'PgaA': safe(pgaA[i]),
                # 'BcsC': safe(celluloseC[i]),
            },
            'projection': {
                'total': safe(projection[i]),
                'FLGB_ROD': safe(FLGB_ROD[i]),
                'FLGC_ROD': safe(FLGC_ROD[i]),
                'FLGF_ROD': safe(FLGF_ROD[i]),
                'FLGH_RING': safe(FLGH_RING[i]),
                'FLGI_RING': safe(FLGI_RING[i]),
                'FliQ': safe(FliQ[i]),
                'FliO': safe(FliO[i]),
                #'Flg_export_app': safe(Flg_Export_app[i]),
                'FliL': safe(FliL[i]),
                #'Flg_Motor': safe(Flg_Motor[i]),
                #'Flagellum': safe(Flagellum[i]),
            },
            'membrane': {
                'total': safe(membrane[i]),
                'FLIN_SWITCH': safe(FLIN_SWITCH[i]),
                # 'MsbA': safe(msbA[i]),
            },
            'inner_membrane': {
                'total': safe(inner_mem[i]),
                'FLIF_RING': safe(FLIF_RING[i]),
                'FLIG_SWITCH': safe(FLIG_SWITCH[i]),
                'FLIM_SWITCH': safe(FLIM_SWITCH[i]),
                'FlhB': safe(FlhB[i]),
                'FlhA': safe(FlhA[i]),
                'FliR': safe(FliR[i]),
                'FliP': safe(FliP[i]),
                'MotA': safe(MotA[i]),
                'MotB': safe(MotB[i]),
                # 'CsgD': safe(csgD[i]),
                # 'WaaY': safe(waaY[i]),
                # 'WaaG': safe(waaG[i]),
                # 'WaaR': safe(waaR[i]),
                # 'PgaC': safe(pgaC[i]),
                # 'PgaD': safe(pgaD[i]),
                # 'BcsA': safe(celluloseA[i]),
                # 'BcsB': safe(celluloseB[i]),
                # 'BcsF': safe(celluloseF[i]),
                # 'BcsG': safe(celluloseG[i]),
            }
        }

        for compart_id, compart_dict in compartments.items():
            total = compart_dict.pop('total')
            used = sum(list(compart_dict.values()))
            remaining = total - used
            compart_dict[compart_id] = safe(remaining)

        dictionaries.append(compartments)


    #indexing to get the first and second elements appended
    initial_dict = dictionaries[0]
    final_dict = dictionaries[1]

    extra_colors = [list(plt.cm.tab20(i % 20))[:3] for i in range(200)] + \
                   [list(np.random.rand(3)) for _ in range(200)]

    COLORS.extend(extra_colors)

    vm = VoronoiMaster()
    vm.plot(
        [[initial_dict, final_dict]],
        title=[["Initial biomass components", "Final biomass components"]],
        ax_shape=(1, 2),
        chained=True,
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
