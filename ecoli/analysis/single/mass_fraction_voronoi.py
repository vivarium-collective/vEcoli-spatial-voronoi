#MAYA Notes to understand script
#Loads simulation data from WCM through parquet history data,compute mass of different molecule types
#and creates Voronoi diagram viz of inital vs. final biomass




from typing import Any
import os

#DuckDB to query parquet files
from duckdb import DuckDBPyConnection

#Polars for fast dataframe handling
import polars as pl
import numpy as np
import matplotlib.pyplot as plt

from ecoli.library.parquet_emitter import read_stacked_columns, field_metadata
from ecoli.library.sim_data import LoadSimData

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
    #Saves SQL query to output folder
    with open(os.path.join(outdir, "history_sql.txt"), "w") as f:
        f.write(history_sql)

    #Columns to load from history parquet
    voronoi_columns = [
        "listeners__mass__rna_mass",
        "listeners__mass__protein_mass",
        "listeners__mass__tRna_mass",
        "listeners__mass__rRna_mass",
        "listeners__mass__mRna_mass",
        "listeners__mass__dna_mass",
        "listeners__mass__smallMolecule_mass",
        "bulk",
    ]


    #Read columns into polars df
    #read_stacked_columns - runs DuckDB query on simulation parquet files and returns a python
    #dictionary of lists so all mass are 1D overtime EXCEPT for bulk because contains the counts for every bulk molecule at every timepoint

    voronoi_data = pl.DataFrame(
        read_stacked_columns(history_sql, voronoi_columns, conn=conn)
    )

    #Extract bulk molecule counts (2D matrix)
    #np.stack - converts the list of lists into a np 2D array
    #only for bulk because it's a nested data column for many molecules and many times
    #RAW simulation output!!! does not have labels right now
    bulk_molecule_counts = np.stack(voronoi_data["bulk"])

    #field_stacked queries metadata from simulation schema stores in parquet config files
    #finds the names of every molecule whose count is included in bulk - need this to know which index belongs to which molecule
    #each string corresponds to the same index in the stacked count matrix
    bulk_molecule_ids = field_metadata(conn, config_sql, "bulk")

    #Makes a lookup dictionary after having the IDS
    bulk_molecule_idx = {name: idx for idx, name in enumerate(bulk_molecule_ids)}

    #Get the first experiment key
    exp_id = list(sim_data_paths.keys())[0]
    #Get the first value inside that experiment
    sim_data_path = list(sim_data_paths[exp_id].values())[0]
    #Load simulation metadata object - need to do these to get the MW, group definitions, AVO
    sim_data = LoadSimData(sim_data_path).sim_data
    nAvogadro = sim_data.constants.n_avogadro

    #This is a function to compute the group mass
    def find_mass_molecule_group(group_id):
        temp_ids = getattr(sim_data.molecule_groups, str(group_id)) #This is a predefined list in sim_data that contains all molecule IDs of that group, like lipids group is ALL Lipids
        temp_indexes = np.array([bulk_molecule_idx[temp] for temp in temp_ids]) #convert molecule names to indexes
        temp_counts = bulk_molecule_counts[:, temp_indexes] #extract counts for molecule Ids across all simulation timepoints
        temp_mw = sim_data.getter.get_masses(temp_ids)  #Gets the MW of each molecule in group

        #temp_counts is a 2D array with number of timepoints and number of molecules
        #temp_mw is a 1D array of molecular weights in g/mol
        #units.dot produces a single value per timepoint, converts g/mol x molecules into grams because:
        # grams = (molecules x g/mol) / molecules per mol --> as.Number then converts raw float into fg
        return (units.dot(temp_counts, temp_mw) / nAvogadro).asNumber(units.fg)

    #This is a function to compute the single-molecule mass - reference function above for comments this is just for single molcules
    def find_mass_single_molecule(molecule_id):
        temp_id = getattr(sim_data.molecule_ids, str(molecule_id))
        temp_index = bulk_molecule_idx[temp_id]
        temp_counts = bulk_molecule_counts[:, temp_index]
        temp_mw = sim_data.getter.get_mass(temp_id)
        return (units.multiply(temp_counts, temp_mw) / nAvogadro).asNumber(units.fg)

    lipid = find_mass_molecule_group("lipids")
    polyamines = find_mass_molecule_group("polyamines")
    lps = find_mass_single_molecule("LPS")
    murein = find_mass_single_molecule("murein")
    glycogen = find_mass_single_molecule("glycogen")

    protein = voronoi_data["listeners__mass__protein_mass"]
    rna = voronoi_data["listeners__mass__rna_mass"]
    tRna = voronoi_data["listeners__mass__tRna_mass"]
    rRna = voronoi_data["listeners__mass__rRna_mass"]
    mRna = voronoi_data["listeners__mass__mRna_mass"]
    miscRna = rna - (tRna + rRna + mRna)
    dna = voronoi_data["listeners__mass__dna_mass"]
    smallMolecules = voronoi_data["listeners__mass__smallMolecule_mass"]
    metabolites = smallMolecules - (lipid + lps + murein + polyamines + glycogen)

    dic_initial = {
        "nucleic_acid": {
            "DNA": dna[0],
            "mRNA": mRna[0],
            "miscRNA": miscRna[0],
            "rRNA": rRna[0],
            "tRNA": tRna[0],
        },
        "metabolites": {
            "LPS": lps[0],
            "glycogen": glycogen[0],
            "lipid": lipid[0],
            "metabolites": metabolites[0],
            "peptidoglycan": murein[0],
            "polyamines": polyamines[0],
        },
        "protein": protein[0],
    }
    dic_final = {
        "nucleic_acid": {
            "DNA": dna[-1],
            "mRNA": mRna[-1],
            "miscRNA": miscRna[-1],
            "rRNA": rRna[-1],
            "tRNA": tRna[-1],
        },
        "metabolites": {
            "LPS": lps[-1],
            "glycogen": glycogen[-1],
            "lipid": lipid[-1],
            "metabolites": metabolites[-1],
            "peptidoglycan": murein[-1],
            "polyamines": polyamines[-1],
        },
        "protein": protein[-1],
    }
    vm = VoronoiMaster()
    vm.plot(
        [[dic_initial, dic_final]],
        title=[["Initial biomass components", "Final biomass components"]],
        ax_shape=(1, 2),
        chained=True,
    )

    plotOutFileName = "mass_fractions_voronoi"

    # Save figure in main workspace (optional fallback)
    plt.savefig(f"{plotOutFileName}.png", dpi=200)

    # Save figure into analysis output directory
    full_path = os.path.join(outdir, f"{plotOutFileName}.png")
    plt.savefig(full_path, dpi=200)

    print(f"\nSaved Voronoi biomass plot to:\n {full_path}\n")

    plt.close()

    #exportFigure(plt, outdir, plotOutFileName, extension=".png")
