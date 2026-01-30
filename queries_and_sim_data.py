#%%
import numpy as np

from ecoli.library.sim_data import LoadSimData

sim_data_default = "reconstruction/sim_data/kb/simData.cPickle"
sim_data = LoadSimData(sim_data_default).sim_data

# %%
import duckdb
from duckdb import DuckDBPyConnection
import numpy as np


# Query of bulk
conn = DuckDBPyConnection

query_dict = {
    "experiment_id": "test_installation",
    "variant": 0,
    "lineage_seed": 0,
    "generation": 1,
}

query = f"""
    SELECT listeners__,time FROM read_parquet("out/{query_dict["experiment_id"]}/history/*/*/*/*/*/*.pq", hive_partitioning=true)
    WHERE variant={query_dict["variant"]}
    AND lineage_seed={query_dict["lineage_seed"]}
    AND generation={query_dict["generation"]}
    ORDER BY time
"""

# Query to dataframe of listeners__mass__extracellular_mass and values
db_bulk = duckdb.sql(query)
db_bulk1 = db_bulk.df()

# Map values from db_listener_emass to a matrix and stack to make numpy matrix,
db_bulk_stack = np.stack(db_bulk1["bulk"].values)

# %%
import duckdb
from duckdb import DuckDBPyConnection
import numpy as np


# Query of bulk
conn = DuckDBPyConnection

query_dict = {
    "experiment_id": "test_installation",
    "variant": 0,
    "lineage_seed": 0,
    "generation": 1,
}

query = f"""
    SELECT listeners__mass__dna_mass,time FROM read_parquet("out/{query_dict["experiment_id"]}/history/*/*/*/*/*/*.pq", hive_partitioning=true)
    WHERE variant={query_dict["variant"]}
    AND lineage_seed={query_dict["lineage_seed"]}
    AND generation={query_dict["generation"]}
    ORDER BY time
"""

# Query to dataframe of listeners__mass__extracellular_mass and values
db_bulk = duckdb.sql(query)
db_bulk1 = db_bulk.df()

# # Map values from db_listener_emass to a matrix and stack to make numpy matrix,
# db_bulk_stack = np.stack(db_bulk1["bulk"].values)

#%%
import numpy as np

from ecoli.library.sim_data import LoadSimData

sim_data_default_new = "runscripts/out/kb/simData.cPickle"
sim_data_new = LoadSimData(sim_data_default_new).sim_data

flg_molecule = sim_data_new.getter.get_compartment("EG10322-MONOMER")

print(flg_molecule)


#%%


# Import required packages
import duckdb
import numpy as np
from duckdb import DuckDBPyConnection

conn = DuckDBPyConnection

# Dictionary to define which experiment data to load
query_dict = {
    "experiment_id": "c41ad91a-fbb2-11f0-9de6-2bfaea4f42f9",
    "variant": 0,
    "lineage_seed": 0,
    "generation": 1,
}

# SQL query to read Parquet files with simulatikon output
# Selecting only "bulk" and "time" columns from history data
query = f"""
    SELECT bulk,time FROM read_parquet("out/{query_dict["experiment_id"]}/history/*/*/*/*/*/*.pq", hive_partitioning=true)
    WHERE variant={query_dict["variant"]}
    AND lineage_seed={query_dict["lineage_seed"]}
    AND generation={query_dict["generation"]}
    ORDER BY time
"""
# Execute the SQL Query using DuckDB's SQl interface
# Returns a DuckDB relation object containing the result
db_bulk = duckdb.sql(query)

# Convert the DuckDB relation into a Pandas Dataframe for easier handling
db_bulk = db_bulk.df()

# Convert the "bulk" column, which contains a vector of molecule counts at each time step into a 2D NumPy array
# Each row corresponds to a single time point and each column corresponds to a molecule
bulk_state_mtx = np.stack(db_bulk["bulk"].values)