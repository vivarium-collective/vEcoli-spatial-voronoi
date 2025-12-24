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

#%%

