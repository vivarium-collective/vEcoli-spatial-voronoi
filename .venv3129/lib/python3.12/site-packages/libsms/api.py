import requests
import tempfile
import json
from pathlib import Path
from zipfile import ZipFile

import polars as pl

__all__ = ['get_observables_data']

def unzip_parquet(zip_file_path: Path, local_dirpath: Path):
    extraction_path = local_dirpath
    try:
        with ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extraction_path)
        print(f"Successfully unzipped '{zip_file_path}' to '{extraction_path}'")
    except FileNotFoundError:
        print(f"Error: The file '{zip_file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


def download_parquet(local_dirpath: Path, experiment_id: str) -> Path:
    url = f"https://sms.cam.uchc.edu/core/download/parquet?experiment_id={experiment_id}"

    response = requests.post(url, headers={"Accept": "*/*"})

    if response.status_code != 200:
        raise Exception(f"HTTP error! status: {response.status_code}")

    zippath = local_dirpath / f"{experiment_id}.zip"
    with open(zippath, "wb") as f:
        f.write(response.content)

    return zippath


def get_observables_data(
        observables: list[str] | None = None,
        experiment_id: str | None = None
) -> pl.DataFrame:
    """Get the output data from parquet files generated from a given vEcoli simulation as a dataframe containing all simulation timepoints.

    :param observables: list of observables(dataframe columns) to include. If None is passed, defaults to all columns.
    :param experiment_id: the experiment ID for the simulation that you wish to query. If None is passed, defaults to the exampled
        pinned simulation: "sms_single".

    :rtype: polars.DataFrame
    :return: A dataframe containing all simulation timepoints.

    """
    expid = experiment_id or "sms_single"
    tmpdir = tempfile.TemporaryDirectory()
    dirpath = Path(tmpdir.name)
    zippath = download_parquet(dirpath, expid)
    unzip_parquet(zippath, dirpath)
    df = pl.scan_parquet(f"{str(dirpath)}/*.pq").select(observables).collect()
    tmpdir.cleanup()
    return df
