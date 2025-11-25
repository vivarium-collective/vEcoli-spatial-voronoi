from api import get_observables_data


def test_download_parquet():
    expid = "sms_single"
    obs = ["bulk", "time"]
    df = get_observables_data(experiment_id=expid, observables=obs)
    print(df.head())

if __name__ == "__main__":
    test_download_parquet()