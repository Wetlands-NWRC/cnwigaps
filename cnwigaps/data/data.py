from pathlib import Path
import pandas as pd

ZONES = Path(__file__).parent / "zones.csv"


def load_eco_zone_data() -> pd.DataFrame:
    """loads the data from the data directory"""
    return pd.read_csv(ZONES)
