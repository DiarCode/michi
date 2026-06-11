import tempfile
from pathlib import Path

from data.osm_parser import assign_district, load_parsed_network, save_parsed_network


def test_assign_district():
    assert assign_district(51.15, 71.40) == "Esil"
    assert assign_district(51.00, 71.35) == "Saryarka"
    assert assign_district(0.0, 0.0) is None


def test_save_load_roundtrip():
    network = {
        "stops": [{"stop_id": "1", "name": "Test", "lat": 51.0, "lon": 71.0}],
        "routes": [],
        "adjacency": {},
    }
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        save_parsed_network(network, d)
        loaded = load_parsed_network(d)
    assert loaded["stops"][0]["name"] == "Test"
