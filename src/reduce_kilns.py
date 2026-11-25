from pathlib import Path
import geopandas as gpd

def exclude_kilns(gpkg, exclude, id_column="ID_PAROLINI", out=None):
    """Write a copy of `gpkg` without the kilns whose ID is in `exclude`."""
    kilns = gpd.read_file(gpkg)

    # Compare as strings so 12 and "12" both match
    ids = kilns[id_column].astype(str)
    exclude = {str(e) for e in exclude}

    missing = exclude - set(ids)
    if missing:
        print(f"Not found in {id_column}: {sorted(missing)}")

    reduced = kilns[~ids.isin(exclude)]
    out = out or Path(gpkg).with_name(Path(gpkg).stem + "_reduced.gpkg")
    reduced.to_file(out, driver="GPKG")

    print(f"{len(kilns)} kilns -> {len(reduced)} kept, saved to {out}")
    return reduced