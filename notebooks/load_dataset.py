# Dataset from https://code.earthengine.google.com/
import json
import tempfile
import requests
import rasterio
import osmnx as ox
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import sys
sys.path.append("../")

from config import GOOGLE_DRIVE_CLIENT_SECRETS_JSON, FOLDER_NAME, FILE_NAME, ADMINISTRATIVE_AREA, KEEP_COLS


# ----------------------------------------------------------------------
# A: LANDSAT Dataset: original from Google Earth Engine, download the London_LST_NDVI_NDBI.tif file from Google Drive using PyDrive2
# ----------------------------------------------------------------------

def download_landsat(GOOGLE_DRIVE_CLIENT_SECRETS_JSON, FOLDER_NAME, FILE_NAME):

    gauth = GoogleAuth()

    gauth.settings = {
        "oauth_scope": ["https://www.googleapis.com/auth/drive"]
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(GOOGLE_DRIVE_CLIENT_SECRETS_JSON, f)
        client_secrets_path = f.name

    gauth.LoadClientConfigFile(client_secrets_path)
    os.remove(client_secrets_path)
    gauth.LocalWebserverAuth()

    drive = GoogleDrive(gauth)

    folder_list = drive.ListFile({
        'q': f"title='{FOLDER_NAME}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    }).GetList()

    folder_id = folder_list[0]['id']
    print(folder_id)


    file_list = drive.ListFile({
        'q': f"'{folder_id}' in parents and trashed=false"
    }).GetList()


    for file in file_list:
        if file['title'] == FILE_NAME:
            file_id = file['id']
            break


    url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"

    headers = {"Authorization": f"Bearer {gauth.credentials.access_token}"}


    r = requests.get(url, headers=headers)

    with open(FILE_NAME, "wb") as f:
        f.write(r.content)

    print(os.path.getsize(FILE_NAME))

    # Read original
    with rasterio.open(FILE_NAME) as src:
        data = src.read()
        meta = src.meta.copy()
        band_descriptions = src.descriptions

    # Update metadata
    meta.update({
        "count": data.shape[0]
    })

    # Save file
    out_path = "../data/" + FILE_NAME

    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(data)
        dst.descriptions = band_descriptions
        

def load_landsat_data(FILE_NAME):
            
    with rasterio.open("../data/" + FILE_NAME) as src:
        data = src.read()
        transform = src.transform
        band_names = src.descriptions

    rows, cols = np.meshgrid(
        np.arange(data.shape[1]),
        np.arange(data.shape[2]),
        indexing="ij"
    )

    xs, ys = rasterio.transform.xy(transform, rows, cols)

    df = pd.DataFrame({
        "lon": np.array(xs).ravel(),
        "lat": np.array(ys).ravel(),
        **{name: data[i].ravel() for i, name in enumerate(band_names)},
    })

    df.head()

    # Transform to Geopandas
    landsat_gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.lon, df.lat),
        crs="EPSG:4326"
    )
    
    return landsat_gdf

# ----------------------------------------------------------------------
# B: BUILDINGS Dataset: original from OSMnx, download building footprints for Greater London
# ----------------------------------------------------------------------

def download_buildings_osmnx(ADMINISTRATIVE_AREA, KEEP_COLS, req_timeout=600):

    ox.settings.requests_timeout = req_timeout
    # This gets the actual administrative polygon for Greater London
    london_gdf = ox.geocode_to_gdf(ADMINISTRATIVE_AREA)
    london_polygon = london_gdf.geometry.iloc[0]


    # Download building footprints
    buildings = ox.features_from_polygon(london_polygon, tags={"building": True})

    print(buildings.shape)
    
    cols_present = [c for c in KEEP_COLS if c in buildings.columns]
    missing = [c for c in KEEP_COLS if c not in buildings.columns]
    if missing:
        print(f"Note: these requested columns weren't found and will be skipped: {missing}")

    buildings = buildings[cols_present].copy()
    buildings = buildings[buildings.geometry.type.isin(["Polygon", "MultiPolygon"])]
    buildings = buildings.drop_duplicates(subset=["geometry"])

    # Save file
    out_path = "../data/osmnx_buildings.gpkg"
    
    buildings.to_file(out_path, driver="GPKG")


if __name__ == "__main__":
    pass
