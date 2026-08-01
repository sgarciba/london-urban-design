import geopandas as gpd
import osmnx as ox


def borough_spatial_split(data, train_boroughts):
    
    """Given a list of borough names and a GeoDataFrame, return a boolean mask for the rows that belong to those boroughs.
    """

    train_polygons = []

    for borough_name in train_boroughts:
        borough_gdf = ox.geocode_to_gdf(borough_name)
        borough_polygon = borough_gdf.geometry.iloc[0]
        # Append the borough polygon to the list of train polygons
        train_polygons.append(borough_polygon)
    
    borough_mask = data.within(train_polygons)
    X_train = data[borough_mask].drop(columns=["LST"]).reset_index(drop=True) 
    y_train = data[borough_mask]["LST"].reset_index(drop=True)
    
    X_test = data[~borough_mask].drop(columns=["LST"]).reset_index(drop=True)
    y_test = data[~borough_mask]["LST"].reset_index(drop=True)
    
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    pass