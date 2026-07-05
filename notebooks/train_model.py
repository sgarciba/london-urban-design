import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
 
import sys
sys.path.append("../")

# ---------------------- 
# 1. Load the data
# ----------------------
data = gpd.read_file("../data/london_urban_design_data.gpkg")

borough_name = "Harrow, London, UK"
"City of London, London, UK"

borough_gdf = ox.geocode_to_gdf(borough_name)
borough_polygon = borough_gdf.geometry.iloc[0]

borough_mask = data.within(borough_polygon)
borough_data = data[borough_mask]

borough_data.shape
borough_data.head()
 

# ----------------------------------------------------------------------
# Split strategy 1: random (baseline, for comparison)
# ----------------------------------------------------------------------
X = borough_data.copy().drop(columns=["LST", "geometry", "lon", "lat"])
y = borough_data["LST"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=203)

print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)

# ----------------------------------------------------------------------
# Split strategy 2: spatial blocks (grid-based GroupKFold)
# ----------------------------------------------------------------------
X_spatial = borough_data.drop(columns=["LST", "geometry", "lon", "lat"]).reset_index(drop=True)
y_spatial = borough_data["LST"].reset_index(drop=True)
geom = borough_data.geometry.reset_index(drop=True)

# Assign each point to a spatial grid cell (5x5 = 25 blocks)
bounds = geom.total_bounds  # [minx, miny, maxx, maxy]
x_bins = np.linspace(bounds[0], bounds[2], 6)
y_bins = np.linspace(bounds[1], bounds[3], 6)
x_idx = np.digitize(geom.x, x_bins, right=True)
y_idx = np.digitize(geom.y, y_bins, right=True)
spatial_groups = x_idx * 5 + y_idx

skcv = GroupKFold(n_splits=5).split(X_spatial, y_spatial, groups=spatial_groups)


# ----------------------------------------------------------------------
# Train + evaluate: random split
# ----------------------------------------------------------------------
random_model = RandomForestRegressor(n_estimators=300, random_state=203, n_jobs=-1)
random_model.fit(X_train, y_train)
random_preds = random_model.predict(X_test)

random_metrics = {
    "MAE": mean_absolute_error(y_test, random_preds),
    "RMSE": mean_squared_error(y_test, random_preds) ** 0.5,
    "R2": r2_score(y_test, random_preds),
    "n_train": len(X_train),
    "n_test": len(X_test),
    "feature_importances": pd.Series(
        random_model.feature_importances_, index=X_train.columns
    ).sort_values(ascending=False),
}
print(random_metrics)


# ----------------------------------------------------------------------
# Train + evaluate: spatial split (grid-based GroupKFold folds)
# ----------------------------------------------------------------------
spatial_fold_metrics = []
fold_assignments = np.full(len(X_spatial), -1, dtype=int)
fold_r2 = np.full(len(X_spatial), np.nan)
point_preds = np.full(len(X_spatial), np.nan)

for fold_num, (train_idx, test_idx) in enumerate(skcv):
    X_fold_train, X_fold_test = X_spatial.iloc[train_idx], X_spatial.iloc[test_idx]
    y_fold_train, y_fold_test = y_spatial.iloc[train_idx], y_spatial.iloc[test_idx]

    spatial_model = RandomForestRegressor(n_estimators=300, random_state=203, n_jobs=-1)
    spatial_model.fit(X_fold_train, y_fold_train)
    fold_preds = spatial_model.predict(X_fold_test)

    fold_r2_val = r2_score(y_fold_test, fold_preds)
    spatial_fold_metrics.append({
        "fold": fold_num,
        "MAE": mean_absolute_error(y_fold_test, fold_preds),
        "RMSE": mean_squared_error(y_fold_test, fold_preds) ** 0.5,
        "R2": fold_r2_val,
        "n_train": len(X_fold_train),
        "n_test": len(X_fold_test),
    })

    fold_assignments[test_idx] = fold_num
    fold_r2[test_idx] = fold_r2_val
    point_preds[test_idx] = fold_preds

spatial_metrics_df = pd.DataFrame(spatial_fold_metrics)
print(spatial_metrics_df)
print("\nMean spatial metrics:")
print(spatial_metrics_df.mean())


# ----------------------------------------------------------------------
# Diagnostic: spatial fold maps — where is the bad fold?
# ----------------------------------------------------------------------
diag_gdf = gpd.GeoDataFrame(
    {"fold": fold_assignments, "r2": fold_r2},
    geometry=geom,
    crs=borough_data.crs,
)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

diag_gdf.plot(ax=axes[0], column="fold", categorical=True, legend=True,
              cmap="tab10", markersize=4, alpha=0.8)
borough_gdf.to_crs(borough_data.crs).boundary.plot(ax=axes[0], color="black", linewidth=1)
axes[0].set_title("Test-fold assignment")
axes[0].set_axis_off()

diag_gdf.plot(ax=axes[1], column="r2", legend=True, cmap="RdYlGn",
              vmin=-0.5, vmax=1.0, markersize=4, alpha=0.8)
borough_gdf.to_crs(borough_data.crs).boundary.plot(ax=axes[1], color="black", linewidth=1)
axes[1].set_title("R² by fold")
axes[1].set_axis_off()

plt.tight_layout()
plt.show()


# ----------------------------------------------------------------------
# Diagnostic: worst-fold feature profile & residual analysis
# ----------------------------------------------------------------------
worst_fold = int(spatial_metrics_df.loc[spatial_metrics_df["R2"].idxmin(), "fold"])

df = gpd.GeoDataFrame(
    X_spatial[["building_density", "NDVI", "NDBI"]].copy(),
    geometry=geom,
    crs=borough_data.crs,
)
df["fold"] = fold_assignments

worst = df.loc[df["fold"] == worst_fold, ["geometry", "building_density", "NDVI", "NDBI"]].copy()
worst["residual"] = (
    point_preds[df["fold"] == worst_fold]
    - y_spatial[df["fold"] == worst_fold].values
)

anomaly_mask = (
    (worst["building_density"] > worst["building_density"].quantile(0.75)) &
    (worst["NDVI"] < worst["NDVI"].quantile(0.25)) &
    (worst["NDBI"] > worst["NDBI"].quantile(0.75))
)

worst_r2 = spatial_metrics_df.loc[spatial_metrics_df["fold"] == worst_fold, "R2"].iloc[0]
print(f"Worst fold: {worst_fold}  (R²={worst_r2:.3f})")
print(f"  Anomalous profile mean residual : {worst.loc[anomaly_mask, 'residual'].mean():.3f}")
print(f"  Rest of fold mean residual      : {worst.loc[~anomaly_mask, 'residual'].mean():.3f}")
print(f"  Anomalous points                : {anomaly_mask.sum()} / {len(worst)}")


# test to see if the model is picking up the relationship with being near the water
# How strongly do existing features already encode "near the river"?
# Use y-coordinate as a rough proxy for northing (southern boundary = river)
df = borough_data.copy()
df["northing"] = df.geometry.y
print(df[["northing", "NDVI", "NDBI", "building_density"]].corr()["northing"])