import os
import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import sys
sys.path.append("../")

FIGURES_DIR = "../figures"

# ----------------------------------------------------------------------
# Areas to analyze. Canary Wharf has no OSM administrative boundary
# (geocode_to_gdf only resolves it to a single building/POI), so it's
# defined as a 1km buffer around its centre instead — that radius gives
# ~5.6k points, similar sample size to City of London, for comparability.
# ----------------------------------------------------------------------
AREAS = {
    "City of London": {"kind": "boundary", "query": "City of London, London, UK"},
    "Kensington and Chelsea": {"kind": "boundary", "query": "Kensington and Chelsea, London, UK"},
    "Canary Wharf": {"kind": "buffer", "center": (51.5049, -0.0199), "radius_m": 1000},
}


def slugify(name):
    return name.lower().replace(" ", "_")


def get_area_geometry(spec):
    if spec["kind"] == "boundary":
        boundary_gdf = ox.geocode_to_gdf(spec["query"])
        polygon = boundary_gdf.geometry.iloc[0]
    else:
        lat, lon = spec["center"]
        center = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326")
        polygon = center.to_crs(epsg=27700).buffer(spec["radius_m"]).to_crs(epsg=4326).iloc[0]
        boundary_gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")
    return boundary_gdf, polygon


# ----------------------------------------------------------------------
# 1. Load the data (shared across all areas)
# ----------------------------------------------------------------------
data = gpd.read_file("../data/london_urban_design_data.gpkg")


# ----------------------------------------------------------------------
# Spatial-CV training/evaluation for one model, reused per area/config.
# ----------------------------------------------------------------------
def run_cv(model, X_spatial, y_spatial, cv_splits):
    fold_metrics = []
    fold_assign = np.full(len(X_spatial), -1, dtype=int)
    fold_r2_arr = np.full(len(X_spatial), np.nan)
    preds_arr = np.full(len(X_spatial), np.nan)

    for fold_num, (train_idx, test_idx) in enumerate(cv_splits):
        X_fold_train, X_fold_test = X_spatial.iloc[train_idx], X_spatial.iloc[test_idx]
        y_fold_train, y_fold_test = y_spatial.iloc[train_idx], y_spatial.iloc[test_idx]

        model.fit(X_fold_train, y_fold_train)
        fold_preds = model.predict(X_fold_test)
        fold_train_preds = model.predict(X_fold_train)

        fold_r2_val = r2_score(y_fold_test, fold_preds)
        fold_train_r2_val = r2_score(y_fold_train, fold_train_preds)
        fold_metrics.append({
            "fold": fold_num,
            "MAE": mean_absolute_error(y_fold_test, fold_preds),
            "RMSE": mean_squared_error(y_fold_test, fold_preds) ** 0.5,
            "R2": fold_r2_val,
            "train_R2": fold_train_r2_val,
            "overfit_gap": fold_train_r2_val - fold_r2_val,
            "n_train": len(X_fold_train),
            "n_test": len(X_fold_test),
        })

        fold_assign[test_idx] = fold_num
        fold_r2_arr[test_idx] = fold_r2_val
        preds_arr[test_idx] = fold_preds

    return pd.DataFrame(fold_metrics), fold_assign, fold_r2_arr, preds_arr


def analyze_area(area_name, spec):
    fig_dir = os.path.join(FIGURES_DIR, slugify(area_name))
    os.makedirs(fig_dir, exist_ok=True)

    boundary_gdf, polygon = get_area_geometry(spec)
    area_mask = data.within(polygon)
    area_data = data[area_mask]
    print(f"\n=== {area_name} ({len(area_data)} points) ===")

    # ------------------------------------------------------------------
    # Spatial split by blocks (grid-based GroupKFold)
    # ------------------------------------------------------------------
    X_spatial = area_data.drop(columns=["LST", "geometry", "lon", "lat"]).reset_index(drop=True)
    y_spatial = area_data["LST"].reset_index(drop=True)
    geom = area_data.geometry.reset_index(drop=True)

    bounds = geom.total_bounds  # [minx, miny, maxx, maxy]
    x_bins = np.linspace(bounds[0], bounds[2], 6)
    y_bins = np.linspace(bounds[1], bounds[3], 6)
    x_idx = np.digitize(geom.x, x_bins, right=True)
    y_idx = np.digitize(geom.y, y_bins, right=True)
    spatial_groups = x_idx * 5 + y_idx

    cv_splits = list(GroupKFold(n_splits=5).split(X_spatial, y_spatial, groups=spatial_groups))

    # ------------------------------------------------------------------
    # Train + evaluate: baseline (unconstrained) vs regularized RF
    # ------------------------------------------------------------------
    model_configs = {
        "baseline": RandomForestRegressor(n_estimators=300, random_state=203, n_jobs=-1),
        "regularized": RandomForestRegressor(
            n_estimators=300, random_state=203, n_jobs=-1,
            max_depth=10, min_samples_leaf=5, max_features="sqrt",
        ),
    }
    cv_results = {name: run_cv(model, X_spatial, y_spatial, cv_splits) for name, model in model_configs.items()}

    comparison = pd.DataFrame({
        name: metrics_df[["MAE", "RMSE", "R2", "train_R2", "overfit_gap"]].mean()
        for name, (metrics_df, _, _, _) in cv_results.items()
    })
    print("Baseline vs regularized RF — mean across spatial folds:")
    print(comparison)

    # Downstream diagnostics use the baseline model's fold assignments/predictions.
    spatial_metrics_df, fold_assignments, fold_r2, point_preds = cv_results["baseline"]

    # ------------------------------------------------------------------
    # Diagnostic: spatial fold maps — where is the bad fold?
    # ------------------------------------------------------------------
    diag_gdf = gpd.GeoDataFrame(
        {"fold": fold_assignments, "r2": fold_r2},
        geometry=geom,
        crs=area_data.crs,
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    diag_gdf.plot(ax=axes[0], column="fold", categorical=True, legend=True,
                  cmap="tab10", markersize=4, alpha=0.8)
    boundary_gdf.to_crs(area_data.crs).boundary.plot(ax=axes[0], color="black", linewidth=1)
    axes[0].set_title("Test-fold assignment")
    axes[0].set_axis_off()

    diag_gdf.plot(ax=axes[1], column="r2", legend=True, cmap="RdYlGn",
                  vmin=-0.5, vmax=1.0, markersize=4, alpha=0.8)
    boundary_gdf.to_crs(area_data.crs).boundary.plot(ax=axes[1], color="black", linewidth=1)
    axes[1].set_title("R² by fold")
    axes[1].set_axis_off()

    fig.suptitle(f"Spatial CV diagnostics — {area_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "spatial_cv_fold_r2_map.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Diagnostic: worst-fold feature profile & residual analysis
    # ------------------------------------------------------------------
    worst_fold = int(spatial_metrics_df.loc[spatial_metrics_df["R2"].idxmin(), "fold"])

    df = gpd.GeoDataFrame(
        X_spatial[["building_density", "NDVI", "NDBI", "MNDWI", "building_volume_density"]].copy(),
        geometry=geom,
        crs=area_data.crs,
    )
    df["fold"] = fold_assignments

    worst = df.loc[df["fold"] == worst_fold, ["geometry", "building_density", "NDVI", "NDBI", "MNDWI", "building_volume_density"]].copy()
    worst["residual"] = (
        point_preds[df["fold"] == worst_fold]
        - y_spatial[df["fold"] == worst_fold].values
    )

    anomaly_mask = (
        (worst["building_density"] > worst["building_density"].quantile(0.75)) &
        (worst["NDVI"] < worst["NDVI"].quantile(0.25)) &
        (worst["NDBI"] > worst["NDBI"].quantile(0.75)) &
        (worst["MNDWI"] < worst["MNDWI"].quantile(0.25)) &
        (worst["building_volume_density"] > worst["building_volume_density"].quantile(0.75))
    )

    worst_r2 = spatial_metrics_df.loc[spatial_metrics_df["fold"] == worst_fold, "R2"].iloc[0]
    anomaly_residual = worst.loc[anomaly_mask, "residual"].mean()
    rest_residual = worst.loc[~anomaly_mask, "residual"].mean()
    print(f"Worst fold: {worst_fold}  (R²={worst_r2:.3f})")
    print(f"  Anomalous profile mean residual : {anomaly_residual:.3f}")
    print(f"  Rest of fold mean residual      : {rest_residual:.3f}")
    print(f"  Anomalous points                : {anomaly_mask.sum()} / {len(worst)}")

    # ------------------------------------------------------------------
    # Diagnostic: is the worst fold's error a level-shift (unmodeled
    # baseline temperature) or a feature-distribution mismatch? Compare
    # actual LST and feature means between the worst fold and the rest.
    # ------------------------------------------------------------------
    worst_mask = fold_assignments == worst_fold
    rest_mask = ~worst_mask

    lst_worst = y_spatial[worst_mask]
    lst_rest = y_spatial[rest_mask]
    lst_diff = lst_worst.mean() - lst_rest.mean()

    print(f"Mean actual LST — fold {worst_fold}: {lst_worst.mean():.3f}  |  rest: {lst_rest.mean():.3f}  |  diff: {lst_diff:+.3f}")

    feature_cols = ["NDVI", "MNDWI", "NDBI", "building_density", "building_volume_density"]
    level_shift_check = pd.DataFrame({
        f"fold_{worst_fold}_mean": X_spatial.loc[worst_mask, feature_cols].mean(),
        "rest_mean": X_spatial.loc[rest_mask, feature_cols].mean(),
    })
    level_shift_check["diff"] = level_shift_check[f"fold_{worst_fold}_mean"] - level_shift_check["rest_mean"]
    print(level_shift_check)

    # test to see if the model is picking up the relationship with being near the water
    # How strongly do existing features already encode "near the river"?
    # Use y-coordinate as a rough proxy for northing (southern boundary = river)
    northing_df = area_data.copy()
    northing_df["northing"] = northing_df.geometry.y
    print(northing_df[["northing", "NDVI", "MNDWI", "NDBI", "building_density", "building_volume_density"]].corr()["northing"])

    return {
        "area": area_name,
        "n_points": len(area_data),
        "test_R2_baseline": comparison.loc["R2", "baseline"],
        "test_R2_regularized": comparison.loc["R2", "regularized"],
        "overfit_gap_baseline": comparison.loc["overfit_gap", "baseline"],
        "overfit_gap_regularized": comparison.loc["overfit_gap", "regularized"],
        "worst_fold": worst_fold,
        "worst_fold_R2": worst_r2,
        "worst_fold_mean_residual": worst["residual"].mean(),
        "worst_fold_lst_diff_vs_rest": lst_diff,
    }


summaries = [analyze_area(name, spec) for name, spec in AREAS.items()]
summary_df = pd.DataFrame(summaries)
print("\n=== Cross-area summary ===")
print(summary_df.to_string(index=False))
