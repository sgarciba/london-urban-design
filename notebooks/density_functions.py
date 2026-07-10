import time
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
import sys
sys.path.append("../")

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

# Columns worth keeping out of the ~1700+ sparse OSM tag columns.
# Add/remove freely depending on what you actually need downstream.
KEEP_COLS = ["geometry", "building", "name", "building:levels", "height"]


# ----------------------------------------------------------------------
# STEP 1: Select only the columns you need
# ----------------------------------------------------------------------

def select_columns(buildings: gpd.GeoDataFrame, keep_cols=KEEP_COLS) -> gpd.GeoDataFrame:
    """
    Reduces a wide OSM buildings GeoDataFrame (often 1000+ sparse tag
    columns) down to just the ones you need, drops non-polygon
    geometries and exact duplicates, and saves the result.
    """
    print(f"Input shape: {buildings.shape}")

    cols_present = [c for c in keep_cols if c in buildings.columns]
    missing = [c for c in keep_cols if c not in buildings.columns]
    if missing:
        print(f"Note: these requested columns weren't found and will be skipped: {missing}")

    buildings = buildings[cols_present].copy()

    before = len(buildings)
    buildings = buildings[buildings.geometry.type.isin(["Polygon", "MultiPolygon"])]
    buildings = buildings.drop_duplicates(subset=["geometry"])
    after = len(buildings)
    print(f"Dropped {before - after} non-polygon/duplicate rows ({before} -> {after})")
    print(f"Output shape: {buildings.shape}")

    return buildings


# ----------------------------------------------------------------------
# STEP 2: Build cells centered on your raster points (not an arbitrary grid)
# ----------------------------------------------------------------------

def infer_resolution(points_gdf: gpd.GeoDataFrame, lon_col="lon", lat_col="lat"):
    """
    Infers the raster's pixel spacing directly from the point coordinates,
    by looking at the smallest non-zero gap between consecutive unique
    lon values, and separately for lat.

    Returns (lon_res, lat_res) as a tuple — these are often genuinely
    different. This shows up specifically when data was exported from
    Google Earth Engine with a metric `scale` (e.g. scale=30) but
    `crs='EPSG:4326'`: GEE computes the longitude step and latitude step
    needed to each represent ~30m on the ground at the export region's
    reference latitude, but a degree of longitude is a shorter ground
    distance than a degree of latitude away from the equator (the
    difference scales with cos(latitude)). So at ~51.5N, a true 30m
    pixel ends up roughly 0.00027 degrees tall but ~0.00043 degrees
    wide — genuinely rectangular in degree-space despite being ~square
    in meters. Treating these as one shared resolution would distort
    your cells, so callers should use both values rather than forcing
    a single number.
    """
    lons = sorted(points_gdf[lon_col].unique())
    lats = sorted(points_gdf[lat_col].unique())

    lon_diffs = [round(lons[i + 1] - lons[i], 10) for i in range(len(lons) - 1)]
    lat_diffs = [round(lats[i + 1] - lats[i], 10) for i in range(len(lats) - 1)]

    lon_res = min((d for d in lon_diffs if d > 0), default=None)
    lat_res = min((d for d in lat_diffs if d > 0), default=None)

    if lon_res is None and lat_res is None:
        raise ValueError(
            "Couldn't infer resolution: only one unique lon and one unique "
            "lat value found. Pass lon_resolution=/lat_resolution= explicitly "
            "to points_to_cells()."
        )

    if lon_res is None:
        print(f"Only one unique lon value found; using lat resolution ({lat_res}) for lon too.")
        lon_res = lat_res
    if lat_res is None:
        print(f"Only one unique lat value found; using lon resolution ({lon_res}) for lat too.")
        lat_res = lon_res

    print(f"Inferred resolution: lon={lon_res}, lat={lat_res}")
    if abs(lon_res - lat_res) > 1e-6:
        ratio = lon_res / lat_res
        print(
            f"Note: lon and lat resolutions differ (ratio {ratio:.3f}). This is "
            f"expected for GEE exports using a metric scale with crs='EPSG:4326' "
            f"— pixels are ~square in meters but rectangular in degrees. Both "
            f"values will be used (lon width, lat height) rather than forcing "
            f"one shared value."
        )

    return lon_res, lat_res



def points_to_cells(
    points_gdf: gpd.GeoDataFrame,
    lon_resolution: float = None,
    lat_resolution: float = None,
    lon_col: str = "lon",
    lat_col: str = "lat",
) -> gpd.GeoDataFrame:
    """
    Converts a table of raster pixel-center points into a GeoDataFrame
    of rectangular cells, one per point, centered on that point. This is
    what lets you compute building density per pixel and attach it back
    to your NDVI/NDBI/LST table by matching row/cell_id.
 
    Cells use lon_resolution for width and lat_resolution for height —
    these are inferred separately from the data if not provided, since
    GEE exports with a metric scale + crs='EPSG:4326' produce pixels
    that are genuinely rectangular in degrees (see infer_resolution).
    """
    if lon_resolution is None or lat_resolution is None:
        inferred_lon, inferred_lat = infer_resolution(points_gdf, lon_col, lat_col)
        lon_resolution = lon_resolution or inferred_lon
        lat_resolution = lat_resolution or inferred_lat
 
    half_lon = lon_resolution / 2
    half_lat = lat_resolution / 2
 
    points_gdf = points_gdf.copy()
 
    # cell_id is derived directly from the index, and compute_density later
    # does groupby("cell_id") to sum building area per cell. If the index
    # has duplicates, two genuinely different pixels share one cell_id,
    # and their building areas get summed together — silently doubling
    # (or worse) the reported density for both. This produced exactly a
    # systematic "X% of cells over 1.0" pattern in practice, so it's
    # checked explicitly rather than left to surface downstream.
    n_dupes = points_gdf.index.duplicated().sum()
    if n_dupes > 0:
        raise ValueError(
            f"points_gdf has {n_dupes} duplicate index values. cell_id is "
            f"derived from the index, so duplicates will cause building "
            f"area from different pixels to be summed together under one "
            f"cell_id, inflating building_density for all affected rows. "
            f"Fix with points_gdf = points_gdf.reset_index(drop=True) (if "
            f"the index isn't meaningful) or points_gdf.index = range(len("
            f"points_gdf)) before calling points_to_cells(), then merge "
            f"results back by position rather than by the original index."
        )
 
    points_gdf["cell_id"] = points_gdf.index.astype(str)
 
    cells = points_gdf.apply(
        lambda row: box(
            row[lon_col] - half_lon,
            row[lat_col] - half_lat,
            row[lon_col] + half_lon,
            row[lat_col] + half_lat,
        ),
        axis=1,
    )
 
    grid = gpd.GeoDataFrame(
        {"cell_id": points_gdf["cell_id"].values},
        geometry=cells.values,
        crs=points_gdf.crs if points_gdf.crs else "EPSG:4326",
    )
    print(
        f"Built {len(grid)} cells (lon_resolution={lon_resolution}, "
        f"lat_resolution={lat_resolution}) centered on raster points"
    )
    return grid
 
 
# ----------------------------------------------------------------------
# STEP 3: Compute density per cell (vectorized via overlay)
# ----------------------------------------------------------------------
 
def compute_density(
    buildings: gpd.GeoDataFrame,
    grid: gpd.GeoDataFrame,
    projected_crs="EPSG:27700",
    clip_to_one: bool = True,
) -> gpd.GeoDataFrame:
    """
    projected_crs defaults to EPSG:27700 (OSGB36 / British National Grid),
    the standard metric projection for the UK. Areas/intersections are
    computed in this CRS so building_area ends up in square meters, then
    the result is reprojected back to the grid's original CRS for output.
 
    clip_to_one: if True (default), caps building_density at 1.0 in the
        output. Set to False if you want to inspect raw values above 1
        first — e.g. with diagnose_high_density_cells() — before deciding
        whether clipping is appropriate for your data.
 
    Returns a GeoDataFrame indexed by cell_id with building_area and
    building_density columns — merge this back onto your raster table
    using cell_id (see attach_density_to_points below).
    """
    output_crs = grid.crs
 
    buildings_m = buildings.to_crs(projected_crs)
    grid_m = grid.to_crs(projected_crs)
 
    grid_m["cell_area"] = grid_m.geometry.area
 
    # OSM building data frequently contains overlapping polygons for the
    # same physical building — e.g. an outer footprint plus a separately
    # tagged inner courtyard/wing, or duplicated outline/inner-ring parts
    # from multipolygon relations. If left as-is, gpd.overlay sums each
    # overlapping polygon's intersection area independently, so a cell
    # covered by two stacked polygons gets counted twice — producing
    # building_density > 1, which is geometrically impossible.
    #
    # unary_union merges all overlapping/touching geometries into a single
    # combined shape per contiguous cluster, so overlapping area is only
    # counted once. This is a one-time cost up front but avoids
    # double-counting throughout the rest of the pipeline.
    print("Dissolving overlapping building geometries (this can take a while for 1M+ buildings)...")
    t0 = time.time()
    merged = buildings_m.geometry.union_all() if hasattr(
        buildings_m.geometry, "union_all"
    ) else buildings_m.geometry.unary_union
    if merged.geom_type == "Polygon":
        merged_geoms = [merged]
    else:
        merged_geoms = list(merged.geoms)
    buildings_m = gpd.GeoDataFrame({"geometry": merged_geoms}, crs=projected_crs)
    print(f"  dissolve done in {time.time() - t0:.1f}s, {len(buildings_m)} merged shapes")
 
    print("Computing intersections (this uses spatial indexing internally)...")
    t0 = time.time()
    intersections = gpd.overlay(
        buildings_m[["geometry"]], grid_m, how="intersection", keep_geom_type=False
    )
    print(f"  overlay done in {time.time() - t0:.1f}s, {len(intersections)} pieces")
 
    intersections["area"] = intersections.geometry.area  # now in square meters
 
    area_per_cell = intersections.groupby("cell_id")["area"].sum()
 
    grid_m = grid_m.set_index("cell_id")
    grid_m["building_area"] = area_per_cell
    grid_m["building_area"] = grid_m["building_area"].fillna(0)
    grid_m["building_density"] = grid_m["building_area"] / grid_m["cell_area"]
    grid_m = grid_m.reset_index()
 
    # Sanity check: after dissolving, density should never exceed 1.
    # Any remaining excess should only be tiny floating-point noise from
    # the overlay/reprojection, not a real geometric impossibility.
    n_over = (grid_m["building_density"] > 1.0001).sum()
    if n_over > 0:
        pct_over = n_over / len(grid_m) * 100
        max_val = grid_m["building_density"].max()
        print(
            f"Warning: {n_over} cells ({pct_over:.2f}%) still show "
            f"building_density > 1 (max={max_val:.4f}) even after dissolving. "
            f"If this is more than a handful of cells, don't just clip — use "
            f"diagnose_high_density_cells() to find out why before deciding "
            f"how to handle it."
        )
    if clip_to_one:
        # Clip tiny floating-point overshoot (e.g. 1.0000003), and anything
        # larger if the caller has explicitly accepted clip_to_one=True
        # after investigating the warning above.
        grid_m["building_density"] = grid_m["building_density"].clip(upper=1.0)
 
    grid_out = grid_m.to_crs(output_crs)

    return grid_out
 
 
# ----------------------------------------------------------------------
# STEP 3b: Levels-weighted building volume density (footprint density
# alone can't tell a low sprawling building apart from a tower on the
# same footprint)
# ----------------------------------------------------------------------

def compute_volume_density(
    buildings: gpd.GeoDataFrame,
    grid: gpd.GeoDataFrame,
    projected_crs="EPSG:27700",
) -> gpd.GeoDataFrame:
    """
    Adds a building_volume_density column: footprint area weighted by
    the `building:levels` OSM tag, summed per cell and divided by cell
    area. Units are "levels" (footprint-covered fraction of the cell,
    scaled by storey count), not meters — `height` is tagged on <1% of
    OSM buildings so it isn't usable, and `building:levels` coverage
    itself varies a lot by borough (check it for your area of interest
    before trusting this feature there).

    Unlike compute_density(), this does NOT dissolve overlapping raw
    building geometries first — dissolving discards per-building
    attributes, and building:levels is exactly what's being weighted by
    here. So genuinely overlapping polygons (duplicate outlines, inner
    rings tagged as separate features) will double-count. Treat this as
    a directional signal, not an exact volume measurement.

    Buildings with a missing or non-numeric building:levels value are
    excluded from the numerator entirely (not treated as 0 levels).

    Returns a GeoDataFrame indexed by cell_id with building_volume and
    building_volume_density columns — merge onto your raster table with
    attach_volume_density_to_points().
    """
    if "building:levels" not in buildings.columns:
        raise ValueError(
            "buildings is missing a 'building:levels' column — pass the "
            "output of select_columns() with 'building:levels' in KEEP_COLS."
        )

    output_crs = grid.crs

    levels = pd.to_numeric(buildings["building:levels"], errors="coerce")
    # A building can't have zero or negative storeys — a handful of OSM
    # buildings are tagged this way (bad data entry), and left as-is they'd
    # silently zero out that building's contribution instead of being
    # excluded like any other missing value.
    levels = levels.where(levels > 0)
    n_usable = levels.notna().sum()
    print(f"{n_usable} / {len(buildings)} buildings have a usable building:levels value ({n_usable / len(buildings) * 100:.1f}%)")

    buildings_m = buildings.loc[levels.notna(), ["geometry"]].copy()
    buildings_m["levels"] = levels.loc[levels.notna()].values
    buildings_m = buildings_m.to_crs(projected_crs)

    grid_m = grid.to_crs(projected_crs)
    grid_m["cell_area"] = grid_m.geometry.area

    intersections = gpd.overlay(
        buildings_m, grid_m, how="intersection", keep_geom_type=False
    )
    intersections["weighted_area"] = intersections.geometry.area * intersections["levels"]

    weighted_area_per_cell = intersections.groupby("cell_id")["weighted_area"].sum()

    grid_m = grid_m.set_index("cell_id")
    grid_m["building_volume"] = weighted_area_per_cell
    grid_m["building_volume"] = grid_m["building_volume"].fillna(0)
    grid_m["building_volume_density"] = grid_m["building_volume"] / grid_m["cell_area"]
    grid_m = grid_m.reset_index()

    grid_out = grid_m.to_crs(output_crs)

    return grid_out


def diagnose_high_density_cells(
    buildings: gpd.GeoDataFrame,
    grid: gpd.GeoDataFrame,
    density_grid: gpd.GeoDataFrame,
    threshold: float = 1.0001,
    n_samples: int = 5,
    projected_crs: str = "EPSG:27700",
):
    """
    Pulls a handful of cells where building_density still exceeds
    `threshold` and reports concrete numbers about what's happening
    geometrically in each one: how many buildings intersect the cell,
    whether those buildings overlap EACH OTHER (which dissolving should
    have already fixed), and how the raw vs. dissolved building area
    compares to the cell area.
 
    This is meant to be run BEFORE deciding to clip/replace high values —
    a high rate (e.g. 6% of cells) is too large to treat as noise, so
    it's worth confirming what's actually causing it.
 
    Note: compute_density() already clips its OUTPUT to a max of 1.0, so
    pass it the density_grid you actually want to inspect for values
    over 1 BEFORE any clipping — e.g. call this with the un-clipped
    density numbers, or temporarily disable the clip in compute_density
    while debugging.
    """
    problem = density_grid[density_grid["building_density"] > threshold].copy()
    pct = len(problem) / len(density_grid) * 100
    print(f"{len(problem)} / {len(density_grid)} cells ({pct:.2f}%) exceed density={threshold}")
 
    if len(problem) == 0:
        print("Nothing to diagnose — either there's no real issue, or density_grid was already clipped.")
        return
 
    buildings_m = buildings.to_crs(projected_crs)
    grid_m = grid.to_crs(projected_crs).set_index("cell_id")
 
    sample_ids = problem.sort_values("building_density", ascending=False)["cell_id"].head(n_samples)
 
    for cid in sample_ids:
        cell_geom = grid_m.loc[cid, "geometry"]
        cell_area = cell_geom.area
 
        # All buildings whose RAW (un-dissolved) geometry intersects this cell
        intersecting = buildings_m[buildings_m.geometry.intersects(cell_geom)]
        n_buildings = len(intersecting)
 
        # Sum of each building's individual intersection area with the cell
        # (this is what the OLD, pre-dissolve code effectively computed)
        raw_areas = intersecting.geometry.intersection(cell_geom).area
        raw_total = raw_areas.sum()
 
        # Dissolved: union the intersecting buildings first, THEN intersect
        # with the cell (this is what compute_density does post-fix)
        if n_buildings > 0:
            merged = intersecting.geometry.union_all() if hasattr(
                intersecting.geometry, "union_all"
            ) else intersecting.geometry.unary_union
            dissolved_area = merged.intersection(cell_geom).area
        else:
            dissolved_area = 0
 
        reported_density = density_grid.loc[
            density_grid["cell_id"] == cid, "building_density"
        ].values[0]
 
        print(f"\ncell_id={cid}")
        print(f"  cell_area (m^2):            {cell_area:.2f}")
        print(f"  buildings intersecting:     {n_buildings}")
        print(f"  raw summed area (m^2):      {raw_total:.2f}  (ratio to cell: {raw_total/cell_area:.3f})")
        print(f"  dissolved area (m^2):       {dissolved_area:.2f}  (ratio to cell: {dissolved_area/cell_area:.3f})")
        print(f"  reported building_density:  {reported_density:.4f}")
 
        if dissolved_area / cell_area > 1.0001:
            print(
                "  -> STILL over 1 after manual dissolve here. This is NOT an "
                "overlapping-polygon issue — points to a grid/CRS/cell geometry "
                "problem. Worth checking this cell's coordinates directly."
            )
        elif abs(raw_total - dissolved_area) > 1:
            print(
                "  -> Raw and dissolved areas differ meaningfully — overlapping "
                "polygons WERE present here. If your density_grid still shows "
                "this cell over 1, your compute_density() call may not be "
                "using the dissolved version (check you're running the "
                "updated function, not a stale cached import)."
            )
        else:
            print(
                "  -> Raw and dissolved areas match. The over-1.0 value in "
                "density_grid doesn't match a manual recomputation — worth "
                "checking for a stale grid/buildings mismatch (e.g. grid built "
                "with different resolution than density_grid was computed with)."
            )
 
 
def attach_density_to_points(
    points_gdf: gpd.GeoDataFrame, density_grid: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Merges building_area / building_density back onto your original
    raster points table (NDVI/NDBI/LST/etc.) by matching cell_id, which
    was set to match points_gdf's original index in points_to_cells().
 
    Returns a copy of points_gdf — with its original index preserved —
    plus two new columns: building_area and building_density.
    """
    original_index_name = points_gdf.index.name
 
    points_gdf = points_gdf.copy()
    points_gdf["cell_id"] = points_gdf.index.astype(str)
 
    density_lookup = density_grid.set_index("cell_id")[["building_area", "building_density"]]
 
    result = points_gdf.merge(
        density_lookup, on="cell_id", how="left", left_index=False, right_index=False
    )
    # merge() resets the index, so restore the original one explicitly
    result.index = points_gdf.index
    result.index.name = original_index_name
 
    result["building_area"] = result["building_area"].fillna(0)
    result["building_density"] = result["building_density"].fillna(0)
    result = result.drop(columns="cell_id")

    return result
 
 
def attach_volume_density_to_points(
    points_gdf: gpd.GeoDataFrame, volume_density_grid: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Merges building_volume / building_volume_density back onto your
    raster points table by matching cell_id, the same way
    attach_density_to_points() does for building_density. Kept as a
    separate function/output rather than folded into
    attach_density_to_points() so you can add this feature to an
    existing table without recomputing building_density.

    Returns a copy of points_gdf — with its original index preserved —
    plus building_volume and building_volume_density columns.
    """
    original_index_name = points_gdf.index.name

    points_gdf = points_gdf.copy()
    points_gdf["cell_id"] = points_gdf.index.astype(str)

    volume_lookup = volume_density_grid.set_index("cell_id")[["building_volume", "building_volume_density"]]

    result = points_gdf.merge(
        volume_lookup, on="cell_id", how="left", left_index=False, right_index=False
    )
    result.index = points_gdf.index
    result.index.name = original_index_name

    result["building_volume"] = result["building_volume"].fillna(0)
    result["building_volume_density"] = result["building_volume_density"].fillna(0)
    result = result.drop(columns="cell_id")

    return result


# ----------------------------------------------------------------------
# MAIN (notebook usage)
# ----------------------------------------------------------------------
 
if __name__ == "__main__":
    print(
        "This module is meant to be imported into your notebook, since "
        "your `buildings` and raster point table already exist there.\n\n"
        "from london_building_density import (\n"
        "    select_columns, points_to_cells, compute_density, attach_density_to_points,\n"
        "    compute_volume_density, attach_volume_density_to_points\n"
        ")\n\n"
        "buildings_clean = select_columns(buildings)\n"
        "cells = points_to_cells(raster_df)  # raster_df has lon/lat/NDVI/etc.\n"
        "density_grid = compute_density(buildings_clean, cells)\n"
        "final = attach_density_to_points(raster_df, density_grid)\n\n"
        "# optional: levels-weighted building volume, check building:levels\n"
        "# coverage for your area first — it varies a lot by borough\n"
        "volume_grid = compute_volume_density(buildings_clean, cells)\n"
        "final = attach_volume_density_to_points(final, volume_grid)"
    )