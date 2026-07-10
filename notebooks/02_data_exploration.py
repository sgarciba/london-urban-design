import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import osmnx as ox

# ---------------------- 
# 1. Load the data
# ----------------------
data = gpd.read_file("../data/london_urban_design_data.gpkg")

borough_name = "City of London, London, UK"

borough_gdf = ox.geocode_to_gdf(borough_name)
borough_polygon = borough_gdf.geometry.iloc[0]

borough_mask = data.within(borough_polygon)
borough_data = data[borough_mask]

print(borough_data.shape)
borough_data.head()

# ---------------------- 
# 2. Plot Maps and feature distributions
# ----------------------

feature_cmaps = {
    "LST": "hot",
    "NDVI": "YlGn",
    "NDBI": "RdGy",
    "MNDWI": "Blues",
    "building_density": "viridis",
    "building_volume_density": "plasma",
}

ncols = 2
nrows = (len(feature_cmaps) + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(14, 12))
axes = axes.flatten()

for ax, (col, cmap) in zip(axes, feature_cmaps.items()):
    borough_data.plot(column=col, ax=ax, markersize=1, legend=True, cmap=cmap)
    ax.set_title(col)
    ax.set_axis_off()

for ax in axes[len(feature_cmaps):]:
    ax.set_visible(False)

fig.suptitle(f"Feature Maps — {borough_name}", fontsize=14)
plt.tight_layout()
plt.savefig(
    f"../figures/feature_maps_{borough_name.replace(' ', '_').replace(',', '')}.png",
    dpi=150,
    bbox_inches="tight",
)
plt.show()

# ---------------------- 
# 3. Correlation Heatmap
# ----------------------
corr = borough_data[["NDVI", "NDBI", "LST", "MNDWI", "building_density", "building_volume_density"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig(f"../figures/correlation_heatmap_{borough_name.replace(' ', '_')}.png")
plt.show()



# ---------------------- 
# 4. Histograms of Features
# ----------------------

features = ["NDVI", "NDBI", "LST", "MNDWI", "building_density", "building_volume_density"]

borough_data.hist(column=features, bins=30, figsize=(12, 8), layout=(2, 3), grid=False)
plt.suptitle("Feature Distributions in " + borough_name)
#plt.savefig(f"../figures/feature_distributions_{borough_name.replace(' ', '_')}.png")
plt.show()


