import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

gdf = gpd.read_file('../data/flood_data/bgd_nhr_floods_sparsso.shp')


def map_flood_risk(flood_risk_cat):
  if flood_risk_cat == 0:  # not flood prone
    return 0
  elif flood_risk_cat in [3, 6]:  # low flooding
    return 1
  elif flood_risk_cat in [2, 5, 8]:  # moderate flooding
    return 2
  elif flood_risk_cat in [1, 4, 7]:  # severe flooding
    return 3


gdf['flood_risk'] = gdf['FLOODCAT'].apply(map_flood_risk)

# Plot the GeoDataFrame
ax = gdf.plot(column='flood_risk', cmap='Blues', legend=True, figsize=(10, 10))

# Load the DataFrame
df = pd.read_csv('../data/N1N2.csv')

# Scatter plot on the same axes with smaller dots and outline
sc = ax.scatter(df['lon'], df['lat'], c=df['flood_risk'], cmap='Greens', s=20, alpha=0.55, edgecolor='none')
plt.colorbar(sc, label='Flood Risk')

# Additional plot settings
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Flood Risk by Location')
plt.grid(True)

# Show the plot
plt.show()