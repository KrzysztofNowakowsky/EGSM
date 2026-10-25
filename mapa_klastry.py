"""Mapa 3 klastrów K-means z obrysem lawy -> mapa_klastry_kmeans.png (slajd 16:9).

Kolory dobrane pod słaby projektor: mocna czerwień, mocny niebieski, jasnoszare tło
klastra stabilnego i gruba czarna granica lawy.

Uruchomienie z folderu repo:
    python mapa_klastry.py
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
from shapely.geometry import box

OUTPUT = "mapa_klastry_kmeans.png"
VENT_LONLAT = (-17.866, 28.613)  # przybliżone położenie stożka Tajogaite

COLORS = {
    "stabilne": "#d4d4d8",
    "osiadanie po erupcji": "#c81e1e",
    "wypiętrzenie 2020–2022": "#1d4ed8",
}
DESCRIPTIONS = {
    "stabilne": "ok. −1 mm/rok, większość obszaru",
    "osiadanie po erupcji": "−27 mm/rok, prawie wyłącznie na lawie",
    "wypiętrzenie 2020–2022": "do ok. +6 cm, na południe od stożka",
}

ps = pd.read_csv("Canaria_results.csv")
lava = gpd.read_file("lawa/EMSR546_lava_2021-12-17.geojson").to_crs(3035)

# Nazwy klastrów jak w notebooku 2: po medianie mean_velocity, nie po numerze
cluster_velocity = ps.groupby("cluster")["mean_velocity"].median().sort_values()
names = {cluster_velocity.index[0]: "osiadanie po erupcji",
         cluster_velocity.index[-1]: "wypiętrzenie 2020–2022"}
for c in cluster_velocity.index[1:-1]:
    names[c] = "stabilne"
ps["cluster_name"] = ps["cluster"].map(names)

cells = gpd.GeoDataFrame(
    ps,
    geometry=[box(e - 50, n - 50, e + 50, n + 50) for e, n in zip(ps["easting"], ps["northing"])],
    crs=3035,
)
vent = gpd.GeoSeries(gpd.points_from_xy([VENT_LONLAT[0]], [VENT_LONLAT[1]]), crs=4326).to_crs(3035)

plt.rcParams["font.family"] = "DejaVu Sans"
fig = plt.figure(figsize=(16, 9), dpi=150, facecolor="white")
ax = fig.add_axes([0.02, 0.03, 0.50, 0.94])

for name in ["stabilne", "wypiętrzenie 2020–2022", "osiadanie po erupcji"]:
    cells[cells["cluster_name"] == name].plot(ax=ax, color=COLORS[name], linewidth=0)
lava.boundary.plot(ax=ax, color="black", linewidth=2.2)
ax.plot(vent.x, vent.y, marker="^", color="black", markersize=16, markeredgecolor="white", markeredgewidth=1.5)
ax.annotate("stożek\nTajogaite", (vent.x.iloc[0], vent.y.iloc[0]), xytext=(14, -6),
            textcoords="offset points", fontsize=15, fontweight="bold", va="top")

# Podziałka 2 km i strzałka północy
xmin, ymin, xmax, ymax = cells.total_bounds
x0, y0 = xmin + 300, ymin + 600
ax.plot([x0, x0 + 2000], [y0, y0], color="black", linewidth=5, solid_capstyle="butt")
ax.text(x0 + 1000, y0 + 250, "2 km", ha="center", va="bottom", fontsize=15)
ax.annotate("N", xy=(xmax - 400, ymax - 300), xytext=(xmax - 400, ymax - 1900),
            ha="center", fontsize=18, fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color="black", linewidth=2.5, mutation_scale=25))
ax.set_aspect("equal")
ax.set_axis_off()

# Legenda i opis po prawej stronie
fig.text(0.56, 0.88, "Klastry K-means (k = 3)", fontsize=30, fontweight="bold")
fig.text(0.56, 0.83, "9155 punktów EGMS, komórki 100 × 100 m, 2020–2024", fontsize=16, color="#4b5563")

counts = ps["cluster_name"].value_counts()
y = 0.70
for name in ["stabilne", "osiadanie po erupcji", "wypiętrzenie 2020–2022"]:
    fig.patches.append(plt.Rectangle((0.56, y - 0.012), 0.03, 0.05, transform=fig.transFigure,
                                     facecolor=COLORS[name], edgecolor="black", linewidth=1))
    fig.text(0.605, y + 0.018, f"{name}  ({counts[name]} pkt)", fontsize=22, fontweight="bold", va="center")
    fig.text(0.605, y - 0.025, DESCRIPTIONS[name], fontsize=17, color="#374151", va="center")
    y -= 0.15

fig.legend(handles=[Patch(facecolor="none", edgecolor="black", linewidth=2.2, label="lawa 2021 (Copernicus EMSR546)")],
           loc="lower left", bbox_to_anchor=(0.555, 0.12), frameon=False, fontsize=17)

fig.savefig(OUTPUT, dpi=150, facecolor="white")
print(f"Zapisano {OUTPUT}")
