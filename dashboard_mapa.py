"""Interaktywna mapa wyników dla La Palmy (folium) -> dashboard_la_palma.html.

Warstwy: lawa 2021 (Copernicus EMSR546), budynki wg priorytetu monitoringu,
klastry K-means (komórki 100 m), wypiętrzenie XII 2020 -> XII 2022, anomalie
Isolation Forest. Popupy budynków z priorytetem 1 i 2 oraz anomalii zawierają
wykres szeregu czasowego przypisanego punktu EGMS.

Uruchomienie (z folderu repo, po przeliczeniu EGMS_buildings_analysis.ipynb):
    python dashboard_mapa.py [ścieżka do punkty_cleaned.csv]

Bez punkty_cleaned.csv (nie ma go w repo) mapa powstaje bez wykresów
i bez warstwy wypiętrzenia.
"""

import sys
from pathlib import Path

import branca.colormap as cm
import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import box

OUTPUT = "dashboard_la_palma.html"
ERUPTION = ("2021-09-19", "2021-12-13")
VENT_LONLAT = (-17.866, 28.613)  # przybliżone położenie stożka Tajogaite

PRIORITY_COLORS = {
    "1 - wysoki": "#b2182b",
    "2 - średni": "#ef8a62",
    "3 - niski": "#bdbdbd",
    "brak danych": "#f0f0f0",
    "zniszczony (pod lawą)": "#facc15",  # żółty: dobrze widoczny na lawie, także z projektora
}
LAVA_COLOR = "#9a3412"  # rdzawa czerwień lawy
LAVA_EDGE = "#431407"
TYPE_PL = {"residential": "mieszkalny", "non-residential": "niemieszkalny"}
SUBTYPE_PL = {
    "detached": "dom wolnostojący", "semi-detached": "bliźniak", "terraced": "szeregowiec",
    "apartment": "budynek wielorodzinny", "agricultural": "rolniczy", "industrial": "przemysłowy",
    "commercial": "usługowy", "public": "publiczny", "others": "inny",
}
CLUSTER_COLORS = {
    "osiadanie po erupcji": "#b2182b",
    "stabilne": "#d9d9d9",
    "wypiętrzenie 2020–2022": "#2166ac",
}


# ---------- dane ----------

ps = pd.read_csv("Canaria_results.csv")
buildings = gpd.read_file("budynki_LaPalma_results.gpkg", layer="budynki_EGMS")
lava = gpd.read_file("lawa/EMSR546_lava_2021-12-17.geojson").to_crs(3035)

# Nazwy klastrów jak w notebooku 2: po medianie mean_velocity, nie po numerze
cluster_velocity = ps.groupby("cluster")["mean_velocity"].median().sort_values()
cluster_names = {cluster_velocity.index[0]: "osiadanie po erupcji",
                 cluster_velocity.index[-1]: "wypiętrzenie 2020–2022"}
for c in cluster_velocity.index[1:-1]:
    cluster_names[c] = "stabilne"
ps["cluster_name"] = ps["cluster"].map(cluster_names)

ts_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("punkty_cleaned.csv")
if not ts_path.exists():
    ts_path = Path("..") / "punkty_cleaned.csv"

series = None
if ts_path.exists():
    ts = pd.read_csv(ts_path)
    date_cols = [c for c in ts.columns if c.isdigit() and len(c) == 8]
    dates = pd.to_datetime(date_cols, format="%Y%m%d")
    series = ts.set_index("pid")[date_cols].astype(float)

    def window_mean(start, end):
        mask = (dates >= start) & (dates <= end)
        return series.loc[:, mask].mean(axis=1)

    uplift = window_mean("2022-10-01", "2022-12-31") - window_mean("2020-11-01", "2021-01-31")
    ps["uplift_mm"] = ps["pid"].map(uplift).round(1)
else:
    print(f"Brak {ts_path} — mapa bez wykresów szeregów i bez warstwy wypiętrzenia.")


# ---------- wykres szeregu jako SVG w popupie ----------

def series_svg(pid, width=300, height=130):
    if series is None or pid not in series.index:
        return ""
    values = series.loc[pid].to_numpy()[::3]
    t = dates[::3]
    t0, t1 = pd.Timestamp("2020-01-01"), pd.Timestamp("2025-01-01")
    pad_l, pad_r, pad_t, pad_b = 34, 16, 8, 18
    w, h = width - pad_l - pad_r, height - pad_t - pad_b
    lo, hi = min(values.min(), 0), max(values.max(), 0)
    span = (hi - lo) or 1.0
    lo, hi = lo - 0.08 * span, hi + 0.08 * span

    def x(day):
        return pad_l + (day - t0) / (t1 - t0) * w

    def y(v):
        return pad_t + (hi - v) / (hi - lo) * h

    points = " ".join(f"{x(d):.0f},{y(v):.0f}" for d, v in zip(t, values))
    e0, e1 = x(pd.Timestamp(ERUPTION[0])), x(pd.Timestamp(ERUPTION[1]))
    years = "".join(
        f'<text x="{x(pd.Timestamp(f"{yr}-01-01")):.0f}" y="{height - 4}" font-size="9" '
        f'text-anchor="middle" fill="#666">{yr}</text>'
        for yr in range(2020, 2026)
    )
    return (
        f'<svg width="{width}" height="{height}" style="font-family:sans-serif">'
        f'<rect x="{e0:.1f}" y="{pad_t}" width="{e1 - e0:.1f}" height="{h}" fill="#fdd0c4"/>'
        f'<line x1="{pad_l}" x2="{pad_l + w}" y1="{y(0):.1f}" y2="{y(0):.1f}" stroke="#999" stroke-dasharray="3,3"/>'
        f'<polyline points="{points}" fill="none" stroke="#1f6fb4" stroke-width="1.5"/>'
        f'<text x="{pad_l - 4}" y="{pad_t + 8}" font-size="9" text-anchor="end" fill="#666">{hi:.0f}</text>'
        f'<text x="{pad_l - 4}" y="{pad_t + h}" font-size="9" text-anchor="end" fill="#666">{lo:.0f}</text>'
        f'<text x="{pad_l - 4}" y="{y(0) + 3:.1f}" font-size="9" text-anchor="end" fill="#666">0</text>'
        f"{years}</svg>"
        '<div style="font-size:10px;color:#666">przemieszczenie [mm]; różowe tło = erupcja 19.09–13.12.2021</div>'
    )


def fmt(value, digits=1, unit=""):
    if pd.isna(value):
        return "—"
    return f"{value:.{digits}f}{unit}"


def building_popup(row):
    rows = [
        ("Typ", f"{TYPE_PL.get(row['type'], '—')}, {SUBTYPE_PL.get(row['subtype'], '—')}"),
        ("Kondygnacje, rok budowy", f"{fmt(row['floors'], 0)}, {fmt(row['const_yr'], 0)}"),
        ("Strefa lawy", row["lava_zone"]),
        ("Odległość od lawy", fmt(row["dist_lava"], 0, " m")),
        ("Lawa w komórce 100 m", fmt(row["cell_lava_share"] * 100, 0, "%")),
        ("Punkt EGMS", f"{row['pid']} ({fmt(row['distance_PS'], 0, ' m')})"),
        ("Średnia prędkość", fmt(row["mean_velocity"], 1, " mm/rok")),
        ("Zmiana trendu (ost. 12 mies.)", fmt(row["trend_change"], 1, " mm/rok")),
        ("Anomalia (Isolation Forest)", "tak" if row["anomaly"] else "nie"),
        ("Klaster", row["cluster_name"]),
    ]
    table = "".join(f"<tr><td style='color:#666;padding-right:8px'>{k}</td><td><b>{v}</b></td></tr>" for k, v in rows)
    return (f"<div style='font-family:sans-serif;font-size:12px'>"
            f"<div style='font-size:14px;font-weight:bold;color:{PRIORITY_COLORS[row['priority_label']]}'>"
            f"Priorytet {row['priority_label']}</div><table>{table}</table>{series_svg(row['pid'])}</div>")


def point_popup(row):
    return (f"<div style='font-family:sans-serif;font-size:12px'><b>Punkt EGMS {row['pid']}</b><br>"
            f"klaster: {row['cluster_name']}<br>prędkość: {row['mean_velocity']:.1f} mm/rok<br>"
            f"anomaly_score: {row['anomaly_score']:.3f}{series_svg(row['pid'])}</div>")


# ---------- warstwy geometryczne w WGS84 ----------

def to_wgs(gdf, simplify=None):
    # Zaokrąglenie do 1e-6° (ok. 0,1 m) mocno zmniejsza rozmiar HTML
    gdf = gdf.copy()
    if simplify:
        gdf["geometry"] = gdf.geometry.simplify(simplify)
    gdf = gdf.to_crs(4326)
    gdf["geometry"] = shapely.set_precision(gdf.geometry.values, 1e-6)
    return gdf


cells = gpd.GeoDataFrame(
    ps.copy(),
    geometry=[box(e - 50, n - 50, e + 50, n + 50) for e, n in zip(ps["easting"], ps["northing"])],
    crs=3035,
)
points = gpd.GeoDataFrame(ps.copy(), geometry=gpd.points_from_xy(ps["easting"], ps["northing"]), crs=3035)

b = buildings.copy()
b["color"] = b["priority_label"].map(PRIORITY_COLORS)
b["typ"] = b["subtype"].map(SUBTYPE_PL).fillna(b["type"].map(TYPE_PL))
important = b[b["priority"].isin([1, 2])].copy()
important["popup"] = important.apply(building_popup, axis=1)

lava_wgs = to_wgs(lava, simplify=2)
center = lava_wgs.geometry.union_all().centroid

m = folium.Map(location=[center.y - 0.03, center.x], zoom_start=13, tiles=None, control_scale=True)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri World Imagery", name="Zdjęcia satelitarne",
).add_to(m)
folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
# Jasny podkład Esri (bez klucza API; CARTO wymaga teraz klucza). Dodany jako ostatni = domyślny.
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
    attr="Esri World Light Gray Canvas", name="Mapa jasna", max_native_zoom=16, max_zoom=19,
).add_to(m)

# Klastry K-means: komórki tego samego klastra łączymy w jeden wielokąt (mniejszy plik)
cluster_areas = (
    cells.groupby("cluster_name")
    .agg(punkty=("pid", "count"), mediana_v=("mean_velocity", "median"), geometry=("geometry", lambda g: g.union_all()))
    .reset_index()
)
cluster_areas = gpd.GeoDataFrame(cluster_areas, geometry="geometry", crs=3035)
cluster_areas["mediana_v"] = cluster_areas["mediana_v"].round(1)
fg = folium.FeatureGroup(name="Klastry K-means (komórki 100 m)", show=False)
folium.GeoJson(
    to_wgs(cluster_areas),
    style_function=lambda f: {"fillColor": CLUSTER_COLORS[f["properties"]["cluster_name"]],
                              "color": None, "weight": 0, "fillOpacity": 0.55},
    tooltip=folium.GeoJsonTooltip(["cluster_name", "punkty", "mediana_v"],
                                  aliases=["klaster", "punktów", "mediana prędkości [mm/rok]"]),
).add_to(fg)
fg.add_to(m)

# Wypiętrzenie XII 2020 -> XII 2022
if "uplift_mm" in ps:
    uplift_cmap = cm.LinearColormap(["#b2182b", "#f7f7f7", "#2166ac"], vmin=-60, vmax=60,
                                    caption="Zmiana wysokości XII 2020 → XII 2022 [mm]")
    # Zmianę wysokości grupujemy w przedziały co 10 mm i łączymy komórki z tego samego przedziału
    edges = np.arange(-60, 70, 10)
    cells["uplift_bin"] = np.clip(np.digitize(cells["pid"].map(ps.set_index("pid")["uplift_mm"]), edges), 1, len(edges) - 1)
    uplift_areas = (
        cells.groupby("uplift_bin")
        .agg(punkty=("pid", "count"), geometry=("geometry", lambda g: g.union_all()))
        .reset_index()
    )
    uplift_areas = gpd.GeoDataFrame(uplift_areas, geometry="geometry", crs=3035)
    uplift_areas["przedzial"] = [f"{edges[i - 1]} do {edges[i]} mm" for i in uplift_areas["uplift_bin"]]
    uplift_areas["srodek"] = [(edges[i - 1] + edges[i]) / 2 for i in uplift_areas["uplift_bin"]]
    fg = folium.FeatureGroup(name="Wypiętrzenie XII 2020 → XII 2022", show=False)
    folium.GeoJson(
        to_wgs(uplift_areas[["przedzial", "punkty", "srodek", "geometry"]]),
        style_function=lambda f: {"fillColor": uplift_cmap(f["properties"]["srodek"]),
                                  "color": None, "weight": 0, "fillOpacity": 0.7},
        tooltip=folium.GeoJsonTooltip(["przedzial", "punkty"], aliases=["zmiana wysokości", "punktów"]),
    ).add_to(fg)
    fg.add_to(m)

# Lawa
folium.GeoJson(
    lava_wgs[["geometry"]], name="Lawa 2021 (Copernicus EMSR546)",
    style_function=lambda f: {"fillColor": LAVA_COLOR, "color": LAVA_EDGE, "weight": 1.5, "fillOpacity": 0.6},
).add_to(m)

# Budynki
fg = folium.FeatureGroup(name="Budynki: priorytet 3 i brak danych", show=False)
folium.GeoJson(
    to_wgs(b[b["priority"].isin([0, 3])][["priority_label", "typ", "color", "geometry"]], simplify=0.5),
    style_function=lambda f: {"fillColor": f["properties"]["color"], "color": "#999", "weight": 0.3, "fillOpacity": 0.8},
    tooltip=folium.GeoJsonTooltip(["priority_label", "typ"], aliases=["priorytet", "typ"]),
).add_to(fg)
fg.add_to(m)

fg = folium.FeatureGroup(name="Budynki zniszczone (pod lawą)", show=True)
folium.GeoJson(
    to_wgs(b[b["priority"] == -1][["priority_label", "typ", "color", "geometry"]], simplify=0.5),
    style_function=lambda f: {"fillColor": f["properties"]["color"], "color": "#422006", "weight": 0.9, "fillOpacity": 1},
    tooltip=folium.GeoJsonTooltip(["priority_label", "typ"], aliases=["status", "typ"]),
).add_to(fg)
fg.add_to(m)

fg = folium.FeatureGroup(name="Budynki: priorytet 1 i 2 (kliknij)", show=True)
folium.GeoJson(
    to_wgs(important[["priority_label", "color", "popup", "geometry"]], simplify=0.5),
    style_function=lambda f: {"fillColor": f["properties"]["color"], "color": "#000", "weight": 0.8, "fillOpacity": 0.95},
    highlight_function=lambda f: {"weight": 3, "color": "#ffcc00"},
    tooltip=folium.GeoJsonTooltip(["priority_label"], aliases=["priorytet"]),
    popup=folium.GeoJsonPopup(["popup"], labels=False, max_width=340),
).add_to(fg)
fg.add_to(m)

# Anomalie Isolation Forest
anomalies = points[points["anomaly"]].copy()
anomalies["popup"] = anomalies.apply(point_popup, axis=1)
fg = folium.FeatureGroup(name=f"Anomalie Isolation Forest ({len(anomalies)})", show=False)
folium.GeoJson(
    to_wgs(anomalies[["pid", "popup", "geometry"]]),
    marker=folium.CircleMarker(radius=4, fill=True, fill_opacity=0.9, color="#7b3294", weight=1),
    tooltip=folium.GeoJsonTooltip(["pid"], aliases=["anomalia"]),
    popup=folium.GeoJsonPopup(["popup"], labels=False, max_width=340),
).add_to(fg)
fg.add_to(m)

# Trójkąt z CSS zamiast ikony Font Awesome: mapa nie zależy od czcionki z CDN
folium.Marker(
    [VENT_LONLAT[1], VENT_LONLAT[0]], tooltip="Stożek Tajogaite (przybliżone położenie)",
    icon=folium.DivIcon(
        html='<div style="width:0;height:0;border-left:11px solid transparent;border-right:11px solid transparent;'
             'border-bottom:19px solid #111827;filter:drop-shadow(0 0 2px #fff)"></div>',
        icon_size=(22, 19), icon_anchor=(11, 19),
    ),
).add_to(m)

folium.LayerControl(collapsed=False).add_to(m)
xmin, ymin, xmax, ymax = lava_wgs.total_bounds
m.fit_bounds([[ymin, xmin], [ymax, xmax]])


# ---------- układ strony: pasek górny, mapa jako karta, panel boczny ----------

counts = b["priority_label"].value_counts()
n_p1_points = b.loc[b["priority"] == 1, "pid"].nunique()
n_anomalies = int(ps["anomaly"].sum())
n_anomalies_on_lava = int(points.loc[points["anomaly"], "geometry"].within(lava.geometry.union_all()).sum())


def swatch(color, label, shape="square"):
    radius = "50%" if shape == "dot" else "3px"
    return (f'<div class="wiersz"><i style="background:{color};border-radius:{radius}"></i>'
            f"<span>{label}</span></div>")


priority_legend = "".join(
    swatch(c, f"{name}<b>{counts.get(name, 0)}</b>") for name, c in PRIORITY_COLORS.items()
)
cluster_legend = "".join(swatch(c, name) for name, c in CLUSTER_COLORS.items())

page = f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {{
    --tlo: #f4f5f7; --karta: #fff; --ramka: #e3e6ea; --tekst: #111827; --tekst-2: #4b5563;
    --wyciszony: #6b7280; --akcent: #b2182b; --promien: 10px;
    --font: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }}
  html, body {{ background: var(--tlo); color: var(--tekst); font: 14px/1.5 var(--font); }}
  .topbar {{
    position: fixed; top: 0; left: 0; right: 0; height: 64px; z-index: 1000;
    display: flex; align-items: center; justify-content: space-between; gap: 16px;
    padding: 0 24px; background: var(--karta); border-bottom: 1px solid var(--ramka);
  }}
  html, body {{ overflow: hidden; }}
  .marka {{ display: flex; align-items: center; gap: 12px; min-width: 0; }}
  .marka div {{ min-width: 0; }}
  .znak {{
    flex: none; width: 30px; height: 30px; border-radius: 8px; background: var(--akcent);
    clip-path: polygon(50% 8%, 92% 92%, 8% 92%);
  }}
  .topbar h1 {{ margin: 0; font-size: 16px; font-weight: 650; letter-spacing: -.01em;
               white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .topbar p {{ margin: 1px 0 0; font-size: 12.5px; color: var(--wyciszony);
              white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .zrodla {{ flex: none; font-size: 12px; color: var(--wyciszony); }}

  .folium-map {{
    position: fixed !important; top: 76px !important; left: 16px !important; right: 352px !important;
    bottom: 16px !important; width: auto !important; height: auto !important;
    border: 1px solid var(--ramka); border-radius: var(--promien); overflow: hidden;
  }}
  aside.panel {{
    position: fixed; top: 76px; right: 16px; bottom: 16px; width: 324px; z-index: 900;
    padding: 18px; overflow-y: auto; background: var(--karta);
    border: 1px solid var(--ramka); border-radius: var(--promien);
  }}
  .etykieta {{
    display: block; margin: 0 0 8px; color: var(--wyciszony);
    font-size: 11px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase;
  }}
  .sekcja {{ margin-top: 18px; padding-top: 16px; border-top: 1px solid var(--ramka); }}
  .liczby {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
  .liczby div {{ padding: 10px 12px; border: 1px solid var(--ramka); border-radius: 8px; }}
  .liczby b {{ display: block; font-size: 22px; font-weight: 600; line-height: 1.1; letter-spacing: -.02em; }}
  .liczby b.akcent {{ color: var(--akcent); }}
  .liczby small {{ display: block; margin-top: 2px; color: var(--wyciszony); font-size: 11.5px; line-height: 1.35; }}
  .wiersz {{ display: flex; align-items: center; gap: 8px; margin: 5px 0; font-size: 13px; }}
  .wiersz i {{ flex: none; width: 12px; height: 12px; border: 1px solid rgba(0,0,0,.15); }}
  .wiersz span {{ flex: 1; display: flex; justify-content: space-between; }}
  .wiersz span b {{ font-weight: 600; }}
  .rampa {{ height: 8px; border-radius: 4px; background: linear-gradient(to right, #b2182b, #f7f7f7, #2166ac); }}
  .skala {{ display: flex; justify-content: space-between; margin-top: 4px; color: var(--wyciszony); font-size: 11.5px; }}
  .opis {{ margin: 6px 0 0; color: var(--tekst-2); font-size: 12.5px; line-height: 1.6; }}

  .leaflet-container {{ font-family: var(--font); background: #e8ecef; }}
  .leaflet-control-attribution {{ font-size: 10px !important; }}
  .leaflet-control-layers {{
    border: 1px solid var(--ramka) !important; border-radius: var(--promien) !important;
    box-shadow: none !important; font-size: 12.5px; color: var(--tekst);
  }}
  .leaflet-control-layers-expanded {{ padding: 10px 12px 10px 10px; }}
  .leaflet-control-layers-separator {{ border-top-color: var(--ramka); }}
  .leaflet-control-layers label {{ margin: 2px 0; }}
  .leaflet-bar {{ border: 1px solid var(--ramka) !important; border-radius: 8px !important; box-shadow: none !important; }}
  .leaflet-bar a {{ color: var(--tekst); }}
  .leaflet-popup-content-wrapper {{ border-radius: var(--promien); box-shadow: 0 4px 18px rgba(17,24,39,.14); }}
  .leaflet-popup-content {{ margin: 14px 16px; }}
  .leaflet-popup-content table td {{ padding: 1px 0; vertical-align: top; }}
  .leaflet-tooltip {{ border: 1px solid var(--ramka); border-radius: 6px; box-shadow: none; font-family: var(--font); }}

  @media (max-width: 860px) {{
    .topbar {{ height: 56px; padding: 0 12px; }}
    .topbar h1 {{ font-size: 14px; }}
    .topbar p {{ font-size: 11.5px; }}
    .znak {{ width: 24px; height: 24px; }}
    .zrodla {{ display: none; }}
    .folium-map {{ top: 64px !important; left: 8px !important; right: 8px !important; bottom: 42vh !important; }}
    aside.panel {{ top: auto; left: 8px; right: 8px; bottom: 8px; width: auto; height: calc(42vh - 16px); }}
  }}
</style>

<header class="topbar">
  <div class="marka">
    <span class="znak" aria-hidden="true"></span>
    <div>
      <h1>La Palma · deformacje terenu i budynki po erupcji 2021</h1>
      <p>EGMS Ortho 2020–2024 · EUBUCCO · Copernicus EMSR546 · AGH, projekt grupowy</p>
    </div>
  </div>
  <div class="zrodla">Copernicus · EGMS · EUBUCCO · Esri · OpenStreetMap</div>
</header>

<aside class="panel">
  <span class="etykieta">W skrócie</span>
  <div class="liczby">
    <div><b class="akcent">{counts.get('1 - wysoki', 0)}</b><small>budynków z priorytetem 1 ({n_p1_points} punktów EGMS)</small></div>
    <div><b>{counts.get('zniszczony (pod lawą)', 0)}</b><small>budynków pod lawą</small></div>
    <div><b>{n_anomalies_on_lava}/{n_anomalies}</b><small>anomalii leży na lawie</small></div>
    <div><b>{len(ps)}</b><small>punktów EGMS, siatka 100 m</small></div>
  </div>

  <div class="sekcja">
    <span class="etykieta">Priorytet monitoringu</span>
    {priority_legend}
    <p class="opis">Kliknij budynek z priorytetem 1 lub 2, żeby zobaczyć jego dane i szereg czasowy punktu EGMS.</p>
  </div>

  <div class="sekcja">
    <span class="etykieta">Pozostałe warstwy</span>
    {swatch(LAVA_COLOR, "Lawa 2021 (Copernicus EMSR546)")}
    {swatch("#7b3294", "Anomalia Isolation Forest", "dot")}
    <p class="opis" style="margin-top:10px">Klastry K-means</p>
    {cluster_legend}
    <p class="opis" style="margin-top:10px">Zmiana wysokości XII 2020 → XII 2022</p>
    <div class="rampa" style="margin-top:6px"></div>
    <div class="skala"><span>−60 mm</span><span>0</span><span>+60 mm</span></div>
    <p class="opis">Warstwy włączasz w prawym górnym rogu mapy.</p>
  </div>

  <div class="sekcja">
    <span class="etykieta">Metoda</span>
    <p class="opis">Każdy budynek dostaje najbliższy punkt EGMS do 71 m (połowa przekątnej komórki 100 m).
      Priorytet 1: punkt poza lawą, lawa zajmuje mniej niż 25% komórki, osiadanie szybsze niż 10 mm/rok
      albo anomalia. Budynki pod lawą są wyłączone z listy.</p>
  </div>
</aside>

<script>
  // Na małym ekranie lista warstw startuje zwinięta, żeby nie zasłaniała mapy
  window.addEventListener("load", function () {{
    if (window.innerWidth < 860) {{
      document.querySelectorAll(".leaflet-control-layers")
        .forEach(function (el) {{ el.classList.remove("leaflet-control-layers-expanded"); }});
    }}
  }});
</script>
"""
m.get_root().html.add_child(folium.Element(page))

m.save(OUTPUT)
size_mb = Path(OUTPUT).stat().st_size / 1e6
print(f"Zapisano {OUTPUT} ({size_mb:.1f} MB)")
