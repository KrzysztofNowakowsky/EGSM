# EGSM: La Palma, deformacje terenu i budynki po erupcji 2021

Projekt z przedmiotu „Przetwarzanie i analiza cyfrowych danych przestrzennych i budowlanych metodami AI” (AGH).
Analiza danych EGMS (Ortho, składowa pionowa, 2020–2024) dla zachodniego zbocza Cumbre Vieja na La Palmie,
z uwzględnieniem erupcji wulkanu Tajogaite (19.09–13.12.2021).

Autorzy: Jakub Wilk, Krzysztof Nowakowski, Mikołaj Etrych.

## Kolejność

1. **`EGMS_analysis_pipeline.ipynb`**: cechy szeregów czasowych, Isolation Forest (anomalie), K-means (klastry).
   Wejście: `punkty_cleaned.csv` (przycięte dane EGMS, 54 MB, nie ma go w repo). Wynik: `Canaria_results.csv`.
2. **`EGMS_buildings_analysis.ipynb`**: przypisanie punktów do budynków EUBUCCO (najbliższy punkt do 71 m),
   pole lawowe Copernicus EMSR546, strefy (pod lawą / kipuka / przy granicy / dalej), udział lawy w komórce 100 m,
   priorytety monitoringu, walidacja z oceną uszkodzeń Copernicus, wypiętrzenie 2020–2022.
3. **`dashboard_mapa.py`**: interaktywna mapa `dashboard_la_palma.html` (folium).
4. **`mapa_klastry.py`**: mapa 3 klastrów K-means z obrysem lawy do prezentacji (`mapa_klastry_kmeans.png`).

## Uruchomienie

```bash
# budynki EUBUCCO są spakowane; notebook 2 czyta je z folderu budynki/
mkdir budynki && unrar x budynki-cleaned.rar budynki/   # albo dowolny program do .rar

# po przeliczeniu notebooka 2:
python dashboard_mapa.py [ścieżka/do/punkty_cleaned.csv]
```

Bez `punkty_cleaned.csv` notebook 2 i mapa pomijają tylko wykresy szeregów czasowych i warstwę wypiętrzenia.
Wymagane biblioteki: pandas, numpy, geopandas, shapely, matplotlib, scikit-learn, folium.

## Pliki

| Plik | Opis |
| --- | --- |
| `Canaria_results.csv` | 9155 punktów EGMS z cechami, `anomaly_score`, `anomaly`, `cluster` |
| `budynki-cleaned.rar` | budynki EUBUCCO (ES70) przycięte do obszaru |
| `lawa/` | zasięg lawy z 17.12.2021 i ocena uszkodzeń budynków (Copernicus EMS EMSR546) |
| `budynki_LaPalma_results.gpkg` | budynki z cechami punktu, strefą lawy i priorytetem + warstwa lawy |
| `budynki_do_analizy.csv` | budynki z priorytetem 1 i 2 |
| `punkty_priorytet1.csv` | punkty EGMS stojące za priorytetem 1 |
| `interpolation.tif` | interpolacja prędkości (raster 4 m); środek obszaru na południu to brak danych, nie pomiar |
| `*.png` | wykresy do prezentacji: mapa priorytetów, profil odległości od lawy, szeregi, wypiętrzenie |
| `dashboard_la_palma.html` | interaktywna mapa: otwórz w przeglądarce |

## Najważniejsze wyniki

- 417 z 458 anomalii Isolation Forest leży na samym polu lawowym: model wykrył lawę, nie zagrożone budynki.
- 2891 budynków z bazy leży pod lawą; walidacja z Copernicus: 2173 budynki oznaczone jako zniszczone są w strefie „pod lawą”, 0 daleko od lawy.
- Priorytet 1: 137 budynków (44 punkty EGMS) przy krawędzi lawy i w kipukach, osiadających szybciej niż 10 mm/rok.
- Osiadanie maleje z odległością od lawy: −15,3 mm/rok w pasie 0–25 m, −0,9 mm/rok ponad 1 km dalej.
- Na południe od stożka teren unosił się od jesieni 2020 (ok. +3 cm przed erupcją, ok. +6 cm do końca 2022),
  zgodnie z Intrieri i in. (2026), *Remote Sensing* 18, 485, doi:10.3390/rs18030485.

## Źródła danych

- EGMS: https://egms.land.copernicus.eu/
- EUBUCCO: https://eubucco.com/
- Copernicus EMS EMSR546: https://mapping.emergency.copernicus.eu/activations/EMSR546/
