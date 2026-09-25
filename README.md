# NP_Project – Dwarf Napier grass under solar panels

Sample IDs `T<irrigation>B<block>R<replication>(<repeat>)`: T1/T2/T3 = 100/75/50 % irrigation, B1 = outside panel shade,
B2 = under solar panel. Harvest cycles 1–5. B3 samples were removed everywhere.

```
NP_Project/
├── image_data/           original leaf photos, cycle-1 … cycle-5
├── data_preprocessed/    leaf-only PNGs (background removed, portrait 960x1280)
├── data/                 measurement tables
│   ├── SPAD_Data.xlsx           whole-leaf SPAD per sample (mean of available readings)
│   ├── Spectrometer_Data.xlsx   reflectance spectra, 380-950 nm
│   ├── Chl_Lab_Data.xlsx        laboratory chlorophyll a, b, total
│   └── backup_original/         untouched originals (before B3 removal / averaging)
├── results/
│   ├── tables/           Leaf_RGB_Data, Leaf_RGB_Histogram, Vegetation_Indices, Spectrometer_Indices (.xlsx)
│   └── figures/          all plots (.svg)
├── report/               NP_Project_Report.pdf (A4) and the figure images it uses
└── scripts/              one script per step (run from anywhere: python scripts/<name>.py)
```

| Script | Does |
|---|---|
| remove_background.py | image_data → data_preprocessed |
| extract_rgb.py | mean leaf R, G, B → results/tables/Leaf_RGB_Data.xlsx |
| rgb_histogram.py | leaf RGB histograms + statistics |
| analyze_vi.py | RGB vegetation indices vs SPAD / lab chlorophyll |
| spectrometer_analysis.py | SRVI, SRRE, NDRE, NDVI, CCCI vs SPAD / lab chlorophyll |
| spectrometer_plot.py | smoothed mean spectra |
| dnn_feasibility.py | small neural network vs ridge, honest (leave-one-cycle-out) vs leaky validation → DNN_feasibility.xlsx |
| what_could_work.py | paired irrigation effect, within-cycle calibration, shade detection → Promising_analyses.xlsx |
| plot_paired_irrigation.py | figure of the paired irrigation effect (T2, T3 vs T1) → paired_irrigation_effect.svg |
| plot_dnn_scatter.py | neural-network predicted vs observed scatter (honest and leaky) → dnn_scatter_honest / dnn_scatter_leaky.svg |
| pca_analysis.py | PCA structure and PCA-before-regression test → PCA_analysis.xlsx, pca_scores.svg |
| stats_summary.py | group means, ANOVA and SPAD-lab correlations → Statistics_summary.xlsx |
| build_report.py | builds the A4 PDF report → report/NP_Project_Report.pdf (needs reportlab) |
| netto_model.py | Netto et al. (2005) SPAD → Chl a / Chl b / carotenoids, compared with the lab values |
| plot_spad_by_cycle.py, plot_spad_vs_chl.py, plot_spad_vs_chl_by_irrigation.py | SPAD and SPAD-vs-lab plots |
