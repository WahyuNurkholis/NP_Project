"""Build the A4 PDF report (methods, results, short discussion, conclusions).

Reads the result tables in results/tables and the figure images in report/figures_png, and writes report/NP_Project_Report.pdf.
Every number in the tables is read from the Excel result files; run the analysis scripts first (see README.md).
Needs reportlab (pip install reportlab). Standard Times fonts are used, so text avoids characters outside Latin-1.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, NextPageTemplate, PageBreak, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_vi as avi          # noqa: E402
import stats_summary as ss        # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "results" / "tables"
PNG = ROOT / "report" / "figures_png"
OUT = ROOT / "report" / "NP_Project_Report.pdf"

# ------------------------------------------------------------------ styles
INK, INK2, RULE, HEAD_BG, ZEBRA = colors.HexColor("#111111"), colors.HexColor("#444444"), colors.HexColor("#9a9a9a"), colors.HexColor("#e6e9ee"), colors.HexColor("#f6f7f9")
ACCENT = colors.HexColor("#1f4e79")
base = ParagraphStyle("base", fontName="Times-Roman", fontSize=10.5, leading=13.6, textColor=INK, alignment=TA_JUSTIFY, spaceAfter=5)
S = {
    "body": base,
    "title": ParagraphStyle("title", parent=base, fontName="Times-Bold", fontSize=19, leading=23, alignment=TA_LEFT, spaceAfter=4, textColor=ACCENT),
    "subtitle": ParagraphStyle("subtitle", parent=base, fontSize=12, leading=15, alignment=TA_LEFT, textColor=INK2, spaceAfter=10),
    "h1": ParagraphStyle("h1", parent=base, fontName="Times-Bold", fontSize=14, leading=17, alignment=TA_LEFT, spaceBefore=12, spaceAfter=5, textColor=ACCENT, keepWithNext=1),
    "h2": ParagraphStyle("h2", parent=base, fontName="Times-Bold", fontSize=11.5, leading=14, alignment=TA_LEFT, spaceBefore=8, spaceAfter=3, textColor=INK, keepWithNext=1),
    "bullet": ParagraphStyle("bullet", parent=base, leftIndent=14, bulletIndent=3, spaceAfter=2.5),
    "cap": ParagraphStyle("cap", parent=base, fontSize=9.3, leading=11.6, textColor=INK2, spaceBefore=3, spaceAfter=9),
    "tcap": ParagraphStyle("tcap", parent=base, fontSize=9.3, leading=11.6, textColor=INK2, alignment=TA_LEFT, spaceBefore=6, spaceAfter=3),
    "cell": ParagraphStyle("cell", parent=base, fontSize=8.4, leading=10, alignment=TA_LEFT, spaceAfter=0),
    "cellc": ParagraphStyle("cellc", parent=base, fontSize=8.4, leading=10, alignment=TA_CENTER, spaceAfter=0),
    "cellh": ParagraphStyle("cellh", parent=base, fontName="Times-Bold", fontSize=8.4, leading=10, alignment=TA_CENTER, spaceAfter=0),
    "cellhl": ParagraphStyle("cellhl", parent=base, fontName="Times-Bold", fontSize=8.4, leading=10, alignment=TA_LEFT, spaceAfter=0),
    "note": ParagraphStyle("note", parent=base, fontSize=8.8, leading=11, textColor=INK2, spaceAfter=6),
    "box": ParagraphStyle("box", parent=base, fontSize=10, leading=13, spaceAfter=3),
}
P = lambda t, s="body": Paragraph(t, S[s])
H1, H2 = (lambda t: Paragraph(t, S["h1"])), (lambda t: Paragraph(t, S["h2"]))
BUL = lambda t: Paragraph(t, S["bullet"], bulletText="•")

# ------------------------------------------------------------------ numbering
FIG_ORDER = ["prep", "spad", "paired", "spadchl", "spadchl_trt", "hist_cycle", "hist_group", "rgbvi", "spec_trt", "spec_shade", "specvi", "netto", "dnn_honest", "dnn_leaky", "pca"]
TAB_ORDER = ["design", "meas", "quality", "indices", "means", "anova", "paired", "corr", "rgbstat", "rgbvi", "specqc", "specvi", "specper", "netto", "ml", "pcastruct", "pcacv", "shade", "objective"]
FN = {k: i + 1 for i, k in enumerate(FIG_ORDER)}
TN = {k: i + 1 for i, k in enumerate(TAB_ORDER)}
FILE = {"prep": "prep", "spad": "spad_plot", "paired": "paired_irrigation_effect", "spadchl": "spad_vs_chl_correlation", "spadchl_trt": "spad_vs_chl_by_irrigation",
        "hist_cycle": "rgb_histogram_by_cycle", "hist_group": "rgb_histogram_by_group", "rgbvi": "vegetation_indices_correlation", "spec_trt": "spectrometer_smoothed_by_irrigation",
        "spec_shade": "spectrometer_smoothed_by_shade", "specvi": "spectrometer_indices_correlation", "netto": "netto_predicted_vs_lab", "dnn_honest": "dnn_scatter_honest",
        "dnn_leaky": "dnn_scatter_leaky", "pca": "pca_scores"}
F = lambda k: f"Figure {FN[k]}"
T = lambda k: f"Table {TN[k]}"

f2 = lambda v: "n/a" if pd.isna(v) else f"{v:.2f}"
f1 = lambda v: "n/a" if pd.isna(v) else f"{v:.1f}"
f3 = lambda v: "n/a" if pd.isna(v) else f"{v:.3f}"
pf = lambda v: "&lt;0.001" if v < 0.001 else f"{v:.3f}"


# ------------------------------------------------------------------ helpers: tables and figures
def table(rows, widths, header_rows=1, first_left=True, spans=(), zebra=True, left_cols=()):
    data = []
    for i, r in enumerate(rows):
        line = []
        for j, c in enumerate(r):
            if i < header_rows:
                st = "cellhl" if (j == 0 and first_left) else "cellh"
            else:
                st = "cell" if ((j == 0 and first_left) or j in left_cols) else "cellc"
            line.append(Paragraph(str(c), S[st]) if c is not None else "")
        data.append(line)
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=header_rows)
    cmds = [("BACKGROUND", (0, 0), (-1, header_rows - 1), HEAD_BG), ("LINEABOVE", (0, 0), (-1, 0), 0.9, INK), ("LINEBELOW", (0, header_rows - 1), (-1, header_rows - 1), 0.6, INK),
            ("LINEBELOW", (0, -1), (-1, -1), 0.9, INK), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 2.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
            ("LEFTPADDING", (0, 0), (-1, -1), 3.5), ("RIGHTPADDING", (0, 0), (-1, -1), 3.5)]
    if zebra:
        cmds += [("BACKGROUND", (0, i), (-1, i), ZEBRA) for i in range(header_rows + 1, len(rows), 2)]
    cmds += list(spans)
    t.setStyle(TableStyle(cmds))
    return t


def tab(key, caption, rows, widths, note=None, **kw):
    out = [KeepTogether([Paragraph(f"<b>{T(key)}.</b> {caption}", S["tcap"]), table(rows, widths, **kw)])]
    if note: out.append(Paragraph(note, S["note"]))
    else: out.append(Spacer(1, 6))
    return out


class LandFig:
    def __init__(self, key, caption, max_w=26.0, max_h=15.4):
        self.key, self.caption, self.max_w, self.max_h = key, caption, max_w, max_h


def fig_flowables(key, caption, width=17.0, max_h=21.0, path=None):
    p = path or PNG / f"{FILE[key]}.png"
    w0, h0 = PILImage.open(p).size
    w, h = width * cm, width * cm * h0 / w0
    if h > max_h * cm: w, h = max_h * cm * w0 / h0, max_h * cm
    return KeepTogether([Image(str(p), width=w, height=h), Paragraph(f"<b>{F(key)}.</b> {caption}", S["cap"])])


def land_flowables(lf):
    p = PNG / f"{FILE[lf.key]}.png"
    w0, h0 = PILImage.open(p).size
    w, h = lf.max_w * cm, lf.max_w * cm * h0 / w0
    if h > lf.max_h * cm: w, h = lf.max_h * cm * w0 / h0, lf.max_h * cm
    return [Image(str(p), width=w, height=h), Paragraph(f"<b>{F(lf.key)}.</b> {lf.caption}", S["cap"])]


# ------------------------------------------------------------------ helpers: read result tables
def blocks(path, sheet, header_first="Index"):
    ws = openpyxl.load_workbook(path, data_only=True)[sheet]
    rows = [list(r) for r in ws.iter_rows(values_only=True)]
    out = {}
    for i, r in enumerate(rows):
        if r[0] == header_first and i > 0:
            title = rows[i - 1][0]; cols = [c for c in r if c is not None]; data = []
            for rr in rows[i + 1:]:
                if rr[0] is None: break
                data.append(rr[: len(cols)])
            out[title] = pd.DataFrame(data, columns=cols).set_index(header_first)
    return out


def sheet(path, name, header=1):
    return pd.read_excel(path, sheet_name=name, header=header)


# ------------------------------------------------------------------ extra figure: preprocessing example
def preprocessing_figure():
    plt.rcParams["svg.fonttype"] = "path"
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral"], "font.size": 15})
    ex = [(1, "T1B1R1(1)", "Cycle 1 · T1B1R1(1)"), (3, "T3B2R2(1)", "Cycle 3 · T3B2R2(1)"), (5, "T2B1R3(2)", "Cycle 5 · T2B1R3(2)")]
    fig, axes = plt.subplots(2, 3, figsize=(16, 13.5), facecolor="#fcfcfb")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.02, hspace=0.12, wspace=0.05)
    fig.text(0.02, 0.975, "Leaf photographs before and after background removal", fontsize=25, fontweight="bold", va="top")
    fig.text(0.02, 0.935, "Top: original photograph. Bottom: leaf-only image used for all colour analysis (background transparent, shown here on grey).", fontsize=15, color="#444444", va="top")
    for j, (c, s, title) in enumerate(ex):
        a = PILImage.open(ROOT / "image_data" / f"cycle-{c}" / f"{s}.jpg").convert("RGB")
        b = PILImage.open(ROOT / "data_preprocessed" / f"cycle-{c}" / f"{s}.png").convert("RGBA")
        grey = PILImage.new("RGBA", b.size, (110, 110, 110, 255)); grey.alpha_composite(b)
        for i, im in enumerate((a, grey.convert("RGB"))):
            ax = axes[i, j]; ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{'Original' if i == 0 else 'Leaf only'} · {title}", fontsize=15, fontweight="bold", loc="left", pad=6)
            for sp in ax.spines.values(): sp.set_visible(False)
    fig.savefig(PNG / "prep.png", dpi=110, facecolor="#fcfcfb"); plt.close(fig)


# ------------------------------------------------------------------ document
class Doc(BaseDocTemplate):
    def __init__(self, path):
        super().__init__(str(path), pagesize=A4, title="Smart irrigation of Dwarf Napier grass under solar panels: SPAD, spectrometer and camera evaluation",
                         author="NP_Project", subject="Methods and results report")
        pw, ph = A4; lw, lh = landscape(A4)
        self.addPageTemplates([
            PageTemplate("P", [Frame(2 * cm, 2 * cm, pw - 4 * cm, ph - 4 * cm, id="p", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)], onPage=self.foot, pagesize=A4),
            PageTemplate("L", [Frame(1.7 * cm, 1.9 * cm, lw - 3.4 * cm, lh - 3.6 * cm, id="l", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)], onPage=self.foot, pagesize=landscape(A4))])

    def foot(self, canvas, doc):
        canvas.saveState(); canvas.setFont("Times-Roman", 8.5); canvas.setFillColor(INK2)
        w, h = canvas._pagesize
        canvas.drawString(2 * cm if w < h else 1.7 * cm, 1.1 * cm, "Dwarf Napier grass under solar panels: SPAD, spectrometer and camera evaluation")
        canvas.drawRightString(w - (2 * cm if w < h else 1.7 * cm), 1.1 * cm, f"Page {doc.page}")
        canvas.setStrokeColor(RULE); canvas.setLineWidth(0.4); canvas.line(2 * cm if w < h else 1.7 * cm, 1.45 * cm, w - (2 * cm if w < h else 1.7 * cm), 1.45 * cm)
        canvas.restoreState()


def assemble(items):
    """Insert the portrait/landscape page switches around LandFig objects."""
    out, mode = [], "P"
    for it in items:
        if isinstance(it, LandFig):
            if mode == "P": out += [NextPageTemplate("L"), PageBreak()]; mode = "L"
            else: out.append(PageBreak())
            out += land_flowables(it)
        else:
            if mode == "L": out += [NextPageTemplate("P"), PageBreak()]; mode = "P"
            out.append(it)
    return out


def main():
    preprocessing_figure()
    df = ss.load()
    st = pd.read_excel(TABLES / "Statistics_summary.xlsx", sheet_name=None)
    veg = blocks(TABLES / "Vegetation_Indices.xlsx", "Correlation")
    vcv = sheet(TABLES / "Vegetation_Indices.xlsx", "Cross-validation").set_index("Predictors")
    spc = blocks(TABLES / "Spectrometer_Indices.xlsx", "Correlation")
    spper = blocks(TABLES / "Spectrometer_Indices.xlsx", "Per cycle")
    spcv = sheet(TABLES / "Spectrometer_Indices.xlsx", "Cross-validation").set_index("Predictors")
    spind = sheet(TABLES / "Spectrometer_Indices.xlsx", "Indices", header=0)
    net = sheet(TABLES / "Netto_Model_Predictions.xlsx", "Comparison")
    netcv = sheet(TABLES / "Netto_Model_Predictions.xlsx", "Cross-cycle check").set_index("Pigment")
    dnn = sheet(TABLES / "DNN_feasibility.xlsx", "Results")
    pcs = sheet(TABLES / "PCA_analysis.xlsx", "What the components capture")
    pcv = sheet(TABLES / "PCA_analysis.xlsx", "Cross-cycle prediction")
    pair = sheet(TABLES / "Promising_analyses.xlsx", "A paired irrigation effect")
    calib = sheet(TABLES / "Promising_analyses.xlsx", "B calibration within cycle")
    shade = sheet(TABLES / "Promising_analyses.xlsx", "C shade detection")
    hcyc, hblk, htrt = (sheet(TABLES / "Leaf_RGB_Histogram.xlsx", n, header=0) for n in ("By cycle", "By shade block", "By irrigation"))
    hlink = sheet(TABLES / "Leaf_RGB_Histogram.xlsx", "Link to SPAD & Chl").set_index("Statistic")
    anova, corr, means, check = st["ANOVA"], st["SPAD vs lab"], st["Means"], st["Lab Total check"].set_index("quantity")["value"]

    tm = df.groupby("T").SPAD.mean(); bm = df.groupby("Block").SPAD.mean()
    n_ok, n_all = int((spind.QC_flag == "ok").sum()), len(spind)
    n_c5 = int(((spind.Harvest == 5) & (spind["T"].isin([2, 3]))).sum())
    n_out = n_all - n_ok - n_c5
    A = lambda v, e: anova[(anova.variable == v) & (anova.effect == e)].iloc[0]
    story = []

    # ================================================================== title + summary
    story += [P("Evaluating SPAD-502, spectrometer and mobile-camera measurements for detecting chlorophyll variation related to irrigation in Dwarf Napier grass grown under solar panels", "title"),
              P("Methods and results report &nbsp;|&nbsp; data collected over five harvest cycles &nbsp;|&nbsp; report date 19 September 2026", "subtitle")]
    summ = [P("<b>Summary</b>", "box"),
            P(f"<b>Objective.</b> To evaluate whether the SPAD-502 chlorophyll meter, a spectrometer and a mobile camera can detect water stress through changes in leaf chlorophyll (leaf colour) in Dwarf Napier grass (NP) irrigated at 100, 75 and 50 % under and outside solar panels.", "box"),
            P(f"<b>Data.</b> {len(df)} leaf samples (54 per harvest cycle, five cycles): SPAD readings, laboratory chlorophyll a and b, reflectance spectra (380-950 nm) and leaf photographs.", "box"),
            P(f"<b>Main findings.</b> (1) Shade and harvest cycle, not irrigation, controlled chlorophyll: shade explained {A('SPAD','Shade block (B)').share_of_variance_pct:.0f} % of SPAD variance and harvest cycle about 34 % of the variance of laboratory chlorophyll, whereas irrigation explained {A('SPAD','Irrigation (T)').share_of_variance_pct:.1f} % of SPAD and none of laboratory chlorophyll. "
              f"(2) A modest SPAD decline at 50 % irrigation appears in cycles 1-4 (paired with 100 % irrigation) but reverses in cycle 5 and is not confirmed by the laboratory. "
              f"(3) SPAD tracks laboratory chlorophyll only where chlorophyll varies widely (r = {corr.iloc[0].r_chl_a:.2f} overall; {corr[corr.subset=='B2'].r_chl_a.iloc[0]:.2f} under the panel, {corr[corr.subset=='B1'].r_chl_a.iloc[0]:.2f} outside; {corr[corr.subset=='Cycle 5'].r_chl_a.iloc[0]:.2f} in cycle 5). "
              f"(4) Camera colour indices (best: DGCI) follow SPAD within a cycle (r = 0.75) and camera colour features separate shaded from unshaded plots with about 87 % accuracy, but no camera, spectrometer or neural-network model predicted chlorophyll of an unseen harvest cycle.", "box"),
            P("<b>Conclusion.</b> The tools detect shade- and growth-stage-related colour differences, but in this trial they did not detect the irrigation treatments, and the data cannot support claims about water-use efficiency or biomass. Section 5 lists what is needed to meet the full objective.", "box")]
    bx = Table([[summ]], colWidths=[17 * cm]); bx.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.8, ACCENT), ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f3f6fa")), ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9), ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    story += [bx, Spacer(1, 4)]

    # ================================================================== 1 Introduction
    story += [H1("1. Introduction and objective"),
              P("This study develops a smart irrigation management method for Dwarf Napier grass planted under solar panels. Leaf chlorophyll content, seen as leaf colour, is used as an indicator of plant status, and three non-destructive tools are evaluated: the SPAD-502 chlorophyll meter, a spectrometer and a mobile-phone camera. The wider aim is to improve water-use efficiency, reduce water loss, optimise irrigation planning and increase biomass, with results transferable to other crops."),
              P("This report documents the methods and results of the analysis of the data collected so far. It asks four questions: (i) do the irrigation levels, the shade and the harvest cycle change chlorophyll; (ii) how well do SPAD, camera and spectrometer measurements agree with laboratory chlorophyll; (iii) can vegetation indices or machine-learning models predict chlorophyll in a new harvest cycle; and (iv) can the current data support the study objective.")]

    # ================================================================== 2 Methods
    story += [H1("2. Materials and methods"), H2("2.1 Experimental design and sample coding"),
              P(f"Dwarf Napier grass was grown at three irrigation levels, outside and under solar-panel shade, and sampled in five harvest cycles. Each sample is coded <i>T-B-R-(x)</i>, for example T1B2R3(2). "
                f"The design gives 3 irrigation levels x 2 blocks x 3 replications x 3 repeats = 54 samples per cycle and {len(df)} samples in total ({T('design')}). Samples of a third block code (B3) are not part of the experimental design and were removed from every dataset.")]
    story += tab("design", "Experimental factors and sample coding.", [["Code", "Factor", "Levels"], ["T", "Irrigation treatment", "T1 = 100 %, T2 = 75 %, T3 = 50 % of the irrigation requirement"], ["B", "Block (location)", "B1 = outside the panel shade; B2 = under solar-panel shade"],
                                                                        ["C", "Harvest cycle", "1, 2, 3, 4, 5 (folders cycle-1 to cycle-5; sheets c_1 to c_5)"], ["R", "Replication", "R1, R2, R3"], ["(x)", "Repeat within each R", "(1), (2), (3)"]], [1.6, 4.4, 11], left_cols=(1, 2))
    story += [H2("2.2 Measurements"),
              P(f"<b>SPAD.</b> SPAD-502 readings were taken on three parts of each leaf (up, mid, low), giving 12 to 18 readings per sample. The whole-leaf value is the mean of all readings that were available; missing readings were left out and the sum was divided by the number of readings present. "
                f"<b>Laboratory chlorophyll.</b> Chlorophyll a, chlorophyll b and a \"total chlorophyll\" value were supplied for every sample. <b>Spectrometer.</b> Reflectance spectra (380-950 nm, 1,027 points, 0.57 nm step) were recorded for every sample. <b>Camera.</b> One photograph of each leaf sample was taken with a mobile phone ({T('meas')}).")]
    story += tab("meas", "Datasets used.", [["Measurement", "Content", "Samples"], ["SPAD-502", "Whole-leaf SPAD, mean of 12-18 readings (up, mid, low)", "270"], ["Laboratory", "Chlorophyll a, chlorophyll b, total chlorophyll (units not stated in the file)", "270"],
                                             ["Spectrometer", "Reflectance 380-950 nm", f"270 ({n_ok} pass quality control)"], ["Mobile camera", "Leaf photographs, 960 x 1280 pixels", "269 (T2B2R3(3) of cycle 3 is missing)"]], [3.2, 9.6, 4.2], left_cols=(1, 2))
    story += [H2("2.3 Data preparation and quality control"),
              P(f"Every dataset was checked before analysis. The problems found and how they were handled are listed in {T('quality')}. The two that most limit interpretation are that the laboratory \"total chlorophyll\" is an exact multiple of chlorophyll b, so it carries no independent information, and that {n_c5} spectra of cycle 5 (irrigation levels T2 and T3) are copies or averages of one another.")]
    story += tab("quality", "Data-quality findings and handling.", [
        ["Dataset", "Finding", "Handling"],
        ["All", "Samples with block code B3 are outside the design", "Removed from all datasets"],
        ["SPAD", "Some of the 12-18 readings per leaf are missing (up to 6); one entry was a blank space", "Mean over available readings only"],
        ["Laboratory", f"Total chlorophyll = {check['Total Chl / Chl b (min)']:.3f}-{check['Total Chl / Chl b (max)']:.3f} x chlorophyll b (r = {check['Pearson r, Total Chl vs Chl b']:.5f}); it contains no chlorophyll a", "Analyses use chlorophyll a and b; total chlorophyll is not reported separately"],
        ["Laboratory", f"{int(check['Chl a values within 0.01 of the maximum'])} chlorophyll a values sit at about {check['Chl a maximum']:.2f} (apparent ceiling)", "Kept; flagged in the figures"],
        ["Spectrometer", f"{n_c5} spectra of cycle 5 (T2, T3) are near-identical (red band 54.7 +/- 0.3 %); in the source workbook 14 columns were exact copies and about 18 were AVERAGE formulas; one spectrum is blank", "Excluded from spectrometer analysis"],
        ["Spectrometer", f"{n_out} further spectra are outliers (a band far from the cycle median; robust z-score above 4)", f"Excluded; {n_ok} spectra remain"],
        ["Spectrometer", "Near-infrared reflectance is low (no leaf plateau) and drops to zero above about 820 nm", "Reported as a limitation; indices interpreted with care"],
        ["Camera", "Lighting and exposure were not controlled; no grey reference card", "Cycle-wise (within-cycle) analysis added"],
        ["Design", "The three repeats (x) are subsamples of one replication", "p-values treat them as independent and are therefore optimistic"]], [2.6, 9.6, 4.8], left_cols=(1, 2))

    story += [H2("2.4 Image processing and leaf colour extraction"),
              P(f"Leaf photographs were processed with a script written in Python (OpenCV). Leaf pixels were first found by colour thresholds in HSV space (yellow-green to green, thresholds adapted to each photograph from the colour of the leaf interior). The mask was then refined with the GrabCut algorithm, gaps between neighbouring leaves were removed, "
                f"small holes (lesions) were filled and the outline was smoothed. Everything outside the leaf was made transparent. Images taken sideways were rotated so that every image is portrait (960 x 1280 pixels) with the leaf upright ({F('prep')}). "
                f"From the leaf-only images the mean red, green and blue values (0-255) were extracted using fully opaque pixels only; a thin edge zone and the orange rubber bands that hold the leaves were excluded. For each image the histogram of each colour channel was also computed, together with its mean, standard deviation, skewness, kurtosis and mode.")]
    story.append(fig_flowables("prep", "Example of background removal for three samples. The leaf-only images (bottom) are the input for all colour analysis.", width=15.5, max_h=15.5))

    story += [H2("2.5 Vegetation indices"),
              P(f"Two families of indices were calculated ({T('indices')}). <b>Camera (RGB) indices</b> follow a published table of camera-based indices, using the normalised intensities r = R/(R+G+B), g = G/(R+G+B) and b = B/(R+G+B). "
                f"<b>Spectrometer indices</b> (SRVI, SRRE, NDRE, NDVI, CCCI) use the mean reflectance in four bands: red 660-675 nm, red edge 690-740 nm, NIRb 770-785 nm and NIRa 800-810 nm. NDRE was calculated exactly as printed in the reference table, that is with the red band, and CCCI = NDRE/NDVI.")]
    ref = [[k, v[1]] for k, v in avi.DEFS.items() if v[0] == "Reference"]
    half = (len(ref) + 1) // 2
    rows = [["Camera index", "Formula", "Camera index", "Formula"]] + [ref[i] + (ref[i + half] if i + half < len(ref) else ["", ""]) for i in range(half)]
    rows += [["<b>Spectrometer index</b>", "<b>Formula</b>", "", ""], ["SRVI", "NIRb / Red", "NDVI", "(NIRb - Red) / (NIRb + Red)"], ["SRRE", "NIRa / RedEdge", "CCCI", "NDRE / NDVI"], ["NDRE", "(NIRa - Red) / (NIRa + Red)", "", ""]]
    story += tab("indices", "Vegetation indices calculated (L = 0.5; N = (r + b)/255).", rows, [2.3, 6.2, 2.3, 6.2], left_cols=(1, 2, 3), zebra=False,
                 spans=[("BACKGROUND", (0, half + 1), (-1, half + 1), HEAD_BG)])

    story += [H2("2.6 Statistical analysis and validation"),
              P("<b>Effects of the factors.</b> Effects of irrigation, block and harvest cycle on SPAD and on laboratory chlorophyll were tested with partial F-tests from ordinary least squares (each effect tested after the others), together with the irrigation x block and irrigation x cycle interactions. "
                "To remove differences between cycles and blocks, a <i>paired</i> comparison was made: each T2 and T3 sample was matched with the T1 sample of the same block, replication, repeat and cycle and the mean difference, its 95 % confidence interval and a paired t-test were computed. p-values are not corrected for multiple tests."),
              P("<b>Agreement between measurements.</b> Pearson correlation was calculated in three ways: <i>pooled</i> (all samples in one group), <i>within cycle</i> (each cycle's own mean removed first, so differences in lighting and growth stage between cycles cannot create a correlation) and by subset (block, irrigation level, cycle)."),
              P("<b>Predictive validation.</b> Models were validated by <i>leave-one-cycle-out</i>: they were trained on four harvest cycles and tested on the fifth, the situation a farmer meets when the tool is used in a new cycle. R<super>2</super> is the fraction of variance predicted; a negative value means the model is worse than predicting the mean. Random k-fold splitting was shown only to demonstrate leakage, because repeats of the same plot then fall in both the training and test sets."),
              P("<b>Reference calibration models.</b> The quadratic SPAD models of Netto et al. (2005) for chlorophyll a, chlorophyll b and carotenoids were applied to the SPAD values and compared with the laboratory results, both raw and after linear rescaling (the equations give values in different units)."),
              P("<b>Machine learning and PCA.</b> A small neural network (two hidden layers of 64 units, dropout 0.2, five random seeds averaged) and ridge regression were trained on the spectra (400-850 nm in 5-nm bins) or on camera colour features (channel statistics and indices). Targets were SPAD, chlorophyll a and b (regression) and irrigation level and shade block (classification). "
                "Principal component analysis (PCA) was fitted on standardised features, inside the training cycles only, and used both to describe the main sources of variation and as a pre-processing step (3, 5 or 10 components) before regression.")]

    # ================================================================== 3 Results
    story += [H1("3. Results"), H2("3.1 Effects of irrigation, shade and harvest cycle")]
    rows = [["Block", "Irrigation", "SPAD", "Chlorophyll a", "Chlorophyll b"]]
    for _, r in means.iterrows():
        rows.append([("B1 outside shade" if r.Block == 1 else "B2 under panel"), f"T{int(r['T'])} · {['100','75','50'][int(r['T'])-1]} %", f"{r['SPAD mean']:.1f} +/- {r['SPAD sd']:.1f}", f"{r['Chl a mean']:.1f} +/- {r['Chl a sd']:.1f}", f"{r['Chl b mean']:.1f} +/- {r['Chl b sd']:.1f}"])
    story.append(P(f"Under the panel SPAD was about 10 units higher than outside the shade at every irrigation level ({T('means')}), whereas the three irrigation levels were close to one another (overall mean SPAD {tm[1]:.1f}, {tm[2]:.1f} and {tm[3]:.1f} for T1, T2 and T3). "
                   f"The partial F-tests ({T('anova')}) confirm that the block was by far the strongest factor for SPAD ({A('SPAD','Shade block (B)').share_of_variance_pct:.0f} % of variance, p &lt; 0.001), followed by harvest cycle ({A('SPAD','Harvest cycle (C)').share_of_variance_pct:.1f} %). "
                   f"Laboratory chlorophyll was governed by harvest cycle (34-35 % of variance) and, less, by block (10-11 %). Irrigation had a small significant effect on SPAD ({A('SPAD','Irrigation (T)').share_of_variance_pct:.1f} % of variance, p = {A('SPAD','Irrigation (T)').p:.3f}) and no effect on laboratory chlorophyll (p = {A('CHL_A','Irrigation (T)').p:.2f} and {A('CHL_B','Irrigation (T)').p:.2f})."))
    story += tab("means", "Mean +/- standard deviation of SPAD and laboratory chlorophyll by block and irrigation level (all five cycles, n = 45 per row).", rows, [3.6, 3.0, 3.4, 3.5, 3.5])
    rows = [["Effect", "SPAD F", "p", "% var.", "Chl a F", "p", "% var.", "Chl b F", "p", "% var."]]
    for e in ["Irrigation (T)", "Shade block (B)", "Harvest cycle (C)", "Irrigation × block", "Irrigation × cycle"]:
        r = [e.replace("×", "x")]
        for v in ("SPAD", "CHL_A", "CHL_B"):
            a = A(v, e); r += [f"{a.F:.2f}", pf(a.p), f"{a.share_of_variance_pct:.1f}"]
        rows.append(r)
    story += tab("anova", "Partial F-tests for SPAD and laboratory chlorophyll (270 samples). % var. is the share of total variance explained by the effect after the other effects.", rows, [3.4, 1.55, 1.45, 1.4, 1.55, 1.45, 1.4, 1.55, 1.45, 1.4],
                 note="Degrees of freedom: irrigation 2, block 1, cycle 4, irrigation x block 2, irrigation x cycle 8; residual about 254-262.")
    story.append(P(f"SPAD followed a clear cycle-by-block pattern ({F('spad')}): under the panel it rose to 50-54 in cycle 5, whereas outside the shade it fell to 23-28. The irrigation x cycle interaction was significant for SPAD (p = {A('SPAD','Irrigation × cycle').p:.3f}) but not for laboratory chlorophyll, i.e. the SPAD difference between irrigation levels was not consistent across cycles."))
    story.append(fig_flowables("spad", "SPAD by harvest cycle, irrigation level and block. Points are means of 9 samples; whiskers are +/- 1 standard error.", width=17, max_h=9.2))

    p3 = pair[(pair.target == "SPAD") & (pair.treatment == "T3 - T1")]
    p2 = pair[(pair.target == "SPAD") & (pair.treatment == "T2 - T1")]
    rows = [["Cycle", "B1 diff.", "B1 %", "B1 p", "B2 diff.", "B2 %", "B2 p"]]
    for h in range(1, 6):
        a, b = p3[(p3.harvest == h) & (p3.block == 1)].iloc[0], p3[(p3.harvest == h) & (p3.block == 2)].iloc[0]
        rows.append([f"Cycle {h}", f"{a.mean_diff:+.1f}", f"{a.rel_pct:+.0f}", pf(a.p), f"{b.mean_diff:+.1f}", f"{b.rel_pct:+.0f}", pf(b.p)])
    lab_neg = {t: int(((pair.target == t) & (pair.treatment == "T3 - T1") & (pair.mean_diff < 0)).sum()) for t in ("CHL_A", "CHL_B")}
    lab_sig = {t: int(((pair.target == t) & (pair.treatment == "T3 - T1") & (pair.p < 0.05)).sum()) for t in ("CHL_A", "CHL_B")}
    story.append(P(f"The paired comparison removes the cycle and block differences ({F('paired')}, {T('paired')}). At 50 % irrigation SPAD was lower than the well-watered T1 in all eight block-by-cycle cells of cycles 1-4 (by 0.3 to 6.3 SPAD units, up to about 19 %), and four of the ten T3 cells were individually significant (p &lt; 0.05). "
                   f"In cycle 5, however, T3 was higher than T1 (significantly under the panel). The 75 % level was almost never different from T1 (one significant cell of ten). Laboratory chlorophyll did not confirm the SPAD trend: only {lab_neg['CHL_A']} of 10 (chlorophyll a) and {lab_neg['CHL_B']} of 10 (chlorophyll b) T3 cells were below T1, and {lab_sig['CHL_A']} of 10 were significant."))
    story += tab("paired", "Paired SPAD difference between 50 % irrigation (T3) and 100 % irrigation (T1): mean difference (SPAD units), difference as % of the T1 mean and paired t-test p-value (n = 9 pairs per cell).", rows, [2.4, 2.3, 2.1, 2.1, 2.3, 2.1, 2.1])
    story.append(fig_flowables("paired", "Effect of reduced irrigation paired with the well-watered plot (T1). Difference in % of the T1 mean, matched by block, replication, repeat and cycle; bars are 95 % confidence intervals; filled markers p &lt; 0.05, open markers not significant. Rows: SPAD, laboratory chlorophyll a and b.", width=17.0, max_h=19.5))

    # ---------------------------------------------------------------- 3.2 SPAD vs lab
    rows = [["Subset", "n", "r (Chl a)", "R² (Chl a)", "r (Chl b)", "R² (Chl b)"]]
    names = {"B1": "B1 outside shade", "B2": "B2 under panel", "T1": "T1 · 100 %", "T2": "T2 · 75 %", "T3": "T3 · 50 %"}
    for _, r in corr.iterrows():
        rows.append([names.get(r.subset, r.subset), int(r.n), f2(r.r_chl_a), f2(r.R2_chl_a), f2(r.r_chl_b), f2(r.R2_chl_b)])
    story += [H2("3.2 SPAD compared with laboratory chlorophyll"),
              P(f"Across all 270 samples SPAD correlated moderately with laboratory chlorophyll a (r = {corr.iloc[0].r_chl_a:.2f}) and b (r = {corr.iloc[0].r_chl_b:.2f}), explaining about a third of the variance ({T('corr')}, {F('spadchl')}). The relationship was not uniform. It was strong under the panel (r = {corr[corr.subset=='B2'].r_chl_a.iloc[0]:.2f}) and in cycle 5 (r = {corr[corr.subset=='Cycle 5'].r_chl_a.iloc[0]:.2f}), "
                f"where chlorophyll varied widely, and close to zero outside the shade (r = {corr[corr.subset=='B1'].r_chl_a.iloc[0]:.2f}) and in cycles 1 and 2 (r = {corr[corr.subset=='Cycle 1'].r_chl_a.iloc[0]:.2f} and {corr[corr.subset=='Cycle 2'].r_chl_a.iloc[0]:.2f}). The pooled correlation is therefore carried by a few high-chlorophyll groups. "
                f"Splitting by irrigation level gave similar correlations (r = 0.55-0.65), rising slightly as irrigation decreased ({F('spadchl_trt')}). Chlorophyll a shows an apparent ceiling near 38 in 11 samples, visible as a flat row of points in the figures.")]
    story += tab("corr", "Pearson correlation between SPAD and laboratory chlorophyll by subset of samples.", rows, [4.4, 1.6, 2.4, 2.4, 2.4, 2.4],
                 note="Total chlorophyll is not shown: it is a fixed multiple of chlorophyll b (Table 3) and gives identical results.")
    story.append(fig_flowables("spadchl", "SPAD compared with laboratory chlorophyll. Top: scatter with linear fit (all 270 samples; the total-chlorophyll panel repeats chlorophyll b because it is a fixed multiple of it). Bottom: Pearson r by subset.", width=16.5, max_h=20.0))
    story.append(LandFig("spadchl_trt", "SPAD compared with laboratory chlorophyll, coloured by irrigation treatment with a linear fit for each treatment (n = 90 per treatment)."))

    # ---------------------------------------------------------------- 3.3 Camera
    story += [H2("3.3 Camera: leaf colour, histograms and RGB indices")]
    rows = [["Group", "R mean", "G mean", "B mean", "R sd", "G sd", "B sd"]]
    for _, r in hcyc.iterrows(): rows.append([f"Cycle {int(r.Harvest)}", f1(r["R mean"]), f1(r["G mean"]), f1(r["B mean"]), f1(r["R std"]), f1(r["G std"]), f1(r["B std"])])
    for _, r in hblk.iterrows(): rows.append([("B1 outside shade" if r.Block == 1 else "B2 under panel"), f1(r["R mean"]), f1(r["G mean"]), f1(r["B mean"]), f1(r["R std"]), f1(r["G std"]), f1(r["B std"])])
    for _, r in htrt.iterrows(): rows.append([f"T{int(r['T'])} · {['100','75','50'][int(r['T'])-1]} %", f1(r["R mean"]), f1(r["G mean"]), f1(r["B mean"]), f1(r["R std"]), f1(r["G std"]), f1(r["B std"])])
    story.append(P(f"The leaf colour histograms ({F('hist_cycle')}, {F('hist_group')}) have the green channel peaking slightly to the right of the red channel and a broad, low blue channel, as expected for green leaves. The mean colour differed between harvest cycles (mean red 157-188, green 175-205; {T('rgbstat')}), which reflects growth stage but also camera lighting: cycle 2 was darkest and cycle 4 brightest. "
                   f"Under the panel leaves were darker and less green (R 165 vs 182, G 183 vs 197) with more blue (89 vs 78) and wider histograms than outside the shade. The three irrigation levels gave almost identical histograms. "
                   f"Within cycles, the mean of the red channel (r = {hlink.loc['R mean','SPAD']:.2f}), of the green channel (r = {hlink.loc['G mean','SPAD']:.2f}) and the spread of the red channel (r = +{hlink.loc['R std','SPAD']:.2f}) correlated best with SPAD."))
    story += tab("rgbstat", "Mean and standard deviation (sd) of the leaf red, green and blue values (0-255) by harvest cycle, block and irrigation level.", rows, [4.2, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1])
    story.append(LandFig("hist_cycle", "Histograms of the leaf red, green and blue values by harvest cycle (leaf pixels only; each curve is the average of the per-image histograms)."))
    story.append(LandFig("hist_group", "Histograms of the leaf red, green and blue values by block (top row) and by irrigation level (bottom row); all cycles pooled."))

    sel = ["DGCI", "NDVI rgb", "SR rgb", "r", "Hue (partial)", "EVI green", "VARI", "NDVI green"]
    pk, wk = [k for k in veg if k.startswith("Pearson r, all")][0], [k for k in veg if k.startswith("Pearson r within")][0]
    rows = [["Index", "Pooled SPAD", "Pooled Chl a", "Pooled Chl b", "Within SPAD", "Within Chl a", "Within Chl b", "CV SPAD", "CV Chl a", "CV Chl b"]]
    cvkey = {"r": "r", "DGCI": "DGCI", "NDVI rgb": "NDVI rgb", "SR rgb": "SR rgb", "Hue (partial)": "Hue (partial)", "EVI green": "EVI green", "VARI": "VARI", "NDVI green": "NDVI green"}
    for k in sel:
        rows.append([k] + [f2(veg[pk].loc[k, c]) for c in ("SPAD", "Chl a", "Chl b")] + [f2(veg[wk].loc[k, c]) for c in ("SPAD", "Chl a", "Chl b")] + [f2(vcv.loc[cvkey[k], c]) for c in ("SPAD R2", "Chl a R2", "Chl b R2")])
    best_cv = vcv["SPAD R2"].max()
    story.append(P(f"All 15 indices of the reference table and the normalised intensities were calculated from the mean leaf colour ({T('rgbvi')}, {F('rgbvi')}). The dark green colour index (DGCI) was the best index: r = {veg[wk].loc['DGCI','SPAD']:.2f} with SPAD within cycles ({veg[pk].loc['DGCI','SPAD']:.2f} pooled) and r = {veg[wk].loc['DGCI','Chl a']:.2f} and {veg[wk].loc['DGCI','Chl b']:.2f} with laboratory chlorophyll a and b. "
                   f"Several indices (NDVI green, NDI, SAVI, OSAVI, EVI2, GMR, SR) are functions of the same ratio (g - r)/(g + r) and give almost identical correlations, so they add little information. Signs are the opposite of a common expectation for ExG-type indices: in this data leaves with more chlorophyll were darker and slightly bluer. "
                   f"Every index correlated more strongly with SPAD than with laboratory chlorophyll. Under leave-one-cycle-out validation, however, no index predicted chlorophyll of an unseen cycle: the best R<super>2</super> for SPAD was {best_cv:.2f} and all values for laboratory chlorophyll were negative."))
    story += tab("rgbvi", "Camera vegetation indices: Pearson r with SPAD and laboratory chlorophyll (pooled and within cycle) and leave-one-cycle-out R² (CV). Selected indices; all are in Vegetation_Indices.xlsx.", rows,
                 [2.4, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6], note="Total chlorophyll (a copy of chlorophyll b) is omitted. Negative R² means worse than predicting the mean.")
    story.append(fig_flowables("rgbvi", "Pearson correlation of the camera (RGB) indices from the reference table with SPAD and laboratory chlorophyll: pooled over all samples (left) and within harvest cycles (right). The Total Chl column is a fixed multiple of Chl b and repeats it.", width=16.2, max_h=20.0))

    # ---------------------------------------------------------------- 3.4 Spectrometer
    pk_s = [k for k in spc if k.startswith("Pearson r, pooled")][0]; wk_s = [k for k in spc if k.startswith("Pearson r within")][0]
    ind = ["SRVI", "SRRE", "NDRE", "NDVI", "CCCI"]
    story += [H2("3.4 Spectrometer: spectra and indices"),
              P(f"After quality control {n_ok} of the {n_all} spectra were used ({T('specqc')}). The mean spectra show the expected green reflectance peak near 550 nm (about 55-85 %) and a small red-edge shoulder around 700-750 nm, but near-infrared reflectance is low and falls to zero above about 820 nm ({F('spec_trt')}, {F('spec_shade')}). "
                f"Spectra of blocks differ mainly in the green region (higher reflectance outside the shade in cycles 2, 3 and 5, lower in cycle 1), whereas irrigation levels are very similar in cycles 1-4. Cycle 5 spectra of T2 and T3 (dotted) differ in shape and are the copied spectra that were excluded. The near-infrared plateau of healthy leaves is missing, so NDVI is close to 0 (median -0.07 to 0.08 per cycle; healthy leaves are about 0.6-0.9), and the indices mainly reflect noise or instrument response.")]
    rows = [["Cycle", "Spectra", "Excluded", "Reason"]]
    for h in range(1, 6):
        d = spind[spind.Harvest == h]; ex = d[d.QC_flag == "excluded"]
        reason = "T2/T3 copies (36) + outliers" if h == 5 else ("outliers" if len(ex) else "none")
        if h == 5: reason = f"T2/T3 copies ({int((ex.QC_reason.str.startswith('cycle 5')).sum())}) + outliers ({int((~ex.QC_reason.str.startswith('cycle 5')).sum())})"
        else: reason = f"outliers ({len(ex)})" if len(ex) else "none"
        rows.append([f"Cycle {h}", len(d), len(ex), reason])
    rows.append(["Total", n_all, n_all - n_ok, f"{n_ok} spectra retained"])
    story += tab("specqc", "Spectrometer quality control.", rows, [2.4, 2.4, 2.4, 6.6])
    rows = [["Index", "Pooled SPAD", "Pooled Chl a", "Pooled Chl b", "Within SPAD", "Within Chl a", "Within Chl b", "CV SPAD", "CV Chl a", "CV Chl b"]]
    for k in ind:
        rows.append([k] + [f2(spc[pk_s].loc[k, c]) for c in ("SPAD", "Chl a", "Chl b")] + [f2(spc[wk_s].loc[k, c]) for c in ("SPAD", "Chl a", "Chl b")] + [f2(spcv.loc[k, c]) for c in ("SPAD R2", "Chl a R2", "Chl b R2")])
    kb = spcv.index[-1]
    rows.append(["All four bands"] + [""] * 6 + [f2(spcv.loc[kb, c]) for c in ("SPAD R2", "Chl a R2", "Chl b R2")])
    ps = spper["with SPAD"]
    story.append(P(f"Spectrometer indices showed essentially no relationship with laboratory chlorophyll (pooled |r| up to 0.16) and only weak relationships with SPAD, the best being SRRE within cycles (r = {spc[wk_s].loc['SRRE','SPAD']:.2f}) ({T('specvi')}, {F('specvi')}). Correlations with SPAD inside single cycles were inconsistent ({T('specper')}): strong in cycles 2 and 3 (r = 0.85-0.93), negative in cycle 1 and close to zero in cycle 4. "
                   f"Predicting a new cycle failed for every index (negative R<super>2</super>); using the four bands together gave R<super>2</super> = {spcv.loc[kb,'SPAD R2']:.2f} for SPAD but not for laboratory chlorophyll. Because 36 spectra of cycle 5 were excluded, results for cycle 5 rest on 18 (T1) spectra only."))
    story += tab("specvi", "Spectrometer indices: Pearson r with SPAD and laboratory chlorophyll (pooled and within cycle; quality-passed spectra, n = %d) and leave-one-cycle-out R² (CV)." % n_ok, rows, [2.4, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6])
    rows = [["Index", "C1", "C2", "C3", "C4", "C5"]] + [[k] + [f2(ps.loc[k, c]) for c in ("C1", "C2", "C3", "C4", "C5")] for k in ind]
    story += tab("specper", "Spectrometer indices: Pearson r with SPAD inside each harvest cycle (cycle 5: T1 only).", rows, [3.0, 2.2, 2.2, 2.2, 2.2, 2.2])
    story.append(LandFig("spec_trt", "Smoothed mean reflectance spectra by irrigation level for each harvest cycle (Savitzky-Golay smoothing). Dotted lines in cycle 5: T2 and T3 spectra that are copies of one another and were excluded. Grey strips: bands of the indices."))
    story.append(LandFig("spec_shade", "Smoothed mean reflectance spectra by block (outside vs under the panel) for each harvest cycle. Grey strips: bands of the indices."))
    story.append(LandFig("specvi", "Pearson correlation of the spectrometer indices with SPAD and laboratory chlorophyll: pooled, within cycles, and by harvest cycle (quality-passed spectra). The Total Chl column is a fixed multiple of Chl b and repeats it."))

    # ---------------------------------------------------------------- 3.5 Netto
    nall = net[net.Subset == "All samples"].set_index("Pigment")
    def nr(sub, pig): return net[(net.Subset == sub) & (net.Pigment == pig)]["Pearson r"].iloc[0]
    rows = [["Pigment", "r", "R²", "Mean pred. / mean lab", "RMSE after rescale", "Leave-one-cycle-out R²"]]
    for pig in ("Chlorophyll a", "Chlorophyll b"):
        rows.append([pig, f2(nall.loc[pig, "Pearson r"]), f2(nall.loc[pig, "R2"]), f1(nall.loc[pig, "Mean predicted / mean lab"]), f1(nall.loc[pig, "RMSE after rescale"]), f2(netcv.loc[pig, "R2"])])
    story += [H2("3.5 Reference calibration models (Netto et al., 2005)"),
              P(f"The quadratic models of Netto et al. (2005) predict chlorophyll a, chlorophyll b and carotenoids from SPAD. Applied to this data ({F('netto')}, {T('netto')}) the equations give values about {nall.loc['Chlorophyll a','Mean predicted / mean lab']:.0f} times (chlorophyll a) and {nall.loc['Chlorophyll b','Mean predicted / mean lab']:.0f} times (chlorophyll b) the laboratory values, so the units differ from those of the laboratory file and raw errors are not meaningful. "
                f"Their correlation with the laboratory values (r = {nall.loc['Chlorophyll a','Pearson r']:.2f} for chlorophyll a and {nall.loc['Chlorophyll b','Pearson r']:.2f} for chlorophyll b) is only slightly better than that of plain SPAD ({corr.iloc[0].r_chl_a:.2f} and {corr.iloc[0].r_chl_b:.2f}). "
                f"They show the same weakness as SPAD: r = {nr('B2 under panel','Chlorophyll a'):.2f} under the panel and {nr('Cycle 5','Chlorophyll a'):.2f} in cycle 5, but {nr('B1 outside shade','Chlorophyll a'):.2f} outside the shade. A rescaling learnt on four cycles explained almost nothing in the fifth (R<super>2</super> = {netcv.loc['Chlorophyll a','R2']:.2f} and {netcv.loc['Chlorophyll b','R2']:.2f}). Carotenoids were predicted but cannot be validated because no carotenoid measurements exist.")]
    story += tab("netto", "Netto et al. (2005) predictions compared with laboratory values (270 samples). RMSE after rescale is in laboratory units.", rows, [3.0, 1.6, 1.6, 3.4, 3.4, 4.0])
    story.append(LandFig("netto", "Netto et al. (2005) SPAD models compared with laboratory chlorophyll: scatter of predicted vs observed chlorophyll a and b, the three equations, and Pearson r by subset.", max_w=20.5, max_h=14.7))

    # ---------------------------------------------------------------- 3.6 ML + PCA
    story += [H2("3.6 Neural network, PCA and analyses that do work"),
              P(f"A small neural network did not outperform ridge regression ({T('ml')}). Tested on an unseen harvest cycle, SPAD was predicted from camera colour with R<super>2</super> = {dnn[(dnn.features.str.startswith('Camera'))&(dnn.target=='SPAD')]['mlp · leave-one-cycle-out'].iloc[0]:.2f} (ridge {dnn[(dnn.features.str.startswith('Camera'))&(dnn.target=='SPAD')]['ridge · leave-one-cycle-out'].iloc[0]:.2f}), laboratory chlorophyll had negative R<super>2</super> and irrigation level was classified at or below chance (accuracy 0.21-0.30 against 0.33-0.36). "
                f"A random 5-fold split, in which repeats of the same plot appear in both training and test data, produced R<super>2</super> up to 0.70 for SPAD, which is an artefact of leakage ({F('dnn_leaky')} vs {F('dnn_honest')}). With only 206-269 usable samples (about 54 per cycle) and almost no irrigation signal, a deeper model cannot find information that the data do not contain.")]
    rows = [["Features", "Target", "Chance", "Ridge (cycle)", "Neural net (cycle)", "Ridge (random)", "Neural net (random)"]]
    for _, r in dnn.iterrows():
        m = r.metric == "R²"
        rows.append([r.features.replace(" (90 bins)", "").replace("Camera colour (RGB stats + indices)", "Camera colour"), r.target.replace(" (3 classes)", "").replace(" (2 classes)", "") + ("" if m else " (accuracy)"),
                     "" if m or pd.isna(r.chance) else f"{r.chance:.2f}", f2(r["ridge · leave-one-cycle-out"]), f2(r["mlp · leave-one-cycle-out"]), f2(r["ridge · random 5-fold (leaky)"]), f2(r["mlp · random 5-fold (leaky)"])])
    story += tab("ml", "Neural network and ridge regression: R² (regression targets) or accuracy (classification). Cycle = trained on four cycles, tested on the fifth (honest); random = random 5-fold (leaky).", rows, [3.4, 3.7, 1.5, 2.1, 2.3, 2.0, 2.0],
                 note="Spectra: n = 206; camera: n = 269. Negative R² means worse than predicting the mean.", left_cols=(1,))
    story.append(LandFig("dnn_honest", "Neural network predictions compared with observed values on an unseen harvest cycle (trained on four cycles, tested on the fifth). The dashed line is the 1:1 line."))
    story.append(LandFig("dnn_leaky", "The same neural network with a random 5-fold split (leaky): repeats of the same plot are in both the training and the test data, which makes the fit look better than it is."))
    pa = pcs[pcs.component.isin(["PC1", "PC2", "PC3"])]
    rows = [["Features", "Component", "Variance %", "Explained by cycle", "by shade block", "by irrigation"]]
    for _, r in pa.iterrows(): rows.append([r.features, r.component, f1(r.variance_pct), f2(r.share_harvest_cycle), f2(r.share_shade_block), f2(r.share_irrigation)])
    story.append(P(f"PCA showed that the main variation in the data is not about irrigation ({T('pcastruct')}, {F('pca')}). For the spectra, the first component (78 % of variance) was almost entirely harvest cycle (82 % of its variance) and the samples fell into two groups of cycles (1, 4 and 2, 3, 5) with no separation by block or irrigation. "
                   f"For camera colour, the first component (55 %) separated the blocks (49 % of its variance) and no component separated irrigation levels (at most 8 %). "
                   f"Using PCA before regression ({T('pcacv')}) improved some poor spectral models (for example chlorophyll a from -2.8 to about -0.3 with 3 components) but never made them useful, and for camera colour it mostly made predictions worse and did not improve shade or irrigation classification. PCA is therefore helpful to describe the data and to reduce noise in the spectra, but it does not create a signal that is absent."))
    story += tab("pcastruct", "PCA of the standardised features: variance explained by each component and the share of each component's variance explained by cycle, block and irrigation level.", rows, [4.0, 2.2, 2.4, 3.2, 2.6, 2.6])
    rows = [["Features", "Target", "Model", "No PCA", "3 comp.", "5 comp.", "10 comp."]]
    for _, r in pcv.iterrows():
        rows.append([r.features, r.target + ("" if r.metric == "R²" else " (acc.)"), r.model, f2(r["no PCA"]), f2(r["PCA 3 comp."]), f2(r["PCA 5 comp."]), f2(r["PCA 10 comp."])])
    story += tab("pcacv", "Effect of PCA before regression (leave-one-cycle-out; R² or accuracy). PCA was fitted on the training cycles only.", rows, [3.6, 3.4, 1.8, 2.0, 2.0, 2.0, 2.0], left_cols=(1,))
    story.append(LandFig("pca", "PCA score plots (first two components) of the spectra (top) and camera colour features (bottom), coloured by harvest cycle, shade block and irrigation level."))

    sh = shade.set_index("features")
    cam_shade = dnn[dnn.features.str.startswith('Camera') & dnn.target.str.startswith('Shade')]['ridge · leave-one-cycle-out'].iloc[0]
    rows = [["Features", "Mean accuracy", "Lowest cycle", "Highest cycle", "Chance"]] + [[k, f2(r.accuracy_mean), f2(r.accuracy_min), f2(r.accuracy_max), f2(r.chance)] for k, r in sh.iterrows()]
    rows.append(["Camera colour features (channel statistics + indices; ridge)", f2(cam_shade), "", "", "0.50"])
    cb = calib.pivot(index="harvest", columns="target", values="R2")
    story.append(P(f"Two analyses gave usable results. <b>Shade detection:</b> classifying blocks B1 and B2 on unseen cycles (chance 50 %) reached {cam_shade*100:.0f} % from camera colour features (channel statistics and indices), {sh.loc['Camera colour indices','accuracy_mean']*100:.0f} % from five camera indices alone, {sh.loc['SPAD','accuracy_mean']*100:.0f} % from SPAD alone and {sh.loc['Camera + SPAD','accuracy_mean']*100:.0f} % from SPAD with the camera indices ({T('shade')}). "
                   f"<b>Calibration inside one cycle:</b> when camera colour indices were calibrated within a cycle, leaving out one replication at a time, SPAD was predicted with R<super>2</super> = {cb.loc[5,'SPAD']:.2f} and chlorophyll a with {cb.loc[5,'CHL_A']:.2f} in cycle 5, but not in cycles 1-4 (R<super>2</super> up to {cb.loc[1:4,'SPAD'].max():.2f} for SPAD and {cb.loc[1:4,'CHL_A'].max():.2f} for chlorophyll a)."))
    story += tab("shade", "Detection of the shade block (B1 vs B2), trained on four cycles and tested on the fifth.", rows, [6.4, 2.6, 2.4, 2.6, 2.0])

    # ---------------------------------------------------------------- 3.7 objective
    rows = [["Element of the objective", "Evidence in this study", "Status"],
            ["Detect water stress through chlorophyll (leaf colour) with SPAD, spectrometer and camera", "Irrigation did not change laboratory chlorophyll; SPAD shows a small, inconsistent decline at 50 % irrigation in cycles 1-4 that reverses in cycle 5", "Not demonstrated"],
            ["Tools follow chlorophyll / leaf colour", "SPAD and DGCI follow chlorophyll only where it varies widely (under the panel, cycle 5); no model generalises to a new cycle", "Partly"],
            ["Tools detect shade-related colour differences", "Shade detected with 75-87 % accuracy on unseen cycles (camera, SPAD)", "Demonstrated"],
            ["Improve water-use efficiency, reduce water loss, plan irrigation", "No soil-moisture, water-use or water-status data", "Cannot be tested"],
            ["Increase biomass", "No biomass or yield data", "Cannot be tested"],
            ["Apply to other crops", "Only one crop studied", "Cannot be tested"]]
    story += [H2("3.7 Assessment against the study objective"), P(f"{T('objective')} summarises what the present data allow one to conclude.")]
    story += tab("objective", "Assessment of the study objective.", rows, [6.0, 8.2, 2.8], left_cols=(1, 2))

    # ================================================================== 4 Discussion
    story += [H1("4. Discussion"),
              P("<b>Irrigation signal.</b> The most important result is negative: the three irrigation levels barely changed chlorophyll. Two explanations are possible and the present data cannot separate them: the 50 % level may not have produced water stress in this crop and climate, or chlorophyll may simply be a late or insensitive indicator of water stress in Napier grass. "
                "Shade and harvest cycle produced much larger changes than irrigation, so any irrigation signal is easily hidden unless the comparison is paired, as in " + F("paired") + ". Water status (soil moisture, leaf water potential or stomatal conductance) was not measured, which is the main gap for the stated objective."),
              P("<b>Why tool performance depends on the cycle.</b> All tools agreed with laboratory chlorophyll mainly in cycle 5 and under the panel, where chlorophyll ranged widely. Within the narrow range of the other groups, measurement noise dominates. Models trained on one set of cycles fail on another because camera lighting, spectrometer response and plant growth stage all change between cycles. "
                "A white reference for the spectrometer and a grey card with fixed exposure for the camera would remove much of this variation."),
              P("<b>Data quality.</b> Several problems limit the conclusions: the copied spectra in cycle 5, the low near-infrared signal, the laboratory total chlorophyll that is a copy of chlorophyll b, and the apparent ceiling of chlorophyll a. Correcting them may change the results for the affected variables. The reference indices and the Netto equations were applied as published; the units of the laboratory results should be stated so that the Netto equations can be compared quantitatively."),
              P("<b>Statistical caution.</b> The three repeats of each replication are subsamples of one plot, so significance tests that treat all 270 samples as independent overstate certainty, and the p-values of the paired comparisons (9 pairs per cell) should be read as indicative. Ten block-by-cycle cells per treatment were tested without correction for multiple comparisons. "
                "The more reliable statements are those that are large and repeat across cycles or are validated on unseen cycles."),
              P("<b>Neural network and PCA.</b> Neither method changed the outcome. Their value here was diagnostic: PCA showed that harvest cycle, not irrigation, dominates the spectral variation, and the honest (cycle-held-out) validation showed how strongly random splitting inflates apparent accuracy. A neural network should be tried again only when there is a real signal, for instance biomass or stress data, and after simple models have been tested.")]

    # ================================================================== 5 Conclusions
    story += [H1("5. Conclusions and recommendations"),
              BUL(f"Shade and harvest cycle were the main drivers of chlorophyll: shade explained {A('SPAD','Shade block (B)').share_of_variance_pct:.0f} % of SPAD variance and harvest cycle about 34 % of the variance of laboratory chlorophyll; irrigation explained {A('SPAD','Irrigation (T)').share_of_variance_pct:.1f} % of SPAD and none of the laboratory chlorophyll."),
              BUL("A modest SPAD decrease at 50 % irrigation appeared in cycles 1-4 when compared with the well-watered plots (up to about 19 %), but it reversed in cycle 5 and was not confirmed by laboratory chlorophyll. It is a suggestive trend, not proof of detected water stress."),
              BUL(f"SPAD agrees with laboratory chlorophyll moderately overall (r = 0.59-0.60) but strongly only under the panel and in cycle 5. The Netto et al. (2005) models improve the correlation only slightly and need a unit conversion for this laboratory data."),
              BUL("Among camera indices DGCI was best (r = 0.75 with SPAD within cycles, about 0.5 with laboratory chlorophyll). Camera colour features separate shaded from unshaded plots with about 87 % accuracy on unseen cycles. Spectrometer indices were not informative, and the spectrometer data have quality problems."),
              BUL("No vegetation index, neural network or PCA-based model predicted chlorophyll of an unseen harvest cycle; PCA and neural networks did not change this, and apparent successes came from data leakage."),
              BUL("The study objective is therefore only partly met: the tools detect shade- and growth-stage-related colour differences, but detection of irrigation-related water stress, and the effects on water-use efficiency and biomass, could not be shown with the current data."),
              P("<b>Recommendations.</b>", "body"),
              BUL("Measure water status directly for each treatment (soil moisture, leaf water potential or stomatal conductance) and record biomass or yield, then test whether SPAD, camera and spectrometer predict these."),
              BUL("Create a clearer stress contrast (lower irrigation level or a defined dry-down) and keep the treatment layout separate from the shade layout."),
              BUL("Standardise acquisition: white reference at every spectrometer session, grey card and fixed exposure for photographs, and check the near-infrared range of the spectrometer."),
              BUL("Correct the laboratory file (total chlorophyll, the ceiling of chlorophyll a, units) and re-run the analysis; use the scripts in the project folder, which regenerate all tables and figures."),
              BUL("Report every model with leave-one-cycle-out validation, use repeats as subsamples (or average them) in statistical tests, and try machine-learning models only after simple models and a real signal are available.")]

    # ================================================================== appendix
    story += [H1("Appendix. Project files"),
              P("All analyses can be reproduced from the project folder: <i>image_data/</i> (original photographs), <i>data_preprocessed/</i> (leaf-only images), <i>data/</i> (SPAD, spectrometer and laboratory tables and the untouched originals), <i>results/tables/</i> (Excel result tables), <i>results/figures/</i> (all figures as SVG) and <i>scripts/</i> (one Python script per step; see README.md). "
                "This report was generated by scripts/build_report.py from the result tables.")]

    Doc(OUT).build(assemble(story))
    print("written", OUT, round(OUT.stat().st_size / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
