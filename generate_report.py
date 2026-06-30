"""Generate PDF report comparing Run6, Run7, Run8."""
import io, textwrap
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    HRFlowable, PageBreak,
)
from reportlab.platypus.flowables import KeepTogether

ROOT = Path(__file__).parent
RES  = ROOT / "results"
OUT  = ROOT / "results" / "report_run6_7_8.pdf"

# ── colour palette ──────────────────────────────────────────────────────────
C6, C7, C8 = "#4C72B0", "#DD8452", "#55A868"   # blue, orange, green
GREY = "#888888"

RUNS = {
    "Run6\nJoint baseline\n(8 534 Kaggle + 900 DeepPCB ×4)": {
        "color": C6, "short": "Run6",
        "exp": "exp_006_yolov11n_joint_kaggle_deeppcb_cleanval",
        "train_n": "8 534 + 900×4 = 12 134", "val_scheme": "Kaggle-val + DeepPCB 10% slice",
        "kag_map50": 0.8045, "dp_map50": 0.9788,
    },
    "Run7\nEqual data\n(900 Kaggle + 900 DeepPCB)": {
        "color": C7, "short": "Run7",
        "exp": "exp_007_yolov11n_joint_equaldata",
        "train_n": "900 + 900 = 1 800", "val_scheme": "Kaggle-val + DeepPCB 10% slice",
        "kag_map50": 0.7845, "dp_map50": 0.9707,
    },
    "Run8\nKaggle binarized\n(color→BW domain match)": {
        "color": C8, "short": "Run8",
        "exp": "exp_008_yolov11n_joint_kaggle_bw",
        "train_n": "8 534 BW + 900×4 = 12 134", "val_scheme": "Kaggle-val (BW) + DeepPCB 10% slice",
        "kag_map50": 0.9545, "dp_map50": 0.9798,
    },
}

CLASSES = ["mouse_bite", "spur", "missing_hole", "short", "open_circuit", "spurious_copper"]
CLASS_LABELS = ["Mouse\nBite", "Spur", "Missing\nHole", "Short", "Open\nCircuit", "Spurious\nCopper"]


# ── helpers ──────────────────────────────────────────────────────────────────
def fig_to_image(fig, width_cm=17):
    from PIL import Image as PILImage
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    pil = PILImage.open(buf)
    px_w, px_h = pil.size
    w = width_cm * cm
    h = w * px_h / px_w
    buf.seek(0)
    img = Image(buf, width=w, height=h)
    img.hAlign = "CENTER"
    return img


def load_perclass(exp, domain):
    p = RES / f"{exp}_on_{domain}_perclass.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    return dict(zip(df["Class"], df["AP@50"]))


def load_training_curves(exp):
    p = RES / exp / "results.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    return df


# ── Figure 1 — mAP@50 grouped bar (Kaggle + DeepPCB) ───────────────────────
def fig_map50_bars():
    labels = [v["short"] for v in RUNS.values()]
    kag    = [v["kag_map50"] for v in RUNS.values()]
    dp     = [v["dp_map50"]  for v in RUNS.values()]
    clrs   = [v["color"]     for v in RUNS.values()]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    b1 = ax.bar(x - w/2, kag, w, color=clrs, alpha=0.9, label="Kaggle test", edgecolor="white", linewidth=0.5)
    b2 = ax.bar(x + w/2, dp,  w, color=clrs, alpha=0.55, label="DeepPCB test",
                edgecolor="white", linewidth=0.5, hatch="//")

    for bar, v in zip(b1, kag):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar, v in zip(b2, dp):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylim(0.70, 1.02)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("mAP@50", fontsize=11)
    ax.set_title("Cross-Domain Test Performance (mAP@50)", fontsize=13, fontweight="bold", pad=10)
    ax.legend(loc="lower right", fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5); ax.set_axisbelow(True)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    return fig_to_image(fig)


# ── Figure 2 — per-class radar (Kaggle test) ────────────────────────────────
def fig_radar_kaggle():
    N = len(CLASSES)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for meta in RUNS.values():
        pc = load_perclass(meta["exp"], "kaggle")
        if pc is None: continue
        vals = [pc.get(c, 0) for c in CLASSES] + [pc.get(CLASSES[0], 0)]
        ax.plot(angles, vals, color=meta["color"], lw=2, label=meta["short"])
        ax.fill(angles, vals, color=meta["color"], alpha=0.10)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(CLASS_LABELS, fontsize=9)
    ax.set_ylim(0.6, 1.0)
    ax.set_yticks([0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels(["0.7","0.8","0.9","1.0"], fontsize=7, color=GREY)
    ax.set_title("Per-Class AP@50 — Kaggle Test", fontsize=12, fontweight="bold", pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), fontsize=9)
    ax.grid(color=GREY, linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig_to_image(fig, width_cm=12)


# ── Figure 3 — per-class grouped bar (DeepPCB test) ────────────────────────
def fig_perclass_deeppcb():
    x = np.arange(len(CLASSES))
    w = 0.26
    fig, ax = plt.subplots(figsize=(10, 4.5))
    offsets = [-w, 0, w]
    for (meta, offset) in zip(RUNS.values(), offsets):
        pc = load_perclass(meta["exp"], "deeppcb")
        if pc is None: continue
        vals = [pc.get(c, 0) for c in CLASSES]
        ax.bar(x + offset, vals, w, color=meta["color"], alpha=0.85,
               label=meta["short"], edgecolor="white", linewidth=0.4)

    ax.set_xticks(x); ax.set_xticklabels(CLASS_LABELS, fontsize=9)
    ax.set_ylim(0.88, 1.01)
    ax.set_ylabel("AP@50", fontsize=11)
    ax.set_title("Per-Class AP@50 — DeepPCB Test", fontsize=13, fontweight="bold", pad=10)
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5); ax.set_axisbelow(True)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    return fig_to_image(fig)


# ── Figure 4 — training mAP50 curves ────────────────────────────────────────
def fig_training_curves():
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    for meta in RUNS.values():
        df = load_training_curves(meta["exp"])
        if df is None: continue
        col50   = next((c for c in df.columns if "mAP50" in c and "95" not in c), None)
        col5095 = next((c for c in df.columns if "mAP50-95" in c), None)
        ep = df["epoch"]
        if col50:
            axes[0].plot(ep, df[col50], color=meta["color"], lw=1.8, label=meta["short"])
        if col5095:
            axes[1].plot(ep, df[col5095], color=meta["color"], lw=1.8, label=meta["short"])

    for ax, title in zip(axes, ["Val mAP@50", "Val mAP@50-95"]):
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel(title, fontsize=10)
        ax.set_title(title + " During Training", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4); ax.set_axisbelow(True)
        ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    return fig_to_image(fig)


# ── Figure 5 — delta vs Run6 heatmap ─────────────────────────────────────────
def fig_delta_heatmap():
    run6_kag = {c: 0 for c in CLASSES}
    run6_dp  = {c: 0 for c in CLASSES}
    pc6k = load_perclass("exp_006_yolov11n_joint_kaggle_deeppcb_cleanval", "kaggle")
    pc6d = load_perclass("exp_006_yolov11n_joint_kaggle_deeppcb_cleanval", "deeppcb")
    if pc6k: run6_kag = pc6k
    if pc6d: run6_dp  = pc6d

    rows, row_labels = [], []
    for meta in list(RUNS.values())[1:]:   # Run7, Run8 vs Run6
        pc_k = load_perclass(meta["exp"], "kaggle")
        pc_d = load_perclass(meta["exp"], "deeppcb")
        if pc_k:
            rows.append([pc_k.get(c,0) - run6_kag.get(c,0) for c in CLASSES])
            row_labels.append(f"{meta['short']} Kaggle Δ")
        if pc_d:
            rows.append([pc_d.get(c,0) - run6_dp.get(c,0) for c in CLASSES])
            row_labels.append(f"{meta['short']} DeepPCB Δ")

    arr = np.array(rows)
    fig, ax = plt.subplots(figsize=(11, 2.8))
    vmax = max(abs(arr.min()), abs(arr.max())) + 0.005
    im = ax.imshow(arr, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASS_LABELS, fontsize=9)
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels, fontsize=9)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=8.5,
                    color="black" if abs(v) < vmax*0.6 else "white", fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8, label="ΔAP@50 vs Run6")
    ax.set_title("Per-Class AP@50 Delta vs Run6 (green = better)", fontsize=12, fontweight="bold", pad=10)
    fig.tight_layout()
    return fig_to_image(fig)


# ── PDF assembly ─────────────────────────────────────────────────────────────
def build_pdf():
    doc = SimpleDocTemplate(str(OUT), pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=20, spaceAfter=6,
                         textColor=colors.HexColor("#1a1a2e"), alignment=TA_CENTER)
    H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13, spaceBefore=14,
                         spaceAfter=4, textColor=colors.HexColor("#16213e"))
    BODY = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10, leading=15,
                           alignment=TA_JUSTIFY, spaceAfter=6)
    CAPTION = ParagraphStyle("Cap", parent=styles["Normal"], fontSize=8.5,
                              textColor=colors.HexColor(GREY), alignment=TA_CENTER,
                              spaceAfter=10)
    MONO = ParagraphStyle("Mono", parent=styles["Code"], fontSize=9, leading=13)

    story = []

    # ── Title block ──────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("PCB Defect Detection", H1))
    story.append(Paragraph("Joint Multi-Domain Training — Runs 6 · 7 · 8", H1))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#4C72B0"), spaceAfter=10))
    story.append(Paragraph(
        "Abdullah Ashfaq &nbsp;|&nbsp; YOLOv11n &nbsp;|&nbsp; 200 epochs &nbsp;|&nbsp; June 2026",
        ParagraphStyle("sub", parent=styles["Normal"], fontSize=10, alignment=TA_CENTER,
                       textColor=colors.HexColor(GREY), spaceAfter=16)
    ))

    # ── Experiment summary table ─────────────────────────────────────────────
    story.append(Paragraph("1. Experiment Overview", H2))
    tdata = [
        ["", "Run6 (Baseline)", "Run7 (Equal Data)", "Run8 (BW Kaggle)"],
        ["Training images", "12 134", "1 800", "12 134"],
        ["Kaggle images", "8 534 (color)", "900 (color, downsampled)", "8 534 (binarized BW)"],
        ["DeepPCB images", "900 × 4 oversample", "900 (no oversample)", "900 × 4 oversample"],
        ["Val set", "Kaggle-val + DeepPCB 10%", "Kaggle-val + DeepPCB 10%", "Kaggle-val (BW) + DeepPCB 10%"],
        ["Key variable", "Strong Kaggle majority", "Balanced domains", "Domain appearance match"],
    ]
    ts = TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#16213e")),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (-1,-1), 9),
        ("BACKGROUND", (0,1), (0,-1), colors.HexColor("#f0f0f0")),
        ("FONTNAME",   (0,1), (0,-1), "Helvetica-Bold"),
        ("ALIGN",      (1,0), (-1,-1), "CENTER"),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
        ("ROWBACKGROUNDS", (1,1), (-1,-1), [colors.white, colors.HexColor("#f8f8f8")]),
        ("GRID",       (0,0), (-1,-1), 0.4, colors.HexColor("#cccccc")),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ])
    col_w = [3.8*cm, 4.0*cm, 4.2*cm, 4.2*cm]
    t = Table(tdata, colWidths=col_w)
    t.setStyle(ts)
    story.append(t)
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph(
        "All three runs use YOLOv11n trained for 200 epochs (patience 50) with identical "
        "augmentation. The validation set never includes the DeepPCB test split, ensuring "
        "checkpoint selection is clean. The test mAP@50 numbers below are final held-out evaluations.",
        BODY
    ))

    # ── Fig 1 — overall bar ──────────────────────────────────────────────────
    story.append(Paragraph("2. Overall Test Performance", H2))
    story.append(fig_map50_bars())
    story.append(Paragraph(
        "Figure 1. mAP@50 on held-out Kaggle test (solid) and DeepPCB test (hatched) sets. "
        "Run8's binarization strategy lifts Kaggle performance by +15 pp over Run6 while "
        "leaving DeepPCB accuracy virtually unchanged.",
        CAPTION
    ))

    # ── headline numbers table ───────────────────────────────────────────────
    hdata = [
        ["", "Kaggle test mAP@50", "DeepPCB test mAP@50", "Δ Kaggle vs Run6", "Δ DeepPCB vs Run6"],
        ["Run6", "0.8045", "0.9788", "—", "—"],
        ["Run7", "0.7845", "0.9707", "−0.020", "−0.008"],
        ["Run8", "0.9545", "0.9798", "+0.150 ▲", "+0.001"],
    ]
    hs = TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#16213e")),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (-1,-1), 9.5),
        ("ALIGN",      (1,0), (-1,-1), "CENTER"),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
        ("FONTNAME",   (0,1), (0,-1), "Helvetica-Bold"),
        ("BACKGROUND", (0,1), (0,-1), colors.HexColor("#f0f0f0")),
        ("TEXTCOLOR",  (4,3), (4,3), colors.HexColor("#2d6a4f")),
        ("TEXTCOLOR",  (3,3), (3,3), colors.HexColor("#2d6a4f")),
        ("FONTNAME",   (3,3), (4,3), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (1,1), (-1,-1), [colors.white, colors.HexColor("#f8f8f8"), colors.HexColor("#eef6ee")]),
        ("GRID",       (0,0), (-1,-1), 0.4, colors.HexColor("#cccccc")),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ])
    ht = Table(hdata, colWidths=[2.8*cm, 3.8*cm, 3.8*cm, 3.5*cm, 3.5*cm])
    ht.setStyle(hs)
    story.append(ht)
    story.append(Spacer(1, 0.5*cm))

    # ── Fig 2 + 3 — per-class ────────────────────────────────────────────────
    story.append(Paragraph("3. Per-Class Breakdown", H2))
    story.append(fig_radar_kaggle())
    story.append(Paragraph(
        "Figure 2. Radar chart of per-class AP@50 on the Kaggle test set. Run8 (green) "
        "consistently dominates all six defect categories. Run7 (orange) underperforms Run6 "
        "on spurious_copper, likely due to limited Kaggle training samples for that class.",
        CAPTION
    ))
    story.append(fig_perclass_deeppcb())
    story.append(Paragraph(
        "Figure 3. Per-class AP@50 on the DeepPCB test set. All three runs perform similarly "
        "(0.92–0.99), confirming that the DeepPCB domain is well-covered across strategies. "
        "The 'short' class remains the hardest for all models.",
        CAPTION
    ))

    # ── Fig 4 — training curves ──────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("4. Training Dynamics", H2))
    story.append(fig_training_curves())
    story.append(Paragraph(
        "Figure 4. Validation mAP@50 and mAP@50-95 over 200 epochs (clean val set). "
        "Run6 and Run8 converge higher on mAP@50. Run7 converges faster but plateaus lower "
        "due to its smaller 1 800-image training set. Val metrics are on the mixed val set.",
        CAPTION
    ))

    # ── Fig 5 — delta heatmap ────────────────────────────────────────────────
    story.append(Paragraph("5. Per-Class Delta vs Run6", H2))
    story.append(fig_delta_heatmap())
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "Figure 5. Per-class AP@50 change relative to the Run6 baseline. "
        "Green cells indicate improvement; red indicates regression. "
        "Run8 shows large green gains across all Kaggle classes (+8 to +19 pp). "
        "Run7 shows mild regressions on Kaggle, particularly spurious_copper (−6 pp).",
        CAPTION
    ))

    # ── Key findings ─────────────────────────────────────────────────────────
    story.append(Paragraph("6. Key Findings", H2))
    findings = [
        ("<b>Domain gap is the primary bottleneck.</b> Run8 shows that binarizing Kaggle "
         "images to match DeepPCB's grayscale appearance closes the cross-domain gap far "
         "more effectively than rebalancing data quantities."),
        ("<b>Equal-data balancing (Run7) hurts.</b> Cutting Kaggle training data from 8 534 "
         "to 900 images starves the model on color-domain features, causing a −2 pp drop on "
         "Kaggle test with no benefit on DeepPCB."),
        ("<b>DeepPCB is saturated.</b> All three runs score 0.97–0.98 on DeepPCB test, "
         "suggesting the model has learned the binary-image defect patterns well regardless "
         "of the Kaggle training strategy."),
        ("<b>Run8 is the recommended checkpoint</b> for any deployment scenario that must "
         "handle both color PCB images and binary/grayscale inspection scans."),
    ]
    for i, f in enumerate(findings, 1):
        story.append(Paragraph(f"{'●'} {f}", BODY))

    story.append(Spacer(1, 0.4*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc")))
    story.append(Paragraph(
        "Model: YOLOv11n &nbsp;|&nbsp; Framework: Ultralytics &nbsp;|&nbsp; "
        "Datasets: norbertelter/pcb-defect-dataset (Kaggle), DeepPCB (Tang et al.)",
        ParagraphStyle("footer", parent=styles["Normal"], fontSize=8,
                       textColor=colors.HexColor(GREY), alignment=TA_CENTER, spaceBefore=6)
    ))

    doc.build(story)
    print(f"Report saved → {OUT}")


if __name__ == "__main__":
    build_pdf()
