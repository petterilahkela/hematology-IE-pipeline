"""
plot_nonnull_table.py
"""

import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

FIELDS = ["suspected_dg", "prev_cur_dg", "treatment", "cur_dg"]
FIELD_LABELS = {
    "suspected_dg": "Suspected diagnosis",
    "prev_cur_dg":  "Previous diagnosis",
    "treatment":    "Treatment",
    "cur_dg":       "Current diagnosis",
}

GOLD = {
    "suspected_dg": 343/357,
    "prev_cur_dg":  194/357,
    "treatment":    95/357,
    "cur_dg":       357/357,
}

MODELS = ["Llama 3.1", "Poro2", "Poro2 FT", "Poro2 FT RAG"]
DATA = {
    "Llama 3.1":    {"suspected_dg": 344/357, "prev_cur_dg": 153/357, "treatment": 173/357, "cur_dg": 357/357},
    "Poro2":        {"suspected_dg": 332/357, "prev_cur_dg": 178/357, "treatment": 167/357, "cur_dg": 357/357},
    "Poro2 FT":     {"suspected_dg": 343/357, "prev_cur_dg": 169/357, "treatment":  89/357, "cur_dg": 357/357},
    "Poro2 FT RAG": {"suspected_dg": 345/357, "prev_cur_dg": 182/357, "treatment":  68/357, "cur_dg": 357/357},
}

HEADER_BG    = "#5E6F7F"
GOLD_COLOR   = "#4A4A4A"
MODEL_COLORS = ["#9DB8C8", "#7B9DBF", "#5B7FA6", "#3D5A73"]
ROW_BG       = ["#F2F6FA", "#FFFFFF"]

def draw_cell(ax, x, y, w, h, text, bg, text_color="white", fontsize=10, bold=False):
    rect = mpatches.FancyBboxPatch(
        (x + 0.003, y + 0.003), w - 0.006, h - 0.006,
        boxstyle="round,pad=0.005", linewidth=0,
        facecolor=bg, transform=ax.transAxes, clip_on=False
    )
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold" if bold else "normal",
            color=text_color, transform=ax.transAxes)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 3))
    ax.axis("off")
    plt.rcParams.update({"font.family": "Arial"})
    ax.set_title("Non-null extraction rates by field and model", fontsize=10, pad=12)

    # Column layout: [Field label, Gold, Model1, Model2, Model3, Model4]
    col_widths = [0.26, 0.10, 0.16, 0.16, 0.16, 0.16]
    col_x = [sum(col_widths[:i]) for i in range(len(col_widths))]
    row_h = 0.18
    header_y = 0.78

    # Header row
    draw_cell(ax, col_x[0], header_y, col_widths[0], row_h, "Field", HEADER_BG, bold=True)
    draw_cell(ax, col_x[1], header_y, col_widths[1], row_h, "Gold", GOLD_COLOR, bold=True)
    for j, model in enumerate(MODELS):
        draw_cell(ax, col_x[j+2], header_y, col_widths[j+2], row_h, model, MODEL_COLORS[j], bold=True)

    # Data rows
    for i, field in enumerate(FIELDS):
        y = header_y - (i + 1) * row_h
        bg = ROW_BG[i % 2]

        # Field name
        draw_cell(ax, col_x[0], y, col_widths[0], row_h,
                  FIELD_LABELS[field], bg, text_color="#1F4E79", bold=True)

        # Gold
        draw_cell(ax, col_x[1], y, col_widths[1], row_h,
                  f"{GOLD[field]:.0%}", bg, text_color=GOLD_COLOR, bold=True)

        # Models
        for j, model in enumerate(MODELS):
            val = DATA[model][field]
            draw_cell(ax, col_x[j+2], y, col_widths[j+2], row_h,
                      f"{val:.0%}", bg, text_color="#1a1a1a")

    plt.tight_layout()
    plt.savefig(args.output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {args.output}")

if __name__ == "__main__":
    main()