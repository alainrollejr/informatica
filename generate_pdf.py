"""
generate_pdf.py
Converts graphs.py to a well-formatted PDF using fpdf2 + Monaco (monospace).
Run:  python generate_pdf.py
"""
from fpdf import FPDF
import os

# ── settings ────────────────────────────────────────────────────────
SRC        = "graphs.py"
OUT        = "graphs.pdf"
FONT_PATH  = "/System/Library/Fonts/Monaco.ttf"
FONT_SIZE  = 6.5          # pt — fits ~105 chars across portrait A4
LINE_H     = 3.1          # mm line height
MARGIN     = 10           # mm horizontal margin
# ─────────────────────────────────────────────────────────────────────

# Row background and text colours (RGB 0-255)
BG_CODE    = (252, 252, 252)   # near-white for regular code
BG_COMMENT = (240, 248, 240)   # faint green for comment lines
BG_BLANK   = (255, 255, 255)   # white for blank lines
BG_SECTION = (230, 240, 255)   # light blue for section headers (# ===)

FG_COMMENT = (70,  110, 70)    # muted green text for comments
FG_SECTION = (25,  65,  150)   # blue text for section headers
FG_CODE    = (25,  25,  25)    # near-black for code


class CodePDF(FPDF):
    """FPDF subclass — no repeating header, only a page-number footer."""

    def footer(self):
        self.set_y(-10)
        self.set_font("mono", size=6.5)
        self.set_text_color(160, 160, 160)
        self.cell(0, 4, f"{self.page_no()}", align="C")


def classify(line):
    """Return (bg_colour, fg_colour) based on what kind of line this is."""
    s = line.strip()
    if not s:
        return BG_BLANK, FG_CODE
    if s.startswith("# =") or s.startswith("# -") or s.startswith("# ─"):
        return BG_SECTION, FG_SECTION
    if s.startswith("#"):
        return BG_COMMENT, FG_COMMENT
    return BG_CODE, FG_CODE


def main():
    with open(SRC, encoding="utf-8") as fh:
        lines = fh.readlines()

    pdf = CodePDF(orientation="P", unit="mm", format="A4")
    pdf.set_margins(MARGIN, MARGIN, MARGIN)
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_font("mono", style="", fname=FONT_PATH)
    pdf.add_page()

    effective_w = 210 - 2 * MARGIN    # A4 portrait width minus margins

    for raw in lines:
        text = raw.expandtabs(4).rstrip("\n")
        bg, fg = classify(text)

        pdf.set_fill_color(*bg)
        pdf.set_text_color(*fg)
        pdf.set_font("mono", size=FONT_SIZE)

        pdf.multi_cell(
            w=effective_w,
            h=LINE_H,
            text=text,
            border=0,
            fill=True,
            new_x="LMARGIN",
            new_y="NEXT",
        )

    pdf.output(OUT)
    size = os.path.getsize(OUT)
    print(f"Written: {OUT}  ({size:,} bytes,  {pdf.page} pages)")


if __name__ == "__main__":
    main()
