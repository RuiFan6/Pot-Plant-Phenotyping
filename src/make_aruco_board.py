from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from pathlib import Path

OUT_DIR = Path("outputs")
png_path = OUT_DIR / "aruco_board.png"
pdf_path = OUT_DIR / "aruco_board_A4_tiled.pdf"

square_size_mm = 20
squares_x = 8
squares_y = 8

board_w_mm = squares_x * square_size_mm
board_h_mm = squares_y * square_size_mm

page_w_mm = A4[0] / mm
page_h_mm = A4[1] / mm

c = canvas.Canvas(str(pdf_path), pagesize=A4)

# ---- center board ----
x0 = (page_w_mm - board_w_mm) / 2 * mm
y0 = (page_h_mm - board_h_mm) / 2 * mm

c.drawImage(
    str(png_path),
    x0,
    y0,
    width=board_w_mm * mm,
    height=board_h_mm * mm,
    preserveAspectRatio=True,
    mask='auto'
)

# ---- outer boundary ----
c.setLineWidth(1.2)
c.rect(x0, y0, board_w_mm * mm, board_h_mm * mm)

# ---- grid cut lines ----
c.setLineWidth(0.4)

for i in range(1, squares_x):
    x = x0 + i * square_size_mm * mm
    c.line(x, y0, x, y0 + board_h_mm * mm)

for j in range(1, squares_y):
    y = y0 + j * square_size_mm * mm
    c.line(x0, y, x0 + board_w_mm * mm, y)

c.showPage()
c.save()

print("Saved tiled PDF:", pdf_path.resolve())