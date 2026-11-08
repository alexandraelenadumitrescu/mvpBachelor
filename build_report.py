"""
build_report.py — Generates PhotoMatch_Evaluation_Report.docx
Run: python build_report.py
"""

import os
import glob
from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

ROOT     = os.path.dirname(os.path.abspath(__file__))
ASSETS   = os.path.join(ROOT, "evaluation_assets")
OUT_PATH = os.path.join(ASSETS, "PhotoMatch_Evaluation_Report_v5.docx")

doc = Document()

# ── Page margins ──────────────────────────────────────────────────────────────
for section in doc.sections:
    section.top_margin    = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.left_margin   = Cm(3.0)
    section.right_margin  = Cm(2.0)

# ── Helpers ───────────────────────────────────────────────────────────────────
NAVY  = RGBColor(0x1F, 0x35, 0x64)
AMBER = RGBColor(0xC9, 0xA8, 0x4C)
DARK  = RGBColor(0x44, 0x44, 0x44)

def heading(text, level=1):
    p = doc.add_heading(text, level=level)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    for run in p.runs:
        run.font.color.rgb = NAVY
    return p

def body(text, italic=False, size=11):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.italic = italic
    run.font.size = Pt(size)
    p.paragraph_format.space_after = Pt(6)
    return p

def caption(text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.italic = True
    run.font.size = Pt(9.5)
    run.font.color.rgb = DARK
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(12)
    return p

def insert_image(path, width_inches=5.8):
    if os.path.exists(path):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run()
        run.add_picture(path, width=Inches(width_inches))
    else:
        doc.add_paragraph(f"[IMAGE NOT FOUND: {os.path.basename(path)}]")

def bullet(text, bold_prefix=None):
    p = doc.add_paragraph(style="List Bullet")
    if bold_prefix:
        r1 = p.add_run(bold_prefix + ": ")
        r1.bold = True
        r1.font.size = Pt(11)
        r2 = p.add_run(text)
        r2.font.size = Pt(11)
    else:
        r = p.add_run(text)
        r.font.size = Pt(11)
    return p

def hr():
    p = doc.add_paragraph()
    pPr = p._p.get_or_add_pPr()
    pBdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "6")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), "C9A84C")
    pBdr.append(bottom)
    pPr.append(pBdr)
    p.paragraph_format.space_after = Pt(4)


# ══════════════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════════════
doc.add_paragraph()
doc.add_paragraph()

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("PhotoMatch")
r.font.size = Pt(28)
r.bold = True
r.font.color.rgb = NAVY

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("Raport de Evaluare a Modelelor ML")
r.font.size = Pt(18)
r.bold = True
r.font.color.rgb = AMBER

doc.add_paragraph()

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("Evaluare Cantitativa & Explicabilitate (XAI)")
r.font.size = Pt(13)
r.italic = True
r.font.color.rgb = DARK

doc.add_paragraph()
doc.add_paragraph()

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("Dumitrescu Alexandra-Elena\n2025")
r.font.size = Pt(11)

doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — EVALUARE CANTITATIVA
# ══════════════════════════════════════════════════════════════════════════════
heading("1. Evaluare Cantitativa a Modelelor", level=1)
hr()
body(
    "Aceasta sectiune prezinta metricile de performanta obtinute pentru cele "
    "doua modele principale ale sistemului PhotoMatch: capul de regresie "
    "MobileNetV3Small pentru detectarea defectelor fotografice si modelul "
    "YOLOv8n utilizat pentru detectarea continutului sensibil."
)

# ── 1.1 Loss Curve ────────────────────────────────────────────────────────────
heading("1.1  Curba de Pierdere — MobileNetV3Small", level=2)
body(
    "Modelul de regresie este construit pe arhitectura MobileNetV3Small, cu un cap "
    "personalizat: Linear(576→128) → Hardswish → Dropout(0.2) → "
    "Linear(128→5) → Sigmoid. Iesirile reprezinta scoruri continue in "
    "intervalul [0, 1] pentru cei 5 indicatori de calitate: Blur, Noise, "
    "Supraexpunere, Subexpunere si Compresie JPEG."
)
body(
    "Antrenamentul s-a desfasurat pe 15 epoci, cu functie de pierdere MSE "
    "(Mean Squared Error). Curba de mai jos ilustreaza convergenta rapida in "
    "primele 5 epoci, urmata de stabilizare la o valoare de ~0.018 MSE pe "
    "setul de validare."
)

insert_image(os.path.join(ASSETS, "mobilenetv3_loss_curve.png"), width_inches=5.5)
caption("Figura 1.1 — Curba de pierdere MSE pe 15 epoci (train vs. validare).")

# ── 1.2 Scatter Plots ─────────────────────────────────────────────────────────
heading("1.2  Corelatie Predictie vs. Ground Truth — Defecte Fotografice", level=2)
body(
    "Pentru evaluarea calitatii predictiilor, au fost generate seturi sintetice "
    "de test: imaginile din colectia MIT FiveK au fost distorsionate cu niveluri "
    "cunoscute ale fiecarui defect (50 de puncte uniform distribuite in [0, 1]), "
    "reprezentand valorile de ground truth. Modelul a rulat inferenta reala pe "
    "aceste imagini pentru a produce scorurile prezise."
)
body("Metodologia de generare sintetica per defect:")

bullet("GaussianBlur cu sigma in [0, 10], normalizat la [0, 1].", bold_prefix="Blur")
bullet("Zgomot Gaussian aditiv cu std in [0, 80/255].", bold_prefix="Noise")
bullet(
    "Multiplicare de luminozitate cu factor in [1.0, 3.0], clipping la 255.",
    bold_prefix="Supraexpunere",
)
bullet(
    "Factor de intunecare in [0.05, 1.0] (invers proportional cu nivelul).",
    bold_prefix="Subexpunere",
)
bullet(
    "Calitate JPEG in [10, 95] (invers proportional cu nivelul de degradare).",
    bold_prefix="Compresie JPEG",
)

doc.add_paragraph()
body(
    "Linia punctata y=x reprezinta predictia perfecta. Distributia punctelor "
    "fata de aceasta dreapta reflecta capacitatea modelului de a estima "
    "fiecare tip de degradare in mod independent."
)

insert_image(os.path.join(ASSETS, "defect_scatter_plots.png"), width_inches=6.2)
caption(
    "Figura 1.2 — Scatter plots predictie vs. ground truth pentru cei 5 "
    "indicatori de calitate. Linia punctata reprezinta corelatia perfecta (y=x)."
)

# ── 1.3 YOLOv8n ──────────────────────────────────────────────────────────────
heading("1.3  Evaluare YOLOv8n — Detectare Continut Sensibil", level=2)
body(
    "Componenta de detectare a continutului sensibil utilizeaza modelul YOLOv8n "
    "(Ultralytics), antrenat pe setul COCO. Clasele relevante pentru aplicatie "
    "sunt mapate din spatiul COCO: tv/monitor → screen(tv), "
    "laptop → screen(laptop), telefon mobil → phone, carte → document. "
    "Pragul de confidenta operational utilizat in productie este 0.25."
)
body(
    "Validarea a fost efectuata pe subsetul coco128 (128 imagini reprezentative "
    "din COCO train2017), filtrand exclusiv clasele de interes."
)

insert_image(os.path.join(ASSETS, "yolov8n_pr_curve.png"), width_inches=5.5)
caption(
    "Figura 1.3a — Curbe Precizie-Recall per clasa pentru clasele sensibile. "
    "Valorile AP sunt indicate in legenda."
)

doc.add_paragraph()

insert_image(os.path.join(ASSETS, "yolov8n_confusion_matrix.png"), width_inches=5.5)
caption(
    "Figura 1.3b — Matricea de confuzie normalizata (row-normalised) pentru "
    "cele 4 clase sensibile + clasa background."
)

doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — XAI
# ══════════════════════════════════════════════════════════════════════════════
heading("2. Explicabilitate (XAI) — Vizualizari Grad-CAM Personalizate", level=1)
hr()
body(
    "Aceasta sectiune prezinta trei vizualizari de tip Grad-CAM adaptate "
    "contextului fotografic, implementate cu biblioteca pytorch-grad-cam. "
    "Stratul tinta pentru extragerea hartilor de activare este "
    "backbone.features[-1] — ultimul bloc Conv2dNormActivation (576 canale) "
    "inainte de average pooling. Fiecare vizualizare foloseste gradientul "
    "fata de un output specific al modelului."
)
body(
    "Imaginea de intrare utilizata pentru toate cele trei vizualizari: "
    "a0001-jmac_DSC1459.jpg (colectia MIT FiveK, redimensionata la 224x224 px)."
)

# ── 2.1 Zebra ─────────────────────────────────────────────────────────────────
heading("2.1  Zebra Pattern — Detectare Supraexpunere (output idx=2)", level=2)
body(
    "Harta Grad-CAM este calculata fata de outputul de supraexpunere (index 2). "
    "Zonele cu activare >= 0.5 sunt identificate ca regiuni supraexpuse. "
    "Peste aceste regiuni se aplica un pattern diagonal alb-negru (dungi la 45 grade, "
    "pas de 10 pixeli), inspirat din indicatorul de supraexpunere al camerelor "
    "fotografice profesionale (zebra pattern)."
)
body(
    "Formula pentru generarea dungilor: pixel (r, c) este alb daca "
    "((r + c) // 10) % 2 == 0, negru altfel. Dungile sunt aplicate strict "
    "in interiorul mastii binare a zonei cu activare ridicata."
)

insert_image(os.path.join(ASSETS, "xai_zebra_overexposure.png"), width_inches=3.8)
caption(
    "Figura 2.1 — Zebra Pattern suprapus pe zonele de supraexpunere "
    "identificate de Grad-CAM (prag de activare 0.5)."
)

# ── 2.2 Focus Peaking ─────────────────────────────────────────────────────────
heading("2.2  Focus Peaking — Detectare Blur (output idx=0)", level=2)
body(
    "Harta Grad-CAM este calculata fata de outputul de blur (index 0). "
    "Imaginea originala este convertita la tonuri de gri (3 canale pentru "
    "desenare). Contururile regiunilor cu activare in top 30% (prag la "
    "percentila 70) sunt extrase cu cv2.findContours si desenate in doua "
    "treceri pentru efect de stralucire (glow):"
)
bullet("Trecere 1: contur gros (thickness=3), culoare aurie BGR=(55, 175, 212).")
bullet("Trecere 2: contur subtire (thickness=1), culoare mai deschisa BGR=(100, 210, 240).")

doc.add_paragraph()
body(
    "Aceasta tehnica reproduce vizual focus peaking-ul din camerele mirrorless, "
    "unde marginile in focus sunt evidentiate cu o culoare contrastanta."
)

insert_image(os.path.join(ASSETS, "xai_focus_peaking_blur.png"), width_inches=3.8)
caption(
    "Figura 2.2 — Focus Peaking: baza in tonuri de gri cu contururi aurii "
    "trasate pe zonele de blur identificate de Grad-CAM."
)

# ── 2.3 Spotlight Vignette ────────────────────────────────────────────────────
heading("2.3  Spotlight Vignette — Atentie Globala (medie pe toate 5 outputuri)", level=2)
body(
    "Harta finala de activare este media celor 5 harti Grad-CAM individuale "
    "(cate una per defect), normalizata in [0, 1]. Aceasta capteaza zona de "
    "interes global al modelului indiferent de tipul specific de degradare."
)
body("Efectul de vigneta spotlight se obtine astfel:")
bullet(
    "Se creeaza o versiune intunecata a imaginii originale: "
    "pixel_dark = pixel_orig x 0.3  (70% intunecare)."
)
bullet("Se blendeaza: result = pixel_dark + (pixel_orig - pixel_dark) x heatmap_mask.")
bullet("Unde activarea este maxima → imaginea ramane la expunerea originala.")
bullet("Unde activarea este nula → imaginea este 70% mai intunecata (fundal vignette).")

doc.add_paragraph()

insert_image(os.path.join(ASSETS, "xai_spotlight_vignette.png"), width_inches=3.8)
caption(
    "Figura 2.3 — Spotlight Vignette: zona cu cea mai ridicata activare Grad-CAM "
    "ramane la expunerea originala, restul imaginii este intunecat cu 70%."
)

doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2.4 — HEATMAPS TERMICE
# ══════════════════════════════════════════════════════════════════════════════
heading("2.4  Heatmap-uri Termice Grad-CAM — Per Defect", level=2)
body(
    "Pe langa vizualizarile fotografice din sectiunile anterioare, imaginile de "
    "mai jos prezinta heatmap-urile termice clasice (colormap JET) suprapuse pe "
    "imaginea originala, cate unul pentru fiecare din cele 5 outputuri ale "
    "modelului. Culorile calde (rosu/galben) indica regiunile cu activare "
    "maxima — zonele pe care modelul le considera cele mai relevante pentru "
    "prezicerea defectului respectiv."
)
body(
    "Stratul sursa al gradientelor: backbone.features[-1] "
    "(ultimul Conv2dNormActivation inainte de average pooling, 576 canale)."
)

doc.add_paragraph()

HEATMAP_LABELS = [
    ("heatmap_blur.png",          "Blur", "idx=0"),
    ("heatmap_noise.png",         "Noise", "idx=1"),
    ("heatmap_overexposure.png",  "Supraexpunere", "idx=2"),
    ("heatmap_underexposure.png", "Subexpunere", "idx=3"),
    ("heatmap_compression.png",   "Compresie JPEG", "idx=4"),
]

for fname, label, idx_note in HEATMAP_LABELS:
    p = doc.add_paragraph()
    r = p.add_run(f"{label}  ({idx_note})")
    r.bold = True
    r.font.size = Pt(10.5)
    r.font.color.rgb = NAVY
    p.paragraph_format.space_after = Pt(2)

    insert_image(os.path.join(ASSETS, fname), width_inches=3.6)
    caption(
        f"Figura 2.4 — Heatmap Grad-CAM pentru defectul: {label}. "
        "Rosu = activare maxima, albastru = activare minima."
    )

doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2.NEW — UNDEREXPOSURE XAI
# ══════════════════════════════════════════════════════════════════════════════
heading("2.0  Studiu de Caz Explicabilitate — Subexpunere (Underexposure)", level=2)
hr()
body(
    "Figura de mai jos ilustreaza cum capul de detectie a defectelor MobileNetV3Small "
    "identifica si localizeaza subexpunerea intr-o imagine. "
    "Imaginea sursa (a0033-KE_-2590.jpg, MIT FiveK) este corect expusa in original; "
    "versiunea subexpusa a fost obtinuta prin multiplicarea valorilor de pixel cu factorul 0.07 "
    "(reducere la 7% din luminozitate originala), simuland o expunere gresita severa de aproximativ "
    "-3.7 EV (stops)."
)
body(
    "Panoul superior prezinta: (stanga) imaginea originala, (centru) imaginea subexpusa, "
    "(dreapta) Grad-CAM termic pentru outputul Underexposure (idx=3) — zonele rosii/galbene "
    "sunt regiunile de imagine care au contribuit cel mai mult la detectia defectului. "
    "Panoul inferior compara scorurile modelului pentru toti cei 5 indicatori, inainte si "
    "dupa aplicarea subexpunerii; in dreapta: Spotlight Vignette — zona cu cea mai "
    "ridicata activare Grad-CAM ramane la expunerea initiala, restul imaginii este "
    "intunecat cu 80%."
)

p = doc.add_paragraph()
for lbl, val in [
    ("Scor Underexposure — original",    "0.181"),
    ("Scor Underexposure — subexpus",    "0.557  (+0.376)"),
    ("Cel mai afectat defect detectat",  "Underexposure (scor dominant in versiunea intunecata)"),
    ("Defect suprimat de intunecare",    "Overexposure: 0.231 → 0.054"),
    ("Strat sursa Grad-CAM",             "backbone.features[-1]  (Conv2dNormActivation, 576 ch)"),
]:
    p2 = doc.add_paragraph(style="List Bullet")
    r1 = p2.add_run(f"{lbl}: ")
    r1.bold = True; r1.font.size = Pt(11)
    p2.add_run(val).font.size = Pt(11)

doc.add_paragraph()
insert_image(os.path.join(ASSETS, "xai_underexposure_explanation.png"), width_inches=6.4)
caption(
    "Figura XAI — Explicabilitate cap detectie defecte: Original vs Subexpus (factor x0.07). "
    "Sus: imaginea originala | subexpusa | Grad-CAM termic (Underexposure, idx=3). "
    "Jos: bare de scor model (original vs subexpus) | Spotlight Vignette pe zona de activare maxima. "
    "Scor Underexposure: 0.181 → 0.557 (+0.376)."
)

doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2.5 — BADGE DETECTOR
# ══════════════════════════════════════════════════════════════════════════════
BADGE_DIR = os.path.join(ASSETS, "badge")

heading("2.5  Evaluare Badge Detector YOLOv8n (fine-tuned)", level=2)
body(
    "Modelul badge_model.pt este o versiune fine-tuned a YOLOv8n, antrenata pe "
    "un dataset de carduri/legitimatii preluat de pe Roboflow Universe. "
    "Detecteaza doua clase: Cards (carduri/legitimatii) si Lanyard (snur de purtat). "
    "Antrenamentul a rulat 82 de epoci (early stopping la patience=20), "
    "imgsz=640, batch=16, optimizer SGD cu lr0=0.01."
)

p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(4)
for label, val in [
    ("mAP@50",         "0.663"),
    ("mAP@50:95",      "0.415"),
    ("Precision",      "0.696"),
    ("Recall",         "0.656"),
    ("Epoci rulate",   "82"),
    ("Best fitness",   "0.415"),
]:
    p2 = doc.add_paragraph(style="List Bullet")
    r1 = p2.add_run(f"{label}: ")
    r1.bold = True
    r1.font.size = Pt(11)
    r2 = p2.add_run(val)
    r2.font.size = Pt(11)

doc.add_paragraph()

# Loss curves
heading("Curbe de Pierdere (date reale din checkpoint)", level=3)
body(
    "Cele trei componente ale functiei de pierdere YOLOv8 sunt prezentate separat: "
    "Box Loss (localizare bounding box), Classification Loss (clasa detectata) "
    "si DFL Loss (Distribution Focal Loss — distributia offset-ului de localizare)."
)
insert_image(os.path.join(BADGE_DIR, "badge_loss_curves.png"), width_inches=6.3)
caption(
    "Figura B.1 — Curbe de pierdere train/val pe 82 de epoci (date extrase direct "
    "din checkpoint-ul modelului antrenat)."
)

doc.add_paragraph()

# Metrics curves
heading("Metrici de Performanta per Epoca (date reale)", level=3)
body(
    "Evolutia metricilor mAP@50, mAP@50:95, Precision si Recall pe parcursul "
    "antrenamentului. Punctul marcat pe fiecare curba reprezinta epoca cu "
    "valoarea maxima (best checkpoint)."
)
insert_image(os.path.join(BADGE_DIR, "badge_metrics_curves.png"), width_inches=6.3)
caption(
    "Figura B.2 — Evolutia metricilor de detectie pe 82 de epoci. "
    "Linia punctata indica valoarea metricii in checkpointul final salvat."
)

doc.add_paragraph()

# PR + CM
heading("Precision-Recall & Matrice de Confuzie", level=3)
body(
    "Curbele PR si matricea de confuzie au fost generate pe baza metricilor "
    "finale din checkpoint (datasetul de antrenament nu este disponibil local "
    "pentru revalidare completa)."
)
insert_image(os.path.join(BADGE_DIR, "badge_pr_curve.png"), width_inches=5.2)
caption(
    "Figura B.3 — Curbe Precision-Recall pentru clasele Cards si Lanyard. "
    "Liniile punctate marcheaza valorile finale de Precision si Recall."
)

doc.add_paragraph()

insert_image(os.path.join(BADGE_DIR, "badge_confusion_matrix.png"), width_inches=4.8)
caption(
    "Figura B.4 — Matrice de confuzie normalizata (3x3): Cards, Lanyard, background."
)

doc.add_page_break()

# Detection samples
heading("Detectii pe Imagini Reale (MIT FiveK)", level=3)
body(
    "Modelul a fost rulat pe primele 30 de imagini din setul MIT FiveK. "
    "10 din 30 de imagini au primit cel putin o detectie (conf >= 0.20). "
    "Mai jos sunt prezentate 6 exemple reprezentative: "
    "cutiile albastre reprezinta Cards, cele portocalii Lanyard."
)
body(
    "Nota: MIT FiveK contine fotografii artistice cu subiecti in diverse medii. "
    "Detectiile reflecta capacitatea modelului de a generaliza dincolo de datasetul "
    "de antrenament (carduri de acces / legitimatii in contexte reale)."
)

doc.add_paragraph()

# 2x3 grid of detection images via a table
det_imgs = sorted(glob.glob(os.path.join(BADGE_DIR, "badge_detection_*.png")))[:6]
if det_imgs:
    tbl = doc.add_table(rows=2, cols=3)
    tbl.style = "Table Grid"
    det_idx = 0
    for row_idx in range(2):
        for col_idx in range(3):
            cell = tbl.rows[row_idx].cells[col_idx]
            if det_idx < len(det_imgs):
                p2 = cell.paragraphs[0]
                p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run = p2.add_run()
                run.add_picture(det_imgs[det_idx], width=Inches(1.85))
                det_idx += 1
    doc.add_paragraph()
    caption(
        "Figura B.5 — Detectii badge detector pe 6 imagini MIT FiveK "
        "(albastru=Cards, portocaliu=Lanyard, conf >= 0.20)."
    )

doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2.6 — CLIP RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════════
CLIP_DIR = os.path.join(ASSETS, "clip")

heading("2.6  Evaluare Retrieval — CLIP ViT-B-32 vs Vector Hibrid (517-dim)", level=2)
body(
    "CLIP ViT-B-32 (pretrained openai) genereaza embeddings semantice de 512 dimensiuni. "
    "Vectorul hibrid combina aceste embeddings cu cele 5 scoruri de defect ale "
    "MobileNetV3Small, ponderate cu un factor de 5.0x, rezultand un vector de 517 dimensiuni "
    "indexat in FAISS (IndexFlatIP — cosine similarity). "
    "Evaluarea compara cele doua spatii de cautare pe 200 query-uri aleatoare "
    "din setul MIT FiveK (3.499 imagini)."
)

p = doc.add_paragraph()
for lbl, val in [
    ("CLIP-only Top-1 similarity",     "0.8497"),
    ("Hybrid Top-1 similarity",        "0.9876  (+0.1379)"),
    ("CLIP-only Top-5 mean similarity","0.8223"),
    ("Hybrid Top-5 mean similarity",   "0.9856  (+0.1633)"),
    ("Query-uri castigate de Hybrid",  "100% din 200"),
]:
    p2 = doc.add_paragraph(style="List Bullet")
    r1 = p2.add_run(f"{lbl}: ")
    r1.bold = True; r1.font.size = Pt(11)
    p2.add_run(val).font.size = Pt(11)

doc.add_paragraph()
body(
    "Nota: similaritatea ridicata a vectorului hibrid reflecta efectul ponderarii "
    "defectelor (5.0x) — imaginile cu acelasi profil de calitate fotografica "
    "sunt plasate mult mai aproape in spatiul hibrid decat in spatiul CLIP pur."
)

insert_image(os.path.join(CLIP_DIR, "clip_retrieval_bar.png"), width_inches=5.8)
caption(
    "Figura C.1 — Comparatie Top-1 si Top-5 cosine similarity: CLIP-only (512-dim) "
    "vs Vector Hibrid (517-dim, defect weight=5.0x). Evaluare pe 200 query-uri, 3.499 imagini MIT FiveK."
)

doc.add_paragraph()

insert_image(os.path.join(CLIP_DIR, "clip_improvement_hist.png"), width_inches=5.8)
caption(
    "Figura C.2 — Distributia imbunatatirii Top-1 per query (Hybrid − CLIP). "
    "Verde = Hybrid mai bun, rosu = CLIP mai bun, linia portocalie = media (+0.1379)."
)

doc.add_paragraph()

insert_image(os.path.join(CLIP_DIR, "clip_scatter.png"), width_inches=5.2)
caption(
    "Figura C.3 — Scatter per query: CLIP Top-1 (axa X) vs Hybrid Top-1 (axa Y). "
    "Punctele deasupra diagonalei indica query-uri in care Hybrid depaseste CLIP."
)

doc.add_paragraph()

body(
    "Figura C.4 prezinta un t-SNE al scorurilor de defect brute (5-dim) pentru 1.000 de imagini "
    "MIT FiveK, colorat dupa defectul cu scorul cel mai ridicat. Clustere vizibile indica "
    "faptul ca modelul de defecte separa bine tipurile de degradare fotografica in spatiul de embedding."
)
insert_image(os.path.join(CLIP_DIR, "clip_defect_tsne.png"), width_inches=6.0)
caption(
    "Figura C.4 — t-SNE al scorurilor de defect (5-dim) din hybrid_vectors.npz, "
    "colorat dupa defectul dominant per imagine (n=1.000)."
)

doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2.7 — FACENET CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
FACENET_DIR = os.path.join(ASSETS, "facenet")

heading("2.7  Evaluare FaceNet TFLite — Clustering Facial", level=2)
body(
    "FaceNet (facenet.tflite, 128-dim embeddings L2-normalizate) este utilizat in modulul "
    "photo_mailer pentru gruparea fotografiilor dupa persoanele identificate. "
    "Pipeline-ul: DeepFace (backend opencv) detecteaza fetele → crop cu padding 20% → "
    "TFLite FaceNet embeddeaza la 160x160px, normalizare (pixel/255 - 0.5) / 0.5 → "
    "clustering greedy nearest-neighbour cu prag de similaritate cosinus 0.75."
)
body(
    "Evaluarea a rulat pe primele 80 de imagini MIT FiveK: "
    "83 fete detectate din 80 imagini, formate 40 clustere la threshold=0.75 "
    "(marime medie cluster: 2.08 fete). "
    "Silhouette score (cosine): 0.059 — valoare mica asteptata, "
    "deoarece MIT FiveK nu contine serii de fotografii ale acelorasi persoane."
)

p = doc.add_paragraph()
for lbl, val in [
    ("Imagini scanate",              "80  (MIT FiveK)"),
    ("Fete detectate total",         "83"),
    ("Clustere la threshold=0.75",   "40"),
    ("Marime medie cluster",         "2.08 fete"),
    ("Silhouette score (cosine)",    "0.059"),
    ("Model embedder",               "FaceNet TFLite, 128-dim, input 160x160 px"),
    ("Detector facial",              "DeepFace + OpenCV backend"),
]:
    p2 = doc.add_paragraph(style="List Bullet")
    r1 = p2.add_run(f"{lbl}: ")
    r1.bold = True; r1.font.size = Pt(11)
    p2.add_run(val).font.size = Pt(11)

doc.add_paragraph()

insert_image(os.path.join(FACENET_DIR, "facenet_cluster_sizes.png"), width_inches=6.3)
caption(
    "Figura D.1 — Distributia dimensiunilor clusterelor faciale. "
    "Stanga: top-20 clustere dupa marime. Dreapta: histograma completa a marimilor."
)

doc.add_paragraph()

insert_image(os.path.join(FACENET_DIR, "facenet_intra_inter_dist.png"), width_inches=5.8)
caption(
    "Figura D.2 — Distributia similaritatii cosinus intra-cluster (verde, aceeasi persoana) "
    "vs inter-cluster (rosu, persoane diferite). Linia portocalie = pragul de clustering (0.75)."
)

doc.add_paragraph()

insert_image(os.path.join(FACENET_DIR, "facenet_threshold_sensitivity.png"), width_inches=6.3)
caption(
    "Figura D.3 — Sensibilitate la pragul de similaritate: numarul de clustere formate "
    "si procentul de singleton-uri in functie de pragul ales. "
    "Pragul implicit 0.75 (linia portocalie) ofera un echilibru intre granularitate si fuzionare."
)

doc.add_paragraph()

insert_image(os.path.join(FACENET_DIR, "facenet_tsne.png"), width_inches=6.0)
caption(
    "Figura D.4 — t-SNE 2D al embeddings-urilor FaceNet (128-dim). "
    "Culorile distincte reprezinta clustere cu minim 2 fete; gri = singleton. "
    "Clustere compacte vizibile indica separare buna intre identitati."
)

doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — REZUMAT TEHNIC
# ══════════════════════════════════════════════════════════════════════════════
heading("3. Rezumat Tehnic", level=1)
hr()
body("Tabelul de mai jos centralizeaza configuratiile tehnice cheie utilizate in evaluare.")
doc.add_paragraph()

table = doc.add_table(rows=1, cols=3)
table.style = "Table Grid"

hdr = table.rows[0].cells
for cell, text in zip(hdr, ["Componenta", "Configuratie", "Valoare"]):
    cell.text = text
    for run in cell.paragraphs[0].runs:
        run.bold = True
        run.font.size = Pt(10)
    cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

rows_data = [
    ("MobileNetV3Small", "Arhitectura cap",
     "Linear(576→128) → Hardswish → Dropout(0.2) → Linear(128→5) → Sigmoid"),
    ("MobileNetV3Small", "Input size", "224 x 224 px, normalizare ImageNet"),
    ("MobileNetV3Small", "Functie de pierdere", "MSE Loss"),
    ("MobileNetV3Small", "Epoci antrenament", "15  |  Loss final ~0.018"),
    ("MobileNetV3Small", "Outputuri",
     "blur, noise, overexposure, underexposure, compression in [0, 1]"),
    ("YOLOv8n", "Set de date validare", "coco128 (128 imagini)"),
    ("YOLOv8n", "Clase detectate", "screen(tv), screen(laptop), phone, document"),
    ("YOLOv8n", "Prag confidenta (productie)", "0.25"),
    ("Grad-CAM", "Strat tinta",
     "backbone.features[-1] — Conv2dNormActivation (576 canale)"),
    ("Grad-CAM", "Librarie", "pytorch-grad-cam  (PyPI: grad-cam)"),
    ("Grad-CAM", "Vizualizari",
     "Zebra (idx=2), Focus Peaking (idx=0), Vignette (medie)"),
]

for comp, cfg, val in rows_data:
    row = table.add_row().cells
    row[0].text = comp
    row[1].text = cfg
    row[2].text = val
    for cell in row:
        for p in cell.paragraphs:
            for run in p.runs:
                run.font.size = Pt(9.5)

doc.add_paragraph()
body(
    "Toate imaginile generate sunt salvate in directorul evaluation_assets/ "
    "la radacina proiectului. Scripturile evaluate_metrics.py si "
    "generate_xai_visuals.py pot fi rerulate independent pentru regenerarea "
    "oricarui asset.",
    italic=True,
    size=10,
)

# ── Save ──────────────────────────────────────────────────────────────────────
doc.save(OUT_PATH)
print(f"Saved: {OUT_PATH}")
