import streamlit as st
import pdfplumber
import fitz  # pymupdf
import json
import io
import os
import base64
from openai import OpenAI
from dotenv import load_dotenv
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable, KeepTogether,
)
from reportlab.lib import colors

# ── Env ───────────────────────────────────────────────────────────────────────
load_dotenv(".env.local")
_api_key = os.getenv("OPENAI_API_KEY", "")
client   = OpenAI(api_key=_api_key) if _api_key else None

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF Structurer",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { max-width: 780px; padding: 2rem 1.5rem 4rem; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* upload zone */
[data-testid="stFileUploader"] {
    border: 2px dashed #cbd5e1; border-radius: 14px;
    background: #f8fafc; padding: 1rem; transition: border-color .2s;
}
[data-testid="stFileUploader"]:hover { border-color: #6366f1; }
[data-testid="stFileDropzoneInstructions"] { color: #64748b; font-size: 0.88rem; }

/* download button */
div[data-testid="stDownloadButton"] button {
    background: linear-gradient(135deg,#4f46e5,#7c3aed) !important;
    color: #fff !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
    font-size: 0.95rem !important; padding: 0.6rem 1.2rem !important;
    transition: opacity .2s !important;
    box-shadow: 0 4px 14px rgba(79,70,229,.35) !important;
}
div[data-testid="stDownloadButton"] button:hover { opacity: .88 !important; }

/* stat pills */
.stat-row { display:flex; gap:8px; margin-top:8px; flex-wrap:wrap; }
.pill {
    display:inline-flex; align-items:center; gap:4px;
    padding:3px 10px; border-radius:20px;
    font-size:0.73rem; font-weight:600;
}
.pill-ok   { background:#dcfce7; color:#166534; }
.pill-warn { background:#fef9c3; color:#854d0e; }
.pill-miss { background:#fee2e2; color:#991b1b; }

/* expander */
div[data-testid="stExpander"] {
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 12px !important; overflow: hidden;
}
div[data-testid="stExpander"] summary {
    font-weight: 700 !important; font-size: 0.95rem !important;
    color: #1e293b !important; letter-spacing: .3px;
    padding: 0.8rem 1rem !important; background: #f8fafc;
}
div[data-testid="stExpander"] summary:hover { background: #f1f5f9; }

/* section label */
.sec-label {
    font-size:0.68rem; font-weight:700; letter-spacing:1.1px;
    text-transform:uppercase; color:#94a3b8; margin:18px 0 5px;
}
.sec-divider { height:1px; background:#f1f5f9; margin-bottom:10px; }

/* field inputs */
.stTextArea label { font-size:0.8rem !important; font-weight:500 !important; color:#374151 !important; }
.stTextArea textarea {
    font-size:0.875rem !important; border-radius:8px !important;
    border-color:#e2e8f0 !important; resize:vertical;
}
.stTextArea textarea:focus {
    border-color:#6366f1 !important;
    box-shadow:0 0 0 3px rgba(99,102,241,.12) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in {
    "structured": None,
    "edited_fields": {},
    "file_name": "",
    "page_count": 0,
    "img_count": 0,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
MAX_IMG_PAGES = 20
IMG_DPI       = 144


def extract_text(pdf_bytes: bytes) -> str:
    parts = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                parts.append(t)
    return "\n\n".join(parts).strip()


def pdf_to_images_b64(pdf_bytes: bytes):
    """Returns (list_of_b64_jpegs, total_pages, pages_sent)."""
    doc   = fitz.open(stream=pdf_bytes, filetype="pdf")
    total = len(doc)
    mat   = fitz.Matrix(IMG_DPI / 72, IMG_DPI / 72)
    imgs  = []
    for i, page in enumerate(doc):
        if i >= MAX_IMG_PAGES:
            break
        pix = page.get_pixmap(matrix=mat, alpha=False)
        imgs.append(base64.b64encode(pix.tobytes("jpeg", jpg_quality=88)).decode())
    doc.close()
    return imgs, total, len(imgs)


SYSTEM_PROMPT = """You are a senior document analyst with access to BOTH the raw OCR text AND
high-resolution page images of the same PDF. Use both sources together for maximum accuracy —
images reveal tables, stamps, handwriting, logos, form labels, and fine print that OCR misses.

Your tasks:
1. Identify the document type precisely.
2. Extract EVERY piece of information and organise it into logical sections.
3. Flag fields that are: absent, empty, placeholder-like ("[INSERT]", "N/A", "___"),
   ambiguous, inconsistent between text and image, or illegible.
4. Return ONLY valid JSON — no markdown fences, no extra prose, no extra keys.

Strict JSON schema:
{
  "document_type": "<concise type e.g. 'Commercial Invoice', 'Employment Contract'>",
  "sections": [
    {
      "section_name": "<logical group name>",
      "fields": [
        {
          "key":     "<human-readable field name>",
          "value":   "<extracted value as string, or empty string if absent>",
          "missing": <true if absent/placeholder/illegible, else false>,
          "note":    "<short quality note if flagged, else empty string>"
        }
      ]
    }
  ],
  "overall_notes": "<1-2 sentence quality summary, or empty string>"
}"""


def structure_with_ai(raw_text: str, page_images: list) -> dict:
    if not client:
        st.error("OPENAI_API_KEY not found in .env.local")
        st.stop()

    content = [
        {"type": "text",
         "text": f"=== OCR TEXT ===\n\n{raw_text or '[No selectable text — rely on images]'}"}
    ]
    for idx, b64 in enumerate(page_images):
        content.append({"type": "text", "text": f"=== PAGE {idx + 1} IMAGE ==="})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"},
        })

    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": content},
        ],
        max_tokens=4096,
    )
    return json.loads(response.choices[0].message.content)


def generate_pdf(structured: dict, edited: dict) -> io.BytesIO:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
        leftMargin=22*mm, rightMargin=22*mm, topMargin=24*mm, bottomMargin=22*mm)

    BASE   = getSampleStyleSheet()
    INDIGO = colors.HexColor("#4f46e5")
    SLATE  = colors.HexColor("#475569")
    RED    = colors.HexColor("#dc2626")
    AMBER  = colors.HexColor("#d97706")
    LGRAY  = colors.HexColor("#e2e8f0")

    def S(name, **kw):
        return ParagraphStyle(name, parent=BASE["Normal"], **kw)

    ST = {
        "title":   S("T",  fontSize=19, textColor=INDIGO, spaceAfter=4, fontName="Helvetica-Bold"),
        "section": S("SE", fontSize=8,  textColor=SLATE,  spaceBefore=14, spaceAfter=4,
                     fontName="Helvetica-Bold"),
        "label":   S("LA", fontSize=8,  textColor=SLATE,  spaceBefore=6, spaceAfter=1),
        "value":   S("VA", fontSize=10, textColor=colors.HexColor("#1e293b"), spaceAfter=3, leading=14),
        "missing": S("MI", fontSize=10, textColor=RED,    spaceAfter=3, fontName="Helvetica-Oblique"),
        "note":    S("NO", fontSize=8,  textColor=AMBER,  spaceAfter=2),
        "overall": S("OV", fontSize=9,  textColor=SLATE,  spaceBefore=4),
    }

    story = []
    story.append(Paragraph(structured.get("document_type", "Document"), ST["title"]))
    story.append(HRFlowable(width="100%", thickness=2, color=INDIGO, spaceAfter=8))

    for section in structured.get("sections", []):
        sec_name = section.get("section_name", "")
        block    = [
            Paragraph(sec_name.upper(), ST["section"]),
            HRFlowable(width="100%", thickness=0.5, color=LGRAY, spaceAfter=4),
        ]
        for field in section.get("fields", []):
            fkey  = f"{sec_name}::{field['key']}"
            value = edited.get(fkey, field.get("value", "")).strip()
            note  = (field.get("note") or "").strip()
            block.append(Paragraph(field["key"], ST["label"]))
            if not value:
                block.append(Paragraph(f"[MISSING — {note or 'not found in document'}]", ST["missing"]))
            else:
                block.append(Paragraph(value.replace("\n", "<br/>"), ST["value"]))
                if note:
                    block.append(Paragraph(f"⚠ {note}", ST["note"]))
        story.append(KeepTogether(block))
        story.append(Spacer(1, 2*mm))

    overall = (structured.get("overall_notes") or "").strip()
    if overall:
        story.append(HRFlowable(width="100%", thickness=0.5, color=LGRAY))
        story.append(Spacer(1, 2*mm))
        story.append(Paragraph("ANALYST NOTES", ST["section"]))
        story.append(Paragraph(overall, ST["overall"]))

    doc.build(story)
    buf.seek(0)
    return buf


# ═══════════════════════════════════════════════════════════════════════════════
#  UI
# ═══════════════════════════════════════════════════════════════════════════════
if not _api_key:
    st.error("⚠️  `OPENAI_API_KEY` not found in `.env.local` — add it and restart.", icon="🔑")
    st.stop()

# Header
st.markdown("""
<div style="margin-bottom:1.4rem;">
  <div style="font-size:1.55rem;font-weight:800;color:#1e293b;letter-spacing:-.5px;">
    📄 PDF Structurer
  </div>
  <div style="color:#94a3b8;font-size:0.83rem;margin-top:3px;">
    GPT-4o reads both text <em>and</em> every page image for maximum extraction precision
  </div>
</div>
""", unsafe_allow_html=True)

# Uploader
uploaded = st.file_uploader(
    "Upload PDF", type="pdf", label_visibility="collapsed",
    help="Invoices, contracts, forms, reports — any PDF",
)

if not uploaded:
    st.markdown("""
    <div style="text-align:center;padding:3.5rem 1rem;color:#94a3b8;">
      <div style="font-size:3rem;margin-bottom:.6rem;">📂</div>
      <div style="font-size:.92rem;font-weight:500;">Drop or browse a PDF above to start</div>
      <div style="font-size:.78rem;margin-top:.4rem;">Text + page images are sent to GPT-4o automatically</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Auto-process on new upload ────────────────────────────────────────────────
if uploaded.name != st.session_state.file_name:
    st.session_state.structured    = None
    st.session_state.edited_fields = {}
    st.session_state.file_name     = uploaded.name

    pdf_bytes = uploaded.read()

    prog = st.progress(0, text="📖  Reading PDF text…")
    raw_text = extract_text(pdf_bytes)

    prog.progress(30, text="🖼️  Rendering page images…")
    imgs, total_pages, img_pages = pdf_to_images_b64(pdf_bytes)
    st.session_state.page_count = total_pages
    st.session_state.img_count  = img_pages

    prog.progress(55, text=f"🤖  Analysing {img_pages} page image(s) with GPT-4o…")
    try:
        result = structure_with_ai(raw_text, imgs)
    except Exception as exc:
        st.error(f"OpenAI error: {exc}")
        st.stop()

    st.session_state.structured    = result
    st.session_state.edited_fields = {
        f"{sec['section_name']}::{f['key']}": f.get("value", "")
        for sec in result.get("sections", [])
        for f in sec.get("fields", [])
    }
    prog.progress(100, text="✓  Done")
    prog.empty()
    st.rerun()

structured = st.session_state.structured
if not structured:
    st.stop()

# ── Stats ─────────────────────────────────────────────────────────────────────
all_fields  = [f for s in structured.get("sections", []) for f in s.get("fields", [])]
total       = len(all_fields)
missing_cnt = sum(1 for f in all_fields if f.get("missing"))
warned_cnt  = sum(1 for f in all_fields if not f.get("missing") and f.get("note"))
ok_cnt      = total - missing_cnt - warned_cnt
doc_type    = structured.get("document_type", "Document")

pg_label = (
    f"{st.session_state.img_count}/{st.session_state.page_count} pages scanned"
    if st.session_state.page_count > st.session_state.img_count
    else f"{st.session_state.page_count} page{'s' if st.session_state.page_count != 1 else ''} scanned"
)

# ── PDF buffer (reflect latest edits every run) ───────────────────────────────
pdf_buf = generate_pdf(structured, st.session_state.edited_fields)

# ── Top area: download left, doc info right ───────────────────────────────────
dl_col, info_col = st.columns([1, 1.7], gap="large")

with dl_col:
    st.download_button(
        label="⬇️  Download PDF",
        data=pdf_buf,
        file_name=f"structured_{uploaded.name}",
        mime="application/pdf",
        type="primary",
        use_container_width=True,
    )
    warn_pill = (
        f'<span class="pill pill-warn">⚠ {warned_cnt} flagged</span>'
        if warned_cnt else ""
    )
    miss_pill = (
        f'<span class="pill pill-miss">✕ {missing_cnt} missing</span>'
        if missing_cnt else ""
    )
    st.markdown(f"""
    <div class="stat-row">
      <span class="pill pill-ok">✓ {ok_cnt}</span>
      {warn_pill}
      {miss_pill}
    </div>
    <div style="font-size:0.7rem;color:#94a3b8;margin-top:5px;">{pg_label}</div>
    """, unsafe_allow_html=True)

with info_col:
    st.markdown(f"""
    <div style="padding:.4rem 0;">
      <div style="font-size:1.05rem;font-weight:700;color:#1e293b;">{doc_type}</div>
      <div style="font-size:.78rem;color:#94a3b8;margin-top:3px;">{uploaded.name}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-top:1.2rem'></div>", unsafe_allow_html=True)

# ── EDIT expander ─────────────────────────────────────────────────────────────
with st.expander("EDIT", expanded=False):
    overall = (structured.get("overall_notes") or "").strip()
    if overall:
        st.info(overall, icon="📝")

    for section in structured.get("sections", []):
        sec_name = section.get("section_name", "")
        fields   = section.get("fields", [])
        sec_miss = sum(1 for f in fields if f.get("missing"))

        miss_badge = (
            f' &nbsp;<span class="pill pill-miss" style="font-size:.62rem;vertical-align:middle;">'
            f'✕ {sec_miss}</span>'
        ) if sec_miss else ""

        st.markdown(
            f'<div class="sec-label">{sec_name}{miss_badge}</div>'
            f'<div class="sec-divider"></div>',
            unsafe_allow_html=True,
        )

        # 2-column grid
        pairs = [fields[i : i + 2] for i in range(0, len(fields), 2)]
        for pair_idx, pair in enumerate(pairs):
            cols = st.columns(len(pair), gap="medium")
            for col_idx, (col, field) in enumerate(zip(cols, pair)):
                fkey      = f"{sec_name}::{field['key']}"
                # globally unique widget key — section + pair + column position
                widget_key = f"ta__{sec_name}__{pair_idx}__{col_idx}"
                note   = (field.get("note") or "").strip()
                is_mis = field.get("missing", False)
                has_nt = bool(note) and not is_mis

                label = field["key"]
                if is_mis:   label += "  🔴"
                elif has_nt: label += "  ⚠️"

                cur = st.session_state.edited_fields.get(fkey, "")
                with col:
                    new = st.text_area(
                        label       = label,
                        value       = cur,
                        height      = 80,
                        key         = widget_key,
                        help        = note or None,
                        placeholder = "— missing —" if is_mis else "",
                    )
                    st.session_state.edited_fields[fkey] = new
