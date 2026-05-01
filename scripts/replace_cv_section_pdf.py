import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pdfplumber
from pypdf import PdfReader, PdfWriter
from reportlab.lib.colors import white
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


@dataclass(frozen=True)
class ReplaceSpec:
    start_anchor: str
    end_anchor: str
    replacement_lines: Tuple[str, ...]


def _normalize(s: str) -> str:
    return " ".join((s or "").split())


def _try_register_arial() -> None:
    """
    Register macOS Arial TTFs if available so the overlay matches the CV typography.
    Safe to call multiple times.
    """
    candidates = {
        "Arial": "/System/Library/Fonts/Supplemental/Arial.ttf",
        "Arial-Bold": "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "Arial-Italic": "/System/Library/Fonts/Supplemental/Arial Italic.ttf",
        "Arial-BoldItalic": "/System/Library/Fonts/Supplemental/Arial Bold Italic.ttf",
    }

    for name, path in candidates.items():
        try:
            pdfmetrics.getFont(name)
            continue
        except KeyError:
            pass
        p = Path(path)
        if p.exists():
            pdfmetrics.registerFont(TTFont(name, str(p)))

def _map_font_to_builtin(fontname: str, *, bold: bool = False, italic: bool = False) -> str:
    """
    The CV PDF uses embedded subset fonts (e.g. 'AAAAAA+ArialMT').
    We register system Arial TTFs when available; otherwise we approximate with core 14 fonts.
    """
    fn = (fontname or "").lower()
    family = "Helvetica"
    if "arial" in fn or "helvetica" in fn:
        # Prefer real Arial if registered.
        try:
            pdfmetrics.getFont("Arial")
            family = "Arial"
        except KeyError:
            family = "Helvetica"
    elif "times" in fn:
        family = "Times"
    elif "courier" in fn:
        family = "Courier"

    if family == "Arial":
        if bold and italic:
            try:
                pdfmetrics.getFont("Arial-BoldItalic")
                return "Arial-BoldItalic"
            except KeyError:
                pass
        if bold:
            try:
                pdfmetrics.getFont("Arial-Bold")
                return "Arial-Bold"
            except KeyError:
                pass
        if italic:
            try:
                pdfmetrics.getFont("Arial-Italic")
                return "Arial-Italic"
            except KeyError:
                pass
        # Fallback to Arial regular (or Helvetica if not registered).
        try:
            pdfmetrics.getFont("Arial")
            return "Arial"
        except KeyError:
            return "Helvetica"

    if family == "Helvetica":
        if bold and italic:
            return "Helvetica-BoldOblique"
        if bold:
            return "Helvetica-Bold"
        if italic:
            return "Helvetica-Oblique"
        return "Helvetica"
    if family == "Times":
        if bold and italic:
            return "Times-BoldItalic"
        if bold:
            return "Times-Bold"
        if italic:
            return "Times-Italic"
        return "Times-Roman"
    if family == "Courier":
        if bold and italic:
            return "Courier-BoldOblique"
        if bold:
            return "Courier-Bold"
        if italic:
            return "Courier-Oblique"
        return "Courier"
    return "Helvetica"


def _find_bbox_between_anchors(
    page: "pdfplumber.page.Page",
    start_anchor: str,
    end_anchor: str,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Returns (x0, top, x1, bottom) covering text from start_anchor line
    down to just above end_anchor line.
    """
    lines = page.extract_text_lines() or []
    start = None
    end = None
    for i, ln in enumerate(lines):
        txt = _normalize(ln.get("text", ""))
        if start is None and start_anchor in txt:
            start = i
        if start is not None and end is None and end_anchor in txt:
            end = i
            break
    if start is None or end is None or end <= start:
        return None

    slice_lines = lines[start:end]
    x0 = min(ln["x0"] for ln in slice_lines)
    x1 = max(ln["x1"] for ln in slice_lines)
    top = min(ln["top"] for ln in slice_lines)
    bottom = max(ln["bottom"] for ln in slice_lines)
    return (x0, top, x1, bottom)

def _find_bbox_between_markers(
    page: "pdfplumber.page.Page",
    *,
    above_marker: str,
    below_marker: str,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Fallback when an overlay makes the original start anchor unextractable.
    Uses the bottom of `above_marker` line and the top of `below_marker` line.
    """
    lines = page.extract_text_lines() or []
    above = None
    below = None
    for ln in lines:
        txt = _normalize(ln.get("text", ""))
        if above_marker in txt:
            above = ln
        if below is None and below_marker in txt and above is not None:
            below = ln
            break
    if above is None or below is None:
        return None
    # Build bbox spanning the region between those lines.
    # Add a little padding so we don't force font downscaling.
    top = above["bottom"] + 1
    bottom = below["top"] - 1
    if bottom <= top:
        return None
    # Use page-wide left/right for consistent redaction.
    x0 = min(ln["x0"] for ln in lines)
    x1 = max(ln["x1"] for ln in lines)
    return (x0, top, x1, bottom)


def _find_first_line_containing(page: "pdfplumber.page.Page", marker: str) -> Optional[dict]:
    for ln in (page.extract_text_lines() or []):
        if marker in _normalize(ln.get("text", "")):
            return ln
    return None


def _infer_style_from_bbox(
    page: "pdfplumber.page.Page",
    bbox: Tuple[float, float, float, float],
) -> Tuple[str, str, str, float, float, float]:
    """
    Infer (regular_font, bold_font, italic_font, regular_size, bold_size, line_gap)
    used in this PDF region.
    """
    x0, top, x1, bottom = bbox
    chars = [
        c
        for c in (page.chars or [])
        if c["x0"] >= x0 - 1 and c["x1"] <= x1 + 1 and c["top"] >= top - 1 and c["bottom"] <= bottom + 1
    ]
    if not chars:
        return ("Helvetica", "Helvetica-Bold", "Helvetica-Oblique", 9.0, 10.6, 10.8)

    def is_bold_font(fn: str) -> bool:
        return "bold" in (fn or "").lower()

    regular = [c for c in chars if not is_bold_font(c.get("fontname", ""))]
    bold = [c for c in chars if is_bold_font(c.get("fontname", ""))]

    from collections import Counter

    def mode_font_and_size(items):
        ctr = Counter()
        for c in items:
            ctr[(c.get("fontname", ""), round(float(c.get("size", 0.0)), 1))] += 1
        return ctr.most_common(1)[0][0] if ctr else ("Helvetica", 9.0)

    reg_font_raw, reg_size = mode_font_and_size(regular or chars)
    bold_font_raw, bold_size_mode = mode_font_and_size(bold or chars)
    if bold:
        bold_sizes = sorted(float(c.get("size", 0.0)) for c in bold)
        bold_size = bold_sizes[int(0.9 * (len(bold_sizes) - 1))]
    else:
        bold_size = float(bold_size_mode)

    # Estimate line gap from text line tops inside bbox.
    lines = page.extract_text_lines() or []
    in_box = [ln for ln in lines if ln["top"] >= top - 0.5 and ln["bottom"] <= bottom + 0.5]
    tops = sorted(ln["top"] for ln in in_box)
    gaps = [tops[i + 1] - tops[i] for i in range(len(tops) - 1) if (tops[i + 1] - tops[i]) > 3]
    if gaps:
        gaps_sorted = sorted(gaps)
        line_gap = float(gaps_sorted[len(gaps_sorted) // 2])
    else:
        line_gap = 10.8

    reg_is_italic = "italic" in reg_font_raw.lower() or "oblique" in reg_font_raw.lower()
    reg_font = _map_font_to_builtin(reg_font_raw, bold=False, italic=False)
    bold_font = _map_font_to_builtin(bold_font_raw, bold=True, italic=("italic" in bold_font_raw.lower()))
    italic_font = _map_font_to_builtin(reg_font_raw, bold=False, italic=True) if not reg_is_italic else reg_font
    return (reg_font, bold_font, italic_font, float(reg_size), float(bold_size), float(line_gap))


def _build_overlay_pdf(
    out_path: Path,
    page_width: float,
    page_height: float,
    bbox: Tuple[float, float, float, float],
    replacement_lines: Iterable[str],
    *,
    regular_font: str,
    bold_font: str,
    italic_font: str,
    regular_size: float,
    bold_size: float,
    line_gap: float,
) -> None:
    x0, top, x1, bottom = bbox
    c = canvas.Canvas(str(out_path), pagesize=(page_width, page_height))

    # Redact the target region.
    rect_height = bottom - top
    c.setFillColor(white)
    c.setStrokeColor(white)
    c.rect(x0 - 2, page_height - bottom - 2, (x1 - x0) + 4, rect_height + 4, fill=1, stroke=0)

    max_width = (x1 - x0) - 2

    def wrap_line(text: str, font: str, size: float, width: float) -> List[str]:
        words = text.split()
        if not words:
            return [""]
        out: List[str] = []
        cur = words[0]
        for w in words[1:]:
            trial = f"{cur} {w}"
            if pdfmetrics.stringWidth(trial, font, size) <= width:
                cur = trial
            else:
                out.append(cur)
                cur = w
        out.append(cur)
        return out

    def plan_layout(reg_sz: float, bold_sz: float, gap: float) -> Optional[List[Tuple[str, float, float, str]]]:
        # Returns (text, font_size, x_indent, face) per visual line; face in {"bold","regular","italic"}.
        visual: List[Tuple[str, float, float, str]] = []
        for i, raw in enumerate(replacement_lines):
            raw = raw.rstrip()
            if i == 0:
                for ln in wrap_line(raw, bold_font, bold_sz, max_width):
                    visual.append((ln, bold_sz, 0.0, "bold"))
                continue

            if raw.startswith("- "):
                bullet = "- "
                rest = raw[2:].lstrip()
                bullet_w = pdfmetrics.stringWidth(bullet, regular_font, reg_sz)
                indent = bullet_w + 4
                face = "italic" if rest.startswith("Technologies:") else "regular"
                font = italic_font if face == "italic" else regular_font
                wrapped = wrap_line(rest, font, reg_sz, max_width - indent)
                if wrapped:
                    visual.append((bullet + wrapped[0], reg_sz, 0.0, face))
                    for cont in wrapped[1:]:
                        visual.append((cont, reg_sz, indent, face))
                else:
                    visual.append((bullet, reg_sz, 0.0, "regular"))
                continue

            for ln in wrap_line(raw, regular_font, reg_sz, max_width):
                visual.append((ln, reg_sz, 0.0, "regular"))

        needed = (len(visual) * gap) + reg_sz
        available = rect_height
        if needed <= available:
            return visual
        return None

    # Try to keep original font size/leading; shrink only if it doesn't fit.
    layout = plan_layout(regular_size, bold_size, line_gap)
    chosen_font_size = regular_size
    chosen_line_gap = line_gap
    if layout is None:
        for scale in (0.95, 0.9, 0.85):
            layout = plan_layout(regular_size * scale, bold_size * scale, line_gap * scale)
            if layout is not None:
                chosen_font_size = regular_size * scale
                chosen_line_gap = line_gap * scale
                break
    if layout is None:
        # Last resort: force something to render.
        chosen_font_size = max(7.0, regular_size * 0.8)
        chosen_line_gap = max(8.5, line_gap * 0.8)
        layout = plan_layout(chosen_font_size, max(8.0, bold_size * 0.8), chosen_line_gap) or []

    c.setFillColorRGB(0, 0, 0)
    first_size = layout[0][1] if layout else chosen_font_size
    y = page_height - top - first_size
    for (txt, size, indent, face) in layout:
        if face == "bold":
            c.setFont(bold_font, size)
        elif face == "italic":
            c.setFont(italic_font, size)
        else:
            c.setFont(regular_font, size)
        c.drawString(x0 + indent, y, txt)
        y -= chosen_line_gap

    c.showPage()
    c.save()


def replace_section(
    input_pdf: Path,
    output_pdf: Path,
    spec: ReplaceSpec,
    page_index: int = 0,
) -> None:
    with pdfplumber.open(str(input_pdf)) as pdf:
        page = pdf.pages[page_index]
        bbox = _find_bbox_between_anchors(page, spec.start_anchor, spec.end_anchor)
        style_bbox = None
        if bbox is None:
            # If we previously overlaid the PDF, the start anchor may not be extractable anymore.
            bbox = _find_bbox_between_markers(
                page,
                above_marker="Runtime Security Enforcement",
                below_marker=spec.end_anchor,
            )
            below_line = _find_first_line_containing(page, spec.end_anchor)
            if below_line is not None:
                style_bbox = (
                    below_line["x0"],
                    max(0.0, below_line["top"] - 2),
                    below_line["x1"],
                    min(page.height, below_line["bottom"] + 55),
                )
        if bbox is None:
            raise RuntimeError("Could not locate target section area.")
        regular_font, bold_font, italic_font, regular_size, bold_size, line_gap = _infer_style_from_bbox(
            page, style_bbox or bbox
        )
        overlay_path = output_pdf.with_suffix(".overlay.pdf")
        _build_overlay_pdf(
            out_path=overlay_path,
            page_width=page.width,
            page_height=page.height,
            bbox=bbox,
            replacement_lines=spec.replacement_lines,
            regular_font=regular_font,
            bold_font=bold_font,
            italic_font=italic_font,
            regular_size=regular_size,
            bold_size=bold_size,
            line_gap=line_gap,
        )

    reader = PdfReader(str(input_pdf))
    overlay = PdfReader(str(overlay_path))
    writer = PdfWriter()
    for i, pg in enumerate(reader.pages):
        if i == page_index:
            pg.merge_page(overlay.pages[0])
        writer.add_page(pg)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with output_pdf.open("wb") as f:
        writer.write(f)
    overlay_path.unlink(missing_ok=True)


def main() -> None:
    _try_register_arial()
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="input_pdf", required=True)
    ap.add_argument("--out", dest="output_pdf", required=True)
    ap.add_argument("--page", dest="page_index", type=int, default=0)
    args = ap.parse_args()

    replacement_lines = (
        "Security and Compliance Copilot (NIST and CISA RAG Assistant) | Independent Project (Private) | 2026",
        "- Built a grounded RAG copilot over official NIST/CISA guidance with stable citations and fail-closed behavior when evidence was weak.",
        "- Implemented retrieval-first pipeline (query rewriting, hybrid retrieval, reranking, citation-aware context building) to improve evidence quality.",
        "- Added guardrails to refuse prompt injection/jailbreaks, secret extraction, and proprietary text requests; logged decisions for auditability.",
        "- Technologies: Python, FastAPI, ChromaDB, OpenAI API, JSONL, LLM Evaluation",
    )

    spec = ReplaceSpec(
        start_anchor="Security and Compliance Copilot",
        end_anchor="Secure AWS Foundation",
        replacement_lines=replacement_lines,
    )
    replace_section(Path(args.input_pdf), Path(args.output_pdf), spec, page_index=args.page_index)


if __name__ == "__main__":
    main()

