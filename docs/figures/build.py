"""Builds the report/README figures as self-contained HTML (inline SVG). Render with headless Chrome."""
import html
from pathlib import Path

OUT = Path(__file__).parent
PAPER, INK, MUTED, SOFT, RULE = "#ffffff", "#111111", "#333333", "#555555", "#bbbbbb"
FONTS = ("https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600"
         "&family=Geist+Mono:wght@400;500;600&display=swap")
KINDS = {  # fill, stroke, width, dash
    "focal": ("#e4e4e4", INK, 2.2, None), "svc": (PAPER, INK, 1.2, None), "store": ("#f0f0f0", INK, 1.2, None),
    "ext": (PAPER, SOFT, 1.2, "2,3"), "planned": (PAPER, SOFT, 1.2, "5,4"),
}
E = html.escape


def rpath(pts, r=8, dash=None, end=True):
    d = f"M {pts[0][0]} {pts[0][1]}"
    for i in range(1, len(pts) - 1):
        (x0, y0), (x1, y1), (x2, y2) = pts[i - 1], pts[i], pts[i + 1]
        l1, l2 = abs(x1 - x0) + abs(y1 - y0), abs(x2 - x1) + abs(y2 - y1)
        rr = min(r, l1 / 2, l2 / 2)
        ux, uy = (x1 - x0) / l1, (y1 - y0) / l1
        vx, vy = (x2 - x1) / l2, (y2 - y1) / l2
        d += f" L {x1 - ux * rr:g} {y1 - uy * rr:g} Q {x1} {y1} {x1 + vx * rr:g} {y1 + vy * rr:g}"
    d += f" L {pts[-1][0]} {pts[-1][1]}"
    da = f' stroke-dasharray="{dash}"' if dash else ""
    me = ' marker-end="url(#arrow)"' if end else ""
    return f'<path d="{d}" fill="none" stroke="{MUTED}" stroke-width="1.2"{da}{me}/>'


def node(x, y, w, h, tag, name, sub, kind="svc"):
    fill, stroke, sw, dash = KINDS[kind]
    da = f' stroke-dasharray="{dash}"' if dash else ""
    cx = x + w / 2
    top = y + h / 2 - 14
    if not sub:  # header-only box (sequence lifeline heads)
        return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{stroke}" '
                f'stroke-width="{sw}"{da}/><text x="{cx}" y="{y + h / 2 + 5}" class="name" '
                f'text-anchor="middle">{E(name)}</text>')
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{PAPER}"/>'
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{da}/>'
            f'<text x="{x + 10}" y="{y + 14}" class="tag">{E(tag)}</text>'
            f'<text x="{cx}" y="{top + 16}" class="name" text-anchor="middle">{E(name)}</text>'
            f'<text x="{cx}" y="{top + 32}" class="sub" text-anchor="middle">{E(sub)}</text>')


def label(cx, cy, text):
    w = len(text) * 5.9 + 12
    return (f'<rect x="{cx - w / 2:g}" y="{cy - 8}" width="{w:g}" height="16" fill="{PAPER}"/>'
            f'<text x="{cx}" y="{cy + 3}" class="lab" text-anchor="middle">{E(text)}</text>')


def legend(y, items, x=40):
    out = []
    for kind, text in items:
        fill, stroke, sw, dash = KINDS[kind]
        da = f' stroke-dasharray="{dash}"' if dash else ""
        out.append(f'<rect x="{x}" y="{y}" width="22" height="14" rx="3" fill="{fill}" stroke="{stroke}" '
                   f'stroke-width="{sw}"{da}/><text x="{x + 30}" y="{y + 11}" class="leg">{E(text)}</text>')
        x += 40 + len(text) * 6.4
    return f'<line x1="40" y1="{y - 16}" x2="920" y2="{y - 16}" stroke="{RULE}" stroke-width="0.8"/>' + "".join(out)


def page(slug, title, desc, w, h, body):
    svg = (f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg" role="img" '
           f'aria-labelledby="{slug}-title {slug}-desc"><title id="{slug}-title">{E(title)}</title>'
           f'<desc id="{slug}-desc">{E(desc)}</desc><defs><marker id="arrow" markerWidth="8" markerHeight="6" '
           f'refX="7" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="{MUTED}"/></marker></defs>'
           f'<rect width="100%" height="100%" fill="{PAPER}"/>{body}</svg>')
    doc = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0"><title>{E(title)}</title>
<link href="{E(FONTS)}" rel="stylesheet"><style>
@page {{ size: {w}px {h}px; margin: 0; }}
*{{margin:0;padding:0;box-sizing:border-box}} body{{background:{PAPER};color:{INK};font-family:'Geist',system-ui,sans-serif}}
.diagram-container{{width:{w}px;overflow-x:auto}} svg{{display:block;min-width:{w}px}}
.tag{{font:500 8px 'Geist Mono',monospace;letter-spacing:.12em;fill:{SOFT}}}
.name{{font:600 13px 'Geist',sans-serif;fill:{INK}}}
.sub{{font:400 10px 'Geist Mono',monospace;fill:{MUTED}}}
.lab{{font:500 9px 'Geist Mono',monospace;letter-spacing:.06em;fill:{MUTED}}}
.leg{{font:400 10px 'Geist',sans-serif;fill:{MUTED}}}
.zone{{font:500 9px 'Geist Mono',monospace;letter-spacing:.14em;fill:{SOFT}}}
.big{{font:600 14px 'Geist',sans-serif;fill:{INK}}}
</style></head><body><div class="diagram-container">{svg}</div></body></html>"""
    (OUT / f"{slug}.html").write_text(doc)


def fig_architecture():
    b = [f'<rect x="260" y="48" width="440" height="424" rx="8" fill="none" stroke="{SOFT}" stroke-dasharray="4,3"/>',
         '<text x="276" y="68" class="zone">VERCEL CONTAINER · PYTHON 3.13</text>']
    b += [rpath([(450, 228), (476, 228), (476, 128), (520, 128)]),
          rpath([(450, 248), (488, 248), (488, 218), (520, 218)]),
          rpath([(450, 268), (488, 268), (488, 308), (520, 308)]),
          rpath([(450, 288), (464, 288), (464, 398), (520, 398)]),
          rpath([(200, 152), (236, 152), (236, 241), (300, 241)]),
          rpath([(200, 352), (236, 352), (236, 275), (300, 275)], dash="5,4"),
          rpath([(740, 132), (680, 132)]),
          rpath([(830, 240), (830, 164)])]
    b += [node(40, 120, 160, 64, "USER", "Phone web app", "doctor · patient"),
          node(40, 320, 160, 64, "LATER", "IoT stethoscope", "same REST API", "planned"),
          node(300, 208, 150, 100, "API", "FastAPI server", "/api/*", "focal"),
          node(520, 100, 160, 56, "ML", "Analyzer", "OOD gate + 5 heads"),
          node(520, 190, 160, 56, "RULES", "Fusion rules", "heart + lung flags"),
          node(520, 280, 160, 56, "AGENT", "Intake chat", "scripted · 999/111"),
          node(520, 370, 160, 56, "STORE", "Case inbox", "SQLite · passcode", "store"),
          node(740, 100, 180, 64, "GITHUB", "Model release", "public · SHA-256", "ext"),
          node(740, 240, 180, 64, "OFFLINE", "Training pipeline", "5 public datasets")]
    b += [label(205, 212, "HTTPS"), label(710, 118, "AT BUILD"), label(866, 202, "PUBLISH")]
    b.append(legend(520, [("focal", "entry point"), ("svc", "service"), ("store", "data store"),
                          ("ext", "external"), ("planned", "planned")]))
    page("fig-architecture", "System architecture",
         "A phone web app calls a FastAPI server on Vercel that runs the analyzer, fusion rules, intake chat and "
         "case inbox, with models loaded from a pinned public release.", 960, 548, "".join(b))


def fig_flow():
    b = [rpath([(160, 80), (160, 112)]), rpath([(160, 168), (160, 200)]), rpath([(160, 256), (160, 286)]),
         rpath([(160, 374), (160, 416)]), rpath([(160, 472), (160, 504)]), rpath([(160, 560), (160, 592)]),
         rpath([(260, 140), (300, 140), (300, 220), (330, 220)]),
         rpath([(240, 330), (312, 330), (312, 236), (330, 236)])]
    b += [node(60, 24, 200, 56, "INPUT", "Upload or record", "wav · m4a · mp3"),
          node(60, 112, 200, 56, "CHECK", "Decode and validate", "2-120 s · not clipped"),
          node(60, 200, 200, 56, "DSP", "Filter and resample", "4 kHz · band-pass"),
          f'<polygon points="160,286 240,330 160,374 80,330" fill="{PAPER}" stroke="{INK}" stroke-width="1.2"/>'
          '<text x="160" y="328" class="name" text-anchor="middle">Stethoscope-like?</text>'
          '<text x="160" y="343" class="sub" text-anchor="middle">IsolationForest</text>',
          node(60, 416, 200, 56, "DSP", "Cut windows", "heart 5 s · lung 3 or 8 s"),
          node(60, 504, 200, 56, "ML", "Model heads", "LightGBM x 5"),
          node(60, 592, 200, 56, "OUTPUT", "Fused report", "status · flags · conf.", "focal"),
          node(330, 200, 160, 56, "STOP", "Reject with reason", "ask to re-record", "store")]
    b += [label(326, 180, "FAIL"), label(276, 316, "NO"), label(186, 395, "YES")]
    b.append(f'<line x1="40" y1="676" x2="480" y2="676" stroke="{RULE}" stroke-width="0.8"/>'
             f'<rect x="40" y="690" width="22" height="14" rx="3" fill="#e4e4e4" stroke="{INK}" stroke-width="2.2"/>'
             '<text x="70" y="701" class="leg">result shown to user</text>'
             f'<rect x="210" y="690" width="22" height="14" rx="3" fill="#f0f0f0" stroke="{INK}" stroke-width="1.2"/>'
             '<text x="240" y="701" class="leg">request stops</text>')
    page("fig-flow", "Analysis pipeline for one recording",
         "One recording is decoded, checked, filtered, screened by an out-of-distribution gate, windowed, scored "
         "by the model heads and merged into a report, or rejected with a reason.", 520, 720, "".join(b))


def fig_sequence():
    xs = {"P": 80, "A": 240, "M": 400, "G": 560, "I": 720, "D": 880}
    names = {"P": ("PHONE", "Patient"), "A": ("API", "FastAPI"), "M": ("ML", "Analyzer"),
             "G": ("AGENT", "Intake chat"), "I": ("STORE", "Case inbox"), "D": ("PHONE", "Doctor")}
    b = [f'<rect x="24" y="268" width="560" height="120" rx="6" fill="none" stroke="{SOFT}" stroke-dasharray="4,3"/>',
         f'<rect x="24" y="268" width="150" height="16" fill="{PAPER}" stroke="{SOFT}"/>'
         '<text x="32" y="279" class="zone">LOOP · 4 QUESTIONS</text>']
    for k, x in xs.items():
        b.append(f'<line x1="{x}" y1="68" x2="{x}" y2="588" stroke="{RULE}" stroke-dasharray="3,4"/>')
    msgs = [(112, "P", "A", "POST /analyze/combined", 0), (152, "A", "M", "audio + kind", 0),
            (192, "M", "A", "labels + confidence", 1), (232, "A", "P", "report + flags", 1),
            (312, "P", "A", "POST /chat", 0), (344, "A", "G", "history", 0), (376, "G", "A", "next question", 1),
            (432, "P", "A", "POST /cases/from-chat", 0), (472, "A", "I", "summary + urgency", 0),
            (528, "D", "A", "GET /cases + passcode", 0), (568, "A", "D", "case list", 1)]
    labs = []
    for y, f, t, text, dashed in msgs:
        x1, x2 = xs[f], xs[t]
        x2e = x2 - 4 if x2 > x1 else x2 + 4
        b.append(rpath([(x1, y), (x2e, y)], dash="5,4" if dashed else None))
        labs.append(label((x1 + x2) / 2, y - 14, text))
    for k, x in xs.items():
        tag, name = names[k]
        b.append(node(x - 64, 24, 128, 44, tag, name, "", "focal" if k == "A" else "svc"))
    b += labs
    b.append(f'<line x1="40" y1="604" x2="920" y2="604" stroke="{RULE}" stroke-width="0.8"/>'
             f'<line x1="40" y1="624" x2="70" y2="624" stroke="{MUTED}" stroke-width="1.2"/>'
             '<text x="78" y="627" class="leg">request</text>'
             f'<line x1="160" y1="624" x2="190" y2="624" stroke="{MUTED}" stroke-width="1.2" stroke-dasharray="5,4"/>'
             '<text x="198" y="627" class="leg">response</text>')
    page("fig-sequence", "Patient to doctor sequence",
         "The patient's phone sends audio for analysis, answers four intake questions, and sends a case summary "
         "that the doctor later reads from the inbox with a passcode.", 960, 640, "".join(b))


def fig_training():
    b = [rpath([(220, 80), (276, 80)]), rpath([(456, 80), (512, 80)]), rpath([(692, 80), (748, 80)]),
         rpath([(838, 112), (838, 208)]), rpath([(748, 240), (692, 240)]), rpath([(512, 240), (456, 240)])]
    b += [node(40, 48, 180, 64, "DATA", "Public datasets", "heart 3 · lung 2"),
          node(276, 48, 180, 64, "PREP", "Load, group labels", "6 diagnosis groups"),
          node(512, 48, 180, 64, "SPLIT", "Patient-wise split", "ICBHI official 60/40", "focal"),
          node(748, 48, 180, 64, "DSP", "Feature extraction", "MFCC + spectral stats"),
          node(748, 208, 180, 64, "ML", "Train LightGBM", "OOF threshold · OOD"),
          node(512, 208, 180, 64, "EVAL", "Held-out test", "MAcc · ICBHI · F1"),
          node(276, 208, 180, 64, "SHIP", "Public release", "joblib · SHA-256", "ext")]
    b.append(legend(320, [("focal", "leakage control"), ("svc", "step"), ("ext", "published artefact")]))
    page("fig-training", "Training and evaluation pipeline",
         "Public heart and lung datasets are loaded, split by patient, turned into features, used to train "
         "LightGBM heads, scored on held-out patients and published.", 960, 348, "".join(b))


def fig_roadmap():
    stages = [("NOW", "Prototype", "5 models, web app", "scripted intake"),
              ("NEXT", "Better models", "audio-pretrained nets", "noise rejection"),
              ("THEN", "IoT stethoscope", "device posts to /api", "paired heart+lung data"),
              ("PILOT", "Clinical pilot", "NHS site, HRA ethics", "local validation"),
              ("LATER", "Regulated product", "DTAC, DCB0129/0160", "MHRA SaMD, UKCA")]
    b = [f'<line x1="40" y1="130" x2="920" y2="130" stroke="{MUTED}" stroke-width="1.2" marker-end="url(#arrow)"/>']
    for i, (eb, name, l1, l2) in enumerate(stages):
        x = 104 + i * 188
        fill = INK if i == 0 else PAPER
        b.append(f'<text x="{x}" y="88" class="zone" text-anchor="middle">{eb}</text>'
                 f'<text x="{x}" y="108" class="big" text-anchor="middle">{E(name)}</text>'
                 f'<circle cx="{x}" cy="130" r="7" fill="{fill}" stroke="{INK}" stroke-width="1.6"/>'
                 f'<text x="{x}" y="160" class="sub" text-anchor="middle">{E(l1)}</text>'
                 f'<text x="{x}" y="176" class="sub" text-anchor="middle">{E(l2)}</text>')
    b.append(f'<line x1="40" y1="212" x2="920" y2="212" stroke="{RULE}" stroke-width="0.8"/>'
             f'<circle cx="51" cy="232" r="6" fill="{INK}"/><text x="64" y="236" class="leg">done</text>'
             f'<circle cx="131" cy="232" r="6" fill="{PAPER}" stroke="{INK}" stroke-width="1.6"/>'
             '<text x="144" y="236" class="leg">planned</text>')
    page("fig-roadmap", "Roadmap from prototype to product",
         "Five stages: the current prototype, better models, an IoT stethoscope, an NHS clinical pilot and a "
         "regulated product.", 960, 256, "".join(b))


if __name__ == "__main__":
    for f in (fig_architecture, fig_flow, fig_sequence, fig_training, fig_roadmap):
        f()
