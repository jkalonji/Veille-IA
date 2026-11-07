"""
AI layer cake section of the dashboard ("ia" domain only), shared by the
Streamlit app and the static export of dashboard.py.
"""

from collections import Counter
from html import escape

import cake


def layer_counts(articles: list[dict]) -> Counter:
    """Articles per layer code (an article can count in up to 3 layers)."""
    return Counter(code for a in articles for code in (a.get("layers") or []))


def cake_counts_html(articles: list[dict]) -> str:
    """Five stacked bands, energy at the bottom, with the article count of each layer."""
    counts = layer_counts(articles)
    if not counts:
        return ""
    meta = cake.layer_meta()
    top = max(counts.values())
    bands = ""
    for code in reversed(cake.layer_codes()):
        n = counts.get(code, 0)
        width = 8 + 92 * n / top
        bands += f"""
    <div class="cake-band">
      <span class="cake-band__name">{meta[code]['emoji']} {escape(meta[code]['label'])}</span>
      <span class="cake-band__bar"><span style="width:{width:.0f}%"></span></span>
      <span class="cake-band__count">{n}</span>
    </div>"""
    return f"""
<style>
  .cake {{ background:#0c1222; border:1px solid #1a2540; border-radius:10px; padding:12px 14px; margin-bottom:14px; }}
  .cake-band {{ display:flex; align-items:center; gap:10px; padding:5px 0; font-size:13px; color:#c8d4ec; }}
  .cake-band__name {{ flex:0 0 140px; }}
  .cake-band__bar {{ flex:1; height:10px; background:#060810; border-radius:5px; overflow:hidden; }}
  .cake-band__bar span {{ display:block; height:100%; background:linear-gradient(90deg,#0047ab,#7eb8f7); border-radius:5px; }}
  .cake-band__count {{ flex:0 0 40px; text-align:right; color:#e8eef8; }}
  .cake-caption {{ font-size:11px; color:#5a6a8a; margin-top:6px; }}
</style>
<div class="cake">{bands}
  <div class="cake-caption">Articles par couche sur la période · un article peut couvrir jusqu'à 3 couches</div>
</div>"""
