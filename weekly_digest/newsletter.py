"""
AI Radar — Weekly Newsletter
Builds the weekly reader-facing newsletter (IA domain only) from Supabase.

Topic candidates are the stories with articles this week (see "Histoires
sémantiques" in CLAUDE.md). The editor picks which
candidates make it into the edition; Groq only writes the prose (edito + one
paragraph per topic) — titles, chronology and source links are rendered
straight from the DB so no link can be hallucinated.

Usage:
    python weekly_digest/newsletter.py candidates            # list this week's candidates
    python weekly_digest/newsletter.py preview               # build an edition from the top 5
    python weekly_digest/newsletter.py preview --pick story-12,topic-gpt-5-launch
    python weekly_digest/newsletter.py propose [--dry-run]   # Sunday: open the topic-picker issue
    python weekly_digest/newsletter.py build                 # Monday: build from the checked topics

Env vars: SUPABASE_URL, SUPABASE_KEY, GROQ_API_KEY (GROQ_MODEL optional), BUTTONDOWN_API_KEY (build),
GITHUB_TOKEN + GITHUB_REPOSITORY for propose/build (set automatically in Actions).
"""

import argparse
import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

# dashboard.py lives at the repo root — reuse its paginated loader and helpers
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard import (  # noqa: E402
    DOMAIN_CATEGORY_EMOJI,
    _OLD_HOT_REASONS,
    _deduplicate_articles,
    _fetch_stories_lookup,
    _hot_sort_key,
    _story_members,
    load_articles,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DOMAIN = "ia"
WINDOW_DAYS = 7
CATEGORY_EMOJI = DOMAIN_CATEGORY_EMOJI[DOMAIN]
GROQ_MODEL = (os.environ.get("GROQ_MODEL") or "openai/gpt-oss-120b").strip("'\"").strip()
DASHBOARD_URL = os.environ.get("DASHBOARD_URL", "")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")

# The edition is written in English; DB category names stay French
CATEGORY_EN = {
    "Innovation / Tech": "Innovation & Tech",
    "Politique / Regulation": "Policy & Regulation",
    "Business / Industrie": "Business & Industry",
    "Societe / Ethique": "Society & Ethics",
    "Recherche Academique": "Academic Research",
    "Drama / Controverses": "Drama & Controversies",
    "Energie / Environnement": "Energy & Environment",
    "Semiconducteurs / Hardware": "Semiconductors & Hardware",
}


def _en_date(iso_day: str) -> str:
    """'2026-10-05' -> 'Mon, Oct 5'."""
    try:
        d = datetime.strptime(iso_day[:10], "%Y-%m-%d")
    except ValueError:
        return iso_day
    return f"{d:%a, %b} {d.day}"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:50] or "sans-titre"


def _truncate(text: str, n: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------

def select_candidates(articles: list[dict], stories_lookup: dict[int, dict], limit: int | None = 15) -> list[dict]:
    """Group this week's articles into topic candidates, best first.

    A candidate is a story with at least 2 articles this week (an article can
    feed several candidates). Before topic-based stories exist, falls back to
    hot articles grouped by legacy `story_id`, else by `hot_reason`. Returns
    dicts: {id, title, blurb, days: [{date, articles}], article_count,
    span_days, has_supra, emoji}.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    titles: dict[str, str] = {}
    members = _story_members(articles, stories_lookup)
    for sid, arts in (members or {}).items():
        if len(_deduplicate_articles(arts)) < 2:
            continue
        key = f"story-{sid}"
        groups[key] = arts
        meta = stories_lookup[sid]
        titles[key] = meta["label"] or "Sans titre"
        if meta.get("summary") and meta["summary"].lower() != meta["label"].lower():
            titles[key] += f" — {meta['summary']}"
    # Pre-topics fallback: hot articles by legacy story_id, else by hot_reason
    for a in (articles if members is None else []):
        reason = (a.get("hot_reason") or "").strip()
        if not a.get("hot_topic") or not reason or reason.lower() in _OLD_HOT_REASONS:
            continue
        sid = a.get("story_id")
        if sid:
            key = f"story-{sid}"
            meta = stories_lookup.get(sid, {})
            if meta.get("label"):
                titles[key] = meta["label"]
                if meta.get("summary") and meta["summary"].lower() != meta["label"].lower():
                    titles[key] += f" — {meta['summary']}"
        else:
            key = f"topic-{_slug(reason)}"
        titles.setdefault(key, reason)
        groups[key].append(a)

    candidates = []
    for key, arts in groups.items():
        by_day: dict[str, list[dict]] = defaultdict(list)
        for a in arts:
            by_day[a.get("published", "")].append(a)
        # Dedup within each day only, so a multi-day story keeps every day
        days = [
            {"date": d, "articles": sorted(_deduplicate_articles(by_day[d]), key=_hot_sort_key)}
            for d in sorted(by_day)
        ]
        article_count = sum(len(d["articles"]) for d in days)
        top_cat = Counter(a.get("category", "") for a in arts).most_common(1)[0][0]
        lead = max(arts, key=lambda a: a.get("mention_count", 0))
        candidates.append({
            "id": key,
            "title": titles[key],
            "blurb": _truncate(
                lead.get("summary") or lead.get("description") or lead.get("title", ""), 160
            ),
            "days": days,
            "article_count": article_count,
            "span_days": len(days),
            "has_supra": any(a.get("supa_hot") for a in arts),
            "emoji": CATEGORY_EMOJI.get(top_cat, "📌"),
        })

    # Breadth (articles) plus persistence (days covered) plus a supa-hot bump
    candidates.sort(
        key=lambda c: c["article_count"] + 2 * (c["span_days"] - 1) + (3 if c["has_supra"] else 0),
        reverse=True,
    )
    return candidates[:limit]


def compute_stats(articles: list[dict], stories_lookup: dict[int, dict] | None = None) -> dict:
    by_cat = Counter(a.get("category", "") for a in articles)
    members = _story_members(articles, stories_lookup)
    return {
        "total": len(articles),
        "hot": sum(1 for a in articles if a.get("hot_topic")),
        "stories": len(members) if members is not None else len({a["story_id"] for a in articles if a.get("story_id")}),
        "top_categories": by_cat.most_common(3),
    }


# ---------------------------------------------------------------------------
# Groq prose
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You write a weekly English-language newsletter about AI news, for curious readers who "
    "are not necessarily technical. You get the topics picked by the editor, each with the "
    "titles and summaries of the week's articles, day by day, and sometimes an editor's note. "
    "Some summaries and notes are in French: always write in English.\n"
    "Reply in strict JSON: {\"edito\": str, \"topics\": {\"<id>\": str}}.\n"
    "- edito: 3 to 4 sentences drawing the common thread of the week from the picked topics.\n"
    "- topics: for each id, one paragraph of 3 to 5 sentences explaining what happened, "
    "how the story evolved over the days and why it matters.\n"
    "If an editor's note is given for a topic, follow it first (angle, emphasis).\n"
    "Clear, direct, factual tone. Do not invent any fact absent from the articles provided. "
    "No links, no Markdown, no headings."
)


def _topic_context(c: dict, note: str) -> str:
    lines = [f"[{c['id']}] {c['title']}"]
    if note:
        lines.append(f"Editor's note: {note}")
    for day in c["days"]:
        for a in day["articles"][:4]:
            extra = _truncate(a.get("summary") or a.get("description") or "", 200)
            lines.append(f"  {day['date']} · {a.get('title', '')}" + (f" — {extra}" if extra else ""))
    return "\n".join(lines)


def generate_prose(picked: list[dict], notes: dict[str, str]) -> dict:
    """Return {"edito": str, "topics": {id: str}}; falls back to article summaries on failure."""
    # Fallback text: the source's own description (mostly English), not our French summary
    fallback = {
        "edito": "",
        "topics": {
            c["id"]: _truncate(
                c["days"][-1]["articles"][0].get("description")
                or c["days"][-1]["articles"][0].get("summary") or c["blurb"], 400
            )
            for c in picked
        },
    }
    if not os.environ.get("GROQ_API_KEY"):
        logging.warning("GROQ_API_KEY not set — using article summaries instead of Groq prose.")
        return fallback

    from groq import Groq

    user_msg = "\n\n".join(_topic_context(c, notes.get(c["id"], "")) for c in picked)
    # Groq sometimes emits malformed JSON (json_validate_failed): one retry
    for attempt in (1, 2):
        try:
            resp = Groq(api_key=os.environ["GROQ_API_KEY"]).chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.4,
                max_tokens=4000,
                reasoning_effort="low",
                response_format={"type": "json_object"},
            )
            result = json.loads(resp.choices[0].message.content)
            break
        except Exception as e:
            logging.error(f"Groq newsletter prose failed (attempt {attempt}/2): {e}")
    else:
        return fallback

    topics = result.get("topics") or {}
    return {
        "edito": (result.get("edito") or "").strip(),
        "topics": {c["id"]: (topics.get(c["id"]) or fallback["topics"][c["id"]]).strip() for c in picked},
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _link(a: dict) -> str:
    title = a.get("title", "").replace("[", "(").replace("]", ")")
    return f"[{title}]({a.get('url', '')}) — *{a.get('source', '')}*"


def render_markdown(picked: list[dict], prose: dict, stats: dict, week_start: str, week_end: str) -> str:
    out = [
        f"# 🤖 AI Radar — week of {week_start} to {week_end}",
        "",
    ]
    if prose["edito"]:
        out += [prose["edito"], ""]
    out += ["**In this issue:** " + " · ".join(f"{c['emoji']} {c['title']}" for c in picked), "", "---", ""]

    for i, c in enumerate(picked, 1):
        badge = " 🌋" if c["has_supra"] else ""
        span = f"{c['span_days']} days of coverage · " if c["span_days"] > 1 else ""
        out += [
            f"## {i}. {c['emoji']} {c['title']}{badge}",
            f"*{span}{c['article_count']} articles*",
            "",
            prose["topics"][c["id"]],
            "",
        ]
        if c["span_days"] > 1:
            out.append("**📅 Timeline**")
            out += [f"- **{_en_date(d['date'])}** — {_link(d['articles'][0])}" for d in c["days"]]
        else:
            out.append("**🔗 Read more**")
            out += [f"- {_link(a)}" for a in c["days"][0]["articles"][:3]]
        out += ["", "---", ""]

    cats = " · ".join(
        f"{CATEGORY_EMOJI.get(cat, '📌')} {CATEGORY_EN.get(cat, cat)} ({n})" for cat, n in stats["top_categories"]
    )
    out += [
        "## 📊 The week in numbers",
        f"- 📰 **{stats['total']}** articles analyzed, **{stats['hot']}** of them on hot topics",
        f"- 📖 **{stats['stories']}** stor{'ies' if stats['stories'] != 1 else 'y'} tracked",
        f"- 🏆 Most active categories: {cats}",
    ]
    if DASHBOARD_URL:
        out += ["", f"👉 [Explore all the news on the dashboard]({DASHBOARD_URL})"]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# GitHub issue — the editor's topic picker
# ---------------------------------------------------------------------------

ISSUE_LABEL = "newsletter"
# "- [x] **3.** 🚀 **Title** · 2 jours · 9 articles <!-- id:story-12 -->"
_ITEM_RE = re.compile(r"^\s*[-*]\s*\[([ xX])\]\s*\*\*(\d+)\.\*\*.*?<!--\s*id:(\S+)\s*-->", re.M)
# "3: insister sur l'angle prix" (one note per line, in an issue comment)
_NOTE_RE = re.compile(r"^\s*#?(\d{1,2})\s*[:.)\-–—]\s*(.+?)\s*$", re.M)
# Only people with write access may steer the prose — on a public repo anyone
# can comment, and a note goes straight into the Groq prompt.
_TRUSTED_AUTHORS = {"OWNER", "MEMBER", "COLLABORATOR"}


def _github(method: str, path: str, **kwargs):
    import requests

    token, repo = os.environ.get("GITHUB_TOKEN"), os.environ.get("GITHUB_REPOSITORY")
    if not token or not repo:
        sys.exit("GITHUB_TOKEN and GITHUB_REPOSITORY (owner/repo) must be set.")
    resp = requests.request(
        method,
        f"https://api.github.com/repos/{repo}{path}",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
        timeout=15,
        **kwargs,
    )
    resp.raise_for_status()
    return resp.json() if resp.content else None


def _open_issues() -> list[dict]:
    return _github("GET", "/issues", params={"labels": ISSUE_LABEL, "state": "open"})


def _close_issue(number: int, comment: str) -> None:
    _github("POST", f"/issues/{number}/comments", json={"body": comment})
    _github("PATCH", f"/issues/{number}", json={"state": "closed"})


def render_issue_body(candidates: list[dict]) -> str:
    lines = [
        "Coche les sujets à traiter dans la newsletter de lundi — ils paraîtront dans l'ordre de la liste.",
        "",
        "💬 Pour orienter la rédaction d'un sujet, ajoute un commentaire avec une note par ligne, "
        "précédée du numéro du sujet : `3: insister sur l'angle prix`.",
        "",
        "Si rien n'est coché lundi matin, pas de newsletter cette semaine.",
        "",
    ]
    for i, c in enumerate(candidates, 1):
        badge = " 🌋" if c["has_supra"] else ""
        span = f"{c['span_days']} jours · " if c["span_days"] > 1 else ""
        lead = c["days"][-1]["articles"][0]
        lines += [
            f"- [ ] **{i}.** {c['emoji']} **{c['title']}**{badge} · {span}{c['article_count']} articles"
            f" <!-- id:{c['id']} -->",
            f"  {c['blurb']} — [article principal]({lead.get('url', '')})",
        ]
    return "\n".join(lines) + "\n"


def parse_selection(body: str, comments: list[dict]) -> tuple[list[str], dict[str, str]]:
    """Return (checked ids in list order, {id: editor note})."""
    items = _ITEM_RE.findall(body or "")
    by_number = {int(n): cid for _, n, cid in items}
    picked = [cid for mark, _, cid in items if mark.lower() == "x"]
    notes: dict[str, list[str]] = defaultdict(list)
    for comment in comments:
        if comment.get("author_association") not in _TRUSTED_AUTHORS:
            continue
        for n, note in _NOTE_RE.findall(comment.get("body") or ""):
            if int(n) in by_number:
                notes[by_number[int(n)]].append(note)
    return picked, {cid: " ".join(v) for cid, v in notes.items()}


def propose(dry_run: bool) -> None:
    """Open this week's topic-picker issue and retire any older open one."""
    _, candidates = load_week()
    if not candidates:
        logging.warning("No hot topic candidates this week — no issue opened.")
        return
    today = datetime.now(timezone.utc)
    # Next Monday's build — on a Monday the scheduled build already ran (or is about to)
    monday = today + timedelta(days=(0 - today.weekday()) % 7 or 7)
    title = f"📰 Newsletter du {monday:%d/%m} — choisis tes sujets"
    body = render_issue_body(candidates)
    if dry_run:
        print(f"{title}\n\n{body}")
        return
    previous = _open_issues()
    try:
        _github("POST", "/labels", json={"name": ISSUE_LABEL, "color": "f4a261"})
    except Exception:
        pass  # 422 = label already exists
    issue = _github("POST", "/issues", json={"title": title, "body": body, "labels": [ISSUE_LABEL]})
    logging.info(f"Opened issue #{issue['number']}: {issue['html_url']}")
    for old in previous:
        _close_issue(old["number"], f"Remplacée par #{issue['number']}.")


def build() -> str | None:
    """Build the edition from the latest open picker issue; return the output path."""
    issues = _open_issues()
    if not issues:
        logging.warning("No open newsletter issue — nothing to build.")
        return None
    issue = max(issues, key=lambda i: i["created_at"])
    comments = _github("GET", f"/issues/{issue['number']}/comments", params={"per_page": 100})
    picked_ids, notes = parse_selection(issue.get("body") or "", comments)
    if not picked_ids:
        _close_issue(issue["number"], "Aucun sujet coché — pas de newsletter cette semaine.")
        logging.info("No topic picked — no edition this week.")
        return None

    articles, _ = load_week()
    # Re-rank without the cap: Monday's collection may push a Sunday pick out of the top 15
    by_id = {c["id"]: c for c in select_candidates(articles, _fetch_stories_lookup(DOMAIN), limit=None)}
    picked = [by_id[cid] for cid in picked_ids if cid in by_id]
    missing = [cid for cid in picked_ids if cid not in by_id]
    if not picked:
        _close_issue(issue["number"], "⚠️ Aucun des sujets cochés n'a été retrouvé en base — édition annulée.")
        return None

    markdown = build_edition(articles, picked, notes)
    path = write_edition(markdown)
    warning = f"\n\n⚠️ Sujets introuvables, ignorés : {', '.join(missing)}" if missing else ""
    _close_issue(
        issue["number"],
        f"✅ Édition générée avec {len(picked)} sujet(s).{warning}\n\n{publish_draft(markdown)}\n\n"
        f"<details><summary>Aperçu</summary>\n\n{markdown}\n</details>",
    )
    return path


# ---------------------------------------------------------------------------
# Buttondown
# ---------------------------------------------------------------------------

def publish_draft(markdown: str) -> str:
    """Create the edition as a Buttondown *draft*; return a status line for the issue.

    `status: "draft"` is mandatory: the API default is `about_to_send`, which
    would email every subscriber immediately.
    """
    key = os.environ.get("BUTTONDOWN_API_KEY")
    if not key:
        return "ℹ️ BUTTONDOWN_API_KEY absent — pas de brouillon Buttondown, l'édition est seulement archivée."

    import requests

    # The H1 becomes the subject; Buttondown shows the subject as the email title
    title, _, body = markdown.partition("\n")
    subject = title.lstrip("# ").strip()
    try:
        resp = requests.post(
            "https://api.buttondown.com/v1/emails",
            headers={"Authorization": f"Token {key}"},
            json={
                "subject": subject,
                "body": "<!-- buttondown-editor-mode: plaintext -->\n" + body.strip(),
                "status": "draft",
            },
            timeout=20,
        )
        resp.raise_for_status()
        email = resp.json()
    except Exception as e:
        logging.error(f"Buttondown draft failed: {e}")
        return f"❌ Échec de la création du brouillon Buttondown : `{e}`"

    if email.get("status") != "draft":
        logging.error(f"Buttondown email {email.get('id')} has status {email.get('status')!r}, not draft!")
        return f"⚠️ Buttondown a créé l'email avec le statut `{email.get('status')}` au lieu de `draft` — vérifie-le."
    logging.info(f"Buttondown draft created: {email.get('id')}")
    return "📝 Brouillon créé dans Buttondown — relis-le et envoie-le depuis https://buttondown.com/emails"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def load_week() -> tuple[list[dict], list[dict]]:
    articles = load_articles(WINDOW_DAYS, domain=DOMAIN)
    logging.info(f"Loaded {len(articles)} '{DOMAIN}' articles from the past {WINDOW_DAYS} days")
    return articles, select_candidates(articles, _fetch_stories_lookup(DOMAIN))


def build_edition(articles: list[dict], picked: list[dict], notes: dict[str, str]) -> str:
    today = datetime.now(timezone.utc)
    start = today - timedelta(days=WINDOW_DAYS)
    week_start = f"{start:%b} {start.day}"
    week_end = f"{today:%b} {today.day}, {today.year}"
    prose = generate_prose(picked, notes)
    return render_markdown(picked, prose, compute_stats(articles, _fetch_stories_lookup(DOMAIN)), week_start, week_end)


def write_edition(markdown: str) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"newsletter-{datetime.now(timezone.utc):%Y-%m-%d}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(markdown)
    logging.info(f"Edition written to {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Radar weekly newsletter")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("candidates", help="List this week's topic candidates")
    p_prev = sub.add_parser("preview", help="Build an edition into output/ without GitHub")
    p_prev.add_argument("--pick", default="", help="Comma-separated candidate ids (default: top 5)")
    p_prop = sub.add_parser("propose", help="Open the GitHub issue listing this week's candidates")
    p_prop.add_argument("--dry-run", action="store_true", help="Print the issue instead of opening it")
    sub.add_parser("build", help="Build the edition from the topics checked in the issue")
    args = parser.parse_args()

    if args.cmd == "propose":
        propose(args.dry_run)
        return
    if args.cmd == "build":
        build()
        return

    articles, candidates = load_week()
    if not candidates:
        logging.warning("No hot topic candidates this week — nothing to build.")
        return

    if args.cmd == "candidates":
        for c in candidates:
            span = f"{c['span_days']}j" if c["span_days"] > 1 else "1j"
            print(f"{c['id']:<40} {span:>3} {c['article_count']:>3} art.  {c['emoji']} {c['title']}")
        return

    if args.pick:
        wanted = [p.strip() for p in args.pick.split(",") if p.strip()]
        by_id = {c["id"]: c for c in select_candidates(articles, _fetch_stories_lookup(DOMAIN), limit=None)}
        unknown = [w for w in wanted if w not in by_id]
        if unknown:
            sys.exit(f"Unknown candidate ids: {', '.join(unknown)}")
        picked = [by_id[w] for w in wanted]
    else:
        picked = candidates[:5]
    write_edition(build_edition(articles, picked, notes={}))


if __name__ == "__main__":
    main()
