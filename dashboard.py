"""
AI Radar - Dashboard de Veille
Affiche des insights sur les articles collectés dans Supabase.
Usage: python dashboard.py [--days N]
"""

import argparse
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box
from supabase import create_client

console = Console()

CATEGORY_EMOJI = {
    "Innovation / Tech":      "🚀",
    "Politique / Regulation": "⚖️",
    "Business / Industrie":   "💼",
    "Societe / Ethique":      "🤝",
    "Recherche Academique":   "🎓",
    "Drama / Controverses":   "💥",
    "Geopolitique":           "🌍",
}
SENTIMENT_EMOJI   = {"Positif": "🟢", "Negatif": "🔴", "Neutre": "⚪"}
SENTIMENT_COLOR   = {"Positif": "green", "Negatif": "red", "Neutre": "white"}

BAR_WIDTH = 30


def bar(value: int, total: int, width: int = BAR_WIDTH) -> str:
    filled = round(value / total * width) if total else 0
    return "█" * filled + "░" * (width - filled)


def load_articles(days: int) -> list[dict]:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        console.print("[red]SUPABASE_URL ou SUPABASE_KEY manquant.[/red]")
        sys.exit(1)

    client = create_client(url, key)
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")

    resp = (
        client.table("articles")
        .select("title, source, country, published, category, sentiment, url, created_at")
        .gte("published", cutoff)
        .order("published", desc=True)
        .execute()
    )
    return resp.data or []


def render_header(articles: list[dict], days: int) -> None:
    total = len(articles)
    dates = sorted({a["published"] for a in articles})
    date_range = f"{dates[0]} → {dates[-1]}" if dates else "—"

    console.print(Panel(
        f"[bold cyan]🤖 AI Radar — Dashboard de Veille[/bold cyan]\n"
        f"[dim]Période : {date_range}  ({days} derniers jours)[/dim]\n"
        f"[bold white]{total} articles recensés[/bold white]",
        box=box.DOUBLE_EDGE,
        border_style="cyan",
    ))


def render_categories(articles: list[dict]) -> Panel:
    counts = Counter(a["category"] for a in articles)
    total = len(articles)

    table = Table(box=None, show_header=False, padding=(0, 1))
    table.add_column(width=3)
    table.add_column(min_width=22)
    table.add_column(width=BAR_WIDTH, no_wrap=True)
    table.add_column(width=6, justify="right")

    for cat, emoji in CATEGORY_EMOJI.items():
        count = counts.get(cat, 0)
        pct = count / total * 100 if total else 0
        table.add_row(
            emoji,
            Text(cat, style="bold" if count else "dim"),
            Text(bar(count, total), style="cyan" if count else "dim"),
            f"[bold]{count}[/bold] [dim]{pct:.0f}%[/dim]",
        )

    return Panel(table, title="[bold]Par catégorie[/bold]", border_style="blue")


def render_sentiments(articles: list[dict]) -> Panel:
    counts = Counter(a["sentiment"] for a in articles)
    total = len(articles)

    table = Table(box=None, show_header=False, padding=(0, 1))
    table.add_column(width=3)
    table.add_column(min_width=10)
    table.add_column(width=BAR_WIDTH, no_wrap=True)
    table.add_column(width=6, justify="right")

    for sent, emoji in SENTIMENT_EMOJI.items():
        count = counts.get(sent, 0)
        pct = count / total * 100 if total else 0
        color = SENTIMENT_COLOR[sent]
        table.add_row(
            emoji,
            Text(sent, style=color),
            Text(bar(count, total), style=color),
            f"[bold]{count}[/bold] [dim]{pct:.0f}%[/dim]",
        )

    return Panel(table, title="[bold]Par sentiment[/bold]", border_style="magenta")


def render_sources(articles: list[dict], top_n: int = 10) -> Panel:
    counts = Counter(f"{a['country']} {a['source']}" for a in articles)
    total = len(articles)

    table = Table(box=None, show_header=False, padding=(0, 1))
    table.add_column(min_width=28)
    table.add_column(width=BAR_WIDTH, no_wrap=True)
    table.add_column(width=5, justify="right")

    for source, count in counts.most_common(top_n):
        table.add_row(
            Text(source, style="bold"),
            Text(bar(count, total), style="yellow"),
            str(count),
        )

    return Panel(table, title=f"[bold]Top {top_n} sources[/bold]", border_style="yellow")


def render_trend(articles: list[dict]) -> Panel:
    by_date: dict[str, int] = defaultdict(int)
    for a in articles:
        by_date[a["published"]] += 1

    if not by_date:
        return Panel("Pas de données", title="[bold]Tendance quotidienne[/bold]")

    max_count = max(by_date.values())
    bar_h = 8  # height in lines

    dates = sorted(by_date)
    cols = []
    for d in dates:
        count = by_date[d]
        filled = round(count / max_count * bar_h) if max_count else 0
        lines = ["   "] * (bar_h - filled) + ["[cyan]███[/cyan]"] * filled
        lines.append(f"[dim]{count:^3}[/dim]")
        day_label = d[5:]  # MM-DD
        lines.append(f"[dim]{day_label}[/dim]")
        cols.append("\n".join(lines))

    return Panel(
        Columns(cols, equal=True, expand=False),
        title="[bold]Tendance quotidienne[/bold]",
        border_style="green",
    )


def render_latest(articles: list[dict], n: int = 10) -> Panel:
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold dim")
    table.add_column("Date", width=10, style="dim")
    table.add_column("S", width=2, justify="center")
    table.add_column("Titre", min_width=45, no_wrap=True)
    table.add_column("Source", width=22)
    table.add_column("Catégorie", width=22)

    for a in articles[:n]:
        sent_emoji = SENTIMENT_EMOJI.get(a["sentiment"], "⚪")
        cat_emoji = CATEGORY_EMOJI.get(a["category"], "📌")
        title = a["title"]
        if len(title) > 60:
            title = title[:57] + "..."
        table.add_row(
            a["published"],
            sent_emoji,
            title,
            f"{a['country']} {a['source']}",
            f"{cat_emoji} {a['category']}",
        )

    return Panel(table, title=f"[bold]{n} derniers articles[/bold]", border_style="white")


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Radar Dashboard")
    parser.add_argument("--days", type=int, default=7, help="Fenêtre d'analyse en jours (défaut: 7)")
    parser.add_argument("--top", type=int, default=10, help="Nb de sources à afficher (défaut: 10)")
    parser.add_argument("--latest", type=int, default=10, help="Nb de derniers articles (défaut: 10)")
    args = parser.parse_args()

    # Load .env if present (local usage)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    console.print()
    with console.status("[cyan]Connexion à Supabase...[/cyan]"):
        articles = load_articles(args.days)

    if not articles:
        console.print(f"[yellow]Aucun article trouvé pour les {args.days} derniers jours.[/yellow]")
        return

    render_header(articles, args.days)
    console.print()

    # Row 1: categories + sentiments
    console.print(Columns([
        render_categories(articles),
        render_sentiments(articles),
    ], equal=False, expand=True))
    console.print()

    # Row 2: trend
    console.print(render_trend(articles))
    console.print()

    # Row 3: top sources
    console.print(render_sources(articles, args.top))
    console.print()

    # Row 4: latest articles
    console.print(render_latest(articles, args.latest))
    console.print()


if __name__ == "__main__":
    main()
