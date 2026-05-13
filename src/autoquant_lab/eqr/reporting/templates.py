"""HTML template helpers for the EQR static reporting site."""

from __future__ import annotations

from html import escape
from typing import Any

from .assets import SITE_CSS


NAV_LINKS = (
    ("index.html", "Index"),
    ("leaderboard.html", "Leaderboard"),
    ("dead_letter.html", "Dead letter"),
    ("coverage.html", "Coverage"),
    ("about.html", "About"),
)


def text(value: object) -> str:
    if value is None:
        return ""
    return escape(str(value), quote=True)


def page(title: str, body: str) -> str:
    nav = "".join(f'<a href="{href}">{label}</a>' for href, label in NAV_LINKS)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{text(title)} · EQR Experiment History</title>
  <style>{SITE_CSS}</style>
</head>
<body>
  <main class="shell">
    <header class="topbar">
      <a class="brand" href="index.html">EQR Lab</a>
      <nav class="nav" aria-label="Primary navigation">{nav}</nav>
    </header>
    {body}
    <footer class="footer">Generated as an offline, static experiment history. No external assets or network calls are required.</footer>
  </main>
</body>
</html>
"""


def hero(title: str, lede: str) -> str:
    return f'<section class="hero"><h1>{text(title)}</h1><p class="lede">{text(lede)}</p></section>'


def stat_cards(cards: list[tuple[str, object]]) -> str:
    items = "".join(
        f'<article class="card"><div class="metric">{text(label)}</div><div class="metric-value">{text(value)}</div></article>'
        for label, value in cards
    )
    return f'<section class="grid cards">{items}</section>'


def table(headers: list[str], rows: list[list[object]]) -> str:
    head = "".join(f"<th>{text(header)}</th>" for header in headers)
    if rows:
        body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    else:
        body = f'<tr><td colspan="{len(headers)}" class="muted">No records found.</td></tr>'
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def section(title: str, content: str) -> str:
    return f'<section class="section"><h2>{text(title)}</h2>{content}</section>'


def badge(label: str, tone: str = "") -> str:
    klass = f"badge {tone}" if tone else "badge"
    return f'<span class="{klass}">{text(label)}</span>'


def code_block(value: Any) -> str:
    import json

    rendered = json.dumps(value, indent=2, sort_keys=True, default=str)
    return f"<pre>{text(rendered)}</pre>"


def link(href: str, label: str) -> str:
    return f'<a href="{text(href)}">{text(label)}</a>'
