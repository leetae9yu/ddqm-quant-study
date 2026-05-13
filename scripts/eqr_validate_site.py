#!/usr/bin/env python3
"""Validate the generated EQR static site."""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
from pathlib import Path
import re
from urllib.parse import urldefrag, urlparse


SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("assignment_secret", re.compile(r"(?i)\b(?:api[_-]?key|secret|token|password|passwd|pwd)\b\s*[:=]\s*['\"]?([^'\"\s<]{8,})")),
    ("cloud_key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
)


class SiteHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []
        self.has_html = False
        self.has_title = False
        self.has_body = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "html":
            self.has_html = True
        if tag == "body":
            self.has_body = True
        if tag == "a":
            for key, value in attrs:
                if key == "href" and value:
                    self.links.append(value)
        if tag == "title":
            self.has_title = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate local links, HTML structure, and secret patterns in a static site.")
    parser.add_argument("site", nargs="?", type=Path, default=Path("site"), help="Static site directory to validate.")
    return parser.parse_args()


def validate_site(site_dir: Path) -> list[str]:
    errors: list[str] = []
    if not site_dir.exists() or not site_dir.is_dir():
        return [f"Site directory does not exist: {site_dir}"]
    html_files = sorted(site_dir.glob("*.html"))
    if not html_files:
        errors.append("No HTML files found")
    for path in html_files:
        content = path.read_text(encoding="utf-8", errors="ignore")
        parser = SiteHTMLParser()
        parser.feed(content)
        if not (parser.has_html and parser.has_title and parser.has_body):
            errors.append(f"Invalid HTML structure: {path.name}")
        errors.extend(_secret_errors(path, content))
        errors.extend(_link_errors(site_dir, path, parser.links))
    return errors


def _secret_errors(path: Path, content: str) -> list[str]:
    errors: list[str] = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        for name, pattern in SECRET_PATTERNS:
            if pattern.search(line):
                errors.append(f"Secret-like token {name} in {path.name}:{line_number}")
    return errors


def _link_errors(site_dir: Path, source: Path, links: list[str]) -> list[str]:
    errors: list[str] = []
    for href in links:
        parsed = urlparse(href)
        if parsed.scheme or parsed.netloc or href.startswith("mailto:"):
            continue
        target_name, fragment = urldefrag(href)
        if not target_name:
            continue
        target = (source.parent / target_name).resolve()
        try:
            target.relative_to(site_dir.resolve())
        except ValueError:
            errors.append(f"Link escapes site root in {source.name}: {href}")
            continue
        if not target.exists():
            errors.append(f"Broken local link in {source.name}: {href}")
        if fragment and target.exists():
            target_content = target.read_text(encoding="utf-8", errors="ignore")
            if f'id="{fragment}"' not in target_content and f"id='{fragment}'" not in target_content:
                errors.append(f"Missing fragment in {source.name}: {href}")
    return errors


def main() -> int:
    args = parse_args()
    errors = validate_site(args.site)
    if errors:
        for error in errors:
            print(error)
        return 1
    print(f"Validated {args.site}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
