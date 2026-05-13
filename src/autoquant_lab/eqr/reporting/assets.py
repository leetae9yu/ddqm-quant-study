"""Inline static assets for the EQR reporting site."""

from __future__ import annotations


SITE_CSS = """
:root {
  --color-bg: #0f172a;
  --color-surface: #111c33;
  --color-panel: #18243d;
  --color-panel-soft: #23314f;
  --color-text: #edf2ff;
  --color-muted: #aab7d4;
  --color-border: #33415f;
  --color-primary: #7dd3fc;
  --color-primary-strong: #38bdf8;
  --color-accent: #fbbf24;
  --color-success: #86efac;
  --color-warning: #fde68a;
  --color-danger: #fda4af;
  --space-1: 0.25rem;
  --space-2: 0.5rem;
  --space-3: 0.75rem;
  --space-4: 1rem;
  --space-5: 1.5rem;
  --space-6: 2rem;
  --space-7: 3rem;
  --radius-sm: 0.45rem;
  --radius-md: 0.75rem;
  --radius-lg: 1rem;
  --shadow-card: 0 1rem 2.5rem rgb(2 6 23 / 0.28);
  --font-body: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  --font-mono: "SFMono-Regular", "Cascadia Code", Consolas, monospace;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  min-height: 100vh;
  color: var(--color-text);
  background:
    radial-gradient(circle at top left, rgb(56 189 248 / 0.18), transparent 28rem),
    linear-gradient(135deg, var(--color-bg), #172554 48%, var(--color-bg));
  font-family: var(--font-body);
  line-height: 1.55;
}

a { color: var(--color-primary); text-decoration: none; }
a:hover { color: var(--color-primary-strong); text-decoration: underline; }

.shell { width: min(1180px, calc(100% - var(--space-6))); margin: 0 auto; padding: var(--space-6) 0 var(--space-7); }
.topbar { display: flex; justify-content: space-between; gap: var(--space-4); align-items: center; margin-bottom: var(--space-6); }
.brand { font-weight: 800; letter-spacing: 0.04em; text-transform: uppercase; color: var(--color-accent); }
.nav { display: flex; flex-wrap: wrap; gap: var(--space-3); color: var(--color-muted); }
.hero { padding: var(--space-6); border: 1px solid var(--color-border); border-radius: var(--radius-lg); background: linear-gradient(135deg, rgb(24 36 61 / 0.94), rgb(17 28 51 / 0.82)); box-shadow: var(--shadow-card); margin-bottom: var(--space-6); }
.hero h1 { margin: 0 0 var(--space-2); font-size: clamp(2rem, 6vw, 4.2rem); line-height: 0.95; letter-spacing: -0.06em; }
.lede { color: var(--color-muted); max-width: 72ch; margin: 0; }
.grid { display: grid; gap: var(--space-4); }
.grid.cards { grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); }
.card, .panel { border: 1px solid var(--color-border); border-radius: var(--radius-md); background: rgb(24 36 61 / 0.88); padding: var(--space-5); box-shadow: var(--shadow-card); }
.metric { color: var(--color-muted); text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.78rem; }
.metric-value { font-size: 2rem; font-weight: 800; margin-top: var(--space-1); }
.section { margin-top: var(--space-6); }
.section h2 { margin: 0 0 var(--space-3); font-size: 1.35rem; }
.table-wrap { overflow-x: auto; border-radius: var(--radius-md); border: 1px solid var(--color-border); }
table { width: 100%; border-collapse: collapse; background: rgb(15 23 42 / 0.42); }
th, td { padding: var(--space-3) var(--space-4); text-align: left; border-bottom: 1px solid var(--color-border); vertical-align: top; }
th { color: var(--color-muted); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; }
tr:last-child td { border-bottom: 0; }
.badge { display: inline-flex; align-items: center; gap: var(--space-1); border-radius: 999px; padding: var(--space-1) var(--space-2); background: var(--color-panel-soft); color: var(--color-text); font-size: 0.82rem; }
.badge.success { color: var(--color-success); }
.badge.warning { color: var(--color-warning); }
.badge.danger { color: var(--color-danger); }
pre, code { font-family: var(--font-mono); }
pre { overflow-x: auto; margin: 0; padding: var(--space-4); border-radius: var(--radius-sm); background: rgb(2 6 23 / 0.48); border: 1px solid var(--color-border); color: var(--color-text); }
.sparkline { width: 100%; height: 180px; border-radius: var(--radius-md); background: rgb(2 6 23 / 0.32); border: 1px solid var(--color-border); }
.muted { color: var(--color-muted); }
.split { display: grid; grid-template-columns: minmax(0, 1.2fr) minmax(260px, 0.8fr); gap: var(--space-4); }
.footer { margin-top: var(--space-7); color: var(--color-muted); font-size: 0.9rem; }
@media (max-width: 800px) { .topbar, .split { grid-template-columns: 1fr; display: grid; } .hero { padding: var(--space-5); } }
""".strip()
