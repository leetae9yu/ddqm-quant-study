# DDQM2 S&P 500 Data Inventory

This report consolidates the CRSP, Compustat, and IBES mapping notes for the DDQM2 S&P 500 adaptation. The key point is simple, each source plays a different role and they only work cleanly when joined through the right identifiers and date logic.

## Dependency chain

* **Compustat** provides company fundamentals, indexed by **GVKEY**.
* **CRSP** provides returns, prices, volume, and security labels, indexed by **PERMNO**.
* **IBES** provides analyst estimates, usually reached through **TICKER**, **CUSIP**, or **IKEY**.
* The **CCM link table** is what unites Compustat and CRSP, mapping **GVKEY** to **PERMNO** and **PERMCO** through historical link dates.

That chain matters because the model needs fundamentals, market data, and analyst expectations on the same security timeline. Do not join by ticker.

## S&P 500 universe constraint

This work is for the **S&P 500 universe only**, not the full CRSP universe. The universe has to be filtered first, then the data sources have to be aligned to that subset.

Use a membership or constituent filter, then map those names into CRSP with CCM. The correct security key for the panel is **PERMNO**, with **PERMCO** used only when you want company level consolidation across share classes.

## Source summary

### CRSP

CRSP is the return and trading data layer.

Core fields called out in the mapping notes:

* `permno`, `permco`
* `date`
* `prc`
* `ret`
* `vol`
* `shrout`
* `cfacpr`, `cfacshr`
* `dlret` when available

Use `crsp.dsf` for daily work and `crsp.msf` for monthly work. For labels, the minimum monthly set is `permno`, `date`, and `ret`, with forward shifting inside each `permno` panel.

### Compustat

Compustat is the fundamentals layer.

Primary tables:

* `comp.funda`
* `comp.fundq`
* `comp.ccmxpf_lnkhist`

Important filters for the fundamentals files are `indfmt = 'INDL'`, `datafmt = 'STD'`, `popsrc = 'D'`, and `consol = 'C'`.

The Compustat side supplies annual and quarterly fundamentals, plus point in time availability rules. Use `datadate` for the accounting period end, and `rdq` when it exists as the release date. If `rdq` is missing, apply a conservative lag and keep the data flagged as fallback timing.

### IBES

IBES is the analyst estimates layer.

Use IBES Summary History first, then Detail History when you need per analyst values.

Required estimate features include:

* consensus level
* dispersion
* revision counts
* coverage breadth
* optional high, low, median, and mean values when available

IBES security identifiers are not a substitute for CRSP identifiers. Map IBES to CRSP through `wrdsapps.ibcrsphist`, and only keep links that are valid on the estimate date.

## CCM link role

The CCM link table is the bridge between the accounting and market layers.

Use `comp.ccmxpf_lnkhist` with:

* `gvkey`
* `lpermno`
* `lpermco`
* `linktype`
* `linkprim`
* `linkdt`
* `linkenddt`

The working rule is to keep historically valid links, prefer primary links, and never rely on ticker joins.

## Fallback and export paths

The repo should treat WRDS as the source of record, then preserve separate fallback and prototype paths so public data never gets mixed into final research by accident.

### Manual WRDS export paths

Use separate landing zones for manual WRDS exports, one per source:

* `data/manual_exports/crsp/`
* `data/manual_exports/compustat/`
* `data/manual_exports/ibes/`

These paths are the handoff point for exported WRDS extracts when direct querying is not available in the current run.

### Public fallback and prototype paths

Keep the public fallback data separate from WRDS data:

* `data/public_fallback/`
* `data/public_fallback/yfinance/`
* `prototypes/yfinance_sp500/`

The `yfinance` path is only a prototype fallback. Use it to keep the workflow moving, then remap back to `PERMNO` and `PERMCO` when WRDS access returns.

## Fallback stack

1. **WRDS source tables** for the final research path.
2. **Manual WRDS exports** when direct access is blocked.
3. **Public fallback sources** for prototype work only.
4. **`yfinance` prototype path** for temporary CRSP-like price and volume coverage.

Treat anything in the fallback stack as `prototype_only` or equivalent source flagged data. Do not merge it into final backtests without a clear provenance flag.

## Bottom line

The DDQM2 S&P 500 build depends on a clean chain: Compustat fundamentals by GVKEY, CRSP market data by PERMNO, IBES estimates by TICKER or CUSIP through a historical link, and CCM as the bridge. The universe must be filtered down to S&P 500 names first, then mapped through historical links, not pulled from the full CRSP population.
