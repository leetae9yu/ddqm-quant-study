# WRDS manual export guide

Use this guide when `wrds.Connection()` fails and you still need to move data into the Python pipeline.

## 1. Start with a quick check

1. Retry the job once.
2. Confirm your WRDS username, password, VPN, and network access.
3. If the API still fails, switch to the WRDS web interface and export the tables manually.
4. Keep the manual export tagged as a fallback source so it does not get mixed with normal API pulls.

## 2. Web interface paths to use

Use the WRDS web portal and navigate to these paths:

| Dataset | WRDS web path |
| --- | --- |
| CRSP Monthly Stock | `Data` > `CRSP` > `Stock / Security Files` > `Monthly Stock` |
| Compustat Fundamentals Annual | `Data` > `Compustat` > `North America` > `Fundamentals Annual` |
| Compustat Fundamentals Quarterly | `Data` > `Compustat` > `North America` > `Fundamentals Quarterly` |
| IBES Summary History | `Data` > `IBES` > `Summary History` |
| CRSP / Compustat Merged | `Data` > `CRSP / Compustat Merged` > `CCM Link History` |

If the menu labels vary slightly, look for the table name itself, especially `Monthly Stock`, `Fundamentals Annual`, `Fundamentals Quarterly`, `Summary History`, and `CCM Link History`.

## 3. Export settings the pipeline expects

### Date format

Always select or enter dates as `YYYYMMDD`.

Use that format for:

- date filters in the WRDS query form
- export file columns that store dates
- any manual notes you keep beside the download

Do not export dates as `MM/DD/YYYY` or text like `Jan 31 2020`. The Python pipeline parses `YYYYMMDD` cleanly.

### File format

Prefer `CSV` first.

If CSV is not available, use `SAS` output, then convert it in Python with a reader such as `pyreadstat`.

Avoid Excel files for the pipeline unless there is no other option.

## 4. Manual fallback process

### Step 1, download CRSP Monthly Stock

1. Open the WRDS `Monthly Stock` table.
2. Set the date range you need.
3. Export as `CSV`.
4. Keep date fields in `YYYYMMDD`.
5. Save the file with a clear name, such as `crsp_monthly_stock_YYYYMMDD_YYYYMMDD.csv`.

### Step 2, download Compustat fundamentals

1. Open `Fundamentals Annual`.
2. Export the same date range if possible.
3. Repeat for `Fundamentals Quarterly`.
4. Use `CSV` and `YYYYMMDD` for all date fields.
5. Keep the annual and quarterly files separate.

### Step 3, download IBES Summary History

1. Open `Summary History` under IBES.
2. Export the needed estimate window.
3. Use `CSV` if possible.
4. Keep the date or period stamp in `YYYYMMDD` when the portal lets you choose the output format.
5. Preserve any fiscal period fields exactly as WRDS supplies them.

### Step 4, download CRSP / Compustat Merged link history

1. Open the `CRSP / Compustat Merged` area.
2. Download `CCM Link History` manually.
3. Export as `CSV` or `SAS`.
4. Keep the link dates in `YYYYMMDD`.
5. Use this file to resolve `GVKEY` to `PERMNO` and `PERMCO` links.

### Step 5, stage the files for Python

1. Put every export in a single staging folder.
2. Rename each file so the source table is obvious.
3. Load CSV files directly with pandas.
4. Load SAS files with a SAS reader if needed.
5. Check that `GVKEY`, `PERMNO`, `PERMCO`, `DATADATE`, `RDQ`, and other date columns still parse as dates or date-like strings.

## 5. Practical parsing notes

- CRSP monthly data should join on `PERMNO`, not `ticker`.
- Compustat annual and quarterly data should stay separate until your pipeline decides how to align them.
- The CCM link history file is the bridge between Compustat `GVKEY` and CRSP `PERMNO`.
- Keep the export date window and file name in a small text note so you can rebuild the same dataset later.

## 6. Recommended fallback order

1. `wrds.Connection()`
2. WRDS web export in `CSV`
3. WRDS web export in `SAS`
4. Parse in Python and keep the source marked as manual export

If the manual path works, it is still a fallback. Re-sync from the API later when WRDS access is restored.
