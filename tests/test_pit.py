from __future__ import annotations
# pyright: reportMissingImports=false

from pathlib import Path
import sys

import pandas as pd

from autoquant_lab.eqr.pit import CCMLinkResolver, CRSPNameResolver, EntityMapper, IBESLinkResolver, overlap_count


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import eqr_build_links  # noqa: E402


def _crsp_names() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "permno": [10001, 10002, 10003, 10004],
            "comnam": ["A", "B", "C", "D"],
            "ncusip": ["11111111", "22222222", "33333333", "44444444"],
            "ticker": ["AAA", "BBB", "CCC", "DDD"],
            "shrcd": [10, 11, 12, 10],
            "exchcd": [1, 3, 1, 4],
            "namedt": pd.to_datetime(["2000-01-01", "2000-01-01", "2000-01-01", "2000-01-01"]),
            "nameenddt": pd.to_datetime([None, "2005-12-31", None, None]),
        }
    )


def test_open_ended_linkenddt_handling() -> None:
    resolver = CCMLinkResolver(
        pd.DataFrame(
            {
                "gvkey": ["001004"],
                "linkprim": ["P"],
                "liid": ["01"],
                "linktype": ["LU"],
                "lpermno": [10001],
                "lpermco": [20000],
                "usedflag": [1],
                "linkdt": pd.to_datetime(["2000-01-01"]),
                "linkenddt": pd.to_datetime([None]),
            }
        )
    )

    assert resolver.resolve("001004", "2020-06-30") == 10001


def test_expired_links_are_not_returned_after_end_date() -> None:
    resolver = CCMLinkResolver(
        pd.DataFrame(
            {
                "gvkey": ["001009"],
                "linkprim": ["P"],
                "liid": ["01"],
                "linktype": ["LC"],
                "lpermno": [10002],
                "lpermco": [20001],
                "usedflag": [1],
                "linkdt": pd.to_datetime(["2000-01-01"]),
                "linkenddt": pd.to_datetime(["2001-12-31"]),
            }
        )
    )

    assert resolver.resolve("001009", "2002-01-01") is None


def test_overlapping_links_are_reported_as_duplicate_active_links() -> None:
    links = pd.DataFrame(
        {
            "gvkey": ["001010", "001010"],
            "linkprim": ["P", "C"],
            "liid": ["01", "01"],
            "linktype": ["LC", "LU"],
            "lpermno": [10001, 10002],
            "lpermco": [20000, 20001],
            "usedflag": [1, 1],
            "linkdt": pd.to_datetime(["2000-01-01", "2000-06-01"]),
            "linkenddt": pd.to_datetime(["2001-12-31", "2002-12-31"]),
        }
    )

    resolver = CCMLinkResolver(links)

    assert resolver.resolve("001010", "2000-07-01") == 10001
    assert resolver.diagnostics()["duplicate_active_link_count"] == 1
    assert overlap_count(links, ["gvkey"], "linkdt", "linkenddt") == 1


def test_ticker_reuse_resolves_different_permnos_by_date() -> None:
    resolver = IBESLinkResolver(
        pd.DataFrame(
            {
                "ticker": ["XYZ", "XYZ"],
                "permno": [10001, 10002],
                "ncusip": ["11111111", "22222222"],
                "sdate": pd.to_datetime(["2000-01-01", "2005-01-01"]),
                "edate": pd.to_datetime(["2004-12-31", None]),
                "score": [0, 1],
            }
        )
    )

    assert resolver.resolve("xyz", "2001-06-30") == 10001
    assert resolver.resolve("XYZ", "2006-06-30") == 10002


def test_crsp_filter_correctness_for_mapper_and_build_links() -> None:
    names = _crsp_names()
    name_resolver = CRSPNameResolver(names)
    mapper = EntityMapper(
        crsp_names=name_resolver,
        ccm_links=CCMLinkResolver(
            pd.DataFrame(
                {
                    "gvkey": ["GOOD", "BAD_SHR", "BAD_EXCH"],
                    "linkprim": ["P", "P", "P"],
                    "liid": ["01", "01", "01"],
                    "linktype": ["LC", "LC", "LC"],
                    "lpermno": [10001, 10003, 10004],
                    "lpermco": [1, 2, 3],
                    "usedflag": [1, 1, 1],
                    "linkdt": pd.to_datetime(["2000-01-01", "2000-01-01", "2000-01-01"]),
                    "linkenddt": pd.to_datetime([None, None, None]),
                }
            )
        ),
        ibes_links=IBESLinkResolver(
            pd.DataFrame(
                {
                    "ticker": ["AAA"],
                    "permno": [10001],
                    "ncusip": ["11111111"],
                    "sdate": pd.to_datetime(["2000-01-01"]),
                    "edate": pd.to_datetime([None]),
                    "score": [1],
                }
            )
        ),
    )

    assert mapper.resolve_permno("GOOD", "2001-01-31", "gvkey") == 10001
    assert mapper.resolve_permno("BAD_SHR", "2001-01-31", "gvkey") is None
    assert mapper.resolve_permno("BAD_EXCH", "2001-01-31", "gvkey") is None

    ccm_resolved, diagnostics = eqr_build_links.build_ccm_links(mapper.ccm_links.links, eqr_build_links._prepare_crsp_names(names))
    assert set(ccm_resolved["gvkey"]) == {"GOOD"}
    assert diagnostics["unmatched_count"] == 2


def test_ibes_score_filtering() -> None:
    links = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "permno": [10001, 10002],
            "ncusip": ["11111111", "22222222"],
            "sdate": pd.to_datetime(["2000-01-01", "2000-01-01"]),
            "edate": pd.to_datetime([None, None]),
            "score": [1, 2],
        }
    )
    resolver = IBESLinkResolver(links)

    assert resolver.resolve("AAA", "2001-01-31") == 10001
    assert resolver.resolve("BBB", "2001-01-31") is None

    resolved, diagnostics = eqr_build_links.build_ibes_links(links, eqr_build_links._prepare_crsp_names(_crsp_names()))
    assert set(resolved["ibes_ticker"]) == {"AAA"}
    assert diagnostics["score_filtered_out_count"] == 1
