"""Typed data contracts for offline EQR raw artifacts.

The resolver accepts the current local layout under ``data/`` while the target
canonical layout is documented as ``data/raw/eqr/<artifact_name>/`` for
partitioned vendor extracts and ``data/raw/eqr/<artifact_name>.parquet`` for
single-file artifacts.  Stale JSON manifests are intentionally advisory only;
the active pipeline resolves actual Parquet files present under the supplied
data directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class DateColumnSpec:
    """Date parsing contract for one column."""

    name: str
    nullable: bool = False


@dataclass(frozen=True)
class ArtifactContract:
    """Schema contract for one logical raw artifact."""

    artifact_name: str
    artifact_type: str
    current_locations: tuple[str, ...]
    canonical_location: str
    required_columns: tuple[str, ...]
    date_columns: tuple[DateColumnSpec, ...]
    key_columns: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class CRSPMonthly:
    """CRSP monthly stock file contract."""

    contract: ClassVar[ArtifactContract]


@dataclass(frozen=True)
class CRSPNames:
    """CRSP monthly security names contract."""

    contract: ClassVar[ArtifactContract]


@dataclass(frozen=True)
class CCMLink:
    """CRSP/Compustat merged link table contract."""

    contract: ClassVar[ArtifactContract]


@dataclass(frozen=True)
class CompCompany:
    """Compustat company metadata contract."""

    contract: ClassVar[ArtifactContract]


@dataclass(frozen=True)
class CompFundQ:
    """Compustat quarterly fundamentals contract."""

    contract: ClassVar[ArtifactContract]


@dataclass(frozen=True)
class IBESLink:
    """IBES to CRSP link contract."""

    contract: ClassVar[ArtifactContract]


@dataclass(frozen=True)
class IBESSummary:
    """IBES EPS summary statistics contract."""

    contract: ClassVar[ArtifactContract]


@dataclass(frozen=True)
class IBESDetail:
    """IBES EPS detailed estimates contract."""

    contract: ClassVar[ArtifactContract]


@dataclass(frozen=True)
class IBESActual:
    """IBES EPS actuals contract."""

    contract: ClassVar[ArtifactContract]


@dataclass(frozen=True)
class IBESTarget:
    """IBES price target summary contract."""

    contract: ClassVar[ArtifactContract]


@dataclass(frozen=True)
class MacroFeatures:
    """Macro feature table contract."""

    contract: ClassVar[ArtifactContract]


CRSP_MONTHLY = ArtifactContract(
    artifact_name="crsp_monthly",
    artifact_type="CRSPMonthly",
    current_locations=("monthly_crsp_msf_by_year", "monthly_crsp_msf.parquet", "crsp_msf.parquet"),
    canonical_location="data/raw/eqr/crsp_monthly/",
    required_columns=("permno", "permco", "date", "prc", "ret", "retx", "shrout", "vol", "cfacpr", "cfacshr", "market_cap"),
    date_columns=(DateColumnSpec("date"),),
    key_columns=("permno", "date"),
    description="CRSP monthly security returns and market data.",
)

CRSP_NAMES = ArtifactContract(
    artifact_name="crsp_names",
    artifact_type="CRSPNames",
    current_locations=("crsp_msenames.parquet", "crsp_names.parquet"),
    canonical_location="data/raw/eqr/crsp_names.parquet",
    required_columns=("permno", "comnam", "ncusip", "ticker", "shrcd", "exchcd", "namedt", "nameenddt"),
    date_columns=(DateColumnSpec("namedt"), DateColumnSpec("nameenddt")),
    key_columns=("permno", "namedt", "nameenddt"),
    description="CRSP security names and exchange/share-code history.",
)

CCM_LINK = ArtifactContract(
    artifact_name="ccm_link",
    artifact_type="CCMLink",
    current_locations=("ccm_linktable.parquet", "ccm_link.parquet"),
    canonical_location="data/raw/eqr/ccm_link.parquet",
    required_columns=("gvkey", "linkprim", "liid", "linktype", "lpermno", "lpermco", "usedflag", "linkdt", "linkenddt"),
    date_columns=(DateColumnSpec("linkdt"), DateColumnSpec("linkenddt", nullable=True)),
    key_columns=("gvkey", "liid", "lpermno", "linkdt", "linkenddt", "linktype", "linkprim"),
    description="CCM security-level link history between Compustat and CRSP.",
)

COMP_COMPANY = ArtifactContract(
    artifact_name="comp_company",
    artifact_type="CompCompany",
    current_locations=("comp_company.parquet",),
    canonical_location="data/raw/eqr/comp_company.parquet",
    required_columns=("gvkey", "conm", "cik", "sic", "naics", "gsector", "gind", "gsubind", "fic", "loc", "ipodate"),
    date_columns=(DateColumnSpec("ipodate", nullable=True),),
    key_columns=("gvkey",),
    description="Compustat company metadata.",
)

COMP_FUNDQ = ArtifactContract(
    artifact_name="comp_fundq",
    artifact_type="CompFundQ",
    current_locations=("comp_fundq_by_year", "comp_fundq.parquet"),
    canonical_location="data/raw/eqr/comp_fundq/",
    required_columns=(
        "gvkey",
        "datadate",
        "rdq",
        "fyearq",
        "fqtr",
        "fyr",
        "indfmt",
        "datafmt",
        "popsrc",
        "consol",
        "curcdq",
        "oiadpq",
        "niq",
        "ceqq",
        "oancfy",
        "revtq",
        "atq",
        "ltq",
        "dlttq",
        "dlcq",
        "cogsq",
        "xsgaq",
        "dpq",
        "pstkq",
        "txditcq",
        "cshoq",
        "prccq",
        "epspxq",
        "epsfxq",
        "saleq",
    ),
    date_columns=(DateColumnSpec("datadate"), DateColumnSpec("rdq", nullable=True)),
    key_columns=("gvkey", "datadate", "fyearq", "fqtr", "indfmt", "datafmt", "popsrc", "consol"),
    description="Compustat quarterly fundamentals.",
)

IBES_LINK = ArtifactContract(
    artifact_name="ibes_link",
    artifact_type="IBESLink",
    current_locations=("ibes_link.parquet",),
    canonical_location="data/raw/eqr/ibes_link.parquet",
    required_columns=("ticker", "permno", "ncusip", "sdate", "edate", "score"),
    date_columns=(DateColumnSpec("sdate"), DateColumnSpec("edate")),
    key_columns=("ticker", "permno", "sdate", "edate"),
    description="IBES ticker to CRSP PERMNO link history.",
)

IBES_SUMMARY = ArtifactContract(
    artifact_name="ibes_summary",
    artifact_type="IBESSummary",
    current_locations=("ibes_statsum_epsus_by_year", "ibes_statsum_epsus.parquet"),
    canonical_location="data/raw/eqr/ibes_summary/",
    required_columns=(
        "ticker",
        "cusip",
        "oftic",
        "cname",
        "statpers",
        "fiscalp",
        "measure",
        "fpi",
        "numest",
        "meanest",
        "medest",
        "stdev",
        "highest",
        "lowest",
        "actual",
        "usfirm",
        "fpedats",
        "actdats_act",
        "anndats_act",
    ),
    date_columns=(DateColumnSpec("statpers"), DateColumnSpec("fpedats"), DateColumnSpec("actdats_act", nullable=True), DateColumnSpec("anndats_act", nullable=True)),
    key_columns=("ticker", "statpers", "measure", "fpi", "fpedats"),
    description="IBES EPS summary statistics.",
)

IBES_DETAIL = ArtifactContract(
    artifact_name="ibes_detail",
    artifact_type="IBESDetail",
    current_locations=("ibes_det_epsus_by_year", "ibes_det_epsus.parquet"),
    canonical_location="data/raw/eqr/ibes_detail/",
    required_columns=(
        "ticker",
        "cusip",
        "oftic",
        "cname",
        "analys",
        "estimator",
        "measure",
        "fpi",
        "value",
        "curr",
        "usfirm",
        "anndats",
        "revdats",
        "fpedats",
        "actual",
        "anndats_act",
    ),
    date_columns=(DateColumnSpec("anndats"), DateColumnSpec("revdats"), DateColumnSpec("fpedats"), DateColumnSpec("anndats_act", nullable=True)),
    key_columns=("ticker", "analys", "estimator", "measure", "fpi", "fpedats", "anndats", "revdats"),
    description="IBES EPS analyst-level estimate detail.",
)

IBES_ACTUAL = ArtifactContract(
    artifact_name="ibes_actual",
    artifact_type="IBESActual",
    current_locations=("ibes_act_epsus_by_year", "ibes_act_epsus.parquet"),
    canonical_location="data/raw/eqr/ibes_actual/",
    required_columns=("ticker", "cusip", "oftic", "cname", "anndats", "measure", "pends", "value", "curr_act", "usfirm", "actdats"),
    date_columns=(DateColumnSpec("anndats"), DateColumnSpec("pends"), DateColumnSpec("actdats")),
    key_columns=("ticker", "measure", "pends", "anndats", "actdats"),
    description="IBES reported EPS actuals.",
)

IBES_TARGET = ArtifactContract(
    artifact_name="ibes_target",
    artifact_type="IBESTarget",
    current_locations=("ibes_ptgsum_by_year", "ibes_ptgsum.parquet"),
    canonical_location="data/raw/eqr/ibes_target/",
    required_columns=(
        "ticker",
        "cusip",
        "oftic",
        "cname",
        "statpers",
        "numest",
        "numup4w",
        "numdown4w",
        "numup1m",
        "numdown1m",
        "meanptg",
        "medptg",
        "stdev",
        "ptghigh",
        "ptglow",
        "curr",
        "usfirm",
        "measure",
    ),
    date_columns=(DateColumnSpec("statpers"),),
    key_columns=("ticker", "statpers", "measure"),
    description="IBES price target summary statistics.",
)

MACRO_FEATURES = ArtifactContract(
    artifact_name="macro_features",
    artifact_type="MacroFeatures",
    current_locations=("macro_features.parquet",),
    canonical_location="data/raw/eqr/macro_features.parquet",
    required_columns=("date", "sp500", "nasdaq", "treasury_2y", "treasury_10y"),
    date_columns=(DateColumnSpec("date"),),
    key_columns=("date",),
    description="Offline macro and market features.",
)


ARTIFACT_CONTRACTS: tuple[ArtifactContract, ...] = (
    CRSP_MONTHLY,
    CRSP_NAMES,
    CCM_LINK,
    COMP_COMPANY,
    COMP_FUNDQ,
    IBES_LINK,
    IBES_SUMMARY,
    IBES_DETAIL,
    IBES_ACTUAL,
    IBES_TARGET,
    MACRO_FEATURES,
)

CONTRACTS_BY_NAME: dict[str, ArtifactContract] = {contract.artifact_name: contract for contract in ARTIFACT_CONTRACTS}

REQUIRED_COLUMNS_BY_ARTIFACT: dict[str, tuple[str, ...]] = {
    contract.artifact_name: contract.required_columns for contract in ARTIFACT_CONTRACTS
}
DATE_COLUMNS_BY_ARTIFACT: dict[str, tuple[DateColumnSpec, ...]] = {
    contract.artifact_name: contract.date_columns for contract in ARTIFACT_CONTRACTS
}
KEY_COLUMNS_BY_ARTIFACT: dict[str, tuple[str, ...]] = {
    contract.artifact_name: contract.key_columns for contract in ARTIFACT_CONTRACTS
}

CRSPMonthly.contract = CRSP_MONTHLY
CRSPNames.contract = CRSP_NAMES
CCMLink.contract = CCM_LINK
CompCompany.contract = COMP_COMPANY
CompFundQ.contract = COMP_FUNDQ
IBESLink.contract = IBES_LINK
IBESSummary.contract = IBES_SUMMARY
IBESDetail.contract = IBES_DETAIL
IBESActual.contract = IBES_ACTUAL
IBESTarget.contract = IBES_TARGET
MacroFeatures.contract = MACRO_FEATURES


__all__ = [
    "ARTIFACT_CONTRACTS",
    "CCM_LINK",
    "CONTRACTS_BY_NAME",
    "CRSP_MONTHLY",
    "CRSP_NAMES",
    "COMP_COMPANY",
    "COMP_FUNDQ",
    "DATE_COLUMNS_BY_ARTIFACT",
    "IBES_ACTUAL",
    "IBES_DETAIL",
    "IBES_LINK",
    "IBES_SUMMARY",
    "IBES_TARGET",
    "KEY_COLUMNS_BY_ARTIFACT",
    "MACRO_FEATURES",
    "REQUIRED_COLUMNS_BY_ARTIFACT",
    "ArtifactContract",
    "CCMLink",
    "CRSPMonthly",
    "CRSPNames",
    "CompCompany",
    "CompFundQ",
    "DateColumnSpec",
    "IBESActual",
    "IBESDetail",
    "IBESLink",
    "IBESSummary",
    "IBESTarget",
    "MacroFeatures",
]
