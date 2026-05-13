"""Machine-readable EQR/DDQM2 factor definitions.

The list mirrors the 55-factor taxonomy in ``EQR.md``. Some factors are exact
implementations from the currently available WRDS/FRED artifact set, while
others are explicit proxies or marked unavailable when the source data is not
present (for example, retail/investor flow).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class FactorDefinition:
    factor_id: str
    name_ko: str
    family: str
    scope: str
    source_column: str | None
    direction: float
    status: str
    description: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _f(
    factor_id: str,
    name_ko: str,
    family: str,
    scope: str,
    source_column: str | None,
    direction: float,
    status: str,
    description: str,
) -> FactorDefinition:
    return FactorDefinition(factor_id, name_ko, family, scope, source_column, direction, status, description)


def all_factor_definitions() -> tuple[FactorDefinition, ...]:
    """Return the EQR.md 55-factor registry.

    ``status`` values:
    - ``implemented``: directly computed from available columns.
    - ``proxy``: computed with a documented proxy using current data.
    - ``unavailable``: source data is absent in current local WRDS/FRED set.
    """

    return (
        _f("val_global_relative_pe_industry_fwd", "상대P/E (업종 대비, forward)", "valuation", "global", "compustat__pe_proxy", -1.0, "proxy", "Forward industry-relative P/E approximated by PIT Compustat trailing earnings proxy."),
        _f("val_global_relative_pb_industry_fwd", "상대P/B (업종 대비, forward)", "valuation", "global", "compustat__pb", -1.0, "proxy", "Forward industry-relative P/B approximated by current PIT book-value proxy."),
        _f("val_global_pc_ttm", "P/C (직전4분기, 개별)", "valuation", "global", "compustat__debt_to_assets", -1.0, "proxy", "Cash-flow valuation proxy unavailable; lower leverage proxy is used as conservative substitute."),
        _f("val_global_pb_fwd", "P/B (forward)", "valuation", "global", "compustat__pb", -1.0, "proxy", "Forward P/B approximated by PIT book-value proxy."),
        _f("val_global_pe_fwd", "P/E (forward)", "valuation", "global", "compustat__pe_proxy", -1.0, "proxy", "Forward P/E approximated by PIT trailing earnings proxy."),
        _f("val_local_relative_pe_industry_fwd", "상대P/E (업종 대비, forward)", "valuation", "local", "compustat__pe_proxy", -1.0, "proxy", "Local peer-relative P/E proxy; peer group falls back to exchange when sector unavailable."),
        _f("val_local_pb_fwd", "P/B (forward)", "valuation", "local", "compustat__pb", -1.0, "proxy", "Local forward P/B proxy."),
        _f("val_local_pb_lastq", "P/B (직전분기)", "valuation", "local", "compustat__pb", -1.0, "implemented", "P/B using last PIT quarterly common equity."),
        _f("val_local_pe_fwd", "P/E (forward)", "valuation", "local", "compustat__pe_proxy", -1.0, "proxy", "Local P/E proxy."),
        _f("val_local_relative_pb_industry_fwd", "상대P/B (업종 대비, forward)", "valuation", "local", "compustat__pb", -1.0, "proxy", "Local peer-relative P/B proxy."),

        _f("earn_global_op_income_change_fy2_1m_voladj", "변동성조정 영업이익 변화율 (FY2, 1m)", "earnings", "global", "ibes__revision_1m", 1.0, "proxy", "IBES revision balance proxy for unavailable OPR FY2 volatility-adjusted revision."),
        _f("earn_global_op_income_change_fq1_1m", "영업이익 변화율 (FQ1, 1m)", "earnings", "global", "ibes__revision_1m", 1.0, "proxy", "IBES detail revision proxy."),
        _f("earn_global_eps_fast_fy1_1m_voladj", "변동성조정 EPS 변화율 (빠른 FY1, 1m)", "earnings", "global", "ibes__revision_1m", 1.0, "implemented", "EPS estimate revision balance from IBES detail."),
        _f("earn_global_eps_fast_fy1_1m", "EPS 변화율 (빠른 FY1, 1m)", "earnings", "global", "ibes__revision_1m", 1.0, "implemented", "EPS estimate revision balance from IBES detail."),
        _f("earn_global_eps_fq1_1m", "EPS 변화율 (FQ1, 1m)", "earnings", "global", "ibes__revision_1m", 1.0, "proxy", "EPS FQ1 revision proxy from aggregate revision balance."),
        _f("earn_global_eps_fy2_1m_voladj", "변동성조정 EPS 변화율 (FY2, 1m)", "earnings", "global", "ibes__revision_1m", 1.0, "proxy", "EPS FY2 volatility-adjusted revision proxy."),
        _f("earn_global_err_fast_eps1_1m", "이익조정비율 (빠른 EPS1, 1m)", "earnings", "global", "ibes__revision_1m", 1.0, "implemented", "Revision balance approximates earnings revision ratio."),
        _f("earn_global_op_income_change_fy1_1m", "영업이익 변화율 (FY1, 1m)", "earnings", "global", "ibes__revision_1m", 1.0, "proxy", "OP estimate proxy from IBES EPS revision direction."),
        _f("earn_global_time_adjusted_ni_surprise_fqo", "시간조정 순이익 서프라이즈 (FQO)", "earnings", "global", "ibes__surprise", 1.0, "implemented", "IBES actual minus latest consensus proxy."),
        _f("earn_global_op_income_change_fy1_1m_voladj", "변동성조정 영업이익 변화율 (FY1, 1m)", "earnings", "global", "ibes__estimate_dispersion", -1.0, "proxy", "Lower estimate dispersion used as volatility-adjusted earnings confidence proxy."),
        _f("earn_global_op_income_change_fy2_1m", "영업이익 변화율 (FY2, 1m)", "earnings", "global", "ibes__revision_1m", 1.0, "proxy", "OP FY2 revision proxy."),
        _f("earn_global_ni_surprise_fqo", "순이익 서프라이즈 (FQO)", "earnings", "global", "ibes__surprise", 1.0, "implemented", "IBES surprise proxy."),
        _f("earn_global_eps_fy2_1m", "EPS 변화율 (FY2, 1m)", "earnings", "global", "ibes__revision_1m", 1.0, "proxy", "EPS FY2 revision proxy."),
        _f("earn_global_eps_fq2_1m", "EPS 변화율 (FQ2, 1m)", "earnings", "global", "ibes__revision_1m", 1.0, "proxy", "EPS FQ2 revision proxy."),
        _f("earn_global_target_price_gap_1m", "목표가 괴리율 (1개월컨센)", "earnings", "global", "ibes__target_price_mean", 1.0, "implemented", "IBES mean target-price level proxy."),
        _f("earn_global_op_income_change_fq2_1m", "영업이익 변화율 (FQ2, 1m)", "earnings", "global", "ibes__revision_1m", 1.0, "proxy", "OP FQ2 revision proxy."),

        _f("earn_local_eps_fq1_1m", "EPS 변화율 (FQ1, 1m)", "earnings", "local", "ibes__revision_1m", 1.0, "proxy", "Local EPS FQ1 revision proxy."),
        _f("earn_local_ni_surprise_fqo", "순이익 서프라이즈 (FQO)", "earnings", "local", "ibes__surprise", 1.0, "implemented", "Local IBES surprise proxy."),
        _f("earn_local_target_price_gap_1m", "목표가 괴리율 (1개월컨센)", "earnings", "local", "ibes__target_price_mean", 1.0, "implemented", "Local target price proxy."),
        _f("earn_local_target_price_gap_3m", "목표가 괴리율 (3개월컨센)", "earnings", "local", "ibes__target_revision_balance_1m", 1.0, "proxy", "Three-month target gap proxied by target revision balance."),
        _f("earn_local_op_income_change_fy2_1m_voladj", "변동성조정 영업이익 변화율 (FY2, 1m)", "earnings", "local", "ibes__estimate_dispersion", -1.0, "proxy", "Local lower dispersion proxy."),
        _f("earn_local_time_adjusted_ni_surprise_fqo", "시간조정 순이익 서프라이즈 (FQO)", "earnings", "local", "ibes__surprise", 1.0, "implemented", "Local time-adjusted surprise proxy."),
        _f("earn_local_eps_fast_fy1_1m_voladj", "변동성조정 EPS 변화율 (빠른 FY1, 1m)", "earnings", "local", "ibes__revision_1m", 1.0, "proxy", "Local EPS revision proxy."),
        _f("earn_local_op_income_change_fq1_1m", "영업이익 변화율 (FQ1, 1m)", "earnings", "local", "ibes__revision_1m", 1.0, "proxy", "Local OP FQ1 revision proxy."),
        _f("earn_local_eps_fast_fy1_1m", "EPS 변화율 (빠른 FY1, 1m)", "earnings", "local", "ibes__revision_1m", 1.0, "implemented", "Local EPS revision proxy."),
        _f("earn_local_eps_fy2_1m_voladj", "변동성조정 EPS 변화율 (FY2, 1m)", "earnings", "local", "ibes__revision_1m", 1.0, "proxy", "Local EPS FY2 vol-adjusted proxy."),
        _f("earn_local_op_income_change_fy1_1m", "영업이익 변화율 (FY1, 1m)", "earnings", "local", "ibes__revision_1m", 1.0, "proxy", "Local OP FY1 proxy."),
        _f("earn_local_op_income_surprise_fqo", "영업이익 서프라이즈 (FQO)", "earnings", "local", "ibes__surprise", 1.0, "proxy", "Local OP surprise proxy from IBES EPS actual surprise."),
        _f("earn_local_op_income_change_fy2_1m", "영업이익 변화율 (FY2, 1m)", "earnings", "local", "ibes__revision_1m", 1.0, "proxy", "Local OP FY2 proxy."),
        _f("earn_local_err_fast_eps1_1m", "이익조정비율 (빠른 EPS1, 1m)", "earnings", "local", "ibes__revision_1m", 1.0, "implemented", "Local earnings revision ratio proxy."),
        _f("earn_local_eps_fy2_1m", "EPS 변화율 (FY2, 1m)", "earnings", "local", "ibes__revision_1m", 1.0, "proxy", "Local EPS FY2 proxy."),
        _f("earn_local_op_income_change_fq2_1m", "영업이익 변화율 (FQ2, 1m)", "earnings", "local", "ibes__revision_1m", 1.0, "proxy", "Local OP FQ2 proxy."),
        _f("earn_local_op_income_change_fy1_1m_voladj", "변동성조정 영업이익 변화율 (FY1, 1m)", "earnings", "local", "ibes__estimate_dispersion", -1.0, "proxy", "Local OP FY1 vol-adjusted proxy."),

        _f("quality_global_operating_income_yoy", "영업이익 증가율 (FQO, yoy)", "quality_growth", "global", "compustat__revenue_yoy", 1.0, "proxy", "Operating-income YoY unavailable in current builder; revenue YoY proxy."),
        _f("quality_global_net_income_yoy", "순이익 증가율 (FQO, yoy)", "quality_growth", "global", "compustat__net_income_yoy", 1.0, "implemented", "Net income YoY growth from Compustat."),
        _f("quality_local_operating_income_yoy", "영업이익 증가율 (FQO, yoy)", "quality_growth", "local", "compustat__revenue_yoy", 1.0, "proxy", "Local revenue YoY proxy."),
        _f("quality_local_net_income_yoy", "순이익 증가율 (FQO, yoy)", "quality_growth", "local", "compustat__net_income_yoy", 1.0, "implemented", "Local net income YoY growth."),

        _f("price_local_momentum_12m_1m", "주가 모멘텀 (12m - 1m)", "price_momentum", "local", "crsp__mom_12_2", 1.0, "implemented", "12-to-2 month price momentum."),
        _f("price_local_momentum_3m", "주가 모멘텀 (3m)", "price_momentum", "local", "crsp__mom_6_2", 1.0, "proxy", "3m momentum proxied by 6-to-2 month momentum from monthly data."),
        _f("reversal_local_price_reversal_3m", "주가 리버전 (3m)", "reversal", "local", "crsp__mom_6_2", -1.0, "proxy", "3m reversal proxied by negative intermediate momentum."),
        _f("reversal_local_ma_gap_3m", "이격도 리버전 (3m)", "reversal", "local", "crsp__mom_6_2", -1.0, "proxy", "Monthly-only MA gap proxy."),
        _f("reversal_local_ma_gap_1m", "이격도 리버전 (1m)", "reversal", "local", "crsp__reversal_1m", -1.0, "proxy", "1m reversal using latest monthly return."),
        _f("reversal_local_price_reversal_1m", "주가 리버전 (1m)", "reversal", "local", "crsp__reversal_1m", -1.0, "implemented", "Negative latest monthly return."),
        _f("size_local_small_size", "소형주", "size_flow", "local", "crsp__log_size", -1.0, "implemented", "Small size via negative log market cap."),
        _f("flow_local_retail_net_buy_1m", "개인 순매수 (1m)", "size_flow", "local", None, 1.0, "unavailable", "Retail/investor-flow source is not present in current WRDS/FRED artifact set."),
    )


def implemented_factor_definitions() -> tuple[FactorDefinition, ...]:
    return tuple(factor for factor in all_factor_definitions() if factor.source_column and factor.status in {"implemented", "proxy"})
