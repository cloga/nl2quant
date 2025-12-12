from __future__ import annotations

from app.data.valuation import get_dynamic_pe_latest


def test_get_dynamic_pe_latest_smoke():
    # Requires TUSHARE_TOKEN in environment.
    point = get_dynamic_pe_latest("600519.SH")
    assert point.ts_code == "600519.SH"
    assert len(point.trade_date) == 8
    # pe_ttm can be None if company is loss-making or data missing
