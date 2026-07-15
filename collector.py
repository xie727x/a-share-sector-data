"""Build a public, no-secret evidence pack for A-share industry-board research."""

from __future__ import annotations

import json
import math
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import akshare as ak
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
HISTORY_DIR = DATA_DIR / "history"
TZ = ZoneInfo("Asia/Shanghai")
HISTORY_START_DAYS = 430


def now_cn() -> datetime:
    return datetime.now(TZ)


def json_value(value: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def retry(operation: Callable[[], Any], attempts: int = 3) -> Any:
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as exc:  # Public sources can transiently throttle requests.
            error = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(str(error)) from error


def first_column(frame: pd.DataFrame, choices: list[str]) -> str | None:
    return next((item for item in choices if item in frame.columns), None)


def safe_filename(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "_", name)


def load_history(board: str, today: datetime) -> pd.DataFrame:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    path = HISTORY_DIR / f"{safe_filename(board)}.csv"
    start = (today - timedelta(days=HISTORY_START_DAYS)).strftime("%Y%m%d")

    existing = pd.DataFrame()
    if path.exists():
        existing = pd.read_csv(path)
        if not existing.empty and "日期" in existing.columns:
            last = pd.to_datetime(existing["日期"], errors="coerce").max()
            if pd.notna(last):
                start = (last - timedelta(days=7)).strftime("%Y%m%d")

    fetched = retry(
        lambda: ak.stock_board_industry_hist_em(
            symbol=board,
            period="日k",
            start_date=start,
            end_date=today.strftime("%Y%m%d"),
            adjust="",
        )
    )
    if fetched.empty:
        return existing

    merged = pd.concat([existing, fetched], ignore_index=True)
    if "日期" not in merged.columns:
        raise RuntimeError(f"{board} history has no date column")
    merged["日期"] = pd.to_datetime(merged["日期"], errors="coerce")
    merged = merged.dropna(subset=["日期"]).drop_duplicates(subset=["日期"], keep="last")
    merged = merged.sort_values("日期")
    merged.to_csv(path, index=False)
    return merged


def fetch_benchmark(today: datetime) -> pd.DataFrame:
    path = HISTORY_DIR / "csi300.csv"
    start = (today - timedelta(days=HISTORY_START_DAYS)).strftime("%Y%m%d")
    existing = pd.DataFrame()
    if path.exists():
        existing = pd.read_csv(path)
        date_col = first_column(existing, ["日期", "date"])
        if date_col:
            last = pd.to_datetime(existing[date_col], errors="coerce").max()
            if pd.notna(last):
                start = (last - timedelta(days=7)).strftime("%Y%m%d")

    fetched = retry(
        lambda: ak.index_zh_a_hist(
            symbol="000300",
            period="daily",
            start_date=start,
            end_date=today.strftime("%Y%m%d"),
        )
    )
    merged = pd.concat([existing, fetched], ignore_index=True)
    date_col = first_column(merged, ["日期", "date"])
    if not date_col:
        raise RuntimeError("CSI 300 history has no date column")
    merged[date_col] = pd.to_datetime(merged[date_col], errors="coerce")
    merged = merged.dropna(subset=[date_col]).drop_duplicates(subset=[date_col], keep="last")
    merged = merged.sort_values(date_col)
    merged.to_csv(path, index=False)
    return merged


def pct_change(series: pd.Series, days: int) -> float | None:
    if len(series) <= days or series.iloc[-days - 1] == 0:
        return None
    return float((series.iloc[-1] / series.iloc[-days - 1] - 1) * 100)


def rsi(series: pd.Series, period: int = 14) -> float | None:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    if len(loss) == 0 or pd.isna(loss.iloc[-1]) or loss.iloc[-1] == 0:
        return None
    value = 100 - 100 / (1 + gain.iloc[-1] / loss.iloc[-1])
    return float(value)


def technical_metrics(history: pd.DataFrame, benchmark: pd.DataFrame, today: datetime) -> dict[str, Any]:
    date_col = first_column(history, ["日期", "date"])
    close_col = first_column(history, ["收盘", "close"])
    volume_col = first_column(history, ["成交额", "amount"])
    if not date_col or not close_col:
        raise RuntimeError("missing daily price columns")

    frame = history.copy()
    frame[date_col] = pd.to_datetime(frame[date_col])
    # Midday does not count as a completed daily bar.
    frame = frame[frame[date_col].dt.date < today.date()].tail(260)
    close = pd.to_numeric(frame[close_col], errors="coerce").dropna()
    if len(close) < 60:
        raise RuntimeError("fewer than 60 completed daily bars")

    bench_date = first_column(benchmark, ["日期", "date"])
    bench_close = first_column(benchmark, ["收盘", "close"])
    bench = benchmark.copy()
    bench[bench_date] = pd.to_datetime(bench[bench_date])
    bench = bench[bench[bench_date].dt.date < today.date()]
    benchmark_close = pd.to_numeric(bench[bench_close], errors="coerce").dropna()

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    rolling_max = close.tail(60).cummax()
    drawdown = close.tail(60) / rolling_max - 1
    returns = close.pct_change().dropna()
    amount_ratio = None
    if volume_col and volume_col in frame.columns:
        amount = pd.to_numeric(frame[volume_col], errors="coerce").dropna()
        if len(amount) >= 21 and amount.tail(20).mean() != 0:
            amount_ratio = float(amount.iloc[-1] / amount.tail(20).mean())

    ma = {f"ma{days}": float(close.tail(days).mean()) if len(close) >= days else None for days in [20, 60, 120, 200]}
    ma_slope = {
        f"ma{days}_slope_5d": (
            float((close.tail(days).mean() / close.iloc[-days - 5 : -5].mean() - 1) * 100)
            if len(close) >= days + 5 else None
        )
        for days in [20, 60, 120, 200]
    }
    sector_returns = {f"return_{days}d": pct_change(close, days) for days in [5, 20, 60, 120, 250]}
    benchmark_returns = {days: pct_change(benchmark_close, days) for days in [5, 20, 60, 120]}
    relative = {
        f"relative_csi300_{days}d": (
            sector_returns[f"return_{days}d"] - benchmark_returns[days]
            if sector_returns[f"return_{days}d"] is not None and benchmark_returns[days] is not None else None
        )
        for days in [5, 20, 60, 120]
    }
    return {
        **sector_returns,
        **relative,
        **ma,
        **ma_slope,
        "last_completed_close": float(close.iloc[-1]),
        "macd": float(macd_line.iloc[-1]),
        "macd_signal": float(signal.iloc[-1]),
        "macd_histogram": float((macd_line - signal).iloc[-1]),
        "rsi14": rsi(close),
        "volatility_20d": float(returns.tail(20).std() * math.sqrt(252) * 100) if len(returns) >= 20 else None,
        "max_drawdown_60d": float(drawdown.min() * 100),
        "amount_to_20d_average": amount_ratio,
        "completed_daily_bars": int(len(close)),
    }


def snapshot_for_board(spot: pd.DataFrame, board: str) -> dict[str, Any] | None:
    name_col = first_column(spot, ["板块名称", "名称", "行业名称"])
    if not name_col:
        return None
    matched = spot[spot[name_col].astype(str) == board]
    if matched.empty:
        return None
    row = matched.iloc[0]
    aliases = {
        "last_price": ["最新价", "最新", "现价"],
        "pct_change": ["涨跌幅"],
        "amount": ["成交额"],
        "high": ["最高"],
        "low": ["最低"],
        "open": ["开盘"],
        "previous_close": ["昨收"],
        "amplitude": ["振幅"],
    }
    output: dict[str, Any] = {"board": board}
    for target, columns in aliases.items():
        column = next((item for item in columns if item in spot.columns), None)
        output[target] = json_value(row[column]) if column else None
    return output


def coverage(metrics: dict[str, Any], snapshot: dict[str, Any] | None) -> tuple[float, list[str]]:
    required = ["return_20d", "return_60d", "ma20", "ma60", "macd", "rsi14", "volatility_20d"]
    missing = [key for key in required if metrics.get(key) is None]
    if not snapshot:
        missing.append("midday_snapshot")
    else:
        missing.extend(key for key in ["last_price", "pct_change", "amount"] if snapshot.get(key) is None)
    score = round((1 - len(missing) / (len(required) + 3)) * 100, 1)
    return score, missing


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    now = now_cn()
    errors: list[dict[str, str]] = []
    try:
        board_frame = retry(ak.stock_board_industry_name_em)
        name_col = first_column(board_frame, ["板块名称", "名称", "行业名称"])
        if not name_col:
            raise RuntimeError("industry list has no name column")
        boards = board_frame[name_col].dropna().astype(str).drop_duplicates().tolist()
        spot = retry(ak.stock_board_industry_spot_em)
        benchmark = fetch_benchmark(now)
    except Exception as exc:
        payload = {
            "generated_at": now.isoformat(),
            "data_source": "AkShare / Eastmoney industry boards",
            "data_status": "unavailable",
            "reason": f"source_unavailable: {exc}",
            "sectors": [],
        }
        (DATA_DIR / "latest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    sectors: list[dict[str, Any]] = []
    for index, board in enumerate(boards):
        try:
            history = load_history(board, now)
            metrics = technical_metrics(history, benchmark, now)
            snapshot = snapshot_for_board(spot, board)
            data_coverage, missing = coverage(metrics, snapshot)
            sectors.append({
                "board": board,
                "short_term": snapshot,
                "medium_long_term": metrics,
                "valuation": None,
                "data_coverage": data_coverage,
                "confidence": "medium" if data_coverage >= 80 else "low",
                "missing_fields": missing + ["pe_pb_percentile", "policy_industry_evidence"],
            })
        except Exception as exc:
            errors.append({"board": board, "error": str(exc)[:240]})
        time.sleep(0.15)

    expected_count = len(boards)
    valid_count = len(sectors)
    mid_day = now.weekday() < 5 and (now.hour == 11 and now.minute >= 25 or now.hour == 12 and now.minute <= 10)
    required_snapshot = sum(1 for item in sectors if item["short_term"])
    status = "ready"
    reason = None
    if not mid_day:
        status, reason = "unavailable", "not_a_midday_snapshot_window"
    elif valid_count < expected_count * 0.8 or required_snapshot < expected_count * 0.8:
        status, reason = "unavailable", "insufficient_sector_or_snapshot_coverage"

    payload = {
        "generated_at": now.isoformat(),
        "report_type": "midday_three_cycle_evidence",
        "classification": "Eastmoney industry boards",
        "data_source": "AkShare / Eastmoney",
        "data_status": status,
        "reason": reason,
        "expected_sector_count": expected_count,
        "valid_sector_count": valid_count,
        "snapshot_sector_count": required_snapshot,
        "errors": errors,
        "sectors": sectors,
    }
    (DATA_DIR / "latest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_value), encoding="utf-8")


if __name__ == "__main__":
    main()
