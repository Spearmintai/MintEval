"""R008 - Intraday reversal in New York hours (QuantConnect LEAN
Algorithm.Python/Alphas/IntradayReversalCurrencyMarketsAlpha.py, Apache-2.0). Reference port.

Source: https://github.com/QuantConnect/Lean/blob/41c6e603/Algorithm.Python/Alphas/IntradayReversalCurrencyMarketsAlpha.py
        Copyright 2014 QuantConnect Corporation, Apache-2.0. Insight.IsExpired (CloseTimeUtc < now)
        and EqualWeightingPortfolioConstructionModel from the same repository.

Original logic (EURUSD, hourly bars, algorithm time zone America/New_York):
  * SMA(5) of hourly closes. On each hourly bar whose (end) time of day is between 10:00 and 15:00
    NY inclusive: direction = UP if price < round(SMA*1.001, 6) else DOWN (DOWN also while the SMA is
    not ready). If the direction equals the previously emitted direction, nothing happens; otherwise
    the new direction is stored and an insight expiring at 15:01 NY of the same day is emitted.
  * The stored previous direction is NOT reset between days.
  * EqualWeighting PCM: active UP insight -> 100 % long, DOWN -> 100 % short (reversal); expired ->
    flat. Expiry is strict (close 15:01 < now), so the position is closed at the 16:00 NY hourly bar.
  * No risk model (NullRiskManagementModel), zero fees.

Approximations / porting decisions:
  * Symbol: EURUSD -> BTCUSDT; BTC trades 24/7 so weekends are traded too (FX data has none).
  * Bars: hourly LEAN bars = hist.htf(60) completed UTC-aligned hours (NY hours are whole-hour
    offsets of UTC, so the buckets coincide). Decisions are only taken at 15m bars that complete an
    hour; on the three other 15m bars of each hour the strategy holds its position.
  * New York time: UTC-5, or UTC-4 between the second Sunday of March 07:00 UTC and the first Sunday
    of November 06:00 UTC (US rule since 2007), computed from the bar timestamp.
  * Price = hourly close; SMA(5) = ind.sma of hourly closes (identical definition).
  * Fill at the next open (LEAN fills market orders on the next data point as well).
"""
import math

SMA_N = 5
BAND = 1.001
START_H = 10
END_H = 15
CLOSE_H = 15        # insights expire at 15:01 local
HOUR_MS = 3600000
DAY_MS = 86400000


def days_from_civil(y, m, d):
    y = y - 1 if m <= 2 else y
    era = (y if y >= 0 else y - 399) // 400
    yoe = y - era * 400
    mp = m - 3 if m > 2 else m + 9
    doy = (153 * mp + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def year_of_day(z):
    z = z + 719468
    era = (z if z >= 0 else z - 146096) // 146097
    doe = z - era * 146097
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    y = yoe + era * 400
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    m = mp + 3 if mp < 10 else mp - 9
    return y + 1 if m <= 2 else y


def first_sunday(y, m):
    d0 = days_from_civil(y, m, 1)
    wd = (d0 + 3) % 7          # 0 = Monday (1970-01-01 was a Thursday)
    return d0 + (6 - wd) % 7


def ny_offset_ms(utc_ms):
    y = year_of_day(int(utc_ms // DAY_MS))
    start = (first_sunday(y, 3) + 7) * DAY_MS + 7 * HOUR_MS     # 02:00 EST = 07:00 UTC
    end = first_sunday(y, 11) * DAY_MS + 6 * HOUR_MS            # 02:00 EDT = 06:00 UTC
    if start <= utc_ms < end:
        return -4 * HOUR_MS
    return -5 * HOUR_MS


def strategy(hist, state, pos, ind):
    now_utc = int(hist.time[-1]) + 900000           # close time of the current 15m bar
    if now_utc % HOUR_MS != 0:
        return {"target": pos, "stop": None, "take": None}
    local = now_utc + ny_offset_ms(now_utc)
    local_day = local // DAY_MS
    minute_of_day = (local % DAY_MS) // 60000

    hours = hist.htf(60)
    price = hours.close[-1]
    sma = ind.sma(hours.close, SMA_N)[-1] if len(hours) >= SMA_N else float("nan")

    # alpha model (10:00 <= local time <= 15:00)
    if START_H * 60 <= minute_of_day <= END_H * 60:
        up = (not math.isnan(sma)) and price < round(sma * BAND, 6)
        direction = 1 if up else -1
        if direction != state.get("prev_dir", 0):
            state["prev_dir"] = direction
            state["dir"] = direction
            state["expiry_day"] = local_day          # expires 15:01 local of that day

    # portfolio construction: active insight -> +/-100 %, expired (15:01 < now) -> flat
    target = 0.0
    if "dir" in state:
        expiry_local = state["expiry_day"] * DAY_MS + (CLOSE_H * 60 + 1) * 60000
        if expiry_local < local:
            del state["dir"]
            del state["expiry_day"]
        else:
            target = float(state["dir"])
    return {"target": target, "stop": None, "take": None}
