# -*- coding: utf-8 -*-
"""
SUI Clean Pro Bot — Bybit USDT Perps — Render 24/7 (PORT=5000)
- Closed candle entries only
- MACD cross 12/26/9
- Death/Golden cross (EMA50/EMA200) as regime filter (kill switch)
- Dynamic sizing by score: 60% / 35% / 20%
- SMC-lite: Liquidity sweep boost
- Trade management: TP1 partial + BE + ATR trailing + SL/TP3 hard
"""

import time, json, math, threading, traceback
from datetime import datetime

import numpy as np
import pandas as pd
import ccxt
from flask import Flask, jsonify

# =========================
# CONFIG INSIDE CODE (NO ENV)
# =========================
API_KEY    = "PUT_YOUR_BYBIT_KEY"
API_SECRET = "PUT_YOUR_BYBIT_SECRET"

SYMBOL   = "SUI/USDT:USDT"
INTERVAL = "15m"
PORT     = 5000

LEVERAGE = 5

# Indicators
EMA_FAST = 9
EMA_SLOW = 200
EMA_REG_FAST = 50
EMA_REG_SLOW = 200

ATR_LEN = 14
ADX_LEN = 14
RSI_LEN = 14

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIG  = 9

ADX_MIN = 20.0

# Score tiers -> allocation of wallet (then * leverage)
SCORE_STRONG = 9.0
SCORE_MED    = 7.0
SCORE_WEAK   = 6.0

ALLOC_STRONG = 0.60
ALLOC_MED    = 0.35
ALLOC_WEAK   = 0.20

# Risk/targets (ATR multiples)
SL_ATR_MULT  = 1.0
TP1_ATR_MULT = 1.5
TP2_ATR_MULT = 3.0
TP3_ATR_MULT = 4.5
TP1_CLOSE_FRAC = 0.50

# Trailing
TRAIL_ACTIVATE_PCT = 1.2
ATR_TRAIL_MULT     = 1.6

# SMC-lite (sweep)
SWEEP_LOOKBACK = 20
SWEEP_BOOST = 1.5  # score boost if sweep aligns with direction

# Guards / loop
MAX_SPREAD_BPS = 6.0
SLEEP_SEC = 5
POLL_OHLCV_SEC = 20

STATE_PATH = "./state_clean_pro.json"
COOLDOWN_SEC = 30

# =========================
# Utilities
# =========================
def now_utc():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

def log(msg):
    print(f"{now_utc()} | {msg}", flush=True)

def with_retry(fn, tries=3, base_wait=0.4):
    for i in range(tries):
        try:
            return fn()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(base_wait*(2**i) + 0.1)

def load_state():
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_state(s):
    try:
        s["ts"] = int(time.time())
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(s, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log(f"state save failed: {e}")

STATE = load_state()

METRICS = {
    "start_ts": int(time.time()),
    "loops": 0,
    "signals": 0,
    "entries": 0,
    "closes": 0,
    "errors": 0,
    "last_error": "",
    "last_signal": "",
    "last_action": "",
    "open": False,
    "side": "",
}

# =========================
# Exchange
# =========================
ex = ccxt.bybit({
    "apiKey": API_KEY,
    "secret": API_SECRET,
    "enableRateLimit": True,
    "timeout": 20000,
    "options": {"defaultType": "swap"},
})
ex.load_markets()

if SYMBOL not in ex.markets:
    raise Exception(f"Symbol not found on Bybit swap: {SYMBOL}")

def set_leverage():
    try:
        ex.set_leverage(LEVERAGE, SYMBOL)
        log(f"⚙️ Leverage set: {LEVERAGE}x")
    except Exception as e:
        log(f"set_leverage warning: {e}")

def fetch_ohlcv(limit=600):
    rows = with_retry(lambda: ex.fetch_ohlcv(SYMBOL, timeframe=INTERVAL, limit=limit, params={"type":"swap"}))
    return pd.DataFrame(rows, columns=["time","open","high","low","close","volume"])

def fetch_balance_usdt():
    b = with_retry(lambda: ex.fetch_balance(params={"type":"swap"}))
    total = (b.get("total", {}) or {}).get("USDT", None)
    free  = (b.get("free", {})  or {}).get("USDT", None)
    return float(total if total is not None else (free if free is not None else 0.0))

def last_price():
    t = with_retry(lambda: ex.fetch_ticker(SYMBOL))
    return float(t.get("last") or t.get("close") or 0.0)

def spread_bps():
    ob = with_retry(lambda: ex.fetch_order_book(SYMBOL, limit=5))
    if not ob.get("bids") or not ob.get("asks"):
        return None
    bid = float(ob["bids"][0][0])
    ask = float(ob["asks"][0][0])
    mid = (bid + ask) / 2.0
    return ((ask - bid) / mid) * 10000.0

def amount_to_precision(qty: float) -> float:
    try:
        return float(ex.amount_to_precision(SYMBOL, qty))
    except Exception:
        return float(qty)

def market_order(side: str, qty: float, reduce_only=False):
    params = {"reduceOnly": bool(reduce_only), "positionSide": "Both"}
    return with_retry(lambda: ex.create_order(SYMBOL, "market", side, qty, None, params))

def fetch_position():
    try:
        ps = with_retry(lambda: ex.fetch_positions([SYMBOL], params={"type":"swap"}))
        for p in ps:
            if p.get("symbol") == SYMBOL:
                contracts = float(p.get("contracts") or 0.0)
                side = (p.get("side") or "").lower()
                entry = float(p.get("entryPrice") or 0.0)
                return {"ok": True, "contracts": contracts, "side": side, "entry": entry}
        return {"ok": True, "contracts": 0.0, "side": "", "entry": 0.0}
    except Exception as e:
        return {"ok": False, "err": str(e)}

def strict_close(side: str, qty: float):
    close_side = "sell" if side == "buy" else "buy"
    qty = amount_to_precision(qty)
    if qty <= 0:
        return False
    for i in range(8):
        try:
            market_order(close_side, qty, reduce_only=True)
            time.sleep(1.6)
            p = fetch_position()
            if p.get("ok") and float(p.get("contracts") or 0.0) <= 0.0:
                return True
        except Exception as e:
            log(f"close retry {i+1}/8 err: {e}")
            time.sleep(0.8 + i*0.3)
    return False

# =========================
# Indicators
# =========================
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi(close, n=14):
    c = close.astype(float)
    delta = c.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/n, adjust=False).mean()
    ma_down = down.ewm(alpha=1/n, adjust=False).mean()
    rs = ma_up / (ma_down.replace(0, 1e-12))
    return 100 - (100 / (1 + rs))

def atr(df, n=14):
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    prev = c.shift(1)
    tr = pd.concat([(h-l), (h-prev).abs(), (l-prev).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()

def macd(close, fast=12, slow=26, signal=9):
    c = close.astype(float)
    macd_line = ema(c, fast) - ema(c, slow)
    sig_line  = ema(macd_line, signal)
    hist      = macd_line - sig_line
    return macd_line, sig_line, hist

def adx(df, n=14):
    high = df["high"].astype(float)
    low  = df["low"].astype(float)
    close= df["close"].astype(float)

    up = high.diff()
    dn = -low.diff()
    plus_dm  = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)

    tr = pd.concat([(high-low), (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
    atrn = tr.ewm(span=n, adjust=False).mean()

    plus_di  = 100 * pd.Series(plus_dm).ewm(span=n, adjust=False).mean() / atrn.replace(0, 1e-12)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=n, adjust=False).mean() / atrn.replace(0, 1e-12)

    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-12))
    adxv = dx.ewm(span=n, adjust=False).mean()
    return plus_di.values, minus_di.values, adxv.values

def compute_indicators(df):
    c = df["close"].astype(float)
    df["ema_fast"] = ema(c, EMA_FAST)
    df["ema_slow"] = ema(c, EMA_SLOW)
    df["ema_reg_fast"] = ema(c, EMA_REG_FAST)
    df["ema_reg_slow"] = ema(c, EMA_REG_SLOW)
    df["atr"] = atr(df, ATR_LEN)
    df["rsi"] = rsi(c, RSI_LEN)
    m, s, h = macd(c, MACD_FAST, MACD_SLOW, MACD_SIG)
    df["macd"] = m
    df["macd_sig"] = s
    df["macd_hist"] = h
    pdi, mdi, ax = adx(df, ADX_LEN)
    df["plus_di"] = pdi
    df["minus_di"] = mdi
    df["adx"] = ax
    return df

# =========================
# Regime (Death/Golden)
# =========================
def regime(df) -> str:
    cur = df.iloc[-2]
    prev= df.iloc[-3]
    f_c = float(cur["ema_reg_fast"]); s_c = float(cur["ema_reg_slow"])
    f_p = float(prev["ema_reg_fast"]); s_p = float(prev["ema_reg_slow"])

    golden = (f_p <= s_p) and (f_c > s_c)
    death  = (f_p >= s_p) and (f_c < s_c)

    if golden: return "golden_cross"
    if death:  return "death_cross"
    if f_c > s_c: return "bull"
    if f_c < s_c: return "bear"
    return "neutral"

# =========================
# SMC-lite: Liquidity Sweep
# =========================
def liquidity_sweep_boost(df):
    if len(df) < 60:
        return (0.0, 0.0, None)  # boost_buy, boost_sell, reason
    N = SWEEP_LOOKBACK
    cndl = df.iloc[-2]
    low_c, high_c, close_c = float(cndl["low"]), float(cndl["high"]), float(cndl["close"])
    lows  = df["low"].astype(float).iloc[-(N+2):-2]
    highs = df["high"].astype(float).iloc[-(N+2):-2]
    prev_low  = float(lows.min())
    prev_high = float(highs.max())

    # Bull sweep
    if low_c < prev_low and close_c > prev_low:
        return (SWEEP_BOOST, 0.0, f"sweep_buy (broke {prev_low:.6f} then closed above)")
    # Bear sweep
    if high_c > prev_high and close_c < prev_high:
        return (0.0, SWEEP_BOOST, f"sweep_sell (broke {prev_high:.6f} then closed below)")
    return (0.0, 0.0, None)

# =========================
# Score -> Allocation
# =========================
def score_to_alloc(score: float) -> float:
    if score >= SCORE_STRONG: return ALLOC_STRONG
    if score >= SCORE_MED:    return ALLOC_MED
    if score >= SCORE_WEAK:   return ALLOC_WEAK
    return 0.0

# =========================
# Signal (Closed candle)
# =========================
def get_signal(df):
    if len(df) < 250:
        return {"side": None, "why": "short_df", "alloc": 0.0, "score_b":0, "score_s":0, "details":[]}

    cur  = df.iloc[-2]
    prev = df.iloc[-3]

    adx_c = float(cur["adx"])
    if adx_c < ADX_MIN:
        return {"side": None, "why": f"adx_low {adx_c:.1f}", "alloc": 0.0, "score_b":0, "score_s":0, "details":[f"ADX<{ADX_MIN}"]}

    reg = regime(df)

    close_c = float(cur["close"]); close_p = float(prev["close"])
    ema_f_c = float(cur["ema_fast"]); ema_f_p = float(prev["ema_fast"])
    ema_s_c = float(cur["ema_slow"])

    # Trend filter
    trend_up = ema_f_c > ema_s_c
    trend_dn = ema_f_c < ema_s_c

    # Pullback / trigger: price crosses EMA9 on close (closed candle)
    cross_up   = (close_p <= ema_f_p) and (close_c > ema_f_c)
    cross_down = (close_p >= ema_f_p) and (close_c < ema_f_c)

    # MACD cross 12/26/9
    macd_c = float(cur["macd"]); macd_p = float(prev["macd"])
    sig_c  = float(cur["macd_sig"]); sig_p  = float(prev["macd_sig"])
    macd_up = (macd_p <= sig_p) and (macd_c > sig_c)
    macd_dn = (macd_p >= sig_p) and (macd_c < sig_c)

    # RSI context (buy from below / sell from above)
    rsi_c = float(cur["rsi"])

    # Base score
    score_b = 0.0
    score_s = 0.0
    details = [f"reg={reg}", f"adx={adx_c:.1f}", f"rsi={rsi_c:.1f}"]

    # Base BUY: trend up + MACD up + cross up + RSI not overbought
    base_buy = trend_up and macd_up and cross_up and (rsi_c <= 70)
    # Base SELL: trend down + MACD down + cross down + RSI not oversold
    base_sell = trend_dn and macd_dn and cross_down and (rsi_c >= 30)

    if base_buy:
        score_b += 7.0
        details.append("BASE_BUY(trend+macdXup+priceXema9+rsiOK)")
    if base_sell:
        score_s += 7.0
        details.append("BASE_SELL(trend+macdXdn+priceXema9+rsiOK)")

    # Extra confidence: RSI from low/high zones
    if rsi_c < 45: score_b += 0.7
    if rsi_c > 55: score_s += 0.7

    # SMC-lite boost
    b_boost, s_boost, reason = liquidity_sweep_boost(df)
    if b_boost: score_b += b_boost; details.append(reason)
    if s_boost: score_s += s_boost; details.append(reason)

    # Regime kill-switch (death/golden)
    EXTREME = 10.5
    allow_buy = True
    allow_sell = True
    if reg in ("bear", "death_cross"):
        allow_buy = False
        details.append("kill_buy_in_bear_regime")
    if reg in ("bull", "golden_cross"):
        allow_sell = False
        details.append("kill_sell_in_bull_regime")

    # Decide
    side = None
    why = "no_trade"
    if score_b >= score_s + 1.2 and (allow_buy or score_b >= EXTREME) and (base_buy or score_b >= SCORE_STRONG):
        side = "buy"; why = "score_buy"
    elif score_s >= score_b + 1.2 and (allow_sell or score_s >= EXTREME) and (base_sell or score_s >= SCORE_STRONG):
        side = "sell"; why = "score_sell"

    alloc = 0.0
    if side == "buy":  alloc = score_to_alloc(score_b)
    if side == "sell": alloc = score_to_alloc(score_s)

    if side and alloc <= 0:
        side = None
        why = "score_low_no_size"

    return {"side": side, "why": why, "alloc": alloc, "score_b": score_b, "score_s": score_s, "details": details[:12], "atr": float(cur["atr"])}

# =========================
# Trade management
# =========================
def compute_levels(entry, side, atrv):
    d = 1 if side == "buy" else -1
    sl  = entry - d*(atrv*SL_ATR_MULT)
    tp1 = entry + d*(atrv*TP1_ATR_MULT)
    tp2 = entry + d*(atrv*TP2_ATR_MULT)
    tp3 = entry + d*(atrv*TP3_ATR_MULT)
    return sl, tp1, tp2, tp3

def pnl_pct(entry, now, side):
    if entry <= 0: return 0.0
    raw = (now - entry) / entry
    return raw*100.0 if side == "buy" else (-raw*100.0)

def on_closed(reason):
    last_side = STATE.get("side")
    wait_side = "sell" if last_side == "buy" else "buy"

    log(f"✅ Closed({reason}) -> wait opposite: {wait_side.upper()}")
    METRICS["closes"] += 1
    METRICS["open"] = False
    METRICS["side"] = ""
    METRICS["last_action"] = f"closed_{reason}"

    STATE.clear()
    STATE.update({
        "open": False,
        "wait_for_side": wait_side,
        "cooldown_until": int(time.time()) + COOLDOWN_SEC
    })
    save_state(STATE)

def manage_open_trade():
    if not STATE.get("open"):
        return

    side = STATE.get("side")
    entry= float(STATE.get("entry", 0.0))
    qty  = float(STATE.get("qty", 0.0))
    atrv = float(STATE.get("atr", 0.0))
    if qty <= 0 or entry <= 0:
        return

    nowp = last_price()
    p = pnl_pct(entry, nowp, side)

    sl  = float(STATE.get("sl", entry))
    tp1 = float(STATE.get("tp1", 0.0))
    tp3 = float(STATE.get("tp3", 0.0))
    hit_tp1 = bool(STATE.get("hit_tp1", False))

    # Hard SL/TP3
    if side == "buy":
        if nowp <= sl:
            log(f"🛑 SL hit | p={nowp:.6f} sl={sl:.6f}")
            if strict_close(side, qty): on_closed("sl")
            return
        if tp3 and nowp >= tp3:
            log(f"🏁 TP3 hit | p={nowp:.6f} tp3={tp3:.6f}")
            if strict_close(side, qty): on_closed("tp3")
            return
    else:
        if nowp >= sl:
            log(f"🛑 SL hit | p={nowp:.6f} sl={sl:.6f}")
            if strict_close(side, qty): on_closed("sl")
            return
        if tp3 and nowp <= tp3:
            log(f"🏁 TP3 hit | p={nowp:.6f} tp3={tp3:.6f}")
            if strict_close(side, qty): on_closed("tp3")
            return

    # TP1 partial + BE
    if (not hit_tp1) and tp1:
        tp1_hit = (side == "buy" and nowp >= tp1) or (side == "sell" and nowp <= tp1)
        if tp1_hit:
            close_qty = amount_to_precision(qty * TP1_CLOSE_FRAC)
            if close_qty > 0:
                log(f"🎯 TP1 partial {close_qty} | pnl={p:.2f}% -> SL=BE")
                market_order("sell" if side=="buy" else "buy", close_qty, reduce_only=True)
                STATE["qty"] = max(0.0, qty - close_qty)
                STATE["hit_tp1"] = True
                STATE["sl"] = entry
                save_state(STATE)
                METRICS["last_action"] = "tp1_partial_be"
            return

    # ATR trailing
    if atrv > 0 and p >= TRAIL_ACTIVATE_PCT:
        trail = atrv * ATR_TRAIL_MULT
        cur_sl = float(STATE.get("sl", entry))
        if side == "buy":
            new_sl = max(cur_sl, nowp - trail)
            if new_sl > cur_sl:
                STATE["sl"] = new_sl
                save_state(STATE)
                log(f"🧲 Trail up -> SL={new_sl:.6f} | pnl={p:.2f}%")
                METRICS["last_action"] = "trail_up"
        else:
            new_sl = min(cur_sl, nowp + trail)
            if new_sl < cur_sl:
                STATE["sl"] = new_sl
                save_state(STATE)
                log(f"🧲 Trail down -> SL={new_sl:.6f} | pnl={p:.2f}%")
                METRICS["last_action"] = "trail_down"

# =========================
# Entry policy
# =========================
def can_enter(signal_side):
    cd = int(STATE.get("cooldown_until", 0) or 0)
    if time.time() < cd:
        return False, "cooldown"
    w = STATE.get("wait_for_side", None)
    if w and signal_side != w:
        return False, f"waiting_for_{w}"
    return True, "ok"

def calc_qty(price, alloc):
    bal = fetch_balance_usdt()
    if bal <= 0:
        return 0.0
    notional = bal * alloc * LEVERAGE
    qty = notional / price
    return amount_to_precision(qty)

def open_trade(side, atrv, alloc, why, details):
    sp = spread_bps()
    if sp is not None and sp > MAX_SPREAD_BPS:
        log(f"⛔ spread high {sp:.2f}bps > {MAX_SPREAD_BPS}")
        return

    price = last_price()
    qty = calc_qty(price, alloc)
    if qty <= 0:
        log("qty invalid")
        return

    log(f"🚀 OPEN {side.upper()} alloc={alloc:.2f} qty={qty} px~{price:.6f} | {why} | MACD(12,26,9)")
    log("📌 اشتركوا في قناة الواقع ليصلكم كل جديد")
    log(" | ".join(details))

    METRICS["entries"] += 1
    METRICS["last_action"] = f"open_{side}"
    METRICS["open"] = True
    METRICS["side"] = side

    market_order(side, qty, reduce_only=False)

    entry = last_price()
    sl, tp1, tp2, tp3 = compute_levels(entry, side, atrv)

    STATE.clear()
    STATE.update({
        "open": True,
        "side": side,
        "qty": float(qty),
        "entry": float(entry),
        "atr": float(atrv),
        "sl": float(sl),
        "tp1": float(tp1),
        "tp2": float(tp2),
        "tp3": float(tp3),
        "hit_tp1": False,
        "wait_for_side": None
    })
    save_state(STATE)

# =========================
# Adopt exchange position after restart
# =========================
def adopt_position_if_needed():
    pos = fetch_position()
    ex_open = pos.get("ok") and float(pos.get("contracts") or 0.0) > 0.0

    if STATE.get("open") and not ex_open:
        on_closed("exchange_flat")
        return

    if (not STATE.get("open")) and ex_open:
        side = pos.get("side") or ""
        entry= float(pos.get("entry") or 0.0)
        qty  = float(pos.get("contracts") or 0.0)
        df = compute_indicators(fetch_ohlcv())
        atrv = float(df.iloc[-2]["atr"])
        sl, tp1, tp2, tp3 = compute_levels(entry, side, atrv)

        log(f"🧷 Adopted exchange position side={side} qty={qty} entry={entry}")
        STATE.clear()
        STATE.update({
            "open": True, "side": side, "qty": qty, "entry": entry, "atr": atrv,
            "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3,
            "hit_tp1": False, "wait_for_side": None
        })
        save_state(STATE)
        METRICS["open"] = True
        METRICS["side"] = side

# =========================
# Main bot loop
# =========================
def bot_loop():
    set_leverage()
    log(f"🧠 CLEAN PRO READY | {SYMBOL} {INTERVAL} | lev={LEVERAGE}x | CLOSED-CANDLE")
    log("📌 اشتركوا في قناة الواقع ليصلكم كل جديد")

    last_ohlcv_ts = 0

    while True:
        METRICS["loops"] += 1
        try:
            adopt_position_if_needed()

            if STATE.get("open"):
                manage_open_trade()
                time.sleep(SLEEP_SEC)
                continue

            if time.time() - last_ohlcv_ts < POLL_OHLCV_SEC:
                time.sleep(SLEEP_SEC)
                continue

            df = compute_indicators(fetch_ohlcv())
            last_ohlcv_ts = time.time()

            sig = get_signal(df)
            METRICS["signals"] += 1
            METRICS["last_signal"] = f"{sig.get('side') or 'NONE'} | {sig.get('why')} | b={sig.get('score_b',0):.2f} s={sig.get('score_s',0):.2f} alloc={sig.get('alloc',0):.2f}"

            side = sig.get("side")
            if side:
                ok, reason = can_enter(side)
                if not ok:
                    log(f"⏳ Signal {side.upper()} blocked: {reason} | {sig.get('why')}")
                else:
                    open_trade(side, float(sig.get("atr", 0.0)), float(sig.get("alloc", 0.0)),
                               sig.get("why","signal"), sig.get("details",[]))
            time.sleep(SLEEP_SEC)

        except Exception as e:
            METRICS["errors"] += 1
            METRICS["last_error"] = str(e)[:220]
            log(f"ERR: {e}\n{traceback.format_exc()}")
            time.sleep(3)

# =========================
# Flask server for Render (PORT=5000)
# =========================
app = Flask(__name__)

@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "symbol": SYMBOL,
        "tf": INTERVAL,
        "open": bool(STATE.get("open")),
        "side": STATE.get("side",""),
        "entry": STATE.get("entry", 0.0),
        "qty": STATE.get("qty", 0.0),
        "sl": STATE.get("sl", 0.0),
        "uptime_s": int(time.time()) - METRICS["start_ts"],
        "metrics": METRICS
    })

@app.get("/metrics")
def metrics():
    return jsonify({
        **METRICS,
        "state_wait_for": STATE.get("wait_for_side",""),
    })

def run_web():
    log(f"🌐 Web server on PORT={PORT}")
    app.run(host="0.0.0.0", port=PORT)

if __name__ == "__main__":
    t = threading.Thread(target=bot_loop, daemon=True)
    t.start()
    run_web()
