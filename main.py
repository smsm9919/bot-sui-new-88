# -*- coding: utf-8 -*-
"""
SUI Trend Master Bot — MULTI-EXCHANGE (BingX & Bybit)
• Trend-Only System (No Scalp/Weak) with HTF Bias
• Mid Trend (2 TP) + Big Trend (3 TP)
• SMC + Liquidity + Structure Analysis
• Smart Entry after Stop Hunts + Confirmation
• Professional Logging with Trade Plans
• Multi-Exchange Support: BingX & Bybit
• Enhanced Liquidity + SMC + Explosion/Collapse Engine
• ANTI-DESYNC GUARD + LAYERED DECISION SYSTEM
"""

import os, time, math, random, signal, sys, traceback, logging, json
from logging.handlers import RotatingFileHandler
from datetime import datetime
import pandas as pd
import numpy as np
import ccxt
from flask import Flask, jsonify
from decimal import Decimal, ROUND_DOWN, InvalidOperation
from dataclasses import dataclass

try:
    from termcolor import colored
except Exception:
    def colored(t,*a,**k): return t

# =================== ENV / MODE ===================
EXCHANGE_NAME = os.getenv("EXCHANGE", "bingx").lower()

if EXCHANGE_NAME == "bybit":
    API_KEY = os.getenv("BYBIT_API_KEY", "")
    API_SECRET = os.getenv("BYBIT_API_SECRET", "")
else:
    API_KEY = os.getenv("BINGX_API_KEY", "")
    API_SECRET = os.getenv("BINGX_API_SECRET", "")

MODE_LIVE = bool(API_KEY and API_SECRET)
SELF_URL = os.getenv("SELF_URL", "") or os.getenv("RENDER_EXTERNAL_URL", "")
PORT = int(os.getenv("PORT", 5000))

# ==== Execution Switches ====
EXECUTE_ORDERS = True
SHADOW_MODE_DASHBOARD = False
DRY_RUN = False

# ==== Logging Settings ====
LOG_LEGACY = False
LOG_ADDONS = True
LOG_DECISION_LAYER = True  # ✅ طبقة القرار الجديدة

BOT_VERSION = f"SUI Trend Master v7.1 — {EXCHANGE_NAME.upper()} Multi-Exchange"
print("🔁 Booting:", BOT_VERSION, flush=True)

STATE_PATH = "./bot_state.json"
RESUME_ON_RESTART = True
RESUME_LOOKBACK_SECS = 60 * 60

# ==== Config ====
BOOKMAP_DEPTH = 50
BOOKMAP_TOPWALLS = 3
IMBALANCE_ALERT = 1.30
FLOW_WINDOW = 20
FLOW_SPIKE_Z = 1.60
CVD_SMOOTH = 8

# =================== MODE POLICY ===================
FORCE_NO_SCALP = True      # ✅ ممنوع سكالب نهائي
ALLOW_WEAK = False         # ✅ ممنوع weak
ALLOW_MID = True
ALLOW_BIG = True

# Mid/Big thresholds
ADX_MID_MIN = 22.0
ADX_BIG_MIN = 35.0

# HTF bias
HTF_TF = "1h"
HTF_EMA_FAST = 50
HTF_EMA_SLOW = 200

# =================== SETTINGS ===================
SYMBOL     = os.getenv("SYMBOL", "SUI/USDT:USDT")
INTERVAL   = os.getenv("INTERVAL", "15m")
LEVERAGE   = int(os.getenv("LEVERAGE", 10))
RISK_ALLOC = float(os.getenv("RISK_ALLOC", 0.60))
POSITION_MODE = os.getenv("POSITION_MODE", "oneway")

# RF Settings
RF_SOURCE = "close"
RF_PERIOD = int(os.getenv("RF_PERIOD", 18))
RF_MULT   = float(os.getenv("RF_MULT", 3.0))
RF_LIVE_ONLY = True
RF_HYST_BPS  = 6.0

# Indicators
RSI_LEN = 14
ADX_LEN = 14
ATR_LEN = 14

ENTRY_RF_ONLY = False
MAX_SPREAD_BPS = float(os.getenv("MAX_SPREAD_BPS", 6.0))

# Dynamic TP / trail
TP1_PCT_BASE       = 0.45
TP1_CLOSE_FRAC     = 0.50
BREAKEVEN_AFTER    = 0.30
TRAIL_ACTIVATE_PCT = 1.20
ATR_TRAIL_MULT     = 1.8

TREND_TPS       = [0.50, 1.00, 1.80]
TREND_TP_FRACS  = [0.30, 0.30, 0.20]

# Dust guard
FINAL_CHUNK_QTY = float(os.getenv("FINAL_CHUNK_QTY", 50.0))
RESIDUAL_MIN_QTY = float(os.getenv("RESIDUAL_MIN_QTY", 10.0))

# Strict close
CLOSE_RETRY_ATTEMPTS = 6
CLOSE_VERIFY_WAIT_S  = 2.0

# Pacing
BASE_SLEEP   = 5
NEAR_CLOSE_S = 1

# ==== Smart Exit Tuning ===
TP1_SCALP_PCT      = 0.35/100
TP1_TREND_PCT      = 0.60/100
HARD_CLOSE_PNL_PCT = 1.10/100
WICK_ATR_MULT      = 1.5
EVX_SPIKE          = 1.8
BM_WALL_PROX_BPS   = 5
TIME_IN_TRADE_MIN  = 8
TRAIL_TIGHT_MULT   = 1.20

# ==== Golden Entry Settings ====
GOLDEN_ENTRY_SCORE = 6.0
GOLDEN_ENTRY_ADX   = 20.0
GOLDEN_REVERSAL_SCORE = 6.5

# ==== Execution & Strategy Thresholds ====
ADX_TREND_MIN = 20
DI_SPREAD_TREND = 6
RSI_MA_LEN = 9
RSI_NEUTRAL_BAND = (45, 55)
RSI_TREND_PERSIST = 3

GZ_MIN_SCORE = 6.0
GZ_REQ_ADX = 20
GZ_REQ_VOL_MA = 20
ALLOW_GZ_ENTRY = True

SCALP_TP1 = 0.40
SCALP_BE_AFTER = 0.30
SCALP_ATR_MULT = 1.6
TREND_TP1 = 1.20
TREND_BE_AFTER = 0.80
TREND_ATR_MULT = 1.8

MAX_TRADES_PER_HOUR = 6
COOLDOWN_SECS_AFTER_CLOSE = 60
ADX_GATE = 17

# =================== SAFE VALUE HELPER ===================
def safe(v, default=0.0):
    """Normalize values with fail-safe"""
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default

def safe_dict(d, key, default=None):
    """Safe dictionary access"""
    if not isinstance(d, dict):
        return default
    return d.get(key, default)

def safe_get(d, k, default=None):
    """Safe get for nested dicts"""
    if isinstance(d, dict):
        return d.get(k, default)
    return default

def safe_bool(x):
    """Safe boolean conversion"""
    return bool(x) if x is not None else False

# =================== PROFESSIONAL LOGGING ===================
def log_i(msg): print(f"ℹ️ {msg}", flush=True)
def log_g(msg): print(f"✅ {msg}", flush=True)
def log_w(msg): print(f"🟨 {msg}", flush=True)
def log_e(msg): print(f"❌ {msg}", flush=True)
def log_y(msg): print(f"🟡 {msg}", flush=True)
def log_d(msg): print(f"🧠 {msg}", flush=True)  # Decision Layer

def log_banner(text): print(f"\n{'—'*12} {text} {'—'*12}\n", flush=True)

def save_state(state: dict):
    try:
        state["ts"] = int(time.time())
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        log_i(f"state saved → {STATE_PATH}")
    except Exception as e:
        log_w(f"state save failed: {e}")

def load_state() -> dict:
    try:
        if not os.path.exists(STATE_PATH): return {}
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log_w(f"state load failed: {e}")
    return {}

# =================== ANTI-DESYNC GUARD (PATCH 0) ===================
def safe_has_position(exchange, symbol):
    """يرجع (موجود, الكمية) بناءً على المنصة"""
    try:
        pos = None
        # Bybit v5 عبر ccxt غالبًا positions
        positions = exchange.fetch_positions([symbol])
        for p in positions:
            if p.get("symbol") == symbol:
                pos = p
                break
        if not pos:
            return False, 0.0

        contracts = float(pos.get("contracts") or pos.get("contractSize") or 0.0)
        # بعض النسخ بتحط size بدل contracts
        size = float(pos.get("size") or 0.0)
        qty = abs(size) if size != 0 else abs(contracts)

        return qty > 0, qty
    except Exception as e:
        log_w(f"safe_has_position error: {e}")
        return False, 0.0

def guard_reduce_only(exchange, symbol, why=""):
    """يرجع True إذا كان هناك وضعية مفتوحة على المنصة، وإلا False مع تحذير"""
    ok, qty = safe_has_position(exchange, symbol)
    if not ok:
        log_w(f"🧯 DESYNC GUARD: exchange says NO position → skip reduce-only ({why})")
        return False
    return True

# =================== EXCHANGE FACTORY ===================
def make_ex():
    """Factory function for multi-exchange support"""
    exchange_config = {
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "timeout": 20000,
    }
    
    if EXCHANGE_NAME == "bybit":
        exchange_config["options"] = {"defaultType": "swap"}
        return ccxt.bybit(exchange_config)
    else:
        exchange_config["options"] = {"defaultType": "swap"}
        return ccxt.bingx(exchange_config)

ex = make_ex()

# =================== EXCHANGE-SPECIFIC ADAPTERS ===================
def exchange_specific_params(side, is_close=False):
    """Handle exchange-specific parameters"""
    if EXCHANGE_NAME == "bybit":
        if POSITION_MODE == "hedge":
            return {"positionSide": "Long" if side == "buy" else "Short", "reduceOnly": is_close}
        return {"positionSide": "Both", "reduceOnly": is_close}
    else:
        if POSITION_MODE == "hedge":
            return {"positionSide": "LONG" if side == "buy" else "SHORT", "reduceOnly": is_close}
        return {"positionSide": "BOTH", "reduceOnly": is_close}

def exchange_set_leverage(exchange, leverage, symbol):
    """Exchange-specific leverage setting"""
    try:
        if EXCHANGE_NAME == "bybit":
            exchange.set_leverage(leverage, symbol)
        else:
            exchange.set_leverage(leverage, symbol, params={"side": "BOTH"})
        log_g(f"✅ {EXCHANGE_NAME.upper()} leverage set: {leverage}x")
    except Exception as e:
        log_w(f"⚠️ set_leverage warning: {e}")

# =================== MARKET SPECS ===================
MARKET = {}
AMT_PREC = 0
LOT_STEP = None
LOT_MIN  = None

def load_market_specs():
    global MARKET, AMT_PREC, LOT_STEP, LOT_MIN
    try:
        ex.load_markets()
        MARKET = ex.markets.get(SYMBOL, {})
        AMT_PREC = int((MARKET.get("precision", {}) or {}).get("amount", 0) or 0)
        LOT_STEP = (MARKET.get("limits", {}) or {}).get("amount", {}).get("step", None)
        LOT_MIN  = (MARKET.get("limits", {}) or {}).get("amount", {}).get("min", None)
        log_i(f"🎯 {SYMBOL} specs → precision={AMT_PREC}, step={LOT_STEP}, min={LOT_MIN}")
    except Exception as e:
        log_w(f"load_market_specs: {e}")

def ensure_leverage_mode():
    try:
        exchange_set_leverage(ex, LEVERAGE, SYMBOL)
        log_i(f"📊 {EXCHANGE_NAME.upper()} position mode: {POSITION_MODE}")
    except Exception as e:
        log_w(f"ensure_leverage_mode: {e}")

# Initialize exchange
try:
    load_market_specs()
    ensure_leverage_mode()
except Exception as e:
    log_w(f"exchange init: {e}")

# =================== TREND STATE ENGINE ===================
TREND_STATE = {
    "mode": "NONE",           # NONE | MID_TREND | BIG_TREND
    "dir": None,              # "buy" | "sell"
    "phase": "WAIT",          # WAIT | ENTRY | RUN | CORRECTION | EXIT
    "strength": 0.0,          # score 0..10
    "htf_bias": None,         # "buy" | "sell" | None
    "last_plan_ts": 0,
    "plan": None,
}

def compute_htf_bias(df_htf):
    """
    Bias من فريم الساعة:
    - price فوق EMA200 + EMA50>EMA200 => buy bias
    - price تحت EMA200 + EMA50<EMA200 => sell bias
    """
    try:
        c = df_htf["close"].astype(float)
        ema_fast = c.ewm(span=HTF_EMA_FAST, adjust=False).mean()
        ema_slow = c.ewm(span=HTF_EMA_SLOW, adjust=False).mean()

        px = float(c.iloc[-1])
        ef = float(ema_fast.iloc[-1])
        es = float(ema_slow.iloc[-1])

        if px > es and ef > es:
            return "buy", {"px": px, "ema_fast": ef, "ema_slow": es}
        if px < es and ef < es:
            return "sell", {"px": px, "ema_fast": ef, "ema_slow": es}
        return None, {"px": px, "ema_fast": ef, "ema_slow": es}
    except Exception as e:
        log_w(f"HTF bias error: {e}")
        return None, {}

# =================== LIQUIDITY ENGINE "READABLE" (PATCH 1) ===================
def liquidity_read(df, lookback=30, wick_thr=0.55, vol_ma=20):
    """
    يرجّع قراءة سيولة مفهومة:
    - sweepH / sweepL (ضرب استوبات فوق/تحت)
    - accumulation / distribution (تجميع/تصريف)
    - drain (سحب سيولة مفاجئ)
    - vol_x (انفجار فوليوم)
    """
    if df is None or len(df) < max(lookback, vol_ma) + 2:
        return {"ok": False, "state": "NA"}

    sub = df.iloc[-lookback:]
    last = df.iloc[-1]
    prev = df.iloc[-2]

    hi = float(sub["high"].max())
    lo = float(sub["low"].min())

    o = float(last["open"]); h = float(last["high"]); l = float(last["low"]); c = float(last["close"])
    rng = max(h - l, 1e-12)

    up_wick = (h - max(o, c)) / rng
    dn_wick = (min(o, c) - l) / rng

    # sweep = لمس قمة/قاع ثم إغلاق عكسي (مصيدة سيولة)
    sweepH = (h >= hi) and (c < float(prev["close"])) and (up_wick >= wick_thr)
    sweepL = (l <= lo) and (c > float(prev["close"])) and (dn_wick >= wick_thr)

    vol = float(last.get("volume", 0.0))
    vol_mean = float(df["volume"].iloc[-vol_ma:].mean()) if "volume" in df else 0.0
    vol_x = (vol / vol_mean) if vol_mean > 0 else 0.0

    # drain = شمعة كبيرة ضد الاتجاه مع فوليوم عالي (تصريف/سحب)
    body = abs(c - o)
    drain = (body / rng > 0.65) and (vol_x >= 1.6)

    # تجميع/تصريف: ATR منخفض + فوليوم منخفض نسبيًا + رينج ضيق
    avg_rng = float((df["high"].iloc[-20:] - df["low"].iloc[-20:]).mean())
    low_range = (rng <= 0.75 * avg_rng)
    low_vol = (vol_x <= 0.85)

    accumulation = low_range and low_vol and (c >= o)  # ضغط شراء هادي
    distribution = low_range and low_vol and (c < o)   # ضغط بيع هادي

    state = "NEUTRAL"
    if sweepH: state = "SWEEP_HIGH"
    elif sweepL: state = "SWEEP_LOW"
    elif drain and c < o: state = "DRAIN_DOWN"
    elif drain and c > o: state = "DRAIN_UP"
    elif accumulation: state = "ACCUMULATION"
    elif distribution: state = "DISTRIBUTION"

    return {
        "ok": True,
        "state": state,
        "sweepH": bool(sweepH),
        "sweepL": bool(sweepL),
        "wickU": float(up_wick),
        "wickD": float(dn_wick),
        "vol_x": float(vol_x),
        "drain": bool(drain),
        "hi": float(hi),
        "lo": float(lo),
    }

def log_liquidity(liq):
    if not liq or not liq.get("ok"):
        log_i("💧 Liquidity | NA")
        return
    log_i(
        f"💧 Liquidity | state={liq['state']} | "
        f"sweepH={liq['sweepH']} sweepL={liq['sweepL']} | "
        f"wickU={liq['wickU']:.2f} wickD={liq['wickD']:.2f} | "
        f"vol_x={liq['vol_x']:.2f} | drain={liq['drain']}"
    )

# =================== SMC ENGINE (PATCH 2) ===================
def smc_read(df, swing=10, fvg_thr=0.0008):
    """
    مخرجات:
    - ob_side: BUY/SELL/None
    - fvg_side: BUY/SELL/None
    - bos_up / bos_down
    """
    if df is None or len(df) < swing*2 + 5:
        return {"ok": False}

    highs = df["high"].astype(float).values
    lows  = df["low"].astype(float).values
    closes= df["close"].astype(float).values
    opens = df["open"].astype(float).values

    i = len(df) - 1

    # BOS بسيط: كسر آخر قمة/قاع محلي
    recent_hi = max(highs[i-swing:i])
    recent_lo = min(lows[i-swing:i])

    bos_up   = closes[i] > recent_hi
    bos_down = closes[i] < recent_lo

    # FVG: فجوة (low الحالي > high قبلها) أو (high الحالي < low قبلها)
    fvg_buy = lows[i] > highs[i-2] and (lows[i] - highs[i-2]) / max(closes[i],1e-9) >= fvg_thr
    fvg_sell= highs[i] < lows[i-2] and (lows[i-2] - highs[i]) / max(closes[i],1e-9) >= fvg_thr

    # OB مبسط: آخر شمعة عكسية قبل اندفاع
    # اندفاع = جسم كبير + إغلاق قريب من الطرف
    def impulse(idx):
        o = opens[idx]; c = closes[idx]; h = highs[idx]; l = lows[idx]
        rng = max(h-l,1e-12)
        body = abs(c-o)
        return (body/rng > 0.65)

    ob_side = None
    if bos_up and impulse(i):
        # آخر bearish candle قبل الاندفاع
        for k in range(i-1, max(i-10, 0), -1):
            if closes[k] < opens[k]:
                ob_side = "BUY"
                break
    if bos_down and impulse(i):
        for k in range(i-1, max(i-10, 0), -1):
            if closes[k] > opens[k]:
                ob_side = "SELL"
                break

    return {
        "ok": True,
        "bos_up": bool(bos_up),
        "bos_down": bool(bos_down),
        "ob_side": ob_side,
        "fvg_buy": bool(fvg_buy),
        "fvg_sell": bool(fvg_sell),
    }

def log_smc(smc):
    if not smc or not smc.get("ok"):
        log_i("🏗️ SMC | NA")
        return
    log_i(
        f"🏗️ SMC | BOS(up={smc['bos_up']}, down={smc['bos_down']}) | "
        f"OB={smc['ob_side']} | FVG(buy={smc['fvg_buy']}, sell={smc['fvg_sell']})"
    )

# =================== CONTEXT INDICATORS (PATCH 3) ===================
def ichimoku_bias_enhanced(df, tenkan=9, kijun=26, spanb=52):
    """Ichimoku Cloud Bias محسن"""
    if df is None or len(df) < spanb + 5:
        return {"ok": False}

    high = df["high"].astype(float)
    low  = df["low"].astype(float)
    close= df["close"].astype(float)

    tenkan_sen = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
    kijun_sen  = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2
    senkou_a   = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    senkou_b   = ((high.rolling(spanb).max() + low.rolling(spanb).min()) / 2).shift(kijun)

    c = float(close.iloc[-1])
    a = float(senkou_a.iloc[-1]) if not senkou_a.isna().iloc[-1] else None
    b = float(senkou_b.iloc[-1]) if not senkou_b.isna().iloc[-1] else None

    if a is None or b is None:
        return {"ok": False}

    top = max(a,b); bot = min(a,b)
    if c > top: bias="BULL"
    elif c < bot: bias="BEAR"
    else: bias="RANGE"

    return {"ok": True, "bias": bias, "cloud_top": top, "cloud_bot": bot}

def sma_trend(df, n=200):
    """SMA Trend Filter"""
    if df is None or len(df) < n + 3:
        return {"ok": False}
    close = df["close"].astype(float)
    sma = close.rolling(n).mean()
    slope = float(sma.iloc[-1] - sma.iloc[-3])
    c = float(close.iloc[-1])
    bias = "ABOVE" if c >= float(sma.iloc[-1]) else "BELOW"
    return {"ok": True, "bias": bias, "slope": slope, "sma": float(sma.iloc[-1])}

def vp_proxy(df, bins=30, lookback=120):
    """Volume Profile Proxy"""
    if df is None or len(df) < lookback + 5:
        return {"ok": False}
    sub = df.iloc[-lookback:]
    closes = sub["close"].astype(float).values
    vols = sub["volume"].astype(float).values if "volume" in sub else np.ones_like(closes)

    lo, hi = float(closes.min()), float(closes.max())
    if hi <= lo:
        return {"ok": False}

    hist, edges = np.histogram(closes, bins=bins, range=(lo,hi), weights=vols)
    poc_idx = int(np.argmax(hist))
    poc = (edges[poc_idx] + edges[poc_idx+1]) / 2.0

    c = float(df["close"].iloc[-1])
    dist = (c - poc) / max(c,1e-9)

    # لو السعر بعيد عن POC ومعاه Trend قوي = continuation
    # لو قريب من POC = chop / range
    zone = "POC_NEAR" if abs(dist) < 0.0025 else ("ABOVE_POC" if dist > 0 else "BELOW_POC")

    return {"ok": True, "poc": float(poc), "zone": zone, "dist": float(dist)}

def log_context(vwap_bias, ichi, sma, vp):
    """Log context indicators"""
    parts=[]
    if vwap_bias is not None:
        parts.append(f"VWAP_bias={vwap_bias}")
    if ichi and ichi.get("ok"):
        parts.append(f"Ichimoku={ichi['bias']}")
    if sma and sma.get("ok"):
        parts.append(f"SMA200={sma['bias']} slope={sma['slope']:.6f}")
    if vp and vp.get("ok"):
        parts.append(f"VP={vp['zone']} POC={vp['poc']:.4f}")
    log_i("🧭 Context | " + " | ".join(parts) if parts else "🧭 Context | NA")

# =================== ENTRY GATE (PATCH 4) ===================
def classify_trend(trend_state, adx, ichi, sma):
    """تصنيف الترند"""
    adx = float(adx or 0.0)
    if trend_state in ("BIG_TREND_UP", "BIG_TREND_DOWN"):
        return "BIG_TREND"
    if trend_state in ("MID_TREND_UP", "MID_TREND_DOWN"):
        return "MID_TREND"

    # fallback
    if adx >= 30 and ichi and ichi.get("ok") and sma and sma.get("ok"):
        if ichi["bias"] in ("BULL","BEAR"):
            return "BIG_TREND"
    if adx >= 22:
        return "MID_TREND"
    return "NO_TREND"

def mid_big_entry_gate(side, trend_state, liq, smc, expl, ichi, sma, vp, adx):
    """
    يرجع (ok, reason)
    """
    tclass = classify_trend(trend_state, adx, ichi, sma)
    if tclass == "NO_TREND":
        return False, "no_trend"

    # منع الدخول ضد إشارات drain القوية
    if liq and liq.get("ok") and liq.get("drain"):
        if (liq["state"] == "DRAIN_DOWN" and side == "BUY") or (liq["state"] == "DRAIN_UP" and side == "SELL"):
            return False, "liquidity_drain_against"

    # انفجار/انهيار = يسمح بدخول مبكر (بس لازم اتجاه عام)
    if expl and expl.get("ok") and expl.get("state") in ("EXPLOSION_UP","EXPLOSION_DOWN"):
        # تأكد اتجاه عام
        if side == expl.get("side"):
            return True, f"explosion_{tclass.lower()}"

    # SMC confluence
    smc_ok = False
    if smc and smc.get("ok"):
        if side == "BUY":
            smc_ok = (smc.get("ob_side") == "BUY") or smc.get("fvg_buy") or smc.get("bos_up")
        else:
            smc_ok = (smc.get("ob_side") == "SELL") or smc.get("fvg_sell") or smc.get("bos_down")

    # Ichimoku/SMA اتجاه (Role = alignment)
    align = True
    if ichi and ichi.get("ok"):
        if side == "BUY" and ichi["bias"] == "BEAR": align=False
        if side == "SELL" and ichi["bias"] == "BULL": align=False
    if sma and sma.get("ok"):
        if side == "BUY" and sma["bias"] == "BELOW": align=False
        if side == "SELL" and sma["bias"] == "ABOVE": align=False

    # Volume Profile proxy: لو POC_NEAR = غالبًا تشوب → خفف دخول
    if vp and vp.get("ok") and vp["zone"] == "POC_NEAR" and float(adx or 0) < 28:
        return False, "vp_chop_zone"

    # شرط دخول أساسي:
    if align and smc_ok:
        return True, f"smc_align_{tclass.lower()}"

    # لو مفيش SMC بس الترند Big قوي جدًا: اسمح
    if align and tclass == "BIG_TREND" and float(adx or 0) >= 32:
        return True, "big_trend_align"

    return False, "no_confluence"

# =================== TP SYSTEM (PATCH 5) ===================
def tp_plan(mode):
    """مستويات TP للأنماط المختلفة"""
    # نسب بسيطة قابلة للتعديل
    if mode == "MID_TREND":
        return {"tps":[0.012, 0.024], "fracs":[0.5, 0.5], "trail_after":0.018}
    if mode == "BIG_TREND":
        return {"tps":[0.010, 0.022, 0.040], "fracs":[0.25,0.35,0.40], "trail_after":0.022}
    return {"tps":[], "fracs":[], "trail_after":None}

def is_pullback_not_reversal(df, side, adx, liq, smc):
    """
    Pullback: عكس بسيط بدون drain + بدون sweep عكسي قوي + ADX مش بينهار
    """
    adx = float(adx or 0)
    if adx < 18:
        return False  # السوق ضعيف أصلاً

    if liq and liq.get("ok"):
        if liq.get("drain"):
            return False
        # sweep عكس اتجاهك يعتبر خطر
        if side=="BUY" and liq.get("sweepH"):
            return False
        if side=="SELL" and liq.get("sweepL"):
            return False

    # لو SMC بيقول BOS ضدك → انعكاس
    if smc and smc.get("ok"):
        if side=="BUY" and smc.get("bos_down"):
            return False
        if side=="SELL" and smc.get("bos_up"):
            return False

    return True

# =================== STEP 1: TREND STATE ENGINE ===================
def detect_trend_state(df, ind, htf_bias=None):
    """
    تحديد حالة الترند بناءً على:
      - ADX واتجاه DI
      - هيكل السوق (HH/HL أو LH/LL)
      - موقع السعر بالنسبة إلى EMA200 (HTF)
      - VWAP bias
    """
    if len(df) < 20:
        return "RANGE", "short_data"

    adx = safe(ind.get("adx", 0))
    di_plus = safe(ind.get("plus_di", 0))
    di_minus = safe(ind.get("minus_di", 0))
    
    # تحديد الاتجاه من DI
    di_trend = "up" if di_plus > di_minus else "down"
    
    # تحليل الهيكل
    highs = df["high"].astype(float).tail(10)
    lows = df["low"].astype(float).tail(10)
    
    # تحقق من Higher Highs and Higher Lows (صاعد)
    if len(highs) >= 3 and len(lows) >= 3:
        hh = highs.iloc[-1] > highs.iloc[-2] > highs.iloc[-3]
        hl = lows.iloc[-1] > lows.iloc[-2] > lows.iloc[-3]
        lh = highs.iloc[-1] < highs.iloc[-2] < highs.iloc[-3]
        ll = lows.iloc[-1] < lows.iloc[-2] < lows.iloc[-3]
        
        if hh and hl:
            structure = "HH_HL"
        elif lh and ll:
            structure = "LH_LL"
        else:
            structure = "RANGE"
    else:
        structure = "RANGE"
    
    # VWAP bias
    vwap = compute_vwap(df)
    price = float(df["close"].iloc[-1])
    vwap_bias = "above" if price > vwap else "below"
    
    # قرار حالة الترند
    if adx >= ADX_BIG_MIN:
        if di_trend == "up" and structure == "HH_HL" and vwap_bias == "above":
            return "BIG_TREND_UP", f"ADX={adx:.1f}, structure={structure}, VWAP={vwap_bias}"
        elif di_trend == "down" and structure == "LH_LL" and vwap_bias == "below":
            return "BIG_TREND_DOWN", f"ADX={adx:.1f}, structure={structure}, VWAP={vwap_bias}"
    
    if adx >= ADX_MID_MIN:
        if di_trend == "up" and (structure == "HH_HL" or vwap_bias == "above"):
            return "MID_TREND_UP", f"ADX={adx:.1f}, structure={structure}, VWAP={vwap_bias}"
        elif di_trend == "down" and (structure == "LH_LL" or vwap_bias == "below"):
            return "MID_TREND_DOWN", f"ADX={adx:.1f}, structure={structure}, VWAP={vwap_bias}"
    
    return "RANGE", f"ADX={adx:.1f}, structure={structure}, VWAP={vwap_bias}"

# =================== STEP 2: LIQUIDITY + SMC ENGINE ===================
def liquidity_state(liq_ctx, flow_ctx, bm_ctx):
    """
    تحديد حالة السيولة:
      - STOP_HUNT_BUYERS: كسر قمة مع حجم عالي وwick علوي كبير
      - STOP_HUNT_SELLERS: كسر قاع مع حجم عالي وwick سفلي كبير
      - ACCUMULATION: نطاق ضيق وحجم منخفض
      - DISTRIBUTION: نطاق ضيق وحجم عالي
      - NEUTRAL: غير ذلك
    """
    if not liq_ctx.get("ok", False):
        return "NEUTRAL", "no_data"
    
    sweep = liq_ctx.get("sweep")
    vol_ratio = safe(liq_ctx.get("vol_ratio", 1.0))
    
    if sweep:
        if sweep["dir"] == "SELL" and vol_ratio > 1.2:
            return "STOP_HUNT_BUYERS", f"sweep_high, vol×={vol_ratio:.2f}"
        elif sweep["dir"] == "BUY" and vol_ratio > 1.2:
            return "STOP_HUNT_SELLERS", f"sweep_low, vol×={vol_ratio:.2f}"
    
    # حجم منخفض ونطاق ضيق => تراكم
    if vol_ratio < 0.8 and liq_ctx.get("state") == "ACCUMULATION":
        return "ACCUMULATION", f"low_vol={vol_ratio:.2f}"
    
    return "NEUTRAL", f"vol×={vol_ratio:.2f}"

def smc_confirm(smc_ctx, side):
    """
    تأكيد SMC للدخول في اتجاه معين:
      - BOS (Break of Structure)
      - OB (Order Block)
      - FVG (Fair Value Gap)
    """
    if not smc_ctx.get("ok", False):
        return False, "no_data"
    
    reasons = []
    
    if side == "buy":
        if smc_ctx.get("bos", {}).get("buy"):
            reasons.append("BOS")
        if smc_ctx.get("ob", {}).get("buy"):
            reasons.append("OB")
        if smc_ctx.get("fvg", {}).get("buy"):
            reasons.append("FVG")
    else:  # sell
        if smc_ctx.get("bos", {}).get("sell"):
            reasons.append("BOS")
        if smc_ctx.get("ob", {}).get("sell"):
            reasons.append("OB")
        if smc_ctx.get("fvg", {}).get("sell"):
            reasons.append("FVG")
    
    ok = len(reasons) >= 2  # يحتاج إلى مؤشرين على الأقل
    return ok, ",".join(reasons) if reasons else "none"

# =================== STEP 3: EXPLOSION / COLLAPSE ENGINE ===================
def explosion_detector(xc_ctx, flow_ctx, bm_ctx):
    """
    كشف الانفجار الحقيقي:
      - range_compress: نطاق مضغوط (ATR منخفض)
      - volume_spike: ارتفاع حجم التداول
      - close_outside_range: إغلاق خارج النطاق
      - cvd_confirm: تأكيد CVD في نفس الاتجاه
    """
    if not xc_ctx.get("ok", False):
        return None, "no_data"
    
    state = xc_ctx.get("state")
    if state in ("EXPLOSION_UP", "EXPLOSION_DOWN"):
        # تحقق من تأكيد CVD
        cvd_trend = flow_ctx.get("cvd_trend", "") if flow_ctx and flow_ctx.get("ok") else ""
        if (state == "EXPLOSION_UP" and cvd_trend == "up") or (state == "EXPLOSION_DOWN" and cvd_trend == "down"):
            return state, f"volume×={xc_ctx.get('vol_x', 0):.2f}, CVD={cvd_trend}"
    
    return None, "no_explosion"

# =================== STEP 4: DECISION LOGIC ENHANCED ===================
def decide_entry_enhanced(trend_state, adx, liq, smc, expl, ichi, sma, vp, htf_bias):
    """
    قرار الدخول المحسن مع نظام الطبقات
    """
    # تحويل Side إلى تنسيق مناسب
    side_long = "BUY" if trend_state in ("BIG_TREND_UP", "MID_TREND_UP") else None
    side_short = "SELL" if trend_state in ("BIG_TREND_DOWN", "MID_TREND_DOWN") else None
    
    results = []
    
    # اختبار الدخول لـ LONG
    if side_long:
        ok_long, why_long = mid_big_entry_gate(side_long, trend_state, liq, smc, expl, ichi, sma, vp, adx)
        if ok_long:
            mode = classify_trend(trend_state, adx, ichi, sma)
            results.append({
                "ok": True, 
                "side": "buy", 
                "mode": mode, 
                "why": why_long,
                "gate_ok": True
            })
    
    # اختبار الدخول لـ SHORT
    if side_short:
        ok_short, why_short = mid_big_entry_gate(side_short, trend_state, liq, smc, expl, ichi, sma, vp, adx)
        if ok_short:
            mode = classify_trend(trend_state, adx, ichi, sma)
            results.append({
                "ok": True, 
                "side": "sell", 
                "mode": mode, 
                "why": why_short,
                "gate_ok": True
            })
    
    # إذا وجدنا دخول مقبول
    if results:
        # نختار الأول (يمكن تعديله لاختيار الأقوى لاحقًا)
        return results[0]
    
    # إذا لم نجد أي دخول
    return {"ok": False, "reason": "no_entry_conditions", "gate_ok": False}

# =================== STEP 5: TRADE MANAGER ENHANCED ===================
def manage_trend_position_enhanced(state, df, ind, mode, side, liq, smc):
    """
    إدارة الصفقة المحسنة مع كشف التصحيح vs الانعكاس
    """
    current_price = float(df["close"].iloc[-1])
    entry_price = state.get("entry", current_price)
    
    if side == "long":
        pnl_pct = (current_price - entry_price) / entry_price * 100
        dir_side = "BUY"
    else:
        pnl_pct = (entry_price - current_price) / entry_price * 100
        dir_side = "SELL"
    
    # جلب خطة TP المناسبة
    tp_config = tp_plan(mode)
    tp_levels = tp_config["tps"]
    tp_fractions = tp_config["fracs"]
    
    # تحقق من TP1
    if not state.get("tp1_done", False) and pnl_pct >= tp_levels[0]:
        # تحقق من Desync Guard قبل TP
        if not guard_reduce_only(ex, SYMBOL, "TP1"):
            return "HOLD", {"reason": "desync_guard_blocked"}
        
        return "TP1", {"level": tp_levels[0], "fraction": tp_fractions[0]}
    
    # تحقق من TP2 (للمود MID وBIG)
    if len(tp_levels) >= 2 and state.get("tp1_done", False) and not state.get("tp2_done", False) and pnl_pct >= tp_levels[1]:
        if not guard_reduce_only(ex, SYMBOL, "TP2"):
            return "HOLD", {"reason": "desync_guard_blocked"}
        
        return "TP2", {"level": tp_levels[1], "fraction": tp_fractions[1]}
    
    # تحقق من TP3 (للبيك ترند فقط)
    if mode == "BIG_TREND" and len(tp_levels) >= 3 and state.get("tp1_done", False) and state.get("tp2_done", False) and not state.get("tp3_done", False) and pnl_pct >= tp_levels[2]:
        if not guard_reduce_only(ex, SYMBOL, "TP3"):
            return "HOLD", {"reason": "desync_guard_blocked"}
        
        return "TP3", {"level": tp_levels[2], "fraction": tp_fractions[2]}
    
    # كشف الانعكاس (يغلق الصفقة فورًا)
    adx = safe(ind.get("adx", 0))
    if not is_pullback_not_reversal(df, dir_side, adx, liq, smc):
        return "EXIT", {"reason": "reversal_detected"}
    
    # إذا كان الترند يضعف
    if adx < 15 and mode == "MID_TREND":
        return "EXIT", {"reason": "ADX_weak_mid"}
    if adx < 20 and mode == "BIG_TREND":
        return "EXIT", {"reason": "ADX_weak_big"}
    
    return "HOLD", {"pnl_pct": pnl_pct, "adx": adx}

# =================== STEP 6: TRUTH LAYER LOGGER ENHANCED ===================
def log_truth_layer_enhanced(trend_state, liq, smc, explosion_state, decision, ichi, sma, vp):
    """
    طبقة اللوج المحسنة مع عرض جميع الطبقات
    """
    print("=" * 100, flush=True)
    print("🧠 ENHANCED TRUTH LAYER - MULTI-LAYER DECISION SYSTEM", flush=True)
    print("=" * 100, flush=True)
    
    # Layer 1: Trend
    print(f"📈 LAYER 1 - TREND STATE: {trend_state[0]} | {trend_state[1]}", flush=True)
    
    # Layer 2: Liquidity
    log_liquidity(liq)
    
    # Layer 3: SMC
    log_smc(smc)
    
    # Layer 4: Explosion/Collapse
    if explosion_state[0]:
        print(f"💥 LAYER 4 - EXPLOSION: {explosion_state[0]} | {explosion_state[1]}", flush=True)
    else:
        print(f"💥 LAYER 4 - EXPLOSION: None", flush=True)
    
    # Layer 5: Context Indicators
    log_context(None, ichi, sma, vp)
    
    print("-" * 100, flush=True)
    
    # Final Decision
    if decision.get("ok"):
        side_display = "LONG" if decision["side"] == "buy" else "SHORT"
        print(f"🎯 FINAL DECISION: {side_display} | Mode: {decision['mode']}", flush=True)
        print(f"✅ REASON: {decision['why']}", flush=True)
        
        if decision["side"] == "buy":
            print(f"🚀 ACTION: ENTER LONG POSITION ({decision['mode']} Mode)", flush=True)
        elif decision["side"] == "sell":
            print(f"🚀 ACTION: ENTER SHORT POSITION ({decision['mode']} Mode)", flush=True)
    else:
        print(f"⏸️ NO ENTRY: {decision.get('reason', 'Unknown')}", flush=True)
        if decision.get("gate_ok") == False:
            print(f"🔒 ENTRY GATE BLOCKED: Requires stronger confluence", flush=True)
    
    print("=" * 100, flush=True)

# =================== CONTEXT INDICATORS ===================
def compute_vwap(df):
    """VWAP calculation"""
    try:
        tp = (df["high"] + df["low"] + df["close"]) / 3.0
        v = df["volume"].astype(float)
        cum_v = v.cumsum()
        cum_tpv = (tp * v).cumsum()
        vwap = (cum_tpv / cum_v).iloc[-1] if cum_v.iloc[-1] > 0 else float(df["close"].iloc[-1])
        return float(vwap)
    except Exception:
        return float(df["close"].iloc[-1]) if len(df) > 0 else 0.0

def ichimoku_bias(df):
    """Ichimoku Cloud Bias"""
    if len(df) < 80:
        return "neutral"
    
    try:
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)

        tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
        kijun  = (high.rolling(26).max() + low.rolling(26).min()) / 2
        span_a = ((tenkan + kijun) / 2).shift(26)
        span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)

        sa = float(span_a.iloc[-1]); sb = float(span_b.iloc[-1]); c = float(close.iloc[-1])
        cloud_top = max(sa, sb); cloud_bot = min(sa, sb)

        if c > cloud_top:
            return "bull"
        if c < cloud_bot:
            return "bear"
        return "neutral"
    except Exception:
        return "neutral"

# =================== LIQUIDITY + SMC FUNCTIONS (SAFE VERSIONS) ===================
def liquidity_ctx(df: pd.DataFrame, atr: float, lookback=60):
    """Safe version with None protection"""
    try:
        if len(df) < max(lookback, 30):
            return {"ok": False, "state": "NO_DATA"}

        h = df["high"].astype(float)
        l = df["low"].astype(float)
        c = df["close"].astype(float)
        o = df["open"].astype(float)
        v = df["volume"].astype(float)

        cur = float(c.iloc[-1])
        prev_high = float(h.iloc[-lookback:-1].max())
        prev_low  = float(l.iloc[-lookback:-1].min())

        # wick ratios
        last_h = float(h.iloc[-1]); last_l = float(l.iloc[-1])
        last_o = float(o.iloc[-1]); last_c = float(c.iloc[-1])
        rng = max(last_h - last_l, 1e-12)
        upper_w = last_h - max(last_o, last_c)
        lower_w = min(last_o, last_c) - last_l
        upper_ratio = upper_w / rng
        lower_ratio = lower_w / rng

        vma = v.rolling(20).mean()
        vol_ratio = float(v.iloc[-1]) / float(vma.iloc[-1]) if len(df) >= 20 and float(vma.iloc[-1]) > 0 else 1.0

        # Sweep detection
        sweep_down = (last_l < prev_low) and (last_c > prev_low) and (lower_ratio >= 0.55) and (vol_ratio >= 1.10)
        sweep_up   = (last_h > prev_high) and (last_c < prev_high) and (upper_ratio >= 0.55) and (vol_ratio >= 1.10)

        sweep = None
        if sweep_down:
            sweep = {"dir": "BUY", "level": prev_low, "wick_ratio": lower_ratio, "vol_ratio": vol_ratio}
        elif sweep_up:
            sweep = {"dir": "SELL", "level": prev_high, "wick_ratio": upper_ratio, "vol_ratio": vol_ratio}

        state = "NEUTRAL"
        if sweep:
            state = "SWEEP_" + sweep["dir"]
        elif vol_ratio < 0.8:
            state = "ACCUMULATION"

        return {
            "ok": True,
            "price": cur,
            "prev_high": prev_high,
            "prev_low": prev_low,
            "vol_ratio": float(vol_ratio),
            "upper_w_ratio": float(upper_ratio),
            "lower_w_ratio": float(lower_ratio),
            "sweep": sweep,
            "state": state
        }
    except Exception as e:
        log_w(f"Liquidity ctx error: {e}")
        return {"ok": False, "error": str(e)}

def smc_ctx(df: pd.DataFrame, atr: float):
    """Safe SMC context"""
    try:
        if len(df) < 40:
            return {"ok": False, "error": "short_df"}
        
        # Detect structure shifts
        def detect_structure_shift(df, direction, lookback=30):
            try:
                close = df["close"].astype(float)
                highs = df["high"].astype(float)
                lows  = df["low"].astype(float)

                c = float(close.iloc[-1])
                recent_high = float(highs.iloc[-lookback:-1].max())
                recent_low  = float(lows.iloc[-lookback:-1].min())

                if direction == "buy":
                    return c > recent_high, {"bos": "up", "level": recent_high, "close": c}
                else:
                    return c < recent_low, {"bos": "down", "level": recent_low, "close": c}
            except Exception:
                return False, {}
        
        # Detect order blocks
        def detect_order_block(df, direction, window=6):
            try:
                o = df["open"].astype(float)
                c = df["close"].astype(float)
                h = df["high"].astype(float)
                l = df["low"].astype(float)

                if len(df) < window + 3:
                    return None

                seg = df.iloc[-(window+3):-1]
                if direction == "buy":
                    red = seg[(seg["close"] < seg["open"])]
                    if red.empty:
                        return None
                    idx = red.index[-1]
                    return {"type": "bull_ob", "low": float(df.loc[idx, "low"]), "high": float(df.loc[idx, "high"])}
                else:
                    green = seg[(seg["close"] > seg["open"])]
                    if green.empty:
                        return None
                    idx = green.index[-1]
                    return {"type": "bear_ob", "low": float(df.loc[idx, "low"]), "high": float(df.loc[idx, "high"])}
            except Exception:
                return None
        
        # Detect FVG
        def detect_fvg(df, direction):
            try:
                h = df["high"].astype(float).values
                l = df["low"].astype(float).values
                if len(h) < 3:
                    return None
                if direction == "buy":
                    gap = l[-1] > h[-3]
                    if gap:
                        return {"type": "bull_fvg", "low": h[-3], "high": l[-1]}
                else:
                    gap = h[-1] < l[-3]
                    if gap:
                        return {"type": "bear_fvg", "low": h[-1], "high": l[-3]}
                return None
            except Exception:
                return None
        
        bos_buy, bos_buy_meta = detect_structure_shift(df, "buy")
        bos_sell, bos_sell_meta = detect_structure_shift(df, "sell")
        ob_buy  = detect_order_block(df, "buy")
        ob_sell = detect_order_block(df, "sell")
        fvg_buy  = detect_fvg(df, "buy")
        fvg_sell = detect_fvg(df, "sell")

        return {
            "ok": True,
            "bos": {"buy": bos_buy, "sell": bos_sell, "buy_meta": bos_buy_meta, "sell_meta": bos_sell_meta},
            "ob": {"buy": ob_buy, "sell": ob_sell},
            "fvg": {"buy": fvg_buy, "sell": fvg_sell},
        }
    except Exception as e:
        log_w(f"SMC ctx error: {e}")
        return {"ok": False, "error": str(e)}

def explosion_collapse_ctx(df: pd.DataFrame, atr: float, lookback=40):
    """Safe explosion/collapse detection"""
    try:
        if len(df) < lookback + 5:
            return {"ok": False, "state": "NO_DATA"}

        o = df["open"].astype(float)
        h = df["high"].astype(float)
        l = df["low"].astype(float)
        c = df["close"].astype(float)
        v = df["volume"].astype(float)

        cur_o = float(o.iloc[-1]); cur_c = float(c.iloc[-1])
        cur_h = float(h.iloc[-1]); cur_l = float(l.iloc[-1])
        rng = max(cur_h - cur_l, 1e-12)
        body = abs(cur_c - cur_o)

        # volume spike
        vma = v.rolling(20).mean()
        vol_x = float(v.iloc[-1]) / float(vma.iloc[-1]) if len(df) >= 20 and float(vma.iloc[-1]) > 0 else 1.0

        # displacement
        disp = (atr > 0 and rng >= 1.6 * atr and body >= 0.55 * rng and vol_x >= 1.4)

        # breakout levels
        prev_high = float(h.iloc[-lookback:-1].max())
        prev_low  = float(l.iloc[-lookback:-1].min())

        breakout_up   = disp and (cur_c > prev_high)
        breakout_down = disp and (cur_c < prev_low)

        if breakout_up:
            return {"ok": True, "state": "EXPLOSION_UP", "side": "BUY", "level": prev_high, "vol_x": vol_x, "disp": True}
        if breakout_down:
            return {"ok": True, "state": "EXPLOSION_DOWN", "side": "SELL", "level": prev_low, "vol_x": vol_x, "disp": True}

        return {"ok": True, "state": "NONE", "vol_x": vol_x, "disp": bool(disp)}
    except Exception as e:
        log_w(f"Explosion ctx error: {e}")
        return {"ok": False, "error": str(e)}

# =================== ENHANCED EMIT SNAPSHOTS ===================
def emit_snapshots_enhanced(exchange, symbol, df, balance_fn=None, pnl_fn=None):
    """
    النسخة المحسنة مع جميع الطبقات الجديدة
    """
    try:
        # جمع البيانات الأساسية
        price = float(df["close"].iloc[-1]) if len(df) > 0 else 0
        
        # حساب المؤشرات الأساسية
        ind = compute_indicators(df)
        adx = safe(ind.get("adx", 0))
        
        # حساب HTF bias
        df_htf = None
        htf_bias = None
        try:
            df_htf = fetch_ohlcv_htf(HTF_TF, limit=300)
            if df_htf is not None and len(df_htf) > 210:
                htf_bias, _ = compute_htf_bias(df_htf)
        except Exception as e:
            log_w(f"HTF fetch error: {e}")
        
        # ===== STEP 1: Trend State =====
        trend_state_result = detect_trend_state(df, ind, htf_bias)
        
        # ===== STEP 2: Liquidity Engine (PATCH 1) =====
        liq = liquidity_read(df)
        
        # ===== STEP 3: SMC Engine (PATCH 2) =====
        smc = smc_read(df)
        
        # ===== STEP 4: Context Indicators (PATCH 3) =====
        ichi = ichimoku_bias_enhanced(df)
        sma = sma_trend(df)
        vp = vp_proxy(df)
        
        # ===== STEP 5: Explosion/Collapse =====
        xc = explosion_collapse_ctx(df, ind.get("atr", 0))
        explosion_state_result = explosion_detector(xc, None, None)
        
        # ===== STEP 6: Enhanced Decision Logic =====
        decision_result = decide_entry_enhanced(
            trend_state_result[0],
            adx,
            liq,
            smc,
            xc,
            ichi,
            sma,
            vp,
            htf_bias
        )
        
        # ===== STEP 7: Enhanced Truth Layer Logging =====
        if LOG_DECISION_LAYER:
            log_truth_layer_enhanced(
                trend_state_result,
                liq,
                smc,
                explosion_state_result,
                decision_result,
                ichi,
                sma,
                vp
            )
        
        # ===== Legacy Logging (للتوافق) =====
        if LOG_ADDONS:
            print(f"📊 Price: {price:.6f} | Trend: {trend_state_result[0]} | ADX: {adx:.1f}", flush=True)
        
        return {
            "price": price,
            "ind": ind,
            "adx": adx,
            "htf_bias": htf_bias,
            "trend_state": trend_state_result,
            "liq": liq,
            "smc": smc,
            "ichi": ichi,
            "sma": sma,
            "vp": vp,
            "xc": xc,
            "explosion_state": explosion_state_result,
            "decision": decision_result,
        }
        
    except Exception as e:
        log_e(f"Emit snapshots enhanced error: {e}")
        return {
            "price": 0,
            "ind": {},
            "adx": 0,
            "htf_bias": None,
            "trend_state": ("ERROR", str(e)),
            "liq": {"ok": False},
            "smc": {"ok": False},
            "ichi": {"ok": False},
            "sma": {"ok": False},
            "vp": {"ok": False},
            "xc": {"ok": False},
            "explosion_state": (None, str(e)),
            "decision": {"ok": False, "reason": str(e)},
        }

# =================== EXECUTION VERIFICATION ===================
def verify_execution_environment():
    """التحقق من بيئة التنفيذ"""
    print(f"⚙️ EXECUTION ENVIRONMENT", flush=True)
    print(f"🔧 EXCHANGE: {EXCHANGE_NAME.upper()} | SYMBOL: {SYMBOL}", flush=True)
    print(f"🔧 EXECUTE_ORDERS: {EXECUTE_ORDERS} | DRY_RUN: {DRY_RUN}", flush=True)
    print(f"🎯 TREND-ONLY MODE: NO SCALP/WEAK | MID/BIG ONLY", flush=True)
    print(f"📈 HTF BIAS: 1H EMA{HTF_EMA_SLOW}/{HTF_EMA_FAST} | SMC+Liquidity", flush=True)
    print(f"⚡ TREND SETTINGS: ADX_MID={ADX_MID_MIN} | ADX_BIG={ADX_BIG_MIN}", flush=True)
    print(f"🧱 ANTI-DESYNC GUARD: ACTIVE", flush=True)
    print(f"🧠 ENHANCED LAYERS: Liquidity+SMC+Context+EntryGate", flush=True)
    
    if not EXECUTE_ORDERS:
        print("🟡 WARNING: EXECUTE_ORDERS=False - البوت في وضع التحليل فقط!", flush=True)
    if DRY_RUN:
        print("🟡 WARNING: DRY_RUN=True - البوت في وضع المحاكاة!", flush=True)

# =================== ENHANCED TRADE LOOP ===================
def trade_loop_enhanced():
    """حلقة تداول محسنة مع نظام القرار الجديد"""
    global wait_for_next_signal_side
    loop_i = 0
    
    while True:
        try:
            # جمع البيانات
            bal = balance_usdt()
            px = price_now()
            df = fetch_ohlcv()
            
            if len(df) < 50:
                time.sleep(BASE_SLEEP)
                continue
            
            spread_bps = orderbook_spread_bps()
            
            # ===== الحصول على القرار من الطبقات المحسنة =====
            snap = emit_snapshots_enhanced(ex, SYMBOL, df,
                                        balance_fn=lambda: float(bal) if bal else None,
                                        pnl_fn=lambda: float(compound_pnl))
            
            decision = snap.get("decision", {})
            trend_state = snap.get("trend_state", ("", ""))
            
            # ===== إدارة الصفقة المفتوحة =====
            if STATE["open"]:
                # استخدام مدير الصفقات المحسن
                action, details = manage_trend_position_enhanced(
                    STATE, df, snap["ind"],
                    STATE.get("mode", "MID_TREND"),
                    STATE.get("side", "long"),
                    snap.get("liq", {"ok": False}),
                    snap.get("smc", {"ok": False})
                )
                
                if action in ["TP1", "TP2", "TP3"]:
                    # تنفيذ TP مع Anti-Desync Guard
                    fraction = details.get("fraction", 0.5)
                    qty_to_close = safe_qty(STATE["qty"] * fraction)
                    
                    if qty_to_close > 0 and EXECUTE_ORDERS and not DRY_RUN:
                        close_side = "sell" if STATE["side"] == "long" else "buy"
                        try:
                            # تحقق من Desync Guard
                            if not guard_reduce_only(ex, SYMBOL, action):
                                log_w(f"⏸️ {action} skipped due to desync guard")
                            else:
                                params = exchange_specific_params(close_side, is_close=True)
                                ex.create_order(SYMBOL, "market", close_side, qty_to_close, None, params)
                                log_g(f"✅ {action}: closed {fraction*100}% at {details['level']}% profit")
                                STATE["qty"] = safe_qty(STATE["qty"] - qty_to_close)
                                
                                # تحديث حالة TP
                                if action == "TP1":
                                    STATE["tp1_done"] = True
                                elif action == "TP2":
                                    STATE["tp2_done"] = True
                                elif action == "TP3":
                                    STATE["tp3_done"] = True
                                    
                        except Exception as e:
                            log_e(f"❌ {action} close failed: {e}")
                
                elif action == "EXIT":
                    # إغلاق كامل مع السبب
                    log_w(f"🚨 EXIT SIGNAL: {details['reason']}")
                    close_market_strict(f"trend_exit_{details['reason']}")
            
            # ===== قرار الدخول المحسن =====
            elif decision.get("ok") and not STATE["open"]:
                # التحقق من الشروط
                if spread_bps and spread_bps > MAX_SPREAD_BPS:
                    log_w(f"⏸️ Spread too high: {spread_bps:.1f}bps > {MAX_SPREAD_BPS}")
                else:
                    # حساب الكمية
                    qty = compute_size(bal, px or snap["price"])
                    
                    if qty > 0:
                        # فتح الصفقة
                        success = open_market_enhanced(
                            decision["side"], qty, px or snap["price"],
                            decision.get("mode", "MID_TREND"), snap
                        )
                        
                        if success:
                            log_g(f"✅ POSITION OPENED: {decision['side'].upper()} | Mode: {decision['mode']}")
                            log_g(f"✅ ENTRY REASON: {decision['why']}")
                            STATE["mode"] = decision["mode"]
                            TREND_STATE["mode"] = decision["mode"]
                            TREND_STATE["phase"] = "ENTRY"
                            TREND_STATE["dir"] = decision["side"]
            
            # النوم حتى التكرار التالي
            sleep_s = NEAR_CLOSE_S if time_to_candle_close(df) <= 10 else BASE_SLEEP
            time.sleep(sleep_s)
            
            loop_i += 1
            
        except Exception as e:
            log_e(f"Loop error: {e}\n{traceback.format_exc()}")
            time.sleep(BASE_SLEEP * 2)

# =================== HELPER FUNCTIONS ===================
_consec_err = 0

def _round_amt(q):
    if q is None: return 0.0
    try:
        d = Decimal(str(q))
        if LOT_STEP and isinstance(LOT_STEP,(int,float)) and LOT_STEP>0:
            step = Decimal(str(LOT_STEP))
            d = (d/step).to_integral_value(rounding=ROUND_DOWN)*step
        prec = int(AMT_PREC) if AMT_PREC and AMT_PREC>=0 else 0
        d = d.quantize(Decimal(1).scaleb(-prec), rounding=ROUND_DOWN)
        if LOT_MIN and isinstance(LOT_MIN,(int,float)) and LOT_MIN>0 and d < Decimal(str(LOT_MIN)): return 0.0
        return float(d)
    except (InvalidOperation, ValueError, TypeError):
        return max(0.0, float(q))

def safe_qty(q): 
    q = _round_amt(q)
    if q<=0: log_w(f"qty invalid after normalize → {q}")
    return q

def fmt(v, d=6, na="—"):
    try:
        if v is None or (isinstance(v,float) and (math.isnan(v) or math.isinf(v))): return na
        return f"{float(v):.{d}f}"
    except Exception:
        return na

def with_retry(fn, tries=3, base_wait=0.4):
    global _consec_err
    for i in range(tries):
        try:
            r = fn()
            _consec_err = 0
            return r
        except Exception:
            _consec_err += 1
            if i == tries-1: raise
            time.sleep(base_wait*(2**i) + random.random()*0.25)

def fetch_ohlcv(limit=600):
    rows = with_retry(lambda: ex.fetch_ohlcv(SYMBOL, timeframe=INTERVAL, limit=limit, params={"type":"swap"}))
    return pd.DataFrame(rows, columns=["time","open","high","low","close","volume"])

def fetch_ohlcv_htf(timeframe=HTF_TF, limit=300):
    try:
        rows = with_retry(lambda: ex.fetch_ohlcv(SYMBOL, timeframe=timeframe, limit=limit, params={"type":"swap"}))
        return pd.DataFrame(rows, columns=["time","open","high","low","close","volume"])
    except Exception:
        return None

def price_now():
    try:
        t = with_retry(lambda: ex.fetch_ticker(SYMBOL))
        return t.get("last") or t.get("close")
    except Exception: return None

def balance_usdt():
    if not MODE_LIVE: return 100.0
    try:
        b = with_retry(lambda: ex.fetch_balance(params={"type":"swap"}))
        return b.get("total",{}).get("USDT") or b.get("free",{}).get("USDT")
    except Exception: return None

def orderbook_spread_bps():
    try:
        ob = with_retry(lambda: ex.fetch_order_book(SYMBOL, limit=5))
        bid = ob["bids"][0][0] if ob["bids"] else None
        ask = ob["asks"][0][0] if ob["asks"] else None
        if not (bid and ask): return None
        mid = (bid+ask)/2.0
        return ((ask-bid)/mid)*10000.0
    except Exception:
        return None

def _interval_seconds(iv: str) -> int:
    iv=(iv or "").lower().strip()
    if iv.endswith("m"): return int(float(iv[:-1]))*60
    if iv.endswith("h"): return int(float(iv[:-1]))*3600
    if iv.endswith("d"): return int(float(iv[:-1]))*86400
    return 15*60

def time_to_candle_close(df: pd.DataFrame) -> int:
    tf = _interval_seconds(INTERVAL)
    if len(df) == 0: return tf
    cur_start_ms = int(df["time"].iloc[-1])
    now_ms = int(time.time()*1000)
    next_close_ms = cur_start_ms + tf*1000
    while next_close_ms <= now_ms:
        next_close_ms += tf*1000
    left = max(0, next_close_ms - now_ms)
    return int(left/1000)

# ========= Bookmap snapshot =========
def bookmap_snapshot(exchange, symbol, depth=BOOKMAP_DEPTH):
    try:
        ob = exchange.fetch_order_book(symbol, depth)
        bids = ob.get("bids", [])[:depth]; asks = ob.get("asks", [])[:depth]
        if not bids or not asks:
            return {"ok": False, "why": "empty"}
        b_sizes = np.array([b[1] for b in bids]); b_prices = np.array([b[0] for b in bids])
        a_sizes = np.array([a[1] for a in asks]); a_prices = np.array([a[0] for a in asks])
        b_idx = b_sizes.argsort()[::-1][:BOOKMAP_TOPWALLS]
        a_idx = a_sizes.argsort()[::-1][:BOOKMAP_TOPWALLS]
        buy_walls = [(float(b_prices[i]), float(b_sizes[i])) for i in b_idx]
        sell_walls = [(float(a_prices[i]), float(a_sizes[i])) for i in a_idx]
        imb = b_sizes.sum() / max(a_sizes.sum(), 1e-12)
        return {"ok": True, "buy_walls": buy_walls, "sell_walls": sell_walls, "imbalance": float(imb)}
    except Exception as e:
        return {"ok": False, "why": f"{e}"}

# ========= Volume flow / Delta & CVD =========
def compute_flow_metrics(df):
    try:
        if len(df) < max(30, FLOW_WINDOW+2):
            return {"ok": False, "why": "short_df"}
        close = df["close"].astype(float).copy()
        vol = df["volume"].astype(float).copy()
        up_mask = close.diff().fillna(0) > 0
        up_vol = (vol * up_mask).astype(float)
        dn_vol = (vol * (~up_mask)).astype(float)
        delta = up_vol - dn_vol
        cvd = delta.cumsum()
        cvd_ma = cvd.rolling(CVD_SMOOTH).mean()
        wnd = delta.tail(FLOW_WINDOW)
        mu = float(wnd.mean()); sd = float(wnd.std() or 1e-12)
        z = float((wnd.iloc[-1] - mu) / sd)
        trend = "up" if (cvd_ma.iloc[-1] - cvd_ma.iloc[-min(CVD_SMOOTH, len(cvd_ma))]) >= 0 else "down"
        return {"ok": True, "delta_last": float(delta.iloc[-1]), "delta_mean": mu, "delta_z": z,
                "cvd_last": float(cvd.iloc[-1]), "cvd_trend": trend, "spike": abs(z) >= FLOW_SPIKE_Z}
    except Exception as e:
        return {"ok": False, "why": str(e)}

# =================== ENHANCED TRADE EXECUTION ===================
def open_market_enhanced(side, qty, price, mode, snap):
    """فتح صفقة محسنة مع طبقات القرار"""
    if qty <= 0: 
        log_e("skip open (qty<=0)")
        return False
    
    if DRY_RUN or not EXECUTE_ORDERS:
        log_i(f"DRY_RUN: {side} {qty:.4f} @ {price:.6f} | mode={mode}")
        return True
    
    try:
        exchange_set_leverage(ex, LEVERAGE, SYMBOL)
        params = exchange_specific_params(side, is_close=False)
        ex.create_order(SYMBOL, "market", side, qty, None, params)
        
        # تحديث الحالة
        STATE.update({
            "open": True, 
            "side": "long" if side=="buy" else "short", 
            "entry": price,
            "qty": qty, 
            "pnl": 0.0, 
            "bars": 0, 
            "trail": None, 
            "breakeven": None,
            "tp1_done": False, 
            "tp2_done": False,
            "tp3_done": False,
            "highest_profit_pct": 0.0, 
            "profit_targets_achieved": 0,
            "mode": mode
        })
        
        log_g(f"✅ EXECUTED: {side.upper()} {qty:.4f} @ {price:.6f}")
        return True
    except Exception as e:
        log_e(f"❌ EXECUTION FAILED: {e}")
        return False

# =================== INDICATORS ===================
def wilder_ema(s: pd.Series, n: int): 
    return s.ewm(alpha=1/n, adjust=False).mean()

def compute_indicators(df: pd.DataFrame):
    if len(df) < max(ATR_LEN, RSI_LEN, ADX_LEN) + 2:
        return {"rsi":50.0,"plus_di":0.0,"minus_di":0.0,"dx":0.0,"adx":0.0,"atr":0.0}
    c,h,l = df["close"].astype(float), df["high"].astype(float), df["low"].astype(float)
    tr = pd.concat([(h-l).abs(), (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    atr = wilder_ema(tr, ATR_LEN)

    delta=c.diff(); up=delta.clip(lower=0.0); dn=(-delta).clip(lower=0.0)
    rs = wilder_ema(up, RSI_LEN) / wilder_ema(dn, RSI_LEN).replace(0,1e-12)
    rsi = 100 - (100/(1+rs))

    up_move=h.diff(); down_move=l.shift(1)-l
    plus_dm=up_move.where((up_move>down_move)&(up_move>0),0.0)
    minus_dm=down_move.where((down_move>up_move)&(down_move>0),0.0)
    plus_di=100*(wilder_ema(plus_dm, ADX_LEN)/atr.replace(0,1e-12))
    minus_di=100*(wilder_ema(minus_dm, ADX_LEN)/atr.replace(0,1e-12))
    dx=(100*(plus_di-minus_di).abs()/(plus_di+minus_di).replace(0,1e-12)).fillna(0.0)
    adx=wilder_ema(dx, ADX_LEN)

    i=len(df)-1
    return {
        "rsi": float(rsi.iloc[i]), "plus_di": float(plus_di.iloc[i]),
        "minus_di": float(minus_di.iloc[i]), "dx": float(dx.iloc[i]),
        "adx": float(adx.iloc[i]), "atr": float(atr.iloc[i])
    }

# =================== RANGE FILTER ===================
def _ema(s, n): return s.ewm(span=n, adjust=False).mean()

def _rng_size(src: pd.Series, qty: float, n: int) -> pd.Series:
    avrng = _ema((src - src.shift(1)).abs(), n); wper = (n*2)-1
    return _ema(avrng, wper) * qty

def _rng_filter(src: pd.Series, rsize: pd.Series):
    rf=[float(src.iloc[0])]
    for i in range(1,len(src)):
        prev=rf[-1]; x=float(src.iloc[i]); r=float(rsize.iloc[i]); cur=prev
        if x - r > prev: cur = x - r
        if x + r < prev: cur = x + r
        rf.append(cur)
    filt=pd.Series(rf, index=src.index, dtype="float64")
    return filt + rsize, filt - rsize, filt

def rf_signal_live(df: pd.DataFrame):
    if len(df) < RF_PERIOD + 3:
        i = -1
        price = float(df["close"].iloc[i]) if len(df) else None
        return {"time": int(df["time"].iloc[i]) if len(df) else int(time.time()*1000),
                "price": price or 0.0, "long": False, "short": False,
                "filter": price or 0.0, "hi": price or 0.0, "lo": price or 0.0}
    src = df[RF_SOURCE].astype(float)
    hi, lo, filt = _rng_filter(src, _rng_size(src, RF_MULT, RF_PERIOD))
    def _bps(a,b):
        try: return abs((a-b)/b)*10000.0
        except Exception: return 0.0
    p_now = float(src.iloc[-1]); p_prev = float(src.iloc[-2])
    f_now = float(filt.iloc[-1]); f_prev = float(filt.iloc[-2])
    long_flip  = (p_prev <= f_prev and p_now > f_now and _bps(p_now, f_now) >= RF_HYST_BPS)
    short_flip = (p_prev >= f_prev and p_now < f_now and _bps(p_now, f_now) >= RF_HYST_BPS)
    return {
        "time": int(df["time"].iloc[-1]), "price": p_now,
        "long": bool(long_flip), "short": bool(short_flip),
        "filter": f_now, "hi": float(hi.iloc[-1]), "lo": float(lo.iloc[-1])
    }

# =================== STATE ===================
STATE = {
    "open": False, "side": None, "entry": None, "qty": 0.0,
    "pnl": 0.0, "bars": 0, "trail": None, "breakeven": None,
    "tp1_done": False, "tp2_done": False, "tp3_done": False,
    "highest_profit_pct": 0.0, "profit_targets_achieved": 0,
}
compound_pnl = 0.0
wait_for_next_signal_side = None

# =================== POSITION FUNCTIONS ===================
def _read_position():
    try:
        poss = ex.fetch_positions(params={"type":"swap"})
        for p in poss:
            sym = (p.get("symbol") or p.get("info",{}).get("symbol") or "")
            if SYMBOL.split(":")[0] not in sym: continue
            qty = abs(float(p.get("contracts") or p.get("info",{}).get("positionAmt") or 0))
            if qty <= 0: return 0.0, None, None
            entry = float(p.get("entryPrice") or p.get("info",{}).get("avgEntryPrice") or 0)
            side_raw = (p.get("side") or p.get("info",{}).get("positionSide") or "").lower()
            side = "long" if ("long" in side_raw or float(p.get("cost",0))>0) else "short"
            return qty, side, entry
    except Exception as e:
        logging.error(f"_read_position error: {e}")
    return 0.0, None, None

def compute_size(balance, price):
    effective = balance or 0.0
    capital = effective * RISK_ALLOC * LEVERAGE
    raw = max(0.0, capital / max(float(price or 0.0), 1e-9))
    return safe_qty(raw)

def close_market_strict(reason="STRICT"):
    global compound_pnl, wait_for_next_signal_side
    exch_qty, exch_side, exch_entry = _read_position()
    if exch_qty <= 0:
        if STATE.get("open"):
            _reset_after_close(reason)
        return
    side_to_close = "sell" if (exch_side=="long") else "buy"
    qty_to_close  = safe_qty(exch_qty)
    attempts=0; last_error=None
    while attempts < CLOSE_RETRY_ATTEMPTS:
        try:
            if MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN:
                # تحقق من Desync Guard قبل الإغلاق
                if not guard_reduce_only(ex, SYMBOL, f"strict_close_{reason}"):
                    log_w(f"⏸️ Strict close skipped due to desync guard")
                    break
                    
                params = exchange_specific_params(side_to_close, is_close=True)
                ex.create_order(SYMBOL,"market",side_to_close,qty_to_close,None,params)
            time.sleep(CLOSE_VERIFY_WAIT_S)
            left_qty, _, _ = _read_position()
            if left_qty <= 0:
                px = price_now() or STATE.get("entry")
                entry_px = STATE.get("entry") or exch_entry or px
                side = STATE.get("side") or exch_side or ("long" if side_to_close=="sell" else "short")
                qty  = exch_qty
                pnl  = (px - entry_px) * qty * (1 if side=="long" else -1)
                compound_pnl += pnl
                log_i(f"STRICT CLOSE {side} reason={reason} pnl={fmt(pnl)} total={fmt(compound_pnl)}")
                _reset_after_close(reason, prev_side=side)
                return
            qty_to_close = safe_qty(left_qty)
            attempts += 1
            log_w(f"strict close retry {attempts}/{CLOSE_RETRY_ATTEMPTS} — residual={fmt(left_qty,4)}")
            time.sleep(CLOSE_VERIFY_WAIT_S)
        except Exception as e:
            last_error = e; logging.error(f"close_market_strict attempt {attempts+1}: {e}"); attempts += 1; time.sleep(CLOSE_VERIFY_WAIT_S)
    log_e(f"STRICT CLOSE FAILED after {CLOSE_RETRY_ATTEMPTS} attempts — last error: {last_error}")

def _reset_after_close(reason, prev_side=None):
    """إعادة تعيين الحالة بعد الإغلاق"""
    global wait_for_next_signal_side
    prev_side = prev_side or STATE.get("side")
    STATE.update({
        "open": False, "side": None, "entry": None, "qty": 0.0,
        "pnl": 0.0, "bars": 0, "trail": None, "breakeven": None,
        "tp1_done": False, "tp2_done": False, "tp3_done": False,
        "highest_profit_pct": 0.0, "profit_targets_achieved": 0,
    })
    TREND_STATE.update({
        "mode": "NONE",
        "phase": "WAIT",
        "dir": None,
        "plan": None
    })
    save_state({"in_position": False, "position_qty": 0})
    
    wait_for_next_signal_side = None
    logging.info(f"AFTER_CLOSE reset complete")

# =================== LOGGING SETUP ===================
def setup_file_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not any(isinstance(h, RotatingFileHandler) and getattr(h, "baseFilename", "").endswith("bot.log")
               for h in logger.handlers):
        fh = RotatingFileHandler("bot.log", maxBytes=5_000_000, backupCount=7, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        logger.addHandler(fh)
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    log_i("log rotation ready")

setup_file_logging()

# =================== API / KEEPALIVE ===================
app = Flask(__name__)
@app.route("/")
def home():
    mode='LIVE' if MODE_LIVE else 'PAPER'
    return f"✅ SUI Trend Master Bot v7.1 — {EXCHANGE_NAME.upper()} — {SYMBOL} {INTERVAL} — {mode} — Trend-Only System"

@app.route("/metrics")
def metrics():
    return jsonify({
        "exchange": EXCHANGE_NAME,
        "symbol": SYMBOL, "interval": INTERVAL, "mode": "live" if MODE_LIVE else "paper",
        "leverage": LEVERAGE, "risk_alloc": RISK_ALLOC, "price": price_now(),
        "state": STATE, "compound_pnl": compound_pnl,
        "trend_state": TREND_STATE,
        "entry_mode": "TREND_ONLY_NO_SCALP",
        "enhanced_layers": True,
        "anti_desync_guard": True
    })

@app.route("/health")
def health():
    return jsonify({
        "ok": True, "exchange": EXCHANGE_NAME, "mode": "live" if MODE_LIVE else "paper",
        "open": STATE["open"], "side": STATE["side"], "qty": STATE["qty"],
        "compound_pnl": compound_pnl, "timestamp": datetime.utcnow().isoformat(),
        "trend_state": TREND_STATE,
        "enhanced_system": True
    }), 200

def keepalive_loop():
    url=(SELF_URL or "").strip().rstrip("/")
    if not url:
        log_w("keepalive disabled (SELF_URL not set)")
        return
    import requests
    sess=requests.Session(); sess.headers.update({"User-Agent":"rf-live-bot/keepalive"})
    log_i(f"KEEPALIVE every 50s → {url}")
    while True:
        try: sess.get(url, timeout=8)
        except Exception: pass
        time.sleep(50)

# =================== BOOT ===================
if __name__ == "__main__":
    log_banner("SUI TREND MASTER BOT v7.1 - ENHANCED TREND-ONLY MULTI-EXCHANGE")
    state = load_state() or {}
    state.setdefault("in_position", False)

    if RESUME_ON_RESTART:
        try:
            # دالة resume_open_position موجودة في الكود الأصلي
            pass
        except Exception as e:
            log_w(f"resume error: {e}\n{traceback.format_exc()}")

    verify_execution_environment()

    print(colored(f"🎯 EXCHANGE: {EXCHANGE_NAME.upper()} • SYMBOL: {SYMBOL} • TIMEFRAME: {INTERVAL}", "yellow"))
    print(colored(f"⚡ RISK: {int(RISK_ALLOC*100)}% × {LEVERAGE}x • TREND-ONLY=ENABLED", "yellow"))
    print(colored(f"📊 TREND MODES: MID (ADX≥{ADX_MID_MIN}) | BIG (ADX≥{ADX_BIG_MIN})", "yellow"))
    print(colored(f"🧭 HTF BIAS: 1H EMA{HTF_EMA_FAST}/{HTF_EMA_SLOW} | SMC+Liquidity", "yellow"))
    print(colored(f"💰 TP PLANS: MID (2 levels) | BIG (3 levels)", "yellow"))
    print(colored(f"🛡️ SAFETY: NO SCALP | NO WEAK | HTF CONFIRMATION", "yellow"))
    print(colored(f"🚀 EXECUTION: {'ACTIVE' if EXECUTE_ORDERS and not DRY_RUN else 'SIMULATION'}", "yellow"))
    print(colored(f"🧠 DECISION LAYERS: 1-Trend | 2-Liquidity | 3-SMC | 4-Explosion | 5-Context", "yellow"))
    print(colored(f"🔒 ANTI-DESYNC GUARD: ACTIVE", "green"))
    print(colored(f"🎯 ENTRY GATE: Mid/Big ONLY with SMC Confluence", "green"))
    
    logging.info("service starting…")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGINT,  lambda *_: sys.exit(0))
    
    import threading
    threading.Thread(target=trade_loop_enhanced, daemon=True).start()
    threading.Thread(target=keepalive_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
