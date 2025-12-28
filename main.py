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
# Exchange Selection
EXCHANGE_NAME = os.getenv("EXCHANGE", "bingx").lower()

# API Keys - Multi-Exchange Support
if EXCHANGE_NAME == "bybit":
    API_KEY = os.getenv("BYBIT_API_KEY", "")
    API_SECRET = os.getenv("BYBIT_API_SECRET", "")
else:  # Default to BingX
    API_KEY = os.getenv("BINGX_API_KEY", "")
    API_SECRET = os.getenv("BINGX_API_SECRET", "")

MODE_LIVE = bool(API_KEY and API_SECRET)

SELF_URL = os.getenv("SELF_URL", "") or os.getenv("RENDER_EXTERNAL_URL", "")
PORT = int(os.getenv("PORT", 5000))

# ==== Run mode / Logging toggles ====
LOG_LEGACY = False
LOG_ADDONS = True

# ==== Execution Switches ====
EXECUTE_ORDERS = True
SHADOW_MODE_DASHBOARD = False
DRY_RUN = False

# ==== Addon: Logging + Recovery Settings ====
BOT_VERSION = f"SUI Trend Master v6.0 — {EXCHANGE_NAME.upper()} Multi-Exchange"
print("🔁 Booting:", BOT_VERSION, flush=True)

STATE_PATH = "./bot_state.json"
RESUME_ON_RESTART = True
RESUME_LOOKBACK_SECS = 60 * 60

# === Addons config ===
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

# HTF (1H) bias
HTF_TF = "1h"
HTF_EMA_FAST = 50
HTF_EMA_SLOW = 200

# =================== SETTINGS ===================
SYMBOL     = os.getenv("SYMBOL", "SUI/USDT:USDT")
INTERVAL   = os.getenv("INTERVAL", "15m")
LEVERAGE   = int(os.getenv("LEVERAGE", 10))
RISK_ALLOC = float(os.getenv("RISK_ALLOC", 0.60))
POSITION_MODE = os.getenv("POSITION_MODE", "oneway")

# RF Settings - Optimized for SUI
RF_SOURCE = "close"
RF_PERIOD = int(os.getenv("RF_PERIOD", 18))  # Optimized for SUI volatility
RF_MULT   = float(os.getenv("RF_MULT", 3.0))  # Adjusted for SUI
RF_LIVE_ONLY = True
RF_HYST_BPS  = 6.0

# Indicators
RSI_LEN = 14
ADX_LEN = 14
ATR_LEN = 14

ENTRY_RF_ONLY = False  # Now using Council decision
MAX_SPREAD_BPS = float(os.getenv("MAX_SPREAD_BPS", 6.0))

# Dynamic TP / trail - Optimized for SUI
TP1_PCT_BASE       = 0.45  # Increased for SUI volatility
TP1_CLOSE_FRAC     = 0.50
BREAKEVEN_AFTER    = 0.30
TRAIL_ACTIVATE_PCT = 1.20
ATR_TRAIL_MULT     = 1.8   # Adjusted for SUI

TREND_TPS       = [0.50, 1.00, 1.80]
TREND_TP_FRACS  = [0.30, 0.30, 0.20]

# Dust guard - Adjusted for SUI (typically higher min qty)
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
    """تحويل قيمة إلى عدد عائم بأمان، وإرجاع القيمة الافتراضية إذا فشل."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return default

# =================== PROFESSIONAL LOGGING ===================
def log_i(msg): print(f"ℹ️ {msg}", flush=True)
def log_g(msg): print(f"✅ {msg}", flush=True)
def log_w(msg): print(f"🟨 {msg}", flush=True)
def log_e(msg): print(f"❌ {msg}", flush=True)
def log_y(msg): print(f"🟡 {msg}", flush=True)  # Added for trend logging

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

# =================== LIQUIDITY SNAPSHOT BUILDER ===================
@dataclass
class LiquiditySnap:
    side_hint: str              # "BUY" / "SELL" / "NEUTRAL"
    sweep_high: bool
    sweep_low: bool
    sweep_price: float
    sweep_ref: float
    vol_spike_x: float
    delta: float
    cvd: float
    imb: float
    buy_walls: list             # [(price, qty), ...]
    sell_walls: list
    vwap_bias: str              # "Above" / "Below" / "Near"
    htf_bias: str               # "UP" / "DOWN" / "MIX"
    note: str                   # human reasons

def _fmt_walls(walls, n=3):
    if not walls:
        return "[]"
    cut = walls[:n]
    return "[" + ", ".join([f"{p:.6f}@{int(q)}" for p, q in cut]) + "]"

def build_liquidity_snapshot(ctx: dict) -> LiquiditySnap:
    """
    ctx: أي قاموس/هيكل عندك فيه نواتج:
      - orderbook / bookmap: imb, buy_walls, sell_walls
      - flow: delta, cvd, z, flow_side
      - sweep: sweep_high/low, sweep_price, sweep_ref
      - vwap bias
      - htf bias
      - volume spike x
    """
    sweep_high = bool(ctx.get("sweep_high", False))
    sweep_low  = bool(ctx.get("sweep_low", False))
    sweep_price = safe(ctx.get("sweep_price", 0.0))
    sweep_ref   = safe(ctx.get("sweep_ref", 0.0))

    delta = safe(ctx.get("delta", 0.0))
    cvd   = safe(ctx.get("cvd", 0.0))
    imb   = safe(ctx.get("imb", 1.0))

    buy_walls  = ctx.get("buy_walls", []) or []
    sell_walls = ctx.get("sell_walls", []) or []

    vol_spike_x = safe(ctx.get("vol_spike_x", 0.0))
    vwap_bias   = str(ctx.get("vwap_bias", "Near"))
    htf_bias    = str(ctx.get("htf_bias", "MIX"))

    # Side hint (سيولة + فلو + سويب)
    hint = "NEUTRAL"
    reasons = []

    if sweep_low:
        reasons.append("SweepLow")
    if sweep_high:
        reasons.append("SweepHigh")

    # imbalance: >1 يعني ميول شراء، <1 ميول بيع (حسب تعريفك)
    if imb >= 1.15:
        reasons.append("ImbBuy")
    elif imb <= 0.87:
        reasons.append("ImbSell")

    if delta > 0:
        reasons.append("Δ+")
    elif delta < 0:
        reasons.append("Δ-")

    if vol_spike_x >= 1.5:
        reasons.append("VolSpike")

    # قرار hint
    buy_score = 0
    sell_score = 0
    if sweep_low: buy_score += 2
    if sweep_high: sell_score += 2
    if imb >= 1.15: buy_score += 1
    if imb <= 0.87: sell_score += 1
    if delta > 0: buy_score += 1
    if delta < 0: sell_score += 1
    if vwap_bias == "Above": buy_score += 1
    if vwap_bias == "Below": sell_score += 1
    if htf_bias == "UP": buy_score += 1
    if htf_bias == "DOWN": sell_score += 1

    if buy_score >= sell_score + 2:
        hint = "BUY"
    elif sell_score >= buy_score + 2:
        hint = "SELL"

    note = ",".join(reasons) if reasons else "-"

    return LiquiditySnap(
        side_hint=hint,
        sweep_high=sweep_high,
        sweep_low=sweep_low,
        sweep_price=sweep_price,
        sweep_ref=sweep_ref,
        vol_spike_x=vol_spike_x,
        delta=delta,
        cvd=cvd,
        imb=imb,
        buy_walls=buy_walls,
        sell_walls=sell_walls,
        vwap_bias=vwap_bias,
        htf_bias=htf_bias,
        note=note
    )

def log_liquidity_snapshot(logger, snap: LiquiditySnap, symbol: str, price: float):
    """لوج السيولة المحسّن"""
    # Emoji + لون حسب hint
    if snap.side_hint == "BUY":
        tag = "🟢 LIQ"
    elif snap.side_hint == "SELL":
        tag = "🔴 LIQ"
    else:
        tag = "🟡 LIQ"

    msg = (
        f"{tag} {symbol} px={price:.6f} | hint={snap.side_hint} | "
        f"Sweep(H={int(snap.sweep_high)} L={int(snap.sweep_low)} "
        f"{safe(snap.sweep_price):.6f}>{safe(snap.sweep_ref):.6f}) | "
        f"VolSpike×={safe(snap.vol_spike_x, 1.0):.2f} | "
        f"Δ={safe(snap.delta):.0f} CVD={safe(snap.cvd):.0f} | "
        f"Imb={safe(snap.imb):.2f} | "
        f"BuyWalls={_fmt_walls(snap.buy_walls)} | "
        f"SellWalls={_fmt_walls(snap.sell_walls)} | "
        f"VWAP={snap.vwap_bias} HTF={snap.htf_bias} | why={snap.note}"
    )
    logger.info(msg)
    print(f"💧 {msg}", flush=True)

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
    else:  # BingX (default)
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
    else:  # BingX
        if POSITION_MODE == "hedge":
            return {"positionSide": "LONG" if side == "buy" else "SHORT", "reduceOnly": is_close}
        return {"positionSide": "BOTH", "reduceOnly": is_close}

def exchange_set_leverage(exchange, leverage, symbol):
    """Exchange-specific leverage setting"""
    try:
        if EXCHANGE_NAME == "bybit":
            # Bybit uses different leverage setting method
            exchange.set_leverage(leverage, symbol)
        else:  # BingX
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

# =================== CANDLES MODULE ===================
def _body(o,c): return abs(c-o)
def _rng(h,l):  return max(h-l, 1e-12)
def _upper_wick(h,o,c): return h - max(o,c)
def _lower_wick(l,o,c): return min(o,c) - l

def _is_doji(o,c,h,l,th=0.1):
    return _body(o,c) <= th * _rng(h,l)

def _engulfing(po,pc,o,c, min_ratio=1.05):
    bull = (c>o) and (pc<po) and _body(po,pc)>0 and _body(o,c)>=min_ratio*_body(po,pc) and (o<=pc and c>=po)
    bear = (c<o) and (pc>po) and _body(po,pc)>0 and _body(o,c)>=min_ratio*_body(po,pc) and (o>=pc and c<=po)
    return bull, bear

def _hammer_like(o,c,h,l, body_max=0.35, wick_ratio=2.0):
    rng, body = _rng(h,l), _body(o,c)
    lower, upper = _lower_wick(l,o,c), _upper_wick(h,o,c)
    hammer  = (body/rng<=body_max) and (lower>=wick_ratio*body) and (upper<=0.4*body)
    inv_ham = (body/rng<=body_max) and (upper>=wick_ratio*body) and (lower<=0.4*body)
    return hammer, inv_ham

def _shooting_star(o,c,h,l, body_max=0.35, wick_ratio=2.0):
    rng, body = _rng(h,l), _body(o,c)
    return (body/rng<=body_max) and (_upper_wick(h,o,c)>=wick_ratio*body) and (_lower_wick(l,o,c)<=0.4*body)

def _marubozu(o,c,h,l, min_body=0.9): return _body(o,c)/_rng(h,l) >= min_body
def _piercing(po,pc,o,c, min_pen=0.5): return (pc<po) and (c>o) and (c>(po - min_pen*(po-pc))) and (o<pc)
def _dark_cloud(po,pc,o,c, min_pen=0.5): return (pc>po) and (c<o) and (c<(po + min_pen*(pc-po))) and (o>pc)

def _tweezer(ph,pl,h,l, tol=0.15):
    top = abs(h-ph) <= tol*max(h,ph)
    bot = abs(l-pl) <= tol*max(l,pl)
    return top, bot

def compute_candles(df):
    """
    يرجّع: buy/sell + score لكل اتجاه + فتائل كبيرة (exhaustion) + tags
    يعمل على آخر شمعة مغلقة df.iloc[-2]
    """
    if len(df) < 5:
        return {"buy":False,"sell":False,"score_buy":0.0,"score_sell":0.0,
                "wick_up_big":False,"wick_dn_big":False,"doji":False,"pattern":None}

    o1,h1,l1,c1 = float(df["open"].iloc[-2]), float(df["high"].iloc[-2]), float(df["low"].iloc[-2]), float(df["close"].iloc[-2])
    o0,h0,l0,c0 = float(df["open"].iloc[-3]), float(df["high"].iloc[-3]), float(df["low"].iloc[-3]), float(df["close"].iloc[-3])

    strength_b = strength_s = 0.0
    tags = []

    bull_eng, bear_eng = _engulfing(o0,c0,o1,c1)
    if bull_eng: strength_b += 2.0; tags.append("bull_engulf")
    if bear_eng: strength_s += 2.0; tags.append("bear_engulf")

    ham, inv = _hammer_like(o1,c1,h1,l1)
    if ham: strength_b += 1.5; tags.append("hammer")
    if inv: strength_s += 1.5; tags.append("inverted_hammer")

    if _shooting_star(o1,c1,h1,l1): strength_s += 1.5; tags.append("shooting_star")
    if _piercing(o0,c0,o1,c1):      strength_b += 1.2; tags.append("piercing")
    if _dark_cloud(o0,c0,o1,c1):    strength_s += 1.2; tags.append("dark_cloud")

    is_doji = _is_doji(o1,c1,h1,l1)
    if is_doji: tags.append("doji")

    tw_top, tw_bot = _tweezer(h0,l0,h1,l1)
    if tw_bot: strength_b += 1.0; tags.append("tweezer_bottom")
    if tw_top: strength_s += 1.0; tags.append("tweezer_top")

    if _marubozu(o1,c1,h1,l1):
        if c1>o1: strength_b += 1.0; tags.append("marubozu_bull")
        else:     strength_s += 1.0; tags.append("marubozu_bear")

    # فتائل كبيرة = إرهاق
    rng1 = _rng(h1,l1); up = _upper_wick(h1,o1,c1); dn = _lower_wick(l1,o1,c1)
    wick_up_big = (up >= 1.2*_body(o1,c1)) and (up >= 0.4*rng1)
    wick_dn_big = (dn >= 1.2*_body(o1,c1)) and (dn >= 0.4*rng1)

    if is_doji:  # تخفيف ثقة
        strength_b *= 0.8; strength_s *= 0.8

    return {
        "buy": strength_b>0, "sell": strength_s>0,
        "score_buy": round(strength_b,2), "score_sell": round(strength_s,2),
        "wick_up_big": bool(wick_up_big), "wick_dn_big": bool(wick_dn_big),
        "doji": bool(is_doji), "pattern": ",".join(tags) if tags else None
    }

# =================== EXECUTION VERIFICATION ===================
def verify_execution_environment():
    """التحقق من بيئة التنفيذ عند الإقلاع"""
    print(f"⚙️ EXECUTION ENVIRONMENT", flush=True)
    print(f"🔧 EXCHANGE: {EXCHANGE_NAME.upper()} | SYMBOL: {SYMBOL}", flush=True)
    print(f"🔧 EXECUTE_ORDERS: {EXECUTE_ORDERS} | DRY_RUN: {DRY_RUN}", flush=True)
    print(f"🎯 TREND-ONLY MODE: NO SCALP/WEAK | MID/BIG ONLY", flush=True)
    print(f"📈 HTF BIAS: 1H EMA{HTF_EMA_SLOW}/{HTF_EMA_FAST} | SMC+Liquidity", flush=True)
    print(f"⚡ TREND SETTINGS: ADX_MID={ADX_MID_MIN} | ADX_BIG={ADX_BIG_MIN}", flush=True)
    
    if not EXECUTE_ORDERS:
        print("🟡 WARNING: EXECUTE_ORDERS=False - البوت في وضع التحليل فقط!", flush=True)
    if DRY_RUN:
        print("🟡 WARNING: DRY_RUN=True - البوت في وضع المحاكاة!", flush=True)

# =================== ENHANCED INDICATORS ===================
def sma(series, n: int):
    return series.rolling(n, min_periods=1).mean()

def compute_rsi(close, n: int = 14):
    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    roll_up = up.ewm(span=n, adjust=False).mean()
    roll_down = down.ewm(span=n, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, 1e-12)
    rsi = 100 - (100/(1+rs))
    return rsi.fillna(50)

def rsi_ma_context(df):
    if len(df) < max(RSI_MA_LEN, 14):
        return {"rsi": 50, "rsi_ma": 50, "cross": "none", "trendZ": "none", "in_chop": True}
    
    rsi = compute_rsi(df['close'].astype(float), 14)
    rsi_ma = sma(rsi, RSI_MA_LEN)
    
    cross = "none"
    if len(rsi) >= 2:
        if (rsi.iloc[-2] <= rsi_ma.iloc[-2]) and (rsi.iloc[-1] > rsi_ma.iloc[-1]):
            cross = "bull"
        elif (rsi.iloc[-2] >= rsi_ma.iloc[-2]) and (rsi.iloc[-1] < rsi_ma.iloc[-1]):
            cross = "bear"
    
    above = (rsi > rsi_ma)
    below = (rsi < rsi_ma)
    persist_bull = above.tail(RSI_TREND_PERSIST).all() if len(above) >= RSI_TREND_PERSIST else False
    persist_bear = below.tail(RSI_TREND_PERSIST).all() if len(below) >= RSI_TREND_PERSIST else False
    
    current_rsi = float(rsi.iloc[-1])
    in_chop = RSI_NEUTRAL_BAND[0] <= current_rsi <= RSI_NEUTRAL_BAND[1]
    
    return {
        "rsi": current_rsi,
        "rsi_ma": float(rsi_ma.iloc[-1]),
        "cross": cross,
        "trendZ": "bull" if persist_bull else ("bear" if persist_bear else "none"),
        "in_chop": in_chop
    }

def golden_zone_check(df, ind=None, side_hint=None):
    """اكتشاف المناطق الذهبية (فيبو 0.618-0.786) مع تأكيدات"""
    if len(df) < 30:
        return {"ok": False, "score": 0.0, "zone": None, "reasons": ["short_df"]}
    
    try:
        h = df['high'].astype(float)
        l = df['low'].astype(float)
        c = df['close'].astype(float)
        v = df['volume'].astype(float)
        
        swing_hi = h.rolling(10).max().iloc[-1]
        swing_lo = l.rolling(10).min().iloc[-1]
        
        if swing_hi <= swing_lo:
            return {"ok": False, "score": 0.0, "zone": None, "reasons": ["flat_market"]}
        
        f618 = swing_lo + 0.618 * (swing_hi - swing_lo)
        f786 = swing_lo + 0.786 * (swing_hi - swing_lo)
        last_close = float(c.iloc[-1])
        
        vol_ma20 = v.rolling(20).mean().iloc[-1]
        vol_ok = float(v.iloc[-1]) >= vol_ma20 * 0.8
        
        current_open = float(df['open'].iloc[-1])
        current_high = float(h.iloc[-1])
        current_low = float(l.iloc[-1])
        
        body = abs(last_close - current_open)
        wick_up = current_high - max(last_close, current_open)
        wick_down = min(last_close, current_open) - current_low
        
        bull_candle = wick_down > (body * 1.2) and last_close > current_open
        bear_candle = wick_up > (body * 1.2) and last_close < current_open
        
        adx = ind.get('adx', 0) if ind else 0
        rsi_ctx = rsi_ma_context(df)
        
        score = 0.0
        zone_type = None
        reasons = []
        
        if f618 <= last_close <= f786 and bull_candle:
            score += 4.0
            reasons.append("فيبو_قاع+شمعة_صاعدة")
            if adx >= GZ_REQ_ADX:
                score += 2.0
                reasons.append("ADX_قوي")
            if rsi_ctx["cross"] == "bull" or rsi_ctx["trendZ"] == "bull":
                score += 1.5
                reasons.append("RSI_إيجابي")
            if vol_ok:
                score += 0.5
                reasons.append("حجم_مرتفع")
            
            if score >= GZ_MIN_SCORE:
                zone_type = "golden_bottom"
        
        elif f618 <= last_close <= f786 and bear_candle:
            score += 4.0
            reasons.append("فيبو_قمة+شمعة_هابطة")
            if adx >= GZ_REQ_ADX:
                score += 2.0
                reasons.append("ADX_قوي")
            if rsi_ctx["cross"] == "bear" or rsi_ctx["trendZ"] == "bear":
                score += 1.5
                reasons.append("RSI_سلبي")
            if vol_ok:
                score += 0.5
                reasons.append("حجم_مرتفع")
            
            if score >= GZ_MIN_SCORE:
                zone_type = "golden_top"
        
        ok = zone_type is not None and ALLOW_GZ_ENTRY
        return {
            "ok": ok,
            "score": score,
            "zone": {"type": zone_type, "f618": f618, "f786": f786} if zone_type else None,
            "reasons": reasons
        }
        
    except Exception as e:
        return {"ok": False, "score": 0.0, "zone": None, "reasons": [f"error: {e}"]}

def decide_strategy_mode(df, adx=None, di_plus=None, di_minus=None, rsi_ctx=None):
    """تحديد نمط التداول: MID_TREND أم BIG_TREND"""
    if adx is None or di_plus is None or di_minus is None:
        ind = compute_indicators(df)
        adx = ind.get('adx', 0)
        di_plus = ind.get('plus_di', 0)
        di_minus = ind.get('minus_di', 0)
    
    if rsi_ctx is None:
        rsi_ctx = rsi_ma_context(df)
    
    di_spread = abs(di_plus - di_minus)
    
    strong_trend = (
        (adx >= ADX_BIG_MIN and di_spread >= DI_SPREAD_TREND) or
        (rsi_ctx["trendZ"] in ("bull", "bear") and not rsi_ctx["in_chop"])
    )
    
    medium_trend = (
        (adx >= ADX_MID_MIN and di_spread >= DI_SPREAD_TREND/2) or
        (rsi_ctx["trendZ"] in ("bull", "bear"))
    )
    
    if strong_trend:
        mode = "BIG_TREND"
        why = "adx_strong_trend"
    elif medium_trend:
        mode = "MID_TREND"
        why = "adx_mid_trend"
    else:
        mode = "NO_TREND"
        why = "no_trend_detected"
    
    return {"mode": mode, "why": why}

# =================== TREND STATE ENGINE ===================
TREND_STATE = {
    "mode": "NONE",           # NONE | MID_TREND | BIG_TREND
    "dir": None,              # "buy" | "sell"
    "phase": "WAIT",          # WAIT | ENTRY | RUN | CORRECTION | EXIT
    "strength": 0.0,          # score 0..10
    "htf_bias": None,         # "buy" | "sell" | None
    "last_plan_ts": 0,
    "plan": None,             # dict: {tp_levels, fracs, reasons}
}

def _ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def compute_htf_bias(df_htf):
    """
    Bias من فريم الساعة:
    - price فوق EMA200 + EMA50>EMA200 => buy bias
    - price تحت EMA200 + EMA50<EMA200 => sell bias
    """
    try:
        c = df_htf["close"].astype(float)
        ema_fast = _ema(c, HTF_EMA_FAST)
        ema_slow = _ema(c, HTF_EMA_SLOW)

        px = float(c.iloc[-1])
        ef = float(ema_fast.iloc[-1])
        es = float(ema_slow.iloc[-1])

        if px > es and ef > es:
            return "buy", {"px": px, "ema_fast": ef, "ema_slow": es}
        if px < es and ef < es:
            return "sell", {"px": px, "ema_fast": ef, "ema_slow": es}
        return None, {"px": px, "ema_fast": ef, "ema_slow": es}
    except Exception:
        return None, {}

def classify_trend_mode(ind):
    """
    يعتمد على ADX (عندك في ind غالباً).
    """
    adx = float(ind.get("adx", 0.0) or 0.0)
    if adx >= ADX_BIG_MIN:
        return "BIG_TREND"
    if adx >= ADX_MID_MIN:
        return "MID_TREND"
    return "NONE"

def build_tp_plan(mode, atr=None, price=None):
    """
    بناء خطة جني الأرباح حسب النمط
    MID: 2 TP | BIG: 3 TP | مع تعزيز في حالة الانفجار
    """
    atr = max(float(atr or 0), 1e-9)
    price = float(price or 0)
    
    if mode == "BIG_TREND":
        # 3 مراحل للترند القوي
        tp_pcts = [0.012, 0.025, 0.040]  # 1.2%، 2.5%، 4.0%
        fractions = [0.30, 0.35, 0.35]
        trail_mult = 1.6
    elif mode == "MID_TREND":
        # 2 مراحل للترند المتوسط
        tp_pcts = [0.010, 0.022]  # 1.0%، 2.2%
        fractions = [0.45, 0.55]
        trail_mult = 1.4
    else:
        # نمط افتراضي
        tp_pcts = [0.008, 0.015]
        fractions = [0.50, 0.50]
        trail_mult = 1.2
    
    # تحقق أن مجموع الكسور <= 1.0
    if sum(fractions) > 1.0:
        # تعديل الكسور لتناسب المجموع
        total = sum(fractions)
        fractions = [f/total for f in fractions]
    
    return {
        "tp_pcts": tp_pcts,
        "fractions": fractions,
        "trail_mult": trail_mult,
        "mode": mode
    }

# =================== LIQUIDITY + SMC (TBE) ===================
def wick_ratio(df):
    o = float(df["open"].iloc[-1])
    h = float(df["high"].iloc[-1])
    l = float(df["low"].iloc[-1])
    c = float(df["close"].iloc[-1])
    rng = max(h - l, 1e-9)
    upper = h - max(o, c)
    lower = min(o, c) - l
    return upper / rng, lower / rng

def detect_liquidity_sweep(df, side, lookback=20, vol_mult=1.5):
    """
    Sweep = كسر High/Low سابق + wick كبير + volume spike
    side="buy" => sweep low (stop-hunt down)
    side="sell" => sweep high (stop-hunt up)
    """
    try:
        highs = df["high"].astype(float)
        lows  = df["low"].astype(float)
        vols  = df["volume"].astype(float)

        h_last = float(highs.iloc[-1]); l_last = float(lows.iloc[-1])
        prev_high = float(highs.iloc[-lookback:-1].max())
        prev_low  = float(lows.iloc[-lookback:-1].min())

        v = float(vols.iloc[-1])
        vma = float(vols.iloc[-lookback:-1].mean())

        u, d = wick_ratio(df)
        vol_ok = (vma > 0 and v >= vma * vol_mult)

        if side == "buy":
            # sweep low => lower wick كبير + كسر low سابق
            sweep = (l_last < prev_low) and (d >= 0.55) and vol_ok
            return sweep, {"prev_low": prev_low, "wick_d": d, "vol": v, "vma": vma}
        else:
            # sweep high => upper wick كبير + كسر high سابق
            sweep = (h_last > prev_high) and (u >= 0.55) and vol_ok
            return sweep, {"prev_high": prev_high, "wick_u": u, "vol": v, "vma": vma}
    except Exception:
        return False, {}

def detect_structure_shift(df, direction, lookback=30):
    """
    Structure Shift بسيط:
    - BUY: close يكسر آخر swing high
    - SELL: close يكسر آخر swing low
    """
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

def detect_fvg(df, direction):
    """
    FVG بسيط (3 candles):
    BUY FVG: low[0] > high[2] (gap up)
    SELL FVG: high[0] < low[2] (gap down)
    """
    try:
        h = df["high"].astype(float).values
        l = df["low"].astype(float).values
        if len(h) < 3:
            return None
        # آخر 3 شموع: [-3],[-2],[-1]
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

def detect_order_block(df, direction, window=6):
    """
    OB مبسّط:
    - BUY OB: آخر شمعة هبوط قبل اندفاع صاعد (أدنى close ضمن window ثم بعدها صعود)
    - SELL OB: آخر شمعة صعود قبل اندفاع هابط
    """
    try:
        o = df["open"].astype(float)
        c = df["close"].astype(float)
        h = df["high"].astype(float)
        l = df["low"].astype(float)

        if len(df) < window + 3:
            return None

        seg = df.iloc[-(window+3):-1]
        if direction == "buy":
            # شمعة حمراء قوية قبل اندفاع
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

def price_in_zone(px, zone, pad_bps=8):
    if not zone:
        return False
    lo = float(zone["low"]); hi = float(zone["high"])
    pad = (hi - lo) * 0.05
    lo2 = lo - pad
    hi2 = hi + pad
    return lo2 <= px <= hi2

def trend_birth_entry(df15, ind, htf_bias):
    """
    قرار دخول خارج RF (Override):
    - لازم HTF bias يوافق
    - Sweep + Structure Shift + Retest OB/FVG
    """
    mode = classify_trend_mode(ind)
    if mode == "NONE":
        return None

    if not htf_bias:
        return None

    px = float(df15["close"].astype(float).iloc[-1])

    # direction = htf_bias
    direction = htf_bias

    sweep_ok, sweep_info = detect_liquidity_sweep(df15, direction)
    bos_ok, bos_info = detect_structure_shift(df15, direction)

    ob = detect_order_block(df15, direction)
    fvg = detect_fvg(df15, direction)

    retest_ok = price_in_zone(px, ob) or price_in_zone(px, fvg)

    reasons = []
    if sweep_ok: reasons.append("Sweep")
    if bos_ok: reasons.append("BOS/CHoCH")
    if price_in_zone(px, ob): reasons.append("Retest_OB")
    if price_in_zone(px, fvg): reasons.append("Retest_FVG")

    # لازم 3/3: sweep + bos + retest
    if sweep_ok and bos_ok and retest_ok:
        return {
            "side": direction,     # "buy" / "sell"
            "mode": mode,          # MID_TREND / BIG_TREND
            "reasons": reasons,
            "sweep": sweep_info,
            "bos": bos_info,
            "ob": ob,
            "fvg": fvg
        }
    return None

# =================== CORRECTION vs REVERSAL ===================
def _is_correction(df, ind, side):
    """
    تصحيح في Big/Mid:
    - ADX لا ينهار (مش 3 شموع نزول قوي)
    - مفيش كسر هيكلي عكسي واضح
    """
    try:
        adx = float(ind.get("adx", 0.0) or 0.0)
        # لو ADX عالي وثابت => غالباً تصحيح
        if adx >= 25:
            return True
        return False
    except Exception:
        return False

def _is_reversal(df, ind, side):
    """
    انعكاس حقيقي:
    - CHoCH عكسي بسيط + Volume spike
    - DI ينقلب ضدك
    """
    try:
        adx = float(ind.get("adx", 0.0) or 0.0)
        di_p = float(ind.get("di_plus", 0.0) or 0.0)
        di_m = float(ind.get("di_minus", 0.0) or 0.0)

        # لو BUY والـDI- صار أقوى بوضوح
        if side == "buy" and di_m > di_p + 3 and adx >= 20:
            return True
        if side == "sell" and di_p > di_m + 3 and adx >= 20:
            return True
        return False
    except Exception:
        return False

# =================== LIQUIDITY + SMC ENGINE (STEP 2) ===================
def _sma(s: pd.Series, n: int):
    return s.rolling(n).mean()

def liquidity_ctx(df: pd.DataFrame, atr: float, lookback=60):
    """
    Liquidity read:
    - Pools: recent swing highs/lows zones
    - Sweep: wick breach + close back inside
    - Accumulation/Distribution: ATR compression + volume behavior
    """
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

    vma = float(_sma(v, 20).iloc[-1]) if len(df) >= 20 else float(v.mean())
    vol_ratio = (float(v.iloc[-1]) / max(vma, 1e-12))

    # Sweep detection (stop-hunt)
    sweep_down = (last_l < prev_low) and (last_c > prev_low) and (lower_ratio >= 0.55) and (vol_ratio >= 1.10)
    sweep_up   = (last_h > prev_high) and (last_c < prev_high) and (upper_ratio >= 0.55) and (vol_ratio >= 1.10)

    sweep = None
    if sweep_down:
        sweep = {"dir": "BUY", "level": prev_low, "wick_ratio": lower_ratio, "vol_ratio": vol_ratio}
    elif sweep_up:
        sweep = {"dir": "SELL", "level": prev_high, "wick_ratio": upper_ratio, "vol_ratio": vol_ratio}

    # Compression (accumulation) vs expansion
    # compression: ATR low vs its MA, candles small
    atr_series = df["high"].astype(float).sub(df["low"].astype(float)).rolling(14).mean()
    atr_ma = float(_sma(atr_series, 20).iloc[-1]) if len(df) >= 40 else float(atr_series.iloc[-1])
    compression = (atr_ma > 0 and atr_series.iloc[-1] < 0.75 * atr_ma and vol_ratio < 1.2)

    state = "NEUTRAL"
    if sweep:
        state = "SWEEP_" + sweep["dir"]
    elif compression:
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

def smc_ctx(df: pd.DataFrame, atr: float):
    """
    Lightweight SMC:
    - BOS: break recent high/low
    - OB: last opposite candle before displacement
    - FVG: simple 3-candle gap
    """
    out = {"ok": False}
    if len(df) < 40:
        return out

    # BOS
    bos_buy, bos_buy_meta = detect_structure_shift(df, "buy", lookback=30)
    bos_sell, bos_sell_meta = detect_structure_shift(df, "sell", lookback=30)

    # OB
    ob_buy  = detect_order_block(df, "buy", window=6)
    ob_sell = detect_order_block(df, "sell", window=6)

    # FVG
    fvg_buy  = detect_fvg(df, "buy")
    fvg_sell = detect_fvg(df, "sell")

    out.update({
        "ok": True,
        "bos": {"buy": bos_buy, "sell": bos_sell, "buy_meta": bos_buy_meta, "sell_meta": bos_sell_meta},
        "ob": {"buy": ob_buy, "sell": ob_sell},
        "fvg": {"buy": fvg_buy, "sell": fvg_sell},
    })
    return out

def log_liquidity_ctx(liq: dict, smc: dict):
    if not liq or not liq.get("ok"):
        log_i("💧 Liquidity: n/a")
        return

    sweep = liq.get("sweep")
    if sweep:
        log_g(
            f"💧 LIQ SWEEP {sweep['dir']} | level={safe(sweep.get('level')):.6f} "
            f"wick={safe(sweep.get('wick_ratio')):.2f} vol×={safe(sweep.get('vol_ratio'), 1.0):.2f} | state={liq.get('state')}"
        )
    else:
        log_i(
            f"💧 Liquidity | state={liq.get('state')} "
            f"| prevH={safe(liq.get('prev_high')):.6f} prevL={safe(liq.get('prev_low')):.6f} "
            f"| vol×={safe(liq.get('vol_ratio'), 1.0):.2f} wickU={safe(liq.get('upper_w_ratio')):.2f} wickD={safe(liq.get('lower_w_ratio')):.2f}"
        )

    if smc and smc.get("ok"):
        bos = smc.get("bos", {})
        ob  = smc.get("ob", {})
        fvg = smc.get("fvg", {})
        log_i(
            f"🧱 SMC | BOS(buy={bos.get('buy')}, sell={bos.get('sell')}) "
            f"| OB(buy={'Y' if ob.get('buy') else 'n'}, sell={'Y' if ob.get('sell') else 'n'}) "
            f"| FVG(buy={'Y' if fvg.get('buy') else 'n'}, sell={'Y' if fvg.get('sell') else 'n'})"
        )

# =================== EXPLOSION / COLLAPSE ENGINE (STEP 3) ===================
def detect_explosion_collapse(df, ind, flow, bookmap):
    """
    Explosion:
      - Range expansion
      - Volume spike
      - CVD confirm
      - Liquidity walls eaten

    Collapse:
      - Long wicks
      - Negative delta spike
      - Liquidity drain
    """
    if len(df) < 30:
        return {"state": None}
    
    # حماية من القيم None
    flow = flow or {}
    bookmap = bookmap or {}

    # === Price action ===
    atr = safe(ind.get("atr", 0))
    o = float(df["open"].iloc[-1])
    h = float(df["high"].iloc[-1])
    l = float(df["low"].iloc[-1])
    c = float(df["close"].iloc[-1])

    rng = max(h - l, 1e-9)
    body = abs(c - o)

    # === Volume ===
    vol = float(df["volume"].iloc[-1])
    vol_ma = float(df["volume"].rolling(20).mean().iloc[-1])
    vol_spike = vol_ma > 0 and vol >= vol_ma * 1.8

    # === Flow ===
    delta = safe(flow.get("delta_last", 0))
    z = safe(flow.get("delta_z", 0))
    cvd_trend = flow.get("cvd_trend", "")

    # === Bookmap ===
    imb = safe(bookmap.get("imbalance", 1.0))

    # === Wick ratios ===
    upper = h - max(o, c)
    lower = min(o, c) - l

    # ================== EXPLOSION ==================
    if (
        body >= 1.2 * atr and
        vol_spike and
        abs(z) >= 1.6 and
        ((delta > 0 and cvd_trend == "up") or (delta < 0 and cvd_trend == "down")) and
        (imb >= 1.3 or imb <= 0.77)
    ):
        return {
            "state": "EXPLOSION",
            "dir": "buy" if delta > 0 else "sell",
            "reasons": ["ATR_EXPAND", "VOL_SPIKE", "CVD_CONFIRM", "LIQ_EATEN"],
            "atr": atr,
            "delta": delta,
            "z": z,
            "imb": imb
        }

    # ================== COLLAPSE ==================
    if (
        rng >= 1.5 * atr and
        (upper >= 0.45 * rng or lower >= 0.45 * rng) and
        vol_spike and
        abs(z) >= 1.6
    ):
        return {
            "state": "COLLAPSE",
            "dir": "sell" if upper > lower else "buy",
            "reasons": ["LIQ_DRAIN", "LONG_WICK", "VOL_SPIKE"],
            "atr": atr,
            "delta": delta,
            "z": z,
            "imb": imb
        }

    return {"state": None}

def explosion_collapse_ctx(df: pd.DataFrame, atr: float, lookback=40):
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
    vma = float(v.rolling(20).mean().iloc[-1]) if len(df) >= 30 else float(v.mean())
    vol_x = float(v.iloc[-1]) / max(vma, 1e-12)

    # displacement = candle range big vs ATR
    atr = safe(atr or 0.0)
    disp = (atr > 0 and rng >= 1.6 * atr and body >= 0.55 * rng and vol_x >= 1.4)

    # breakout levels
    prev_high = float(h.iloc[-lookback:-1].max())
    prev_low  = float(l.iloc[-lookback:-1].min())

    breakout_up   = disp and (cur_c > prev_high)
    breakout_down = disp and (cur_c < prev_low)

    # Liquidity drain (انهيار): كسر + إغلاق قرب القاع + حجم
    drain_down = (cur_c < prev_low) and (body >= 0.6*rng) and (vol_x >= 1.6)
    drain_up   = (cur_c > prev_high) and (body >= 0.6*rng) and (vol_x >= 1.6)

    if breakout_up:
        return {"ok": True, "state": "EXPLOSION_UP", "side": "BUY", "level": prev_high, "vol_x": vol_x, "disp": True}
    if breakout_down:
        return {"ok": True, "state": "EXPLOSION_DOWN", "side": "SELL", "level": prev_low, "vol_x": vol_x, "disp": True}

    if drain_down:
        return {"ok": True, "state": "DRAIN_DOWN", "side": "SELL", "level": prev_low, "vol_x": vol_x, "disp": False}
    if drain_up:
        return {"ok": True, "state": "DRAIN_UP", "side": "BUY", "level": prev_high, "vol_x": vol_x, "disp": False}

    return {"ok": True, "state": "NONE", "vol_x": vol_x, "disp": bool(disp)}

def log_explode(xc: dict):
    if not xc or not xc.get("ok"):
        return
    st = xc.get("state")
    if st and st != "NONE":
        log_g(f"💥 STEP3 {st} | side={xc.get('side')} level={safe(xc.get('level')):.6f} vol×={safe(xc.get('vol_x'), 1.0):.2f}")
    else:
        log_i(f"💥 STEP3 none | vol×={safe(xc.get('vol_x'), 1.0):.2f} disp={xc.get('disp')}")

# =================== CONTEXT INDICATORS (VWAP + ICHIMOKU) ===================
def compute_vwap(df):
    """حساب VWAP للجلسة"""
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
    """Ichimoku Cloud Bias بسيط"""
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

# =================== ENTRY GATE FROM LIQUIDITY/SMC/STEP3 ===================
def entry_gate_from_liq_smc_xc(snap: dict, desired: str, mode: str):
    """
    بوابة دخول تعتمد على Liquidity + SMC + Step3
    desired: "buy" أو "sell"
    mode: "MID_TREND" أو "BIG_TREND"
    """
    liq = snap.get("liq", {})
    smc = snap.get("smc", {})
    xc  = snap.get("xc", {})

    score = 0
    reasons = []

    # 1) Sweep
    sw = liq.get("sweep")
    if sw and sw.get("dir") == ("BUY" if desired=="buy" else "SELL"):
        score += 1; reasons.append("LIQ_SWEEP")

    # 2) BOS
    bos = smc.get("bos", {})
    if desired=="buy" and bos.get("buy"):
        score += 1; reasons.append("BOS_UP")
    if desired=="sell" and bos.get("sell"):
        score += 1; reasons.append("BOS_DOWN")

    # 3) Step3 explosion/drain
    if xc and xc.get("ok") and xc.get("side"):
        if (desired=="buy" and xc["side"]=="BUY") or (desired=="sell" and xc["side"]=="SELL"):
            score += 1; reasons.append(xc.get("state","STEP3"))

    need = 2 if mode=="MID_TREND" else 3
    ok = score >= need

    return ok, score, reasons

# =================== POSITION SYNC GUARD ===================
def exchange_has_position():
    """التحقق من وجود Position فعلي على Exchange"""
    try:
        if EXCHANGE_NAME == "bybit":
            positions = ex.fetch_positions([SYMBOL])
        else:
            positions = ex.fetch_positions()
        
        for pos in positions:
            symbol = pos.get('symbol') or pos.get('info', {}).get('symbol')
            if symbol and SYMBOL in symbol:
                contracts = float(pos.get('contracts') or pos.get('size') or 0)
                if abs(contracts) > 0:
                    side = "long" if contracts > 0 else "short"
                    return True, side, abs(contracts)
        return False, None, 0.0
    except Exception as e:
        log_w(f"position sync warn: {e}")
        return None, None, 0.0  # unknown

def sync_state_with_exchange():
    """مزامنة حالة البوت مع الواقع على Exchange"""
    has, side, qty = exchange_has_position()
    
    if has is True and not STATE.get("open"):
        log_w(f"🔁 SYNC: exchange has position but STATE closed → repairing | {side} qty={qty}")
        STATE["open"] = True
        STATE["side"] = side
        STATE["qty"] = qty
        # محاولة جلب سعر الدخول
        try:
            ticker = ex.fetch_ticker(SYMBOL)
            STATE["entry"] = ticker.get('last', 0)
        except:
            pass
            
    if has is False and STATE.get("open"):
        log_w("🔁 SYNC: STATE open but exchange has no position → forcing STATE close")
        STATE["open"] = False
        STATE["side"] = None
        STATE["qty"] = 0.0
        STATE["entry"] = None

# =================== ENHANCED COUNCIL VOTING ===================
def council_votes_pro_enhanced(df):
    """مجلس تصويت محسّن مع RSI+MA والمناطق الذهبية + الشموع"""
    try:
        ind = compute_indicators(df)
        rsi_ctx = rsi_ma_context(df)
        gz = golden_zone_check(df, ind)

        # جديد: حساب الشموع
        cd = compute_candles(df)

        votes_b = 0; votes_s = 0
        score_b = 0.0; score_s = 0.0
        logs = []

        adx = ind.get('adx', 0)
        plus_di = ind.get('plus_di', 0)
        minus_di = ind.get('minus_di', 0)
        di_spread = abs(plus_di - minus_di)

        # --- ترند ADX/DI
        if adx > ADX_TREND_MIN:
            if plus_di > minus_di and di_spread > DI_SPREAD_TREND:
                votes_b += 2; score_b += 1.5; logs.append("📈 ترند صاعد قوي")
            elif minus_di > plus_di and di_spread > DI_SPREAD_TREND:
                votes_s += 2; score_s += 1.5; logs.append("📉 ترند هابط قوي")

        # --- RSI-MA cross / Trend-Z
        if rsi_ctx["cross"] == "bull" and rsi_ctx["rsi"] < 70:
            votes_b += 2; score_b += 1.0; logs.append("🟢 RSI-MA إيجابي")
        elif rsi_ctx["cross"] == "bear" and rsi_ctx["rsi"] > 30:
            votes_s += 2; score_s += 1.0; logs.append("🔴 RSI-MA سلبي")

        if rsi_ctx["trendZ"] == "bull":
            votes_b += 3; score_b += 1.5; logs.append("🚀 RSI ترند صاعد مستمر")
        elif rsi_ctx["trendZ"] == "bear":
            votes_s += 3; score_s += 1.5; logs.append("💥 RSI ترند هابط مستمر")

        # --- Golden Zones
        if gz and gz.get("ok"):
            if gz['zone']['type'] == 'golden_bottom':
                votes_b += 3; score_b += 1.5; logs.append(f"🏆 قاع ذهبي (قوة: {gz['score']:.1f})")
            elif gz['zone']['type'] == 'golden_top':
                votes_s += 3; score_s += 1.5; logs.append(f"🏆 قمة ذهبية (قوة: {gz['score']:.1f})")

        # جديد: الشموع
        if cd["score_buy"]>0:
            score_b += min(2.5, cd["score_buy"]); logs.append(f"🕯️ شموع BUY ({cd['pattern']}) +{cd['score_buy']:.1f}")
        if cd["score_sell"]>0:
            score_s += min(2.5, cd["score_sell"]); logs.append(f"🕯️ شموع SELL ({cd['pattern']}) +{cd['score_sell']:.1f}")

        # تخفيف النطاق المحايد
        if rsi_ctx["in_chop"]:
            score_b *= 0.8; score_s *= 0.8; logs.append("⚖️ RSI محايد — تخفيض ثقة")

        # حارس ADX عام
        if adx < ADX_GATE:
            score_b *= 0.85; score_s *= 0.85; logs.append(f"🛡️ ADX Gate ({adx:.1f} < {ADX_GATE})")

        # ضمّ إشارات الشموع ليتوفّر لباقي المنظومة (إدارة/خروج)
        ind.update({
            "rsi_ma": rsi_ctx["rsi_ma"],
            "rsi_trendz": rsi_ctx["trendZ"],
            "di_spread": di_spread,
            "gz": gz,
            "candle_buy_score": cd["score_buy"],
            "candle_sell_score": cd["score_sell"],
            "wick_up_big": cd["wick_up_big"],
            "wick_dn_big": cd["wick_dn_big"],
            "candle_tags": cd["pattern"]
        })

        return {
            "b": votes_b, "s": votes_s,
            "score_b": score_b, "score_s": score_s,
            "logs": logs, "ind": ind, "gz": gz, "candles": cd
        }
    except Exception as e:
        log_w(f"council_votes_pro_enhanced error: {e}")
        return {"b":0,"s":0,"score_b":0.0,"score_s":0.0,"logs":[],"ind":{},"gz":None,"candles":{}}

council_votes_pro = council_votes_pro_enhanced

# =================== POSITION RECOVERY ===================
def _normalize_side(pos):
    side = pos.get("side") or pos.get("positionSide") or ""
    if side: return side.upper()
    qty = float(pos.get("contracts") or pos.get("positionAmt") or pos.get("size") or 0)
    return "LONG" if qty > 0 else ("SHORT" if qty < 0 else "")

def fetch_live_position(exchange, symbol: str):
    try:
        if hasattr(exchange, "fetch_positions"):
            arr = exchange.fetch_positions([symbol])
            for p in arr or []:
                sym = p.get("symbol") or p.get("info", {}).get("symbol")
                if sym and symbol.replace(":","") in sym.replace(":",""):
                    side = _normalize_side(p)
                    qty = abs(float(p.get("contracts") or p.get("positionAmt") or p.get("info",{}).get("size",0) or 0))
                    if qty > 0:
                        entry = float(p.get("entryPrice") or p.get("info",{}).get("entryPrice") or 0.0)
                        lev = float(p.get("leverage") or p.get("info",{}).get("leverage") or 0.0)
                        unr = float(p.get("unrealizedPnl") or 0.0)
                        return {"ok": True, "side": side, "qty": qty, "entry": entry, "unrealized": unr, "leverage": lev, "raw": p}
        if hasattr(exchange, "fetch_position"):
            p = exchange.fetch_position(symbol)
            side = _normalize_side(p); qty = abs(float(p.get("size") or 0))
            if qty > 0:
                entry = float(p.get("entryPrice") or 0.0)
                lev   = float(p.get("leverage") or 0.0)
                unr   = float(p.get("unrealizedPnl") or 0.0)
                return {"ok": True, "side": side, "qty": qty, "entry": entry, "unrealized": unr, "leverage": lev, "raw": p}
    except Exception as e:
        log_w(f"fetch_live_position error: {e}")
    return {"ok": False, "why": "no_open_position"}

def resume_open_position(exchange, symbol: str, state: dict) -> dict:
    if not RESUME_ON_RESTART:
        log_i("resume disabled"); return state

    live = fetch_live_position(exchange, symbol)
    if not live.get("ok"):
        log_i("no live position to resume"); return state

    ts = int(time.time())
    prev = load_state()
    if prev.get("ts") and (ts - int(prev["ts"])) > RESUME_LOOKBACK_SECS:
        log_w("found old local state — will override with exchange live snapshot")

    state.update({
        "in_position": True,
        "side": live["side"],
        "entry_price": live["entry"],
        "position_qty": live["qty"],
        "leverage": live.get("leverage") or state.get("leverage") or 10,
        "partial_taken": prev.get("partial_taken", False),
        "breakeven_armed": prev.get("breakeven_armed", False),
        "trail_active": prev.get("trail_active", False),
        "trail_tightened": prev.get("trail_tightened", False),
        "mode": prev.get("mode", "trend"),
        "gz_snapshot": prev.get("gz_snapshot", {}),
        "cv_snapshot": prev.get("cv_snapshot", {}),
        "opened_at": prev.get("opened_at", ts),
    })
    save_state(state)
    log_g(f"RESUME: {state['side']} qty={state['position_qty']} @ {state['entry_price']:.6f} lev={state['leverage']}x")
    return state

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

# =================== HELPERS ===================
_consec_err = 0
last_loop_ts = time.time()

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
    rows = with_retry(lambda: ex.fetch_ohlcv(SYMBOL, timeframe=timeframe, limit=limit, params={"type":"swap"}))
    return pd.DataFrame(rows, columns=["time","open","high","low","close","volume"])

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

# ========= Professional logging helpers =========
def fmt_walls(walls):
    return ", ".join([f"{p:.6f}@{q:.0f}" for p, q in walls]) if walls else "-"

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

# ========= Liquidity Dashboard Formatter =========
def fmt_liq_dash(liq: dict):
    if not liq or not liq.get("ok"):
        return "Liquidity: n/a"
    return (
        f"Liquidity | state={liq.get('state','n/a')} | "
        f"sweepL={liq.get('sweep_low')} sweepH={liq.get('sweep_high')} | "
        f"wickU={safe(liq.get('wick_up')):.2f} wickD={safe(liq.get('wick_dn')):.2f} | "
        f"vol×={safe(liq.get('vol_x'), 1.0):.2f} | "
        f"drain={liq.get('drain')}"
    )

# ========= Unified snapshot emitter =========
def emit_snapshots(exchange, symbol, df, balance_fn=None, pnl_fn=None):
    """
    يطبع Snapshot موحّد: Bookmap + Flow + Council + Strategy + Balance/PnL
    + Liquidity + SMC + Step3 Engine
    """
    try:
        bm = bookmap_snapshot(exchange, symbol)
        flow = compute_flow_metrics(df)
        cv = council_votes_pro(df)
        mode = decide_strategy_mode(df)
        gz = golden_zone_check(df, {"adx": cv["ind"]["adx"]}, "buy" if cv["b"]>=cv["s"] else "sell")
        
        # حساب Indicators
        ind = compute_indicators(df)
        
        # ===== STEP 2: Liquidity + SMC Engine =====
        liq = liquidity_ctx(df, ind.get("atr", 0.0))
        smc = smc_ctx(df, ind.get("atr", 0.0))
        
        # ===== STEP 3: Explosion/Collapse Engine =====
        xc = explosion_collapse_ctx(df, ind.get("atr", 0.0))
        
        # ===== Context Indicators =====
        vwap = compute_vwap(df)
        vw_bias = "above" if float(df["close"].iloc[-1]) > vwap else "below"
        ichi = ichimoku_bias(df)
        
        # ===== Explosion/Collapse Detection =====
        exp = detect_explosion_collapse(df, ind, flow or {}, bm or {})
        
        bal = None; cpnl = None
        if callable(balance_fn):
            try: bal = balance_fn()
            except: bal = None
        if callable(pnl_fn):
            try: cpnl = pnl_fn()
            except: cpnl = None

        if bm.get("ok"):
            imb_tag = "🟢" if bm["imbalance"]>=IMBALANCE_ALERT else ("🔴" if bm["imbalance"]<=1/IMBALANCE_ALERT else "⚖️")
            bm_note = f"Bookmap: {imb_tag} Imb={safe(bm['imbalance']):.2f} | Buy[{fmt_walls(bm['buy_walls'])}] | Sell[{fmt_walls(bm['sell_walls'])}]"
        else:
            bm_note = f"Bookmap: N/A ({bm.get('why')})"

        if flow.get("ok"):
            dtag = "🟢Buy" if flow["delta_last"]>0 else ("🔴Sell" if flow["delta_last"]<0 else "⚖️Flat")
            spk = " ⚡Spike" if flow["spike"] else ""
            fl_note = f"Flow: {dtag} Δ={safe(flow['delta_last']):.0f} z={safe(flow['delta_z']):.2f}{spk} | CVD {'↗️' if flow['cvd_trend']=='up' else '↘️'} {safe(flow['cvd_last']):.0f}"
        else:
            fl_note = f"Flow: N/A ({flow.get('why')})"

        side_hint = "BUY" if cv["b"]>=cv["s"] else "SELL"
        dash = (f"DASH → hint-{side_hint} | Council BUY({cv['b']},{safe(cv['score_b']):.1f}) "
                f"SELL({cv['s']},{safe(cv['score_s']):.1f}) | "
                f"RSI={safe(cv['ind'].get('rsi')):.1f} ADX={safe(cv['ind'].get('adx')):.1f} "
                f"DI={safe(cv['ind'].get('di_spread')):.1f}")

        strat_icon = "📈" if mode["mode"]=="BIG_TREND" else "↗️" if mode["mode"]=="MID_TREND" else "ℹ️"
        strat = f"Strategy: {strat_icon} {mode['mode']}"

        bal_note = f"Balance={safe(bal):.2f}" if bal is not None else ""
        pnl_note = f"CompoundPnL={safe(cpnl):.6f}" if cpnl is not None else ""
        wallet = (" | ".join(x for x in [bal_note, pnl_note] if x)) or ""

        gz_note = ""
        if gz and gz.get("ok"):
            gz_note = f" | 🟡 {gz['zone']['type']} s={safe(gz['score']):.1f}"

        if LOG_ADDONS:
            print(f"🧱 {bm_note}", flush=True)
            print(f"📦 {fl_note}", flush=True)
            print(f"📊 {dash}{gz_note}", flush=True)
            print(f"{strat}{(' | ' + wallet) if wallet else ''}", flush=True)
            
            # ===== Liquidity/SMC/Step3 Logs =====
            print("🧪 " + fmt_liq_dash(liq), flush=True)
            
            # ===== SMC Logs =====
            if smc and smc.get("ok"):
                ob_buy = smc.get("ob", {}).get("buy")
                ob_sell = smc.get("ob", {}).get("sell")
                fvg_buy = smc.get("fvg", {}).get("buy")
                fvg_sell = smc.get("fvg", {}).get("sell")
                bos_buy = smc.get("bos", {}).get("buy")
                bos_sell = smc.get("bos", {}).get("sell")
                
                print(
                    f"🏗️ SMC | OB(buy={bool(ob_buy)}, sell={bool(ob_sell)}) | "
                    f"FVG(buy={bool(fvg_buy)}, sell={bool(fvg_sell)}) | "
                    f"BOS(buy={bos_buy}, sell={bos_sell})",
                    flush=True
                )
            
            # ===== Explosion/Collapse Logs =====
            if exp.get("state"):
                icon = "🔥" if exp["state"] == "EXPLOSION" else "🧊"
                print(
                    f"{icon} {exp['state']} | dir={exp.get('dir', '').upper()} | "
                    f"Δ={safe(exp.get('delta')):.0f} z={safe(exp.get('z')):.2f} imb={safe(exp.get('imb')):.2f} | "
                    f"reasons={','.join(exp.get('reasons', []))}",
                    flush=True
                )
            
            # ===== Context Logs =====
            print(f"🌫️ Context | VWAP={safe(vwap):.4f} bias={vw_bias} | Ichimoku={ichi}", flush=True)
            
            # ===== Liquidity Snapshot بناءً على السياق =====
            ctx = {
                "imb": bm["imbalance"] if bm.get("ok") else 1.0,
                "buy_walls": bm.get("buy_walls", []) if bm.get("ok") else [],
                "sell_walls": bm.get("sell_walls", []) if bm.get("ok") else [],
                "delta": flow.get("delta_last", 0) if flow.get("ok") else 0,
                "cvd": flow.get("cvd_last", 0) if flow.get("ok") else 0,
                "sweep_high": liq.get("sweep", {}).get("dir") == "SELL" if liq.get("sweep") else False,
                "sweep_low": liq.get("sweep", {}).get("dir") == "BUY" if liq.get("sweep") else False,
                "sweep_price": liq.get("sweep", {}).get("level", 0) if liq.get("sweep") else 0,
                "sweep_ref": liq.get("prev_high", 0) if liq.get("sweep", {}).get("dir") == "SELL" else liq.get("prev_low", 0) if liq.get("sweep", {}).get("dir") == "BUY" else 0,
                "vol_spike_x": liq.get("vol_ratio", 0),
                "vwap_bias": vw_bias.capitalize(),
                "htf_bias": side_hint
            }
            
            snap = build_liquidity_snapshot(ctx)
            log_liquidity_snapshot(logging, snap, SYMBOL, price_now() or df["close"].iloc[-1])
            
            # ===== تحذير لو السيولة ضد المركز المفتوح =====
            if STATE.get("open"):
                position_side = STATE.get("side")
                if position_side == "long" and snap.side_hint == "SELL":
                    log_w("⚠️ LIQ AGAINST LONG: possible trap / reversal")
                if position_side == "short" and snap.side_hint == "BUY":
                    log_w("⚠️ LIQ AGAINST SHORT: possible trap / reversal")

            gz_snap_note = ""
            if gz and gz.get("ok"):
                zone_type = gz["zone"]["type"]
                zone_score = gz["score"]
                gz_snap_note = f" | 🟡{zone_type} s={safe(zone_score):.1f}"
            
            flow_z = flow['delta_z'] if flow and flow.get('ok') else 0.0
            bm_imb = bm['imbalance'] if bm and bm.get('ok') else 1.0
            
            print(f"🧠 SNAP | {side_hint} | votes={cv['b']}/{cv['s']} score={safe(cv['score_b']):.1f}/{safe(cv['score_s']):.1f} "
                  f"| ADX={safe(cv['ind'].get('adx')):.1f} DI={safe(cv['ind'].get('di_spread')):.1f} | "
                  f"z={safe(flow_z):.2f} | imb={safe(bm_imb):.2f}{gz_snap_note}", 
                  flush=True)
            
            print("✅ ADDONS LIVE", flush=True)

        return {
            "bm": bm, "flow": flow, "cv": cv, "mode": mode, "gz": gz, 
            "wallet": wallet, "ind": ind, "liq": liq, "smc": smc, "xc": xc,
            "exp": exp, "vwap": vwap, "vw_bias": vw_bias, "ichi": ichi
        }
    except Exception as e:
        print(f"🟨 AddonLog error: {e}", flush=True)
        return {"bm": None, "flow": None, "cv": {"b":0,"s":0,"score_b":0.0,"score_s":0.0,"ind":{}},
                "mode": {"mode":"n/a"}, "gz": None, "wallet": "", 
                "ind": {}, "liq": {}, "smc": {}, "xc": {}, "exp": {}}

# =================== EXECUTION MANAGER ===================
def execute_trade_decision(side, price, qty, mode, council_data, gz_data):
    """تنفيذ قرار التداول مع التسجيل الواضح"""
    if not EXECUTE_ORDERS or DRY_RUN:
        log_i(f"DRY_RUN: {side} {qty:.4f} @ {price:.6f} | mode={mode}")
        return True
    
    if qty <= 0:
        log_e("❌ كمية غير صالحة للتنفيذ")
        return False

    gz_note = ""
    if gz_data and gz_data.get("ok"):
        gz_note = f" | 🟡 {gz_data['zone']['type']} s={safe(gz_data['score']):.1f}"
    
    votes = council_data
    print(f"🎯 EXECUTE: {side.upper()} {qty:.4f} @ {price:.6f} | "
          f"mode={mode} | votes={votes['b']}/{votes['s']} score={safe(votes['score_b']):.1f}/{safe(votes['score_s']):.1f}"
          f"{gz_note}", flush=True)

    try:
        if MODE_LIVE:
            exchange_set_leverage(ex, LEVERAGE, SYMBOL)
            params = exchange_specific_params(side, is_close=False)
            ex.create_order(SYMBOL, "market", side, qty, None, params)
        
        log_g(f"✅ EXECUTED: {side.upper()} {qty:.4f} @ {price:.6f}")
        return True
    except Exception as e:
        log_e(f"❌ EXECUTION FAILED: {e}")
        return False

def setup_trade_management(mode):
    """تهيئة إدارة الصفقة حسب النمط"""
    if mode == "MID_TREND":
        return {
            "tp_levels": [0.6, 1.2],
            "tp_fracs": [0.5, 0.5],
            "be_activate_pct": 0.4 / 100.0,
            "trail_activate_pct": 0.8 / 100.0,
            "atr_trail_mult": 1.6,
            "close_aggression": "medium"
        }
    else:  # BIG_TREND
        return {
            "tp_levels": [0.8, 1.8, 3.0],
            "tp_fracs": [0.25, 0.35, 0.40],
            "be_activate_pct": 0.5 / 100.0,
            "trail_activate_pct": 1.2 / 100.0,
            "atr_trail_mult": 1.8,
            "close_aggression": "low"
        }

# =================== ENHANCED TRADE EXECUTION ===================
def open_market_enhanced(side, qty, price):
    if qty <= 0: 
        log_e("skip open (qty<=0)")
        return False
    
    df = fetch_ohlcv()
    snap = emit_snapshots(ex, SYMBOL, df)
    
    votes = snap["cv"]
    mode_data = decide_strategy_mode(df, 
                                   adx=votes["ind"].get("adx"),
                                   di_plus=votes["ind"].get("plus_di"),
                                   di_minus=votes["ind"].get("minus_di"),
                                   rsi_ctx=rsi_ma_context(df))
    
    mode = mode_data["mode"]
    gz = snap["gz"]
    
    # ===== التحقق من بوابة الدخول =====
    ok, score, reasons = entry_gate_from_liq_smc_xc(snap, side, mode)
    if not ok:
        log_w(f"⛔ ENTRY BLOCKED by LIQ/SMC/STEP3 | side={side} mode={mode} score={score}/{2 if mode=='MID_TREND' else 3} reasons={reasons}")
        return False
    else:
        log_g(f"✅ ENTRY PASS LIQ/SMC/STEP3 | side={side} mode={mode} score={score} reasons={reasons}")
    
    # ===== بناء خطة TP ديناميكية =====
    ind = compute_indicators(df)
    exp = snap.get("exp", {})
    
    # تحديد وضع الخطة: Explosion يرفع للـBIG_TREND
    plan_mode = "BIG_TREND" if exp.get("state") == "EXPLOSION" else mode
    
    # بناء خطة TP
    plan = build_tp_plan(plan_mode, ind.get("atr"), price)
    
    management_config = {
        "tp_levels": plan["tp_pcts"],
        "tp_fracs": plan["fractions"],
        "be_activate_pct": 0.4/100.0 if mode == "MID_TREND" else 0.5/100.0,
        "trail_activate_pct": 0.8/100.0 if mode == "MID_TREND" else 1.2/100.0,
        "atr_trail_mult": plan["trail_mult"],
        "close_aggression": "medium" if mode == "MID_TREND" else "low"
    }
    
    success = execute_trade_decision(side, price, qty, mode, votes, gz)
    
    if success:
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
            "highest_profit_pct": 0.0, 
            "profit_targets_achieved": 0,
            "mode": mode,
            "plan_mode": plan_mode,
            "tp_pcts": plan["tp_pcts"],
            "tp_fracs": plan["fractions"],
            "trail_mult": plan["trail_mult"],
            "management": management_config,
            "last_entry_source": "trend_master"
        })
        
        save_state({
            "in_position": True,
            "side": "LONG" if side.upper().startswith("B") else "SHORT",
            "entry_price": price,
            "position_qty": qty,
            "leverage": LEVERAGE,
            "mode": mode,
            "plan_mode": plan_mode,
            "management": management_config,
            "gz_snapshot": gz if isinstance(gz, dict) else {},
            "cv_snapshot": votes if isinstance(votes, dict) else {},
            "exp_snapshot": exp if isinstance(exp, dict) else {},
            "opened_at": int(time.time()),
            "partial_taken": False,
            "breakeven_armed": False,
            "trail_active": False,
            "trail_tightened": False,
        })
        
        log_i(f"🧾 Trade Plan: {plan_mode} | TP%={plan['tp_pcts']} | fracs={plan['fractions']} | trail×={plan['trail_mult']}")
        log_g(f"✅ POSITION OPENED: {side.upper()} | mode={mode} | plan={plan_mode}")
        return True
    
    return False

open_market = open_market_enhanced

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

def _ema(s, n): return s.ewm(span=n, adjust=False).mean()

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
    "tp1_done": False, "highest_profit_pct": 0.0,
    "profit_targets_achieved": 0,
}
compound_pnl = 0.0
wait_for_next_signal_side = None

# =================== WAIT FOR NEXT SIGNAL ===================
def _arm_wait_after_close(prev_side):
    """تفعيل انتظار الإشارة التالية بعد الإغلاق"""
    global wait_for_next_signal_side
    wait_for_next_signal_side = "sell" if prev_side=="long" else ("buy" if prev_side=="short" else None)
    log_i(f"🛑 WAIT FOR NEXT SIGNAL: {wait_for_next_signal_side}")

def wait_gate_allow(df, info):
    """التحقق من بوابة الانتظار"""
    if wait_for_next_signal_side is None: 
        return True, ""
    
    bar_ts = int(info.get("time") or 0)
    need = (wait_for_next_signal_side=="buy" and info.get("long")) or (wait_for_next_signal_side=="sell" and info.get("short"))
    
    if need:
        return True, ""
    return False, f"wait-for-next-RF({wait_for_next_signal_side})"

# =================== ORDERS ===================
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
                logging.info(f"STRICT_CLOSE {side} pnl={pnl} total={compound_pnl}")
                _reset_after_close(reason, prev_side=side)
                return
            qty_to_close = safe_qty(left_qty)
            attempts += 1
            log_w(f"strict close retry {attempts}/{CLOSE_RETRY_ATTEMPTS} — residual={fmt(left_qty,4)}")
            time.sleep(CLOSE_VERIFY_WAIT_S)
        except Exception as e:
            last_error = e; logging.error(f"close_market_strict attempt {attempts+1}: {e}"); attempts += 1; time.sleep(CLOSE_VERIFY_WAIT_S)
    log_e(f"STRICT CLOSE FAILED after {CLOSE_RETRY_ATTEMPTS} attempts — last error: {last_error}")
    logging.critical(f"STRICT CLOSE FAILED — last_error={last_error}")

def _reset_after_close(reason, prev_side=None):
    """إعادة تعيين الحالة بعد الإغلاق"""
    global wait_for_next_signal_side
    prev_side = prev_side or STATE.get("side")
    STATE.update({
        "open": False, "side": None, "entry": None, "qty": 0.0,
        "pnl": 0.0, "bars": 0, "trail": None, "breakeven": None,
        "tp1_done": False, "highest_profit_pct": 0.0, "profit_targets_achieved": 0,
        "trail_tightened": False, "partial_taken": False
    })
    TREND_STATE.update({
        "mode": "NONE",
        "phase": "WAIT",
        "dir": None,
        "plan": None
    })
    save_state({"in_position": False, "position_qty": 0})
    
    # تفعيل انتظار الإشارة التالية
    _arm_wait_after_close(prev_side)
    logging.info(f"AFTER_CLOSE waiting_for={wait_for_next_signal_side}")

# =================== ENHANCED TRADE MANAGEMENT ===================
def manage_after_entry_enhanced(df, ind, info):
    """إدارة محسنة للمركز مع خروج ذكي حسب النمط"""
    if not STATE["open"] or STATE["qty"] <= 0:
        return

    px = info["price"]
    entry = STATE["entry"]
    side = STATE["side"]
    qty = STATE["qty"]
    mode = STATE.get("mode", "MID_TREND")
    management = STATE.get("management", {})
    
    pnl_pct = (px - entry) / entry * 100 * (1 if side == "long" else -1)
    STATE["pnl"] = pnl_pct
    
    if pnl_pct > STATE["highest_profit_pct"]:
        STATE["highest_profit_pct"] = pnl_pct

    snap = emit_snapshots(ex, SYMBOL, df)
    gz = snap["gz"]
    flow = snap.get("flow", {})
    bm = snap.get("bm", {})
    
    # ===== الكشف عن الانعكاس الحقيقي باستخدام Explosion/Collapse =====
    exp_now = detect_explosion_collapse(df, ind, flow or {}, bm or {})
    
    # انعكاس حقيقي: Collapse ضد اتجاه المركز
    if exp_now.get("state") == "COLLAPSE" and exp_now.get("dir") != STATE.get("side"):
        log_e(f"❌ HARD REVERSAL — LIQUIDITY COLLAPSE | reasons={exp_now.get('reasons')}")
        close_market_strict("LIQUIDITY_REVERSAL")
        TREND_STATE["phase"] = "EXIT"
        return
    else:
        # تحقق من تصحيح vs انعكاس
        if _is_reversal(df, ind, side):
            log_e("🛑 REVERSAL DETECTED -> STRICT CLOSE NOW")
            close_market_strict(reason="reversal_detected")
            TREND_STATE["phase"] = "EXIT"
            return
        else:
            if TREND_STATE.get("mode") in ("BIG_TREND", "MID_TREND") and _is_correction(df, ind, side):
                TREND_STATE["phase"] = "CORRECTION"
                log_i("🟦 CORRECTION: holding (trend still valid)")
            else:
                TREND_STATE["phase"] = "RUN"
    
    exit_signal = smart_exit_guard(STATE, df, ind, flow, bm, 
                                 px, pnl_pct/100, mode, side, entry, gz)
    
    if exit_signal["log"]:
        print(f"🔔 {exit_signal['log']}", flush=True)

    if exit_signal["action"] == "partial" and not STATE.get("partial_taken"):
        partial_qty = safe_qty(qty * exit_signal.get("qty_pct", 0.3))
        if partial_qty > 0:
            close_side = "sell" if side == "long" else "buy"
            if MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN:
                try:
                    params = exchange_specific_params(close_side, is_close=True)
                    ex.create_order(SYMBOL, "market", close_side, partial_qty, None, params)
                    log_g(f"✅ PARTIAL CLOSE: {partial_qty:.4f} | {exit_signal['why']}")
                    STATE["partial_taken"] = True
                    STATE["qty"] = safe_qty(qty - partial_qty)
                except Exception as e:
                    log_e(f"❌ Partial close failed: {e}")
            else:
                log_i(f"DRY_RUN: Partial close {partial_qty:.4f}")
    
    elif exit_signal["action"] == "tighten" and not STATE.get("trail_tightened"):
        STATE["trail_tightened"] = True
        STATE["trail"] = None
        log_i(f"🔄 TRAIL TIGHTENED: {exit_signal['why']}")
    
    elif exit_signal["action"] == "close":
        log_w(f"🚨 SMART EXIT: {exit_signal['why']}")
        close_market_strict(f"smart_exit_{exit_signal['why']}")
        return

    current_atr = ind.get("atr", 0.0)
    
    # TP Levels حسب النمط
    tp_levels = management.get("tp_levels", [0.6, 1.2] if mode == "MID_TREND" else [0.8, 1.8, 3.0])
    tp_fracs = management.get("tp_fracs", [0.5, 0.5] if mode == "MID_TREND" else [0.25, 0.35, 0.40])
    
    # تنفيذ TP حسب المستويات
    for i, (tp_pct, tp_frac) in enumerate(zip(tp_levels, tp_fracs)):
        if not STATE.get(f"tp{i+1}_done", False) and pnl_pct >= tp_pct:
            close_qty = safe_qty(STATE["qty"] * tp_frac)
            if close_qty > 0:
                close_side = "sell" if STATE["side"] == "long" else "buy"
                if MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN:
                    try:
                        params = exchange_specific_params(close_side, is_close=True)
                        ex.create_order(SYMBOL, "market", close_side, close_qty, None, params)
                        log_g(f"✅ TP{i+1} HIT: closed {tp_frac*100}% at {tp_pct}% profit")
                    except Exception as e:
                        log_e(f"❌ TP{i+1} close failed: {e}")
                STATE["qty"] = safe_qty(STATE["qty"] - close_qty)
                STATE[f"tp{i+1}_done"] = True
                STATE["profit_targets_achieved"] += 1
                break

    be_activate_pct = management.get("be_activate_pct", 0.4/100.0 if mode == "MID_TREND" else 0.5/100.0)
    trail_activate_pct = management.get("trail_activate_pct", 0.8/100.0 if mode == "MID_TREND" else 1.2/100.0)
    atr_trail_mult = management.get("atr_trail_mult", 1.6 if mode == "MID_TREND" else 1.8)

    if not STATE.get("breakeven_armed") and pnl_pct/100 >= be_activate_pct:
        STATE["breakeven_armed"] = True
        STATE["breakeven"] = entry
        log_i("BREAKEVEN ARMED")

    if not STATE.get("trail_active") and pnl_pct/100 >= trail_activate_pct:
        STATE["trail_active"] = True
        log_i("TRAIL ACTIVATED")

    if STATE.get("trail_active"):
        trail_mult = TRAIL_TIGHT_MULT if STATE.get("trail_tightened") else atr_trail_mult
        if side == "long":
            new_trail = px - (current_atr * trail_mult)
            if STATE.get("trail") is None or new_trail > STATE["trail"]:
                STATE["trail"] = new_trail
        else:
            new_trail = px + (current_atr * trail_mult)
            if STATE.get("trail") is None or new_trail < STATE["trail"]:
                STATE["trail"] = new_trail

    if STATE.get("trail"):
        if (side == "long" and px <= STATE["trail"]) or (side == "short" and px >= STATE["trail"]):
            log_w(f"TRAIL STOP: {px} vs trail {STATE['trail']}")
            close_market_strict("trail_stop")

    if STATE.get("breakeven"):
        if (side == "long" and px <= STATE["breakeven"]) or (side == "short" and px >= STATE["breakeven"]):
            log_w(f"BREAKEVEN STOP: {px} vs breakeven {STATE['breakeven']}")
            close_market_strict("breakeven_stop")

    if STATE["qty"] <= FINAL_CHUNK_QTY:
        log_w(f"DUST GUARD: qty {STATE['qty']} <= {FINAL_CHUNK_QTY}, closing...")
        close_market_strict("dust_guard")

manage_after_entry = manage_after_entry_enhanced

def smart_exit_guard(state, df, ind, flow, bm, now_price, pnl_pct, mode, side, entry_price, gz=None):
    """يقرر: Partial / Tighten / Strict Close مع لوج واضح."""
    atr = ind.get('atr', 0.0)
    adx = ind.get('adx', 0.0)
    rsi = ind.get('rsi', 50.0)
    rsi_ma = ind.get('rsi_ma', 50.0)
    
    if len(df) >= 3:
        adx_slope = adx - ind.get('adx_prev', adx)
    else:
        adx_slope = 0.0

    # حساب الفتائل
    wick_signal = False
    if len(df) > 0:
        c = df.iloc[-1]
        wick_up = float(c['high']) - max(float(c['close']), float(c['open']))
        wick_down = min(float(c['close']), float(c['open'])) - float(c['low'])
        wick_signal = (wick_up >= WICK_ATR_MULT * atr) if side == "long" else (wick_down >= WICK_ATR_MULT * atr)

    rsi_cross_down = (rsi < rsi_ma) if side == "long" else (rsi > rsi_ma)
    adx_falling = (adx_slope < 0)
    cvd_down = (flow and flow.get('ok') and flow.get('cvd_trend') == 'down')
    evx_spike = False  # يمكن إضافة حساب EVX لاحقًا
    
    bm_wall_close = False
    if bm and bm.get('ok'):
        if side == "long":
            sell_walls = bm.get('sell_walls', [])
            if sell_walls:
                best_ask = min([p for p, _ in sell_walls])
                bps = abs((best_ask - now_price) / now_price) * 10000.0
                bm_wall_close = (bps <= BM_WALL_PROX_BPS)
        else:
            buy_walls = bm.get('buy_walls', [])
            if buy_walls:
                best_bid = max([p for p, _ in buy_walls])
                bps = abs((best_bid - now_price) / now_price) * 10000.0
                bm_wall_close = (bps <= BM_WALL_PROX_BPS)

    # --- Golden Reversal بعد TP1 ---
    if state.get('tp1_done') and (gz and gz.get('ok')):
        # إغلاق صارم لو تقاطع Golden عكس اتجاهي بعد TP1
        opp = (gz['zone']['type']=='golden_top' and side=='long') or (gz['zone']['type']=='golden_bottom' and side=='short')
        if opp and gz.get('score',0) >= GOLDEN_REVERSAL_SCORE:
            return {
                "action": "close", 
                "why": "golden_reversal",
                "log": f"🔴 CLOSE STRONG | golden reversal after TP1 | score={safe(gz['score']):.1f}"
            }

    # TP Targets حسب النمط
    if mode == "MID_TREND":
        tp_targets = [0.3, 0.6]  # نسب جزئية للـMid Trend
    else:  # BIG_TREND
        tp_targets = [0.25, 0.5, 0.8]  # نسب جزئية للـBig Trend
    
    for i, tp_target in enumerate(tp_targets):
        if pnl_pct >= tp_target and not state.get(f'tp{i+1}_done', False):
            if i == 0:  # أول TP
                qty_pct = 0.25 if mode == "BIG_TREND" else 0.35
            elif i == 1:  # ثاني TP
                qty_pct = 0.35 if mode == "BIG_TREND" else 0.50
            else:  # ثالث TP (لـBig Trend فقط)
                qty_pct = 0.40
            
            return {
                "action": "partial", 
                "why": f"TP{i+1} hit {tp_target*100:.2f}%",
                "qty_pct": qty_pct,
                "log": f"💰 TP{i+1} جزئي {tp_target*100:.2f}% | pnl={pnl_pct*100:.2f}% | mode={mode}"
            }

    # --- Wick exhaustion + Tighten عند إجهاد/تدفق/جدار ---
    if pnl_pct > 0:
        if wick_signal or evx_spike or bm_wall_close or cvd_down:
            return {
                "action": "tighten", 
                "why": "exhaustion/flow/wall",
                "trail_mult": TRAIL_TIGHT_MULT,
                "log": f"🛡️ Tighten | wick={int(bool(wick_signal))} evx={int(bool(evx_spike))} wall={bm_wall_close} cvd_down={cvd_down}"
            }

    bearish_signals = [rsi_cross_down, adx_falling, cvd_down, evx_spike, bm_wall_close]
    bearish_count = sum(bearish_signals)
    
    if pnl_pct >= HARD_CLOSE_PNL_PCT and bearish_count >= 2:
        reasons = []
        if rsi_cross_down: reasons.append("rsi↓")
        if adx_falling: reasons.append("adx↓")
        if cvd_down: reasons.append("cvd↓")
        if evx_spike: reasons.append("evx")
        if bm_wall_close: reasons.append("wall")
        
        return {
            "action": "close", 
            "why": "hard_close_signal",
            "log": f"🔴 CLOSE STRONG | pnl={pnl_pct*100:.2f}% | {', '.join(reasons)}"
        }

    return {
        "action": "hold", 
        "why": "keep_riding", 
        "log": None
    }

# =================== ENHANCED TRADE LOOP ===================
def trade_loop_enhanced():
    """حلقة تداول محسنة مع Trend State Engine وHTF Bias"""
    global wait_for_next_signal_side
    loop_i = 0
    
    while True:
        try:
            # ===== Sync Position with Exchange =====
            sync_state_with_exchange()
            
            # جمع البيانات الأساسية
            bal = balance_usdt()
            px = price_now()
            df = fetch_ohlcv()
            info = rf_signal_live(df)
            ind = compute_indicators(df)
            spread_bps = orderbook_spread_bps()
            
            # ===== HTF Fetch (1H) =====
            df_htf = None
            try:
                df_htf = fetch_ohlcv_htf(HTF_TF, limit=300)
            except Exception as _:
                df_htf = None
            
            # ===== HTF Bias =====
            htf_bias, htf_dbg = (None, {})
            if df_htf is not None and len(df_htf) > 210:
                htf_bias, htf_dbg = compute_htf_bias(df_htf)
            TREND_STATE["htf_bias"] = htf_bias
            
            # تحديث الـ Snapshots
            snap = emit_snapshots(ex, SYMBOL, df,
                                balance_fn=lambda: float(bal) if bal else None,
                                pnl_fn=lambda: float(compound_pnl))
            
            # تحديث حالة الربح/الخسارة
            if STATE["open"] and px:
                STATE["pnl"] = (px-STATE["entry"])*STATE["qty"] if STATE["side"]=="long" else (STATE["entry"]-px)*STATE["qty"]
            
            # إدارة الصفقة المفتوحة
            if STATE["open"]:
                manage_after_entry(df, ind, {
                    "price": px or info["price"], 
                    "bm": snap["bm"],
                    "flow": snap["flow"],
                    **info
                })
            
            # قرار الدخول باستخدام Trend Birth Entry + Council + Explosion Override
            reason = None
            if spread_bps is not None and spread_bps > MAX_SPREAD_BPS:
                reason = f"spread too high ({fmt(spread_bps,2)}bps > {MAX_SPREAD_BPS})"
            
            council_data = council_votes_pro_enhanced(df)
            gz = council_data.get("gz")
            final_signal = None
            entry_reasons = []

            # =================== OVERRIDE ENTRY (TBE) ===================
            tbe = None
            try:
                if not STATE["open"]:
                    tbe = trend_birth_entry(df, ind, TREND_STATE.get("htf_bias"))
            except Exception as _:
                tbe = None

            if tbe and not STATE["open"]:
                # Override يحدد final_signal حتى لو RF مش موجود
                final_signal = tbe["side"]  # "buy"/"sell"
                entry_reasons += [f"TBE:{r}" for r in tbe.get("reasons", [])]
                log_y(f"🎯 TREND BIRTH ENTRY: {final_signal.upper()} | mode={tbe['mode']}")
            
            # =================== Explosion/Collapse Override ===================
            exp = snap.get("exp", {})
            if not final_signal and exp.get("state") in ("EXPLOSION", "COLLAPSE"):
                if exp["dir"] == htf_bias:
                    final_signal = exp["dir"]
                    entry_reasons.append(f"{exp['state']}_OVERRIDE")
                    log_y(f"⚡ {exp['state']} OVERRIDE: {final_signal.upper()} aligned with HTF")
            
            # --- Golden Entry Override ---
            if not final_signal and (gz and gz.get("ok") and ind.get("adx",0) >= GOLDEN_ENTRY_ADX):
                if gz["zone"]["type"]=="golden_bottom" and gz["score"]>=GOLDEN_ENTRY_SCORE:
                    final_signal = "buy"
                    entry_reasons.append(f"GoldenBottom score={safe(gz['score']):.1f}")
                elif gz["zone"]["type"]=="golden_top" and gz["score"]>=GOLDEN_ENTRY_SCORE:
                    final_signal = "sell" 
                    entry_reasons.append(f"GoldenTop score={safe(gz['score']):.1f}")

            # لو مفيش TBE ولا Golden ولا Explosion، استخدم السكور المعتاد
            if final_signal is None:
                if council_data["score_b"] > council_data["score_s"] and council_data["score_b"] >= 8.0:
                    final_signal = "buy"
                    entry_reasons.append(f"Council score={safe(council_data['score_b']):.1f}")
                elif council_data["score_s"] > council_data["score_b"] and council_data["score_s"] >= 8.0:
                    final_signal = "sell"
                    entry_reasons.append(f"Council score={safe(council_data['score_s']):.1f}")
            
            # =================== NO SCALP POLICY ===================
            signal_strength = None
            tp_profile = None

            # 1) لو جاء TBE => Mid/Big فقط
            if tbe:
                if tbe["mode"] == "BIG_TREND":
                    signal_strength = "strong"
                    tp_profile = "TREND_3"
                elif tbe["mode"] == "MID_TREND":
                    signal_strength = "mid"
                    tp_profile = "MID_2"

            # 2) لو جاء Explosion => يعتبر Big Trend
            if exp.get("state") == "EXPLOSION":
                signal_strength = "strong"
                tp_profile = "TREND_3_EXPLOSION"

            # 3) غير كده (مصادر قديمة: COUNCIL) => اسمح Mid/Big فقط
            if signal_strength is None:
                # تحديد قوة الإشارة بناءً على Council score
                if council_data["score_b"] >= 10.0 or council_data["score_s"] >= 10.0:
                    signal_strength = "strong"
                    tp_profile = "TREND_3"
                elif council_data["score_b"] >= 8.0 or council_data["score_s"] >= 8.0:
                    signal_strength = "mid"
                    tp_profile = "MID_2"
                else:
                    # ممنوع weak
                    signal_strength = None
                    tp_profile = None

            # Gate نهائي: ممنوع سكالب/weak
            if FORCE_NO_SCALP:
                if signal_strength not in ("mid", "strong"):
                    final_signal = None
            
            # Gate: HTF bias يجب أن يوافق اتجاه الدخول
            if htf_bias and final_signal:
                if htf_bias != final_signal:
                    log_i(f"🛑 HTF bias mismatch: {htf_bias} vs {final_signal}")
                    final_signal = None
            
            if not STATE["open"] and final_signal and reason is None:
                # ===== بوابة الدخول من Liquidity/SMC/Step3 =====
                mode = classify_trend_mode(ind)
                ok, score, reasons = entry_gate_from_liq_smc_xc(snap, final_signal, mode)
                if not ok:
                    reason = f"LIQ/SMC/STEP3 gate blocked: score={score}"
                    log_w(f"⛔ ENTRY BLOCKED by LIQ/SMC/STEP3 | side={final_signal} mode={mode} score={score}/{2 if mode=='MID_TREND' else 3}")
                else:
                    log_g(f"✅ ENTRY PASS LIQ/SMC/STEP3 | side={final_signal} mode={mode} score={score} reasons={reasons}")
                
                if ok:
                    # التحقق من سياسة الانتظار
                    allow_wait, wait_reason = wait_gate_allow(df, info)
                    if not allow_wait:
                        reason = wait_reason
                    else:
                        qty = compute_size(bal, px or info["price"])
                        if qty > 0:
                            ok_open = open_market(final_signal, qty, px or info["price"])
                            if ok_open:
                                # ✅ Verify position exists on exchange
                                try:
                                    pos = fetch_live_position(ex, SYMBOL)
                                    if not pos.get("ok") or pos.get("qty", 0) <= 0:
                                        log_e("❌ EXEC VERIFY FAILED: order reported ok but exchange position=0. Aborting state/open.")
                                        # reset local state
                                        STATE["open"] = False
                                        STATE["side"] = None
                                        STATE["qty"] = 0.0
                                        STATE["entry"] = None
                                        final_signal = None
                                        TREND_STATE["phase"] = "WAIT"
                                        # تخطي إدارة الصفقة
                                        continue
                                except Exception as _:
                                    log_w("⚠️ EXEC VERIFY WARNING: couldn't read position after open (will continue cautiously).")

                                wait_for_next_signal_side = None
                                # تحديث حالة الترند
                                TREND_STATE["phase"] = "ENTRY"
                                TREND_STATE["dir"] = final_signal
                                # بناء خطة جني الأرباح
                                mode = classify_trend_mode(ind)
                                TREND_STATE["mode"] = mode
                                plan = build_tp_plan(mode)
                                TREND_STATE["plan"] = plan
                                # تسجيل الصفقة
                                color_tag = "🟢" if final_signal=="buy" else "🔴"
                                log_y(
                                    f"{color_tag} TRADE OPEN | {final_signal.upper()} | MODE={mode} "
                                    f"| HTF={htf_bias} | TP={plan['tp_pcts']} "
                                    f"| frac={plan['fractions']} | reasons={entry_reasons}"
                                )
                            else:
                                log_w("❌ Open failed")
                        else:
                            reason = "qty<=0"
            
            # اللوج الاحترافي
            if LOG_LEGACY:
                pretty_snapshot(bal, {"price": px or info["price"], **info}, ind, spread_bps, reason, df)
            
            loop_i += 1
            sleep_s = NEAR_CLOSE_S if time_to_candle_close(df) <= 10 else BASE_SLEEP
            time.sleep(sleep_s)
            
        except Exception as e:
            log_e(f"loop error: {e}\n{traceback.format_exc()}")
            logging.error(f"trade_loop error: {e}\n{traceback.format_exc()}")
            time.sleep(BASE_SLEEP)

# استبدال حلقة التداول الأصلية بالمحسنة
trade_loop = trade_loop_enhanced

# =================== LOOP / LOG ===================
def pretty_snapshot(bal, info, ind, spread_bps, reason=None, df=None):
    if LOG_LEGACY:
        left_s = time_to_candle_close(df) if df is not None else 0
        print(colored("─"*100,"cyan"))
        print(colored(f"📊 {SYMBOL} {INTERVAL} • {EXCHANGE_NAME.upper()} • {'LIVE' if MODE_LIVE else 'PAPER'} • {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC","cyan"))
        print(colored("─"*100,"cyan"))
        print("📈 INDICATORS & RF")
        print(f"   💲 Price {fmt(info.get('price'))} | RF filt={fmt(info.get('filter'))}  hi={fmt(info.get('hi'))} lo={fmt(info.get('lo'))}")
        print(f"   🧮 RSI={fmt(ind.get('rsi'))}  +DI={fmt(ind.get('plus_di'))}  -DI={fmt(ind.get('minus_di'))}  ADX={fmt(ind.get('adx'))}  ATR={fmt(ind.get('atr'))}")
        print(f"   🎯 ENTRY: TREND-ONLY (NO SCALP) | HTF Bias: {TREND_STATE.get('htf_bias')} | spread_bps={fmt(spread_bps,2)}")
        print(f"   ⏱️ closes_in ≈ {left_s}s")
        print("\n🧭 POSITION")
        bal_line = f"Balance={fmt(bal,2)}  Risk={int(RISK_ALLOC*100)}%×{LEVERAGE}x  CompoundPnL={fmt(compound_pnl)}  Eq~{fmt((bal or 0)+compound_pnl,2)}"
        print(colored(f"   {bal_line}", "yellow"))
        if STATE["open"]:
            lamp='🟩 LONG' if STATE['side']=='long' else '🟥 SHORT'
            print(f"   {lamp}  Entry={fmt(STATE['entry'])}  Qty={fmt(STATE['qty'],4)}  Bars={STATE['bars']}  Trail={fmt(STATE['trail'])}  BE={fmt(STATE['breakeven'])}")
            print(f"   🎯 TP_done={STATE['profit_targets_achieved']}  HP={fmt(STATE['highest_profit_pct'],2)}%")
            print(f"   📊 Trend Mode: {TREND_STATE.get('mode')} | Phase: {TREND_STATE.get('phase')}")
            print(f"   📈 TP Plan: {STATE.get('plan_mode')} | Levels: {STATE.get('tp_pcts', [])}")
        else:
            print("   ⚪ FLAT")
            if wait_for_next_signal_side:
                print(colored(f"   ⏳ Waiting for opposite RF: {wait_for_next_signal_side.upper()}", "cyan"))
        if reason: print(colored(f"   ℹ️ reason: {reason}", "white"))
        print(colored("─"*100,"cyan"))

# =================== API / KEEPALIVE ===================
app = Flask(__name__)
@app.route("/")
def home():
    mode='LIVE' if MODE_LIVE else 'PAPER'
    return f"✅ SUI Trend Master Bot — {EXCHANGE_NAME.upper()} — {SYMBOL} {INTERVAL} — {mode} — Trend-Only System"

@app.route("/metrics")
def metrics():
    return jsonify({
        "exchange": EXCHANGE_NAME,
        "symbol": SYMBOL, "interval": INTERVAL, "mode": "live" if MODE_LIVE else "paper",
        "leverage": LEVERAGE, "risk_alloc": RISK_ALLOC, "price": price_now(),
        "state": STATE, "compound_pnl": compound_pnl,
        "trend_state": TREND_STATE,
        "entry_mode": "TREND_ONLY_NO_SCALP", "wait_for_next_signal": wait_for_next_signal_side,
        "guards": {"max_spread_bps": MAX_SPREAD_BPS, "final_chunk_qty": FINAL_CHUNK_QTY}
    })

@app.route("/health")
def health():
    return jsonify({
        "ok": True, "exchange": EXCHANGE_NAME, "mode": "live" if MODE_LIVE else "paper",
        "open": STATE["open"], "side": STATE["side"], "qty": STATE["qty"],
        "compound_pnl": compound_pnl, "timestamp": datetime.utcnow().isoformat(),
        "trend_state": TREND_STATE, "wait_for_next_signal": wait_for_next_signal_side
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
    log_banner("SUI TREND MASTER BOT - TREND-ONLY MULTI-EXCHANGE")
    state = load_state() or {}
    state.setdefault("in_position", False)

    if RESUME_ON_RESTART:
        try:
            state = resume_open_position(ex, SYMBOL, state)
        except Exception as e:
            log_w(f"resume error: {e}\n{traceback.format_exc()}")

    verify_execution_environment()

    print(colored(f"🎯 EXCHANGE: {EXCHANGE_NAME.upper()} • SYMBOL: {SYMBOL} • TIMEFRAME: {INTERVAL}", "yellow"))
    print(colored(f"⚡ RISK: {int(RISK_ALLOC*100)}% × {LEVERAGE}x • TREND-ONLY=ENABLED", "yellow"))
    print(colored(f"📊 TREND MODES: MID (ADX≥{ADX_MID_MIN}) | BIG (ADX≥{ADX_BIG_MIN})", "yellow"))
    print(colored(f"🧭 HTF BIAS: 1H EMA{HTF_EMA_FAST}/{HTF_EMA_SLOW} | SMC+Liquidity", "yellow"))
    print(colored(f"💰 TP PLANS: MID (2 levels) | BIG (3 levels) | Explosion Boost", "yellow"))
    print(colored(f"🛡️ SAFETY: NO SCALP | NO WEAK | HTF CONFIRMATION", "yellow"))
    print(colored(f"🚀 EXECUTION: {'ACTIVE' if EXECUTE_ORDERS and not DRY_RUN else 'SIMULATION'}", "yellow"))
    print(colored(f"💧 LIQUIDITY ENGINE: STEP2 (Sweep/BOS/OB/FVG) + STEP3 (Explosion/Collapse)", "yellow"))
    print(colored(f"🌫️ CONTEXT: VWAP + Ichimoku Cloud Bias", "yellow"))
    
    logging.info("service starting…")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGINT,  lambda *_: sys.exit(0))
    
    import threading
    threading.Thread(target=trade_loop, daemon=True).start()
    threading.Thread(target=keepalive_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
