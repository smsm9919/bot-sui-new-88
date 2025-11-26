# -*- coding: utf-8 -*-
"""
SUI ULTRA PRO AI BOT - الإصدار المحترف الذكي المتقدم
• نظام دخول مرتب بمستويات Tier A/B/C
• حارس صارم يمنع الصفقات الضعيفة
• إدارة ذكية للمراكز بناءً على قوة الإشارات
• الحفاظ على الدوال الأساسية واللوج الملون
"""

import os, time, math, random, signal, sys, traceback, logging, json
from logging.handlers import RotatingFileHandler
from datetime import datetime
import pandas as pd
import numpy as np
import ccxt
from flask import Flask, jsonify
from decimal import Decimal, ROUND_DOWN, InvalidOperation
from collections import deque, defaultdict
import statistics

try:
    from termcolor import colored
except Exception:
    def colored(t,*a,**k): return t

# =================== STRICT ENTRY GUARDS CONFIG ===================

# عتبات الدخول الصارمة
STRICT_SCALP_MIN_SCORE = 7        # ✅ سكالب قوي فقط
STRICT_TREND_MIN_SCORE = 10       # ✅ ترند قوي فقط
STRICT_TREND_TIER_A_MIN = 1       # ✅ لازم Tier A للترند
STRICT_TIER_B_MIN = 2             # ✅ لازم إشارتين Tier B على الأقل

# أوزان الطبقات
TIER_A_WEIGHT = 4      # Golden / SMC قوي / Liquidity Sweep كبيرة
TIER_B_WEIGHT = 2      # FVG / OB / VWAP / Structure / Flow  
TIER_C_WEIGHT = 1      # RSI / ADX / Candles

# عتبات الدخول
TREND_MIN_SCORE = 10      # أقل Score لصفقة ترند
SCALP_MIN_SCORE = 5       # أقل Score لسكالب محترم

TREND_NEED_TIER_A = True    # لازم إشارة Tier A لصفقة ترند
SCALP_NEED_TIER_A = False   # السكالب ممكن بدون Tier A لو النقاط كفاية

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

# ==== Run mode / Logging toggles ====
LOG_LEGACY = True  # تفعيل اللوج الملون القديم
LOG_ADDONS = True

# ==== Execution Switches ====
EXECUTE_ORDERS = True
SHADOW_MODE_DASHBOARD = False
DRY_RUN = False

# ==== Addon: Logging + Recovery Settings ====
BOT_VERSION = f"SUI ULTRA PRO AI v8.0 — {EXCHANGE_NAME.upper()} - PROFESSIONAL STRICT SYSTEM"
print("🚀 Booting:", BOT_VERSION, flush=True)

STATE_PATH = "./bot_state.json"
RESUME_ON_RESTART = True
RESUME_LOOKBACK_SECS = 60 * 60

# =================== SETTINGS ===================
SYMBOL     = os.getenv("SYMBOL", "SUI/USDT:USDT")
INTERVAL   = os.getenv("INTERVAL", "15m")

# ===== RISK / LEVERAGE PROFILE (FIXED) =====
LEVERAGE   = 10
RISK_ALLOC = 0.60

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

MAX_SPREAD_BPS = float(os.getenv("MAX_SPREAD_BPS", 6.0))

# Dynamic TP / trail
TP1_PCT_BASE       = 0.45
TP1_CLOSE_FRAC     = 0.50
BREAKEVEN_AFTER    = 0.30
TRAIL_ACTIVATE_PCT = 1.20
ATR_TRAIL_MULT     = 1.8

# Dust guard
FINAL_CHUNK_QTY = float(os.getenv("FINAL_CHUNK_QTY", 50.0))
RESIDUAL_MIN_QTY = float(os.getenv("RESIDUAL_MIN_QTY", 10.0))

# Strict close
CLOSE_RETRY_ATTEMPTS = 6
CLOSE_VERIFY_WAIT_S  = 2.0

# Pacing
BASE_SLEEP   = 5
NEAR_CLOSE_S = 1

# ===== SMART ENTRY SYSTEM =====
SCALP_MODE = True
COUNCIL_AI_MODE = True
TREND_RIDING_AI = True

# ===== PROFIT PROFILES =====
PROFIT_PROFILE_CONFIG = {
    "SCALP_SMALL": {
        "label": "SCALP_SMALL",
        "tp1_pct": 0.45,
        "tp2_pct": None,
        "tp3_pct": None,
        "trail_start_pct": 0.50,
        "desc": "صفقة سكالب صغيرة"
    },
    "TREND_MEDIUM": {
        "label": "TREND_MEDIUM", 
        "tp1_pct": 0.8,
        "tp2_pct": 1.6,
        "tp3_pct": None,
        "trail_start_pct": 1.0,
        "desc": "ترند متوسط"
    },
    "TREND_STRONG": {
        "label": "TREND_STRONG",
        "tp1_pct": 0.8,
        "tp2_pct": 2.0,
        "tp3_pct": 4.0,
        "trail_start_pct": 1.2,
        "desc": "ترند قوي"
    },
}

# ===== SNAPSHOT & MARK SYSTEM =====
GREEN="🟢"; RED="🔴"
RESET="\x1b[0m"; BOLD="\x1b[1m"
FG_G="\x1b[32m"; FG_R="\x1b[31m"; FG_C="\x1b[36m"; FG_Y="\x1b[33m"; FG_M="\x1b[35m"

# ===== SMART QUANTITY FIX =====
MIN_QTY = 0.1
MIN_BALANCE_FOR_TRADE = 10.0

# =================== PROFESSIONAL LOGGING ===================
def log_i(msg): print(f"ℹ️ {msg}", flush=True)
def log_g(msg): print(f"✅ {msg}", flush=True)
def log_w(msg): print(f"🟨 {msg}", flush=True)
def log_e(msg): print(f"❌ {msg}", flush=True)

def log_banner(text): print(f"\n{'—'*12} {text} {'—'*12}\n", flush=True)

def save_state(state: dict):
    try:
        state["ts"] = int(time.time())
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
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

# =================== EXCHANGE FACTORY ===================
def make_ex():
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
    if EXCHANGE_NAME == "bybit":
        if POSITION_MODE == "hedge":
            return {"positionSide": "Long" if side == "buy" else "Short", "reduceOnly": is_close}
        return {"positionSide": "Both", "reduceOnly": is_close}
    else:
        if POSITION_MODE == "hedge":
            return {"positionSide": "LONG" if side == "buy" else "SHORT", "reduceOnly": is_close}
        return {"positionSide": "BOTH", "reduceOnly": is_close}

def exchange_set_leverage(exchange, leverage, symbol):
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
        LOT_MIN  = (MARKET.get("limits", {}) or {}).get("amount", {}).get("min",  None)
        log_i(f"🎯 {SYMBOL} specs → precision={AMT_PREC}, step={LOT_STEP}, min={LOT_MIN}")
    except Exception as e:
        log_w(f"load_market_specs: {e}")

def ensure_leverage_mode():
    try:
        exchange_set_leverage(ex, LEVERAGE, SYMBOL)
        log_i(f"📊 {EXCHANGE_NAME.upper()} position mode: {POSITION_MODE}")
    except Exception as e:
        log_w(f"ensure_leverage_mode: {e}")

try:
    load_market_specs()
    ensure_leverage_mode()
except Exception as e:
    log_w(f"exchange init: {e}")

# =================== LOGGING SETUP ===================
def setup_file_logging():
    """إعداد التسجيل المهني مع قمع رسائل Werkzeug"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    if not any(isinstance(h, RotatingFileHandler) and getattr(h, "baseFilename", "").endswith("bot.log")
               for h in logger.handlers):
        fh = RotatingFileHandler("bot.log", maxBytes=5_000_000, backupCount=7, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s [%(filename)s:%(lineno)d]"))
        logger.addHandler(fh)
    
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(ch)
    
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    logging.getLogger('ccxt.base.exchange').setLevel(logging.INFO)
    
    log_i("🔄 Professional logging ready")

setup_file_logging()

# =================== HELPERS ===================
_consec_err = 0
last_loop_ts = time.time()

def _fmt(x,n=6):
    try: return f"{float(x):.{n}f}"
    except: return str(x)

def _pct(x):
    try: return f"{float(x):.2f}%"
    except: return str(x)

def last_scalar(x, default=0.0):
    """يرجع float من آخر عنصر"""
    try:
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, pd.Series): 
            return float(x.iloc[-1])
        if isinstance(x, (list, tuple, np.ndarray)): 
            return float(x[-1])
        if isinstance(x, str):
            return None
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        return float(x)
    except Exception:
        return None

def safe_get(ind: dict, key: str, default=0.0):
    """يقرأ مؤشر من dict ويحوّله scalar أخير"""
    if ind is None: 
        return float(default)
    val = ind.get(key, default)
    result = last_scalar(val, default=default)
    return result if result is not None else float(default)

def _round_amt(q):
    """نسخة محسنة من التقريب مع منع القيم الصغيرة"""
    if q is None: 
        return MIN_QTY
        
    try:
        d = Decimal(str(q))
        
        if d < Decimal(str(MIN_QTY)):
            return float(MIN_QTY)
            
        if LOT_STEP and isinstance(LOT_STEP, (int, float)) and LOT_STEP > 0:
            step = Decimal(str(LOT_STEP))
            d = (d / step).to_integral_value(rounding=ROUND_DOWN) * step
            
        prec = int(AMT_PREC) if AMT_PREC and AMT_PREC >= 0 else 0
        d = d.quantize(Decimal(1).scaleb(-prec), rounding=ROUND_DOWN)
        
        if LOT_MIN and isinstance(LOT_MIN, (int, float)) and LOT_MIN > 0 and d < Decimal(str(LOT_MIN)):
            return float(MIN_QTY)
            
        result = float(d)
        
        if result <= 0:
            return float(MIN_QTY)
            
        return result
        
    except (InvalidOperation, ValueError, TypeError):
        return float(MIN_QTY)

def safe_qty(q): 
    """نسخة محسنة مع حماية من القيم الصغيرة جداً"""
    try:
        q_float = float(q) if q else 0.0
        
        if q_float < MIN_QTY:
            log_w(f"🛑 كمية صغيرة جداً: {q_float:.6f} < {MIN_QTY}، رفع إلى الحد الأدنى")
            q_float = MIN_QTY
            
        q_rounded = _round_amt(q_float)
        
        if q_rounded <= 0:
            log_w(f"🛑 الكمية بعد التقريب صفر: {q_float:.6f} → {q_rounded}")
            q_rounded = MIN_QTY
            
        log_i(f"✅ كمية الصفقة النهائية: {q_rounded:.4f}")
        return q_rounded
        
    except Exception as e:
        log_e(f"❌ خطأ في safe_qty: {e}")
        return MIN_QTY

def compute_size(balance, price):
    """
    حجم اللوت ثابت:
    - 60% من رصيد المحفظة
    - ×10x ليفرج
    """
    effective_balance = float(balance or 0.0)
    px = float(price or 0.0)

    if effective_balance <= 0 or px <= 0:
        return 0.0

    capital_usdt = effective_balance * 0.60
    notional_usdt = capital_usdt * 10.0
    raw_qty = notional_usdt / px

    qty = safe_qty(raw_qty)

    log_i(
        f"SIZE_FIXED_60pct_10x | bal={effective_balance:.2f} | "
        f"price={px:.6f} | capital={capital_usdt:.2f} | "
        f"notional={notional_usdt:.2f} | qty={qty:.4f}"
    )

    return qty

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

# =================== SMART CONTEXT SYSTEMS ===================
class SmartTrendContext:
    def __init__(self):
        self.fast_ma = deque(maxlen=20)
        self.slow_ma = deque(maxlen=50)
        self.trend = "flat"
        self.strength = 0.0
        self.momentum = 0.0

    def update(self, close, volume=None):
        self.fast_ma.append(close)
        self.slow_ma.append(close)

        if len(self.slow_ma) < 10:
            self.trend = "flat"
            self.strength = 0
            self.momentum = 0
            return

        fast = sum(self.fast_ma) / len(self.fast_ma)
        slow = sum(self.slow_ma) / len(self.slow_ma)

        delta = fast - slow
        self.strength = abs(delta) / slow * 100 if slow != 0 else 0
        
        if len(self.fast_ma) >= 5:
            recent = list(self.fast_ma)[-5:]
            self.momentum = (recent[-1] - recent[0]) / recent[0] * 100 if recent[0] != 0 else 0

        if delta > 0 and self.strength > 0.1:
            self.trend = "up"
        elif delta < 0 and self.strength > 0.1:
            self.trend = "down"
        else:
            self.trend = "flat"

    def is_strong_trend(self):
        return self.strength > 0.3 and abs(self.momentum) > 0.5

class SMCDetector:
    def __init__(self):
        self.swing_highs = deque(maxlen=10)
        self.swing_lows = deque(maxlen=10)
        self.liquidity_zones = []
        
    def detect_swings(self, df, lookback=20):
        if len(df) < lookback * 2:
            return
            
        highs = df['high'].astype(float).tail(lookback * 2)
        lows = df['low'].astype(float).tail(lookback * 2)
        
        for i in range(lookback, len(highs) - lookback):
            if highs.iloc[i] == highs.iloc[i-lookback:i+lookback].max():
                self.swing_highs.append((i, highs.iloc[i]))
            if lows.iloc[i] == lows.iloc[i-lookback:i+lookback].min():
                self.swing_lows.append((i, lows.iloc[i]))
    
    def detect_liquidity_zones(self, current_price):
        zones = []
        for _, high in self.swing_highs:
            if high > current_price * 1.01:
                zones.append(("sell_liquidity", high))
        
        for _, low in self.swing_lows:
            if low < current_price * 0.99:
                zones.append(("buy_liquidity", low))
                
        return zones

def volume_is_strong(vol_list, window=20, threshold=1.4):
    if len(vol_list) < window:
        return False
    recent = vol_list[-window:]
    avg = sum(recent) / len(recent)
    return recent[-1] > avg * threshold

def detect_ob(candles):
    if len(candles) < 5:
        return None
    
    candle_list = []
    for i in range(len(candles)):
        candle_list.append({
            'open': float(candles['open'].iloc[i]),
            'high': float(candles['high'].iloc[i]),
            'low': float(candles['low'].iloc[i]),
            'close': float(candles['close'].iloc[i])
        })
    
    if len(candle_list) < 5:
        return None
        
    b = candle_list[-4]
    c = candle_list[-3]
    
    if b['close'] < b['open'] and c['close'] > c['open']:
        return ("bullish", b['open'], b['close'])
    
    if b['close'] > b['open'] and c['close'] < c['open']:
        return ("bearish", b['open'], b['close'])
    
    return None

def detect_fvg(candles):
    if len(candles) < 4:
        return None
        
    candle_list = []
    for i in range(len(candles)):
        candle_list.append({
            'open': float(candles['open'].iloc[i]),
            'high': float(candles['high'].iloc[i]),
            'low': float(candles['low'].iloc[i]),
            'close': float(candles['close'].iloc[i])
        })
    
    if len(candle_list) < 4:
        return None
        
    a = candle_list[-4]
    b = candle_list[-3]
    c = candle_list[-2]

    if a['high'] < c['low']:
        return ("bullish", a['high'], c['low'])

    if a['low'] > c['high']:
        return ("bearish", c['high'], a['low'])

    return None

# =================== STRICT ENTRY QUALITY ENGINE ===================
def evaluate_entry_quality(direction, council_data, current_price=None):
    """
    🎯 أقوى نظام تقييم دخول - يجمع كل الإستراتيجيات في Score واحد
    """
    score = 0
    tier_a_hits = []
    tier_b_hits = [] 
    tier_c_hits = []
    
    # استخراج البيانات من مجلس الإدارة
    golden = council_data.get("golden", {})
    smc = council_data.get("smc_analysis", {})
    fvg = council_data.get("fvg_analysis", {})
    ob = council_data.get("order_blocks", {})
    structure = council_data.get("price_structure", {})
    rsi_ctx = council_data.get("rsi_context", {})
    adx_ctx = council_data.get("adx_context", {})
    candles = council_data.get("candle_signals", {})

    # ============= TIER A - الإشارات القوية (إلزامي للترند) =============
    
    # 1. المناطق الذهبية القوية
    if golden.get("ok") and golden.get("score", 0) >= 6.5:
        score += TIER_A_WEIGHT * 2
        tier_a_hits.append(f"GOLDEN_{golden.get('zone_type', '').upper()}({golden.get('score', 0):.1f})")

    # 2. تحليل SMC قوي
    smc_strength = 0
    if smc.get("bos_detected"):
        smc_strength += 2
    if smc.get("choch_detected"): 
        smc_strength += 2
    if smc.get("liquidity_sweep"):
        smc_strength += 2
        
    if smc_strength >= 4:
        score += TIER_A_WEIGHT
        tier_a_hits.append(f"SMC_STRONG({smc_strength})")

    # 3. مناطق السيولة الكبيرة
    liq_zones = smc.get("liquidity_zones", [])
    strong_liquidity = len([z for z in liq_zones if z.get("strength", 0) >= 3]) >= 2
    if strong_liquidity:
        score += TIER_A_WEIGHT
        tier_a_hits.append("STRONG_LIQUIDITY")

    # ============= TIER B - إشارات الدعم القوية =============
    
    # 1. فجوات القيمة العادلة القوية
    if fvg.get("valid") and fvg.get("strength", 0) >= 3:
        score += TIER_B_WEIGHT
        tier_b_hits.append(f"FVG_STRONG({fvg.get('strength', 0):.1f})")

    # 2. كتل الأوامر المحترفة
    if ob.get("valid") and ob.get("strength", 0) >= 3:
        score += TIER_B_WEIGHT  
        tier_b_hits.append(f"OB_STRONG({ob.get('strength', 0):.1f})")

    # 3. هيكل سعري واضح
    if structure.get("trend_strength", 0) >= 2:
        score += TIER_B_WEIGHT
        tier_b_hits.append(f"STRUCTURE_{structure.get('trend', '').upper()}({structure.get('trend_strength', 0):.1f})")

    # ============= TIER C - عوامل تحسين =============
    
    # 1. RSI في منطقة مناسبة
    rsi_val = rsi_ctx.get("value", 50)
    if (direction == "buy" and 30 <= rsi_val <= 45) or (direction == "sell" and 55 <= rsi_val <= 70):
        score += TIER_C_WEIGHT
        tier_c_hits.append(f"RSI_OPTIMAL({rsi_val:.1f})")

    # 2. ADX قوي
    if adx_ctx.get("value", 0) >= 20:
        score += TIER_C_WEIGHT
        tier_c_hits.append(f"ADX_STRONG({adx_ctx.get('value', 0):.1f})")

    # 3. إشارات شموع قوية
    candle_strength = max(candles.get("buy_score", 0), candles.get("sell_score", 0))
    if candle_strength >= 2:
        score += TIER_C_WEIGHT
        tier_c_hits.append(f"CANDLE_STRONG({candle_strength:.1f})")

    # ============= قرار السماح بالدخول =============
    
    has_tier_a = len(tier_a_hits) > 0
    
    return {
        "score": score,
        "tier_a": tier_a_hits,
        "tier_b": tier_b_hits, 
        "tier_c": tier_c_hits,
        "has_tier_a": has_tier_a
    }

def master_entry_engine(council_data, current_price):
    """
    🧠 المحرك الرئيسي للدخول - يقرر الجانب والنمط بناءً على كل البيانات
    """
    # استخراج اتجاه مجلس الإدارة
    score_buy = council_data.get("score_b", 0)
    score_sell = council_data.get("score_s", 0)
    votes_buy = council_data.get("b", 0)
    votes_sell = council_data.get("s", 0)
    
    # تحديد الاتجاه الأساسي
    if score_buy > score_sell and votes_buy > votes_sell:
        direction = "buy"
        confidence = min(1.0, score_buy / 20.0)
    elif score_sell > score_buy and votes_sell > votes_buy:
        direction = "sell" 
        confidence = min(1.0, score_sell / 20.0)
    else:
        return {"allow": False, "reason": "no_clear_direction"}
    
    # تحديد النمط المبدئي
    adx_val = council_data.get("adx", 0)
    if adx_val >= 25 and confidence >= 0.7:
        preliminary_mode = "trend"
    else:
        preliminary_mode = "scalp"
    
    # تقييم جودة الدخول
    quality = evaluate_entry_quality(direction, council_data, current_price)
    
    # تحديد السماح بالدخول بناءً على النمط والجودة
    if preliminary_mode == "trend":
        min_score = TREND_MIN_SCORE
        need_tier_a = TREND_NEED_TIER_A
    else:
        min_score = SCALP_MIN_SCORE
        need_tier_a = SCALP_NEED_TIER_A

    allow = True
    reasons = []

    if quality["score"] < min_score:
        allow = False
        reasons.append(f"low_score({quality['score']:.1f} < {min_score})")

    if need_tier_a and not quality["has_tier_a"]:
        allow = False
        reasons.append("no_tier_A_signal")

    if not allow:
        return {
            "allow": False,
            "reason": "low_quality",
            "details": quality,
            "reasons": reasons
        }
    
    # التعديل النهائي للنمط بناءً على الجودة
    if preliminary_mode == "trend" and not quality["has_tier_a"]:
        final_mode = "scalp"
    else:
        final_mode = preliminary_mode
        
    return {
        "allow": True,
        "direction": direction,
        "mode": final_mode,
        "confidence": confidence,
        "quality": quality,
        "details": {
            "score_buy": score_buy,
            "score_sell": score_sell,
            "votes_buy": votes_buy,
            "votes_sell": votes_sell,
            "adx": adx_val
        }
    }

def apply_strict_entry_guards(entry_decision, current_price=None):
    """
    🛡️ حارس صارم يمنع الصفقات الضعيفة - الإصدار المحترف
    """
    if not entry_decision["allow"]:
        return entry_decision
    
    mode = entry_decision["mode"]
    direction = entry_decision["direction"]
    quality = entry_decision["quality"]
    score = quality["score"]
    tier_a_count = len(quality["tier_a"])
    tier_b_count = len(quality["tier_b"])
    
    reasons = []
    blocked = False
    
    # 🔥 الحارس 1: السكالب الضعيف
    if mode == "scalp" and score < STRICT_SCALP_MIN_SCORE:
        blocked = True
        reasons.append(f"scalp_score_too_low({score} < {STRICT_SCALP_MIN_SCORE})")
    
    # 🔥 الحارس 2: الترند بدون Tier A
    if mode == "trend" and tier_a_count < STRICT_TREND_TIER_A_MIN:
        blocked = True
        reasons.append(f"trend_missing_tier_a({tier_a_count} < {STRICT_TREND_TIER_A_MIN})")
    
    # 🔥 الحارس 3: اتجاه غير واضح
    council_data = entry_decision.get("details", {})
    score_buy = council_data.get("score_buy", 0)
    score_sell = council_data.get("score_sell", 0)
    
    if direction == "buy" and score_buy <= score_sell:
        blocked = True
        reasons.append(f"weak_buy_signal({score_buy} <= {score_sell})")
    elif direction == "sell" and score_sell <= score_buy:
        blocked = True
        reasons.append(f"weak_sell_signal({score_sell} <= {score_buy})")
    
    # 🔥 الحارس 4: SMC ضعيف + VWAP غير محاذي + FVG ضعيف
    weak_structure = (
        tier_a_count == 0 and  # لا يوجد Tier A
        tier_b_count < STRICT_TIER_B_MIN and  # أقل من إشارتين Tier B
        score < 8  # درجة عامة ضعيفة
    )
    
    if weak_structure:
        blocked = True
        reasons.append("weak_structure_analysis")
    
    if blocked:
        return {
            "allow": False,
            "reason": "strict_guard_blocked",
            "details": quality,
            "reasons": reasons,
            "original_decision": entry_decision
        }
    
    return entry_decision

def enhanced_master_entry_engine(council_data, current_price):
    """
    🧠 المحرك المحسن للدخول مع الحارس الصارم
    """
    # القرار الأساسي من المحرك الرئيسي
    base_decision = master_entry_engine(council_data, current_price)
    
    # تطبيق الحارس الصارم
    final_decision = apply_strict_entry_guards(base_decision, current_price)
    
    return final_decision

# =================== PROFESSIONAL TRADE EXECUTION ===================
def open_market_enhanced(side, qty, price):
    """نسخة محسنة من فتح الصفقة مع النظام الجديد"""
    if qty <= 0 or price is None:
        log_e("❌ كمية أو سعر غير صالح")
        return False

    # 🔥 الحارس النهائي قبل التنفيذ
    df = fetch_ohlcv(limit=200)
    council_data = super_council_ai_enhanced(df)
    entry_decision = enhanced_master_entry_engine(council_data, price)
    
    if not entry_decision["allow"]:
        reasons = entry_decision.get("reasons", [])
        print(colored(f"🛑 FINAL GUARD BLOCKED: {', '.join(reasons)}", "red"))
        return False

    # تحقق إضافي من الحجم
    balance = balance_usdt()
    expected_qty = compute_size(balance, price)
    
    if abs(qty - expected_qty) > (expected_qty * 0.1):
        log_w(f"⚠️ تصحيح الحجم: {qty:.4f} → {expected_qty:.4f}")
        qty = expected_qty

    df = fetch_ohlcv(limit=200)
    ind = compute_indicators(df)

    # تحديد المود النهائي
    trend_info = compute_trend_strength(df, ind)
    trend_strength = trend_info.get("strength", "flat")
    
    mode = entry_decision["mode"]
    quality = entry_decision["quality"]

    # تحديد Profit Profile المناسب
    if mode == "scalp":
        profit_profile = PROFIT_PROFILE_CONFIG["SCALP_SMALL"]
    elif trend_strength in ["strong", "very_strong"] and quality["score"] >= 12:
        profit_profile = PROFIT_PROFILE_CONFIG["TREND_STRONG"]
    else:
        profit_profile = PROFIT_PROFILE_CONFIG["TREND_MEDIUM"]

    # إعدادات الإدارة
    management_config = {
        "tp1_pct": profit_profile["tp1_pct"],
        "tp2_pct": profit_profile["tp2_pct"],
        "tp3_pct": profit_profile["tp3_pct"],
        "trail_start_pct": profit_profile["trail_start_pct"],
        "profile": profit_profile["label"],
        "profile_desc": profit_profile["desc"]
    }

    # تنفيذ الأمر
    success = execute_trade_decision(side, price, qty, mode, council_data)

    if success:
        trade_side = "long" if side.lower().startswith("b") else "short"
        
        STATE.update({
            "open": True,
            "side": trade_side,
            "entry": float(price),
            "qty": float(qty),
            "pnl": 0.0,
            "bars": 0,
            "mode": mode,
            "management": management_config,
            "opened_at": time.time(),
            "tp1_done": False,
            "trail_active": False,
            "breakeven_armed": False,
            "highest_profit_pct": 0.0,
            "profit_targets_achieved": 0,
            "profit_profile": profit_profile["label"],
            "entry_quality": quality
        })

        save_state({
            "in_position": True,
            "side": "LONG" if trade_side == "long" else "SHORT",
            "entry_price": price,
            "position_qty": qty,
            "leverage": LEVERAGE,
            "mode": mode,
            "profit_profile": profit_profile["label"],
            "management": management_config,
            "opened_at": int(time.time())
        })

        # 🎯 اللوج الملون القديم مع المعلومات الجديدة
        profile_color = "🟢" if profit_profile["label"] == "TREND_STRONG" else "🟡" if profit_profile["label"] == "TREND_MEDIUM" else "🔵"
        print(colored(f"🎯 EXECUTED: {side.upper()} {qty:.4f} @ {price:.6f} | {mode.upper()} | Score: {quality['score']:.1f}", "green"))
        log_g(
            f"{profile_color} PROFESSIONAL TRADE OPENED | {side.upper()} {qty:.4f} @ {price:.6f} "
            f"| {mode.upper()} | {profit_profile['label']} | "
            f"Quality Score: {quality['score']:.1f} | "
            f"Tier A: {', '.join(quality['tier_a']) or 'None'}"
        )
        
        return True

    return False

def execute_trade_decision(side, price, qty, mode, council_data):
    if not EXECUTE_ORDERS or DRY_RUN:
        log_i(f"DRY_RUN: {side} {qty:.4f} @ {price:.6f} | mode={mode}")
        return True
    
    if qty <= 0:
        log_e("❌ كمية غير صالحة للتنفيذ")
        return False

    print(f"🎯 EXECUTE: {side.upper()} {qty:.4f} @ {price:.6f} | "
          f"mode={mode} | votes={council_data['b']}/{council_data['s']} score={council_data['score_b']:.1f}/{council_data['score_s']:.1f}", flush=True)

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

# =================== SMART TRADE MANAGEMENT ===================
def manage_trade_by_profile(df, ind, info):
    """إدارة الصفقة حسب التصنيف المحدد من النظام الجديد"""
    if not STATE["open"] or STATE["qty"] <= 0:
        return

    px = info["price"]
    entry = STATE["entry"]
    side = STATE["side"]
    mode = STATE.get("mode", "scalp")
    profile = STATE.get("profit_profile", "SCALP_SMALL")
    
    # حساب الربح
    pnl_pct = (px - entry) / entry * 100 * (1 if side == "long" else -1)
    STATE["pnl"] = pnl_pct
    
    if pnl_pct > STATE["highest_profit_pct"]:
        STATE["highest_profit_pct"] = pnl_pct

    # جلب إعدادات الـ Profile
    management = STATE.get("management", {})
    tp1 = management.get("tp1_pct", 0.45)
    tp2 = management.get("tp2_pct")
    tp3 = management.get("tp3_pct")
    
    # تطبيق أهداف الربح حسب الـ Profile
    if profile == "SCALP_SMALL" and not STATE.get("tp1_done") and pnl_pct >= tp1:
        close_market_strict(f"SCALP_SMALL TP: {tp1}%")
        return
        
    elif profile == "TREND_MEDIUM":
        if not STATE.get("tp1_done") and pnl_pct >= tp1:
            close_qty = safe_qty(STATE["qty"] * 0.5)
            if close_qty > 0:
                close_side = "sell" if side == "long" else "buy"
                if MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN:
                    try:
                        params = exchange_specific_params(close_side, is_close=True)
                        ex.create_order(SYMBOL, "market", close_side, close_qty, None, params)
                        log_g(f"🎯 TREND_MEDIUM TP1 | {tp1}% | closed 50%")
                        STATE["qty"] = safe_qty(STATE["qty"] - close_qty)
                        STATE["tp1_done"] = True
                    except Exception as e:
                        log_e(f"❌ TREND_MEDIUM TP1 close failed: {e}")
                        
        elif STATE.get("tp1_done") and not STATE.get("tp2_done") and pnl_pct >= tp2:
            close_market_strict(f"TREND_MEDIUM TP2: {tp2}%")
            return
            
    elif profile == "TREND_STRONG":
        if not STATE.get("tp1_done") and pnl_pct >= tp1:
            close_qty = safe_qty(STATE["qty"] * 0.3)
            if close_qty > 0:
                close_side = "sell" if side == "long" else "buy"
                if MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN:
                    try:
                        params = exchange_specific_params(close_side, is_close=True)
                        ex.create_order(SYMBOL, "market", close_side, close_qty, None, params)
                        log_g(f"🎯 TREND_STRONG TP1 | {tp1}% | closed 30%")
                        STATE["qty"] = safe_qty(STATE["qty"] - close_qty)
                        STATE["tp1_done"] = True
                    except Exception as e:
                        log_e(f"❌ TREND_STRONG TP1 close failed: {e}")
                        
        elif STATE.get("tp1_done") and not STATE.get("tp2_done") and pnl_pct >= tp2:
            close_qty = safe_qty(STATE["qty"] * 0.3)
            if close_qty > 0:
                close_side = "sell" if side == "long" else "buy"
                if MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN:
                    try:
                        params = exchange_specific_params(close_side, is_close=True)
                        ex.create_order(SYMBOL, "market", close_side, close_qty, None, params)
                        log_g(f"🎯 TREND_STRONG TP2 | {tp2}% | closed 30%")
                        STATE["qty"] = safe_qty(STATE["qty"] - close_qty)
                        STATE["tp2_done"] = True
                    except Exception as e:
                        log_e(f"❌ TREND_STRONG TP2 close failed: {e}")
                        
        elif STATE.get("tp2_done") and not STATE.get("tp3_done") and pnl_pct >= tp3:
            close_market_strict(f"TREND_STRONG TP3: {tp3}%")
            return

    STATE["bars"] += 1

def apply_smart_profit_strategy():
    """نسخة مبسطة من نظام جني الأرباح"""
    if not STATE.get("open") or STATE["qty"] <= 0:
        return
        
    try:
        current_price = price_now()
        if not current_price or not STATE.get("entry"):
            return
            
        entry_price = STATE["entry"]
        side = STATE["side"]
        mode = STATE.get("mode", "scalp")
        
        # حساب الربح/الخسارة
        if side == "long":
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
        else:
            pnl_pct = ((entry_price - current_price) / entry_price) * 100
        
        STATE["pnl"] = pnl_pct
        
        # نظام جني الأرباح المبسط
        if mode == "scalp":
            if pnl_pct >= 0.8 and not STATE.get("scalp_tp_done", False):
                log_g(f"💰 SCALP TP FULL | pnl={pnl_pct:.2f}%")
                close_market_strict("scalp_tp_full")
                STATE["scalp_tp_done"] = True
                return
        else:
            # TP1 عند 1.5% - إغلاق 40%
            if (pnl_pct >= 1.5 and 
                not STATE.get("trend_tp1_done", False) and 
                STATE["qty"] > 0):
                
                close_qty = safe_qty(STATE["qty"] * 0.4)
                if close_qty > 0:
                    close_side = "sell" if STATE["side"] == "long" else "buy"
                    if MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN:
                        try:
                            params = exchange_specific_params(close_side, is_close=True)
                            ex.create_order(SYMBOL, "market", close_side, close_qty, None, params)
                            log_g(f"🎯 TREND TP1 | pnl={pnl_pct:.2f}% | closed 40%")
                        except Exception as e:
                            log_e(f"❌ TREND TP1 close failed: {e}")
                    STATE["qty"] = safe_qty(STATE["qty"] - close_qty)
                    STATE["trend_tp1_done"] = True
            
            # TP2 عند 3.0% - إغلاق باقي الصفقة
            if (pnl_pct >= 3.0 and 
                not STATE.get("trend_tp2_done", False) and 
                STATE["qty"] > 0):
                
                log_g(f"🏁 TREND TP2 FULL EXIT | pnl={pnl_pct:.2f}%")
                close_market_strict("trend_tp2_full")
                STATE["trend_tp2_done"] = True
                return
                
    except Exception as e:
        log_w(f"Simple profit strategy error: {e}")

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

def compute_trend_strength(df, ind):
    close = df['close'].astype(float)
    adx = safe_get(ind, 'adx', 0)
    plus_di = safe_get(ind, 'plus_di', 0)
    minus_di = safe_get(ind, 'minus_di', 0)
    
    momentum_5 = ((close.iloc[-1] - close.iloc[-5]) / close.iloc[-5]) * 100 if len(close) >= 5 else 0
    
    if adx > 40 and abs(momentum_5) > 3.0:
        strength = "very_strong"
    elif adx > 30 and abs(momentum_5) > 2.0:
        strength = "strong"
    elif adx > 25 and abs(momentum_5) > 1.0:
        strength = "moderate"
    elif adx > 20:
        strength = "weak"
    else:
        strength = "no_trend"
    
    direction = "up" if plus_di > minus_di else "down"
    
    return {
        "strength": strength,
        "direction": direction,
        "adx": adx,
        "momentum_5": momentum_5
    }

# =================== COUNCIL AI SYSTEM ===================
def super_council_ai_enhanced(df):
    """نسخة مبسطة من مجلس الإدارة"""
    try:
        if len(df) < 50:
            return {"b": 0, "s": 0, "score_b": 0.0, "score_s": 0.0, "confidence": 0.0}
        
        ind = compute_indicators(df)
        
        # استخراج القيم
        adx = safe_get(ind, "adx", 0.0)
        plus_di = safe_get(ind, "plus_di", 0.0)
        minus_di = safe_get(ind, "minus_di", 0.0)
        rsi_val = safe_get(ind, "rsi", 50.0)
        
        votes_b = 0
        votes_s = 0
        score_b = 0.0
        score_s = 0.0

        # تحليل الاتجاه
        if plus_di > minus_di and adx > 20:
            votes_b += 2
            score_b += adx * 0.1
        elif minus_di > plus_di and adx > 20:
            votes_s += 2
            score_s += adx * 0.1

        # تحليل RSI
        if rsi_val < 40:
            votes_b += 1
            score_b += (40 - rsi_val) * 0.1
        elif rsi_val > 60:
            votes_s += 1
            score_s += (rsi_val - 60) * 0.1

        # تحليل المناطق الذهبية
        golden = golden_zone_check(df, ind)
        if golden.get("ok"):
            if golden["zone"]["type"] == "golden_bottom":
                votes_b += 3
                score_b += golden["score"] * 0.2
            elif golden["zone"]["type"] == "golden_top":
                votes_s += 3
                score_s += golden["score"] * 0.2

        # تحليل الشموع
        candles = compute_candles(df)
        if candles["score_buy"] > 0:
            votes_b += 1
            score_b += candles["score_buy"] * 0.3
        if candles["score_sell"] > 0:
            votes_s += 1
            score_s += candles["score_sell"] * 0.3

        # حساب الثقة
        total_score = score_b + score_s
        confidence = min(1.0, total_score / 20.0)

        return {
            "b": votes_b, "s": votes_s,
            "score_b": round(score_b, 2), "score_s": round(score_s, 2),
            "adx": adx,
            "golden": golden,
            "candle_signals": candles,
            "confidence": round(confidence, 2)
        }
    except Exception as e:
        log_w(f"super_council_ai_enhanced error: {e}")
        return {"b":0,"s":0,"score_b":0.0,"score_s":0.0,"confidence":0.0}

def golden_zone_check(df, ind=None):
    """نسخة مبسطة من كشف المناطق الذهبية"""
    try:
        if len(df) < 60:
            return {"ok": False, "score": 0.0}
        
        close = df['close'].astype(float)
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        
        # تحليل مبسط للمناطق الذهبية
        current_price = float(close.iloc[-1])
        recent_high = float(high.tail(20).max())
        recent_low = float(low.tail(20).min())
        
        # حساب المناطق الذهبية
        golden_bottom = recent_low + (recent_high - recent_low) * 0.618
        golden_top = recent_high - (recent_high - recent_low) * 0.618
        
        score = 0.0
        zone_type = None
        
        if current_price <= golden_bottom * 1.01:
            zone_type = "golden_bottom"
            score = 7.0
        elif current_price >= golden_top * 0.99:
            zone_type = "golden_top" 
            score = 7.0
            
        if zone_type:
            return {
                "ok": True,
                "score": score,
                "zone_type": zone_type,
                "zone": {
                    "type": zone_type,
                    "price_level": golden_bottom if zone_type == "golden_bottom" else golden_top
                }
            }
        else:
            return {"ok": False, "score": 0.0}
            
    except Exception as e:
        return {"ok": False, "score": 0.0}

def compute_candles(df):
    """تحليل مبسط للشموع"""
    if len(df) < 3:
        return {"buy":False,"sell":False,"score_buy":0.0,"score_sell":0.0}
    
    try:
        o1 = float(df["open"].iloc[-2])
        h1 = float(df["high"].iloc[-2])
        l1 = float(df["low"].iloc[-2])
        c1 = float(df["close"].iloc[-2])
        
        o0 = float(df["open"].iloc[-3])
        c0 = float(df["close"].iloc[-3])
        
        score_buy = 0.0
        score_sell = 0.0
        
        # شمعة engulfing صاعدة
        if c0 < o0 and c1 > o1 and c1 > o0 and o1 < c0:
            score_buy += 2.0
            
        # شمعة engulfing هابطة  
        if c0 > o0 and c1 < o1 and c1 < o0 and o1 > c0:
            score_sell += 2.0
            
        # شمعة hammer
        body = abs(c1 - o1)
        lower_wick = min(o1, c1) - l1
        if lower_wick >= 2 * body and c1 > o1:
            score_buy += 1.5
            
        # شمعة shooting star
        upper_wick = h1 - max(o1, c1)
        if upper_wick >= 2 * body and c1 < o1:
            score_sell += 1.5
            
        return {
            "buy": score_buy > 0,
            "sell": score_sell > 0,
            "score_buy": round(score_buy, 2),
            "score_sell": round(score_sell, 2)
        }
    except Exception as e:
        return {"buy":False,"sell":False,"score_buy":0.0,"score_sell":0.0}

# =================== RANGE FILTER ===================
def _ema(s, n): return s.ewm(span=n, adjust=False).mean()

def _rng_size(src: pd.Series, qty: float, n: int) -> pd.Series:
    avrng = _ema((src - src.shift(1)).abs(), n)
    wper = (n*2)-1
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

# =================== STATE & POSITION MANAGEMENT ===================
STATE = {
    "open": False, "side": None, "entry": None, "qty": 0.0,
    "pnl": 0.0, "bars": 0, "trail": None, "breakeven": None,
    "tp1_done": False, "highest_profit_pct": 0.0,
    "profit_targets_achieved": 0,
}
compound_pnl = 0.0

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

def close_market_strict(reason="STRICT"):
    global compound_pnl
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
    global compound_pnl
    STATE.update({
        "open": False, "side": None, "entry": None, "qty": 0.0,
        "pnl": 0.0, "bars": 0, "trail": None, "breakeven": None,
        "tp1_done": False, "highest_profit_pct": 0.0, "profit_targets_achieved": 0,
        "trail_tightened": False, "partial_taken": False
    })
    save_state({"in_position": False, "position_qty": 0})
    
    logging.info(f"AFTER_CLOSE: {reason}")

# =================== PROFESSIONAL TRADE LOOP ===================
def professional_trade_loop():
    """
    🎯 الحلقة الرئيسية للتداول المحترف مع نظام الجودة
    """
    global compound_pnl
    
    # تهيئة أنظمة السياق
    trend_ctx = SmartTrendContext()
    smc_detector = SMCDetector()
    
    loop_i = 0
    
    while True:
        try:
            current_time = time.time()
            bal = balance_usdt()
            px = price_now()
            df = fetch_ohlcv()
            
            if df.empty:
                time.sleep(BASE_SLEEP)
                continue
                
            # تحديث أنظمة السياق
            close_prices = df['close'].astype(float).tolist()
            volumes = df['volume'].astype(float).tolist()
            
            trend_ctx.update(close_prices[-1] if close_prices else 0)
            smc_detector.detect_swings(df)
            
            info = rf_signal_live(df)
            ind = compute_indicators(df)
            spread_bps = orderbook_spread_bps()
            
            # ✅ نظام جني الأرباح الذكي
            if STATE.get("open") and px:
                apply_smart_profit_strategy()
                
            # ============================================
            #  PROFESSIONAL ENTRY DECISION ENGINE
            # ============================================

            if not STATE["open"]:
                # جمع بيانات مجلس الإدارة
                council_data = super_council_ai_enhanced(df)
                
                # استخدام المحرك المحسن مع الحارس الصارم
                entry_decision = enhanced_master_entry_engine(council_data, px or info["price"])
                
                if entry_decision["allow"]:
                    direction = entry_decision["direction"]
                    mode = entry_decision["mode"]
                    quality = entry_decision["quality"]
                    
                    # حساب الحجم
                    qty = compute_size(bal, px or info["price"])
                    if qty > 0:
                        # فتح الصفقة
                        success = open_market_enhanced(direction, qty, px or info["price"])
                        if success:
                            # 🎯 اللوج الملون القديم مع المعلومات الجديدة
                            print(colored(f"🎯 EXECUTED: {direction.upper()} {qty:.4f} @ {px:.6f} | {mode.upper()} | Score: {quality['score']:.1f}", "green"))
                            log_i(f"🎯 PROFESSIONAL ENTRY EXECUTED | {direction.upper()} {mode.upper()} | "
                                  f"Quality Score: {quality['score']:.1f} | "
                                  f"Tier A: {', '.join(quality['tier_a']) or 'None'} | "
                                  f"Tier B: {', '.join(quality['tier_b']) or 'None'}")
                        else:
                            log_e("❌ Failed to execute professional entry")
                    else:
                        log_w("❌ Invalid quantity for professional entry")
                        
                else:
                    # 🎯 اللوج الملون القديم لأسباب الرفض
                    quality = entry_decision.get("quality", {})
                    reasons = entry_decision.get("reasons", [])
                    
                    if "strict_guard_blocked" in entry_decision.get("reason", ""):
                        print(colored(f"⛔ BLOCKED: {', '.join(reasons)} | Score: {quality.get('score', 0):.1f}", "red"))
                    else:
                        print(colored(f"⛔ REJECTED: {entry_decision.get('reason', 'unknown')} | Score: {quality.get('score', 0):.1f}", "yellow"))
                    
                    log_i(f"⛔ ENTRY REJECTED | {entry_decision.get('reason', 'unknown')} | "
                          f"Score: {quality.get('score', 0):.1f} | "
                          f"Tier A: {', '.join(quality.get('tier_a', [])) or 'None'} | "
                          f"Reasons: {', '.join(reasons)}")

            # إدارة الصفقة المفتوحة
            if STATE["open"]:
                manage_trade_by_profile(df, ind, {
                    "price": px or info["price"], 
                    "trend_ctx": trend_ctx
                })
            
            # 🎯 اللوج الملون القديم
            if LOG_LEGACY:
                pretty_snapshot(bal, {"price": px or info["price"], **info}, ind, spread_bps, "Professional Strict System", df)
            
            loop_i += 1
            sleep_s = NEAR_CLOSE_S if time_to_candle_close(df) <= 10 else BASE_SLEEP
            time.sleep(sleep_s)
            
        except Exception as e:
            log_e(f"Professional trade loop error: {e}\n{traceback.format_exc()}")
            time.sleep(BASE_SLEEP)

# =================== LEGACY COLORED SNAPSHOT ===================
def pretty_snapshot(bal, info, ind, spread_bps, reason=None, df=None):
    """
    🎯 اللوج الملون القديم - تم الحفاظ عليه مع إضافة معلومات الجودة
    """
    if LOG_LEGACY:
        left_s = time_to_candle_close(df) if df is not None else 0
        print(colored("─"*100,"cyan"))
        print(colored(f"📊 {SYMBOL} {INTERVAL} • {EXCHANGE_NAME.upper()} • {'LIVE' if MODE_LIVE else 'PAPER'} • {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC","cyan"))
        print(colored("─"*100,"cyan"))
        print("📈 INDICATORS & RF")
        print(f"   💲 Price {fmt(info.get('price'))} | RF filt={fmt(info.get('filter'))}  hi={fmt(info.get('hi'))} lo={fmt(info.get('lo'))}")
        print(f"   🧮 RSI={fmt(safe_get(ind, 'rsi'))}  +DI={fmt(safe_get(ind, 'plus_di'))}  -DI={fmt(safe_get(ind, 'minus_di'))}  ADX={fmt(safe_get(ind, 'adx'))}  ATR={fmt(safe_get(ind, 'atr'))}")
        print(f"   🎯 ENTRY: PROFESSIONAL STRICT SYSTEM | spread_bps={fmt(spread_bps,2)}")
        print(f"   ⏱️ closes_in ≈ {left_s}s")
        print("\n🧭 POSITION")
        bal_line = f"Balance={fmt(bal,2)}  Risk={int(RISK_ALLOC*100)}%×{LEVERAGE}x  CompoundPnL={fmt(compound_pnl)}  Eq~{fmt((bal or 0)+compound_pnl,2)}"
        print(colored(f"   {bal_line}", "yellow"))
        if STATE["open"]:
            lamp='🟩 LONG' if STATE['side']=='long' else '🟥 SHORT'
            print(f"   {lamp}  Entry={fmt(STATE['entry'])}  Qty={fmt(STATE['qty'],4)}  Bars={STATE['bars']}  Trail={fmt(STATE['trail'])}  BE={fmt(STATE['breakeven'])}")
            print(f"   🎯 TP_done={STATE['profit_targets_achieved']}  HP={fmt(STATE['highest_profit_pct'],2)}%")
            print(f"   🎯 MODE={STATE.get('mode', 'trend')}  PROFILE={STATE.get('profit_profile', 'none')}")
            
            # إضافة معلومات الجودة إذا كانت متاحة
            quality = STATE.get("entry_quality", {})
            if quality:
                print(f"   🏆 QUALITY: Score={quality.get('score', 0):.1f} | Tier A: {len(quality.get('tier_a', []))} | Tier B: {len(quality.get('tier_b', []))}")
        else:
            print("   ⚪ FLAT")
        if reason: print(colored(f"   ℹ️ reason: {reason}", "white"))
        print(colored("─"*100,"cyan"))

# =================== API / KEEPALIVE ===================
app = Flask(__name__)

@app.get("/mark/<color>")
def mark_position(color):
    color = color.lower()
    if color not in ["green", "red"]:
        return jsonify({"ok": False, "error": "Use /mark/green or /mark/red"}), 400
    
    return jsonify({"ok": True, "marked": color, "timestamp": datetime.utcnow().isoformat()})

@app.route("/")
def home():
    mode='LIVE' if MODE_LIVE else 'PAPER'
    return f"✅ SUI ULTRA PRO AI BOT — {EXCHANGE_NAME.upper()} — {SYMBOL} {INTERVAL} — {mode} — PROFESSIONAL STRICT SYSTEM"

@app.route("/metrics")
def metrics():
    return jsonify({
        "exchange": EXCHANGE_NAME,
        "symbol": SYMBOL, "interval": INTERVAL, "mode": "live" if MODE_LIVE else "paper",
        "leverage": LEVERAGE, "risk_alloc": RISK_ALLOC, "price": price_now(),
        "state": STATE, "compound_pnl": compound_pnl,
        "entry_system": "PROFESSIONAL_STRICT_SYSTEM",
        "strict_guards": {
            "scalp_min_score": STRICT_SCALP_MIN_SCORE,
            "trend_min_score": STRICT_TREND_MIN_SCORE,
            "trend_tier_a_min": STRICT_TREND_TIER_A_MIN,
            "tier_b_min": STRICT_TIER_B_MIN
        }
    })

@app.route("/health")
def health():
    return jsonify({
        "ok": True, "exchange": EXCHANGE_NAME, "mode": "live" if MODE_LIVE else "paper",
        "open": STATE["open"], "side": STATE["side"], "qty": STATE["qty"],
        "compound_pnl": compound_pnl, "timestamp": datetime.utcnow().isoformat(),
        "entry_system": "PROFESSIONAL_STRICT_SYSTEM"
    }), 200

@app.route("/entry_quality")
def get_entry_quality():
    """عرض جودة الدخول الحالية"""
    current_quality = STATE.get("entry_quality", {})
    return jsonify({
        "current_quality": current_quality,
        "strict_guards": {
            "scalp_min_score": STRICT_SCALP_MIN_SCORE,
            "trend_min_score": STRICT_TREND_MIN_SCORE,
            "trend_tier_a_min": STRICT_TREND_TIER_A_MIN,
            "tier_b_min": STRICT_TIER_B_MIN
        }
    })

@app.route("/tier_stats")
def get_tier_stats():
    """إحصائيات الـ Tiers"""
    quality = STATE.get("entry_quality", {})
    return jsonify({
        "current_tiers": {
            "tier_a": quality.get("tier_a", []),
            "tier_b": quality.get("tier_b", []),
            "tier_c": quality.get("tier_c", [])
        },
        "total_score": quality.get("score", 0),
        "entry_mode": STATE.get("mode", "unknown")
    })

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

# =================== EXECUTION VERIFICATION ===================
def verify_execution_environment():
    print(f"⚙️ PROFESSIONAL STRICT EXECUTION ENVIRONMENT", flush=True)
    print(f"🔧 EXCHANGE: {EXCHANGE_NAME.upper()} | SYMBOL: {SYMBOL}", flush=True)
    print(f"🔧 EXECUTE_ORDERS: {EXECUTE_ORDERS} | DRY_RUN: {DRY_RUN}", flush=True)
    print(f"🎯 PROFESSIONAL STRICT SYSTEM ACTIVATED", flush=True)
    print(f"   Strict Scalp Min Score: {STRICT_SCALP_MIN_SCORE}", flush=True)
    print(f"   Strict Trend Min Score: {STRICT_TREND_MIN_SCORE}", flush=True)
    print(f"   Strict Trend Tier A Min: {STRICT_TREND_TIER_A_MIN}", flush=True)
    print(f"   Strict Tier B Min: {STRICT_TIER_B_MIN}", flush=True)

if __name__ == "__main__":
    verify_execution_environment()
    
    import threading
    threading.Thread(target=keepalive_loop, daemon=True).start()
    threading.Thread(target=professional_trade_loop, daemon=True).start()
    
    log_i(f"🚀 SUI ULTRA PRO AI BOT STARTED - {BOT_VERSION}")
    log_i(f"🎯 SYMBOL: {SYMBOL} | INTERVAL: {INTERVAL} | LEVERAGE: {LEVERAGE}x")
    log_i(f"💡 PROFESSIONAL STRICT SYSTEM ACTIVATED")
    log_i(f"   • السكالب: يحتاج score ≥ {STRICT_SCALP_MIN_SCORE}")
    log_i(f"   • الترند: يحتاج score ≥ {STRICT_TREND_MIN_SCORE} + Tier A ≥ {STRICT_TREND_TIER_A_MIN}")
    log_i(f"   • الحارس الصارم: يمنع الصفقات الضعيفة تلقائياً")
    
    app.run(host="0.0.0.0", port=PORT, debug=False)
