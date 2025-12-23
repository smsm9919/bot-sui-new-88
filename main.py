# -*- coding: utf-8 -*-
"""
SUI PROFESSIONAL TRADING BOT — ULTIMATE EDITION
• مجلس إدارة ذكي متكامل مع جميع استراتيجيات التداول
• نظام جني الأرباح المتعدد المستويات (3 مستويات للذهبية، 1 للسكالب)
• تحليل Footprint متقدم لاكتشاف الامتصاص والاندفاع الحقيقي
• إدارة صفقات احترافية مع وقف خسارة متحرك ذكي
• نظام تعلم من الأخطاء وتحسين مستمر
• مكنة أرباح ذكية تعمل بكامل طاقتها
"""

import os, time, math, random, signal, sys, traceback, logging, json
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import ccxt
from flask import Flask, jsonify
from decimal import Decimal, ROUND_DOWN, InvalidOperation

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
BOT_VERSION = f"SUI PROFESSIONAL TRADER v8.0 — {EXCHANGE_NAME.upper()} — MONEY MAKING MACHINE"
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

# =================== PROFESSIONAL FOOTPRINT ANALYSIS SETTINGS ===================
FOOTPRINT_PERIOD = 20
FOOTPRINT_VOLUME_THRESHOLD = 2.0
DELTA_THRESHOLD = 1.5
ABSORPTION_RATIO = 0.65
EFFICIENCY_THRESHOLD = 0.85
FOOTPRINT_MIN_CONFIDENCE = 2.5  # الحد الأدنى لدخول Footprint
FOOTPRINT_EXIT_THRESHOLD = -1.5  # عتبة الخروج عند Footprint سلبي

# =================== SETTINGS ===================
SYMBOL     = os.getenv("SYMBOL", "SUI/USDT:USDT")
INTERVAL   = os.getenv("INTERVAL", "15m")
LEVERAGE   = int(os.getenv("LEVERAGE", 10))
RISK_ALLOC = float(os.getenv("RISK_ALLOC", 0.60))
POSITION_MODE = os.getenv("POSITION_MODE", "oneway")

# RF Settings - Optimized for SUI
RF_SOURCE = "close"
RF_PERIOD = int(os.getenv("RF_PERIOD", 18))
RF_MULT   = float(os.getenv("RF_MULT", 3.0))
RF_LIVE_ONLY = True
RF_HYST_BPS  = 6.0

# =================== CRITICAL VARIABLES FROM ORIGINAL BOT ===================
# Pacing
BASE_SLEEP   = 5
NEAR_CLOSE_S = 1

# Indicators
RSI_LEN = 14
ADX_LEN = 14
ATR_LEN = 14

ENTRY_RF_ONLY = False
MAX_SPREAD_BPS = float(os.getenv("MAX_SPREAD_BPS", 6.0))

# ==== Golden Zone Constants ====
FIB_LOW, FIB_HIGH = 0.618, 0.786
MIN_WICK_PCT = 0.35
VOL_MA_LEN = 20
RSI_LEN_GZ, RSI_MA_LEN_GZ = 14, 9
MIN_DISP = 0.8

# ==== Execution & Strategy Thresholds ====
ADX_TREND_MIN = 22  # Increased
DI_SPREAD_TREND = 6
RSI_MA_LEN = 9
RSI_NEUTRAL_BAND = (45, 55)
RSI_TREND_PERSIST = 3

GZ_MIN_SCORE = 7.0  # Increased
GZ_REQ_ADX = 22  # Increased
GZ_REQ_VOL_MA = 20
ALLOW_GZ_ENTRY = True

MAX_TRADES_PER_HOUR = 4  # Reduced for quality over quantity
COOLDOWN_SECS_AFTER_CLOSE = 90  # Increased cooldown
ADX_GATE = 17

# ==== ULTIMATE COUNCIL SETTINGS ====
ULTIMATE_MIN_CONFIDENCE = 8.0
FOOTPRINT_VOTE_WEIGHT = 4  # Highest weight for Footprint
VOLUME_MOMENTUM_PERIOD = 20
STOCH_RSI_PERIOD = 14
DYNAMIC_PIVOT_PERIOD = 20
TREND_FAST_PERIOD = 10
TREND_SLOW_PERIOD = 20
TREND_SIGNAL_PERIOD = 9

# ==== POSITION MANAGEMENT SETTINGS ====
EARLY_EXIT_IF_WRONG_ZONE = True
WRONG_ZONE_FOOTPRINT_SCORE = -2.0
MIN_PROFIT_FOR_EARLY_EXIT = 0.15  # 0.15% minimum profit to exit early
MAX_LOSS_BEFORE_FORCE_EXIT = -0.8  # -0.8% force exit

# ==== Smart Exit Tuning ===
HARD_CLOSE_PNL_PCT = 1.10/100
WICK_ATR_MULT      = 1.5
EVX_SPIKE          = 1.8
BM_WALL_PROX_BPS   = 5
TIME_IN_TRADE_MIN  = 8
TRAIL_TIGHT_MULT   = 1.20

# ==== Golden Entry Settings ====
GOLDEN_ENTRY_SCORE = 7.0  # Increased for stricter entry
GOLDEN_ENTRY_ADX   = 22.0  # Increased
GOLDEN_REVERSAL_SCORE = 6.5

# =================== PROFESSIONAL PROFIT TAKING SYSTEM ===================
# نظام جني الأرباح الاحترافي حسب نوع الصفقة
class ProfitTakingSystem:
    """نظام جني الأرباح الذكي المتعدد المستويات"""
    
    @staticmethod
    def get_tp_config(trade_type, zone_strength):
        """
        إعدادات جني الأرباح حسب نوع الصفقة وقوة المنطقة
        trade_type: 'GOLDEN_ROCKET', 'SCALP', 'TREND'
        zone_strength: 'VERY_STRONG', 'STRONG', 'MODERATE', 'WEAK', 'VERY_WEAK'
        """
        
        # تعريف قوة المنطقة كأرقام
        strength_map = {
            'VERY_STRONG': 5, 'STRONG': 4, 'MODERATE': 3, 
            'WEAK': 2, 'VERY_WEAK': 1
        }
        
        strength = strength_map.get(zone_strength, 3)
        
        if trade_type == 'GOLDEN_ROCKET':
            # 3 مستويات للصفقات الذهبية الصاروخية
            if strength >= 4:  # منطقة قوية جداً
                return {
                    'tp_levels': [0.8, 1.6, 2.8],  # نسب ربح أعلى
                    'tp_fractions': [0.25, 0.35, 0.40],  # إغلاق تدريجي
                    'trail_start': 1.2,  # بدء التريل بعد 1.2%
                    'atr_trail_mult': 1.5,
                    'partial_close_at_breakeven': True,
                    'move_to_breakeven_after_tp1': True,
                    'description': 'ذهبية صاروخية (منطقة قوية جداً)'
                }
            else:  # منطقة متوسطة
                return {
                    'tp_levels': [0.6, 1.2, 2.0],
                    'tp_fractions': [0.30, 0.30, 0.40],
                    'trail_start': 1.0,
                    'atr_trail_mult': 1.8,
                    'partial_close_at_breakeven': True,
                    'move_to_breakeven_after_tp1': False,
                    'description': 'ذهبية صاروخية (منطقة متوسطة)'
                }
        
        elif trade_type == 'SCALP':
            # مستوى واحد فقط للصفقات السكالب
            if strength >= 4:  # منطقة قوية - نزيد الربح قليلاً
                return {
                    'tp_levels': [0.6],  # مستوى واحد فقط
                    'tp_fractions': [0.5],  # إغلاق 50% فقط
                    'trail_start': 0.8,
                    'atr_trail_mult': 1.6,
                    'partial_close_at_breakeven': True,
                    'move_to_breakeven_after_tp1': False,
                    'description': 'سكالب (منطقة قوية - نغلق جزئياً)'
                }
            elif strength >= 3:  # منطقة متوسطة
                return {
                    'tp_levels': [0.5],  # مستوى واحد فقط
                    'tp_fractions': [1.0],  # إغلاق كامل
                    'trail_start': 0.6,
                    'atr_trail_mult': 1.8,
                    'partial_close_at_breakeven': True,
                    'move_to_breakeven_after_tp1': False,
                    'description': 'سكالب (منطقة متوسطة - نغلق كلياً)'
                }
            else:  # منطقة ضعيفة - لا ندخل أصلاً
                return {
                    'tp_levels': [0.3],  # مستوى واحد مع ربح أقل
                    'tp_fractions': [1.0],  # إغلاق كامل
                    'trail_start': 0.4,
                    'atr_trail_mult': 2.0,
                    'partial_close_at_breakeven': True,
                    'move_to_breakeven_after_tp1': False,
                    'description': 'سكالب (منطقة ضعيفة - خروج سريع)'
                }
        
        else:  # TREND_RIDING
            # مستويين لركوب الترند
            return {
                'tp_levels': [0.8, 1.8],  # مستويين
                'tp_fractions': [0.4, 0.6],  # 40% ثم 60%
                'trail_start': 1.0,
                'atr_trail_mult': 1.7,
                'partial_close_at_breakeven': True,
                'move_to_breakeven_after_tp1': True,
                'description': 'ركوب الترند'
            }
    
    @staticmethod
    def calculate_dynamic_tp_levels(entry_price, atr, trade_type, zone_strength):
        """حساب مستويات جني الأرباح الديناميكية بناءً على ATR"""
        
        config = ProfitTakingSystem.get_tp_config(trade_type, zone_strength)
        
        # تحويل النسب المئوية إلى نقاط سعرية
        tp_levels_price = []
        for tp_percent in config['tp_levels']:
            tp_price = entry_price * (1 + tp_percent / 100) if trade_type != 'SELL' else entry_price * (1 - tp_percent / 100)
            tp_levels_price.append(tp_price)
        
        # إضافة مستويات تعتمد على ATR للمستويات المتقدمة
        if len(tp_levels_price) > 1:
            # المستوى الثاني يعتمد على 2x ATR
            tp_levels_price[1] = entry_price + (atr * 2) if trade_type != 'SELL' else entry_price - (atr * 2)
            
        if len(tp_levels_price) > 2:
            # المستوى الثالث يعتمد على 3.5x ATR
            tp_levels_price[2] = entry_price + (atr * 3.5) if trade_type != 'SELL' else entry_price - (atr * 3.5)
        
        return {
            'tp_levels_price': tp_levels_price,
            'tp_fractions': config['tp_fractions'],
            'trail_start_pct': config['trail_start'],
            'atr_trail_mult': config['atr_trail_mult'],
            'description': config['description']
        }

# =================== PROFESSIONAL LOGGING ===================
def log_i(msg): print(f"ℹ️ {msg}", flush=True)
def log_g(msg): print(f"✅ {msg}", flush=True)
def log_w(msg): print(f"🟨 {msg}", flush=True)
def log_e(msg): print(f"❌ {msg}", flush=True)
def log_f(msg): print(f"👣 {msg}", flush=True)  # Footprint logging

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

# =================== ADVANCED INDICATORS ===================
def enhanced_volume_momentum(df, period=20):
    """الزخم الحجمي المتقدم"""
    if len(df) < period + 5:
        return {"trend": "neutral", "strength": 0, "signal": 0}
    
    volume = df['volume'].astype(float)
    close = df['close'].astype(float)
    
    # متوسط حجم متحرك
    volume_ma = volume.rolling(period).mean()
    volume_ratio = volume / volume_ma.replace(0, 1)
    
    # زخم السعر مع الحجم
    price_change = close.pct_change(period)
    volume_weighted_momentum = price_change * volume_ratio
    
    current_momentum = volume_weighted_momentum.iloc[-1]
    momentum_trend = "bull" if current_momentum > 0.02 else ("bear" if current_momentum < -0.02 else "neutral")
    
    return {
        "trend": momentum_trend,
        "strength": abs(current_momentum) * 100,
        "signal": current_momentum
    }

def stochastic_rsi_enhanced(df, rsi_period=14, stoch_period=14, k_period=3, d_period=3):
    """مؤشر RSI العشوائي المحسن"""
    if len(df) < max(rsi_period, stoch_period) + 10:
        return {"k": 50, "d": 50, "signal": "neutral", "oversold": False, "overbought": False}
    
    # حساب RSI
    rsi = compute_rsi(df['close'].astype(float), rsi_period)
    
    # حساب Stochastic للـ RSI
    rsi_low = rsi.rolling(stoch_period).min()
    rsi_high = rsi.rolling(stoch_period).max()
    
    stoch_k = 100 * (rsi - rsi_low) / (rsi_high - rsi_low).replace(0, 100)
    stoch_k_smooth = stoch_k.rolling(k_period).mean()
    stoch_d = stoch_k_smooth.rolling(d_period).mean()
    
    current_k = stoch_k_smooth.iloc[-1]
    current_d = stoch_d.iloc[-1]
    
    # إشارات التداول
    signal = "neutral"
    if current_k < 20 and current_d < 20:
        signal = "bullish"
    elif current_k > 80 and current_d > 80:
        signal = "bearish"
    elif current_k > current_d and stoch_k_smooth.iloc[-2] <= stoch_d.iloc[-2]:
        signal = "bullish_cross"
    elif current_k < current_d and stoch_k_smooth.iloc[-2] >= stoch_d.iloc[-2]:
        signal = "bearish_cross"
    
    return {
        "k": current_k,
        "d": current_d,
        "signal": signal,
        "oversold": current_k < 20,
        "overbought": current_k > 80
    }

def dynamic_pivot_points(df, period=20):
    """نقاط محورية ديناميكية"""
    if len(df) < period:
        return {"pivot": 0, "r1": 0, "r2": 0, "s1": 0, "s2": 0, "bias": "neutral"}
    
    high = df['high'].astype(float).tail(period)
    low = df['low'].astype(float).tail(period)
    close = df['close'].astype(float).tail(period)
    
    pivot = (high.iloc[-1] + low.iloc[-1] + close.iloc[-1]) / 3
    r1 = 2 * pivot - low.iloc[-1]
    r2 = pivot + (high.iloc[-1] - low.iloc[-1])
    s1 = 2 * pivot - high.iloc[-1]
    s2 = pivot - (high.iloc[-1] - low.iloc[-1])
    
    current_price = close.iloc[-1]
    
    # تحديد الانحياز
    if current_price > r1:
        bias = "strong_bullish"
    elif current_price > pivot:
        bias = "bullish"
    elif current_price < s1:
        bias = "strong_bearish"
    elif current_price < pivot:
        bias = "bearish"
    else:
        bias = "neutral"
    
    return {
        "pivot": pivot,
        "r1": r1, "r2": r2,
        "s1": s1, "s2": s2,
        "bias": bias
    }

def dynamic_trend_indicator(df, fast_period=10, slow_period=20, signal_period=9):
    """مؤشر الاتجاه الديناميكي"""
    if len(df) < slow_period + signal_period:
        return {"trend": "neutral", "momentum": 0, "signal": "hold"}
    
    close = df['close'].astype(float)
    
    # متوسطات متحركة متعددة
    ema_fast = close.ewm(span=fast_period).mean()
    ema_slow = close.ewm(span=slow_period).mean()
    ema_signal = ema_fast.ewm(span=signal_period).mean()
    
    # تقاطعات الاتجاه
    fast_above_slow = ema_fast.iloc[-1] > ema_slow.iloc[-1]
    fast_above_signal = ema_fast.iloc[-1] > ema_signal.iloc[-1]
    
    # زخم الاتجاه
    momentum = (ema_fast.iloc[-1] - ema_slow.iloc[-1]) / ema_slow.iloc[-1] * 100
    
    # تحديد الاتجاه
    if fast_above_slow and fast_above_signal and momentum > 0.1:
        trend = "strong_bull"
    elif fast_above_slow and momentum > 0:
        trend = "bull"
    elif not fast_above_slow and not fast_above_signal and momentum < -0.1:
        trend = "strong_bear"
    elif not fast_above_slow and momentum < 0:
        trend = "bear"
    else:
        trend = "neutral"
    
    # إشارة التداول
    signal = "hold"
    if trend == "strong_bull" and ema_fast.iloc[-2] <= ema_slow.iloc[-2]:
        signal = "strong_buy"
    elif trend == "bull" and ema_fast.iloc[-2] <= ema_signal.iloc[-2]:
        signal = "buy"
    elif trend == "strong_bear" and ema_fast.iloc[-2] >= ema_slow.iloc[-2]:
        signal = "strong_sell"
    elif trend == "bear" and ema_fast.iloc[-2] >= ema_signal.iloc[-2]:
        signal = "sell"
    
    return {
        "trend": trend,
        "momentum": momentum,
        "signal": signal,
        "ema_fast": ema_fast.iloc[-1],
        "ema_slow": ema_slow.iloc[-1]
    }

# =================== PROFESSIONAL FOOTPRINT ANALYSIS ===================
def advanced_footprint_analysis(df, current_price):
    """
    تحليل بصمة السوق المتقدم لاكتشاف:
    - الامتصاص (Absorption)
    - الاندفاع الحقيقي (Real Momentum)
    - نقاط التوقف (Stops)
    - السيولة المخفية (Hidden Liquidity)
    - قوة منطقة الدخول/الخروج
    """
    if len(df) < FOOTPRINT_PERIOD + 5:
        return {"ok": False, "reason": "لا توجد بيانات كافية", "entry_score": 0, "exit_score": 0}
    
    try:
        # تحليل الحجم والسعر المتقدم
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        close = df['close'].astype(float)
        volume = df['volume'].astype(float)
        open_price = df['open'].astype(float)
        
        # المتوسطات الحجمية
        volume_ma = volume.rolling(FOOTPRINT_PERIOD).mean()
        volume_ratio = volume / volume_ma.replace(0, 1)
        
        # حساب دلتا الحجم (الفرق بين الشراء والبيع)
        up_volume = volume.where(close > open_price, 0)
        down_volume = volume.where(close < open_price, 0)
        volume_delta = (up_volume - down_volume).fillna(0)
        
        # كفاءة الحركة (Efficiency)
        body_size = abs(close - open_price)
        total_range = high - low
        efficiency = body_size / total_range.replace(0, 1)
        
        # تحليل الشمعة الحالية
        current_candle = {
            'high': float(high.iloc[-1]),
            'low': float(low.iloc[-1]),
            'close': float(close.iloc[-1]),
            'open': float(open_price.iloc[-1]),
            'volume': float(volume.iloc[-1]),
            'volume_ratio': float(volume_ratio.iloc[-1]),
            'delta': float(volume_delta.iloc[-1]),
            'efficiency': float(efficiency.iloc[-1]),
            'delta_normalized': float(volume_delta.iloc[-1]) / max(volume_ma.iloc[-1], 1)
        }
        
        # ========== تحليل الدخول ==========
        entry_score_bull = 0.0
        entry_score_bear = 0.0
        entry_reasons = []
        
        # 1. امتصاص صاعد قوي للدخول الشراء
        if (current_candle['volume_ratio'] >= FOOTPRINT_VOLUME_THRESHOLD and
            current_candle['efficiency'] < ABSORPTION_RATIO and
            current_candle['delta'] > DELTA_THRESHOLD):
            entry_score_bull += 2.5
            entry_reasons.append("امتصاص صاعد قوي للدخول")
        
        # 2. امتصاص هابط قوي للدخول البيع
        if (current_candle['volume_ratio'] >= FOOTPRINT_VOLUME_THRESHOLD and
            current_candle['efficiency'] < ABSORPTION_RATIO and
            current_candle['delta'] < -DELTA_THRESHOLD):
            entry_score_bear += 2.5
            entry_reasons.append("امتصاص هابط قوي للدخول")
        
        # 3. اندفاع صاعد حقيقي
        if (current_candle['volume_ratio'] >= FOOTPRINT_VOLUME_THRESHOLD and
            current_candle['efficiency'] > EFFICIENCY_THRESHOLD and
            current_candle['delta'] > DELTA_THRESHOLD * 1.5):
            entry_score_bull += 3.0
            entry_reasons.append("اندفاع صاعد حقيقي")
        
        # 4. اندفاع هابط حقيقي
        if (current_candle['volume_ratio'] >= FOOTPRINT_VOLUME_THRESHOLD and
            current_candle['efficiency'] > EFFICIENCY_THRESHOLD and
            current_candle['delta'] < -DELTA_THRESHOLD * 1.5):
            entry_score_bear += 3.0
            entry_reasons.append("اندفاع هابط حقيقي")
        
        # 5. صيد توقف صاعد (Stop Hunt Bullish)
        if len(df) >= 3:
            prev_low = float(low.iloc[-2])
            if current_candle['low'] < prev_low and current_candle['close'] > prev_low:
                entry_score_bull += 2.0
                entry_reasons.append("صيد توقف صاعد")
        
        # 6. صيد توقف هابط (Stop Hunt Bearish)
        if len(df) >= 3:
            prev_high = float(high.iloc[-2])
            if current_candle['high'] > prev_high and current_candle['close'] < prev_high:
                entry_score_bear += 2.0
                entry_reasons.append("صيد توقف هابط")
        
        # ========== تحليل الخروج ==========
        exit_score_bull = 0.0  # سلبي للشراء (إشارة خروج)
        exit_score_bear = 0.0  # سلبي للبيع (إشارة خروج)
        exit_reasons = []
        
        # 1. امتصاص عكسي (إشارة خروج قوية)
        if (current_candle['volume_ratio'] >= FOOTPRINT_VOLUME_THRESHOLD and
            current_candle['efficiency'] < ABSORPTION_RATIO and
            current_candle['delta'] < -DELTA_THRESHOLD * 0.8):
            exit_score_bull += 2.5  # خروج من الشراء
            exit_reasons.append("امتصاص عكسي هابط")
        
        if (current_candle['volume_ratio'] >= FOOTPRINT_VOLUME_THRESHOLD and
            current_candle['efficiency'] < ABSORPTION_RATIO and
            current_candle['delta'] > DELTA_THRESHOLD * 0.8):
            exit_score_bear += 2.5  # خروج من البيع
            exit_reasons.append("امتصاص عكسي صاعد")
        
        # 2. فقدان الزخم (حجم منخفض مع حركة)
        if current_candle['volume_ratio'] < 0.5 and current_candle['efficiency'] > 0.6:
            exit_score_bull += 1.5
            exit_score_bear += 1.5
            exit_reasons.append("فقدان الزخم الحجمي")
        
        # 3. دلتا سلبية قوية بعد حركة
        if current_candle['delta_normalized'] < -1.0:
            exit_score_bull += 2.0
            exit_reasons.append("دلتا سلبية قوية")
        
        if current_candle['delta_normalized'] > 1.0:
            exit_score_bear += 2.0
            exit_reasons.append("دلتا إيجابية قوية")
        
        # ========== حساب النتيجة النهائية ==========
        # نتيجة الدخول: إيجابية للاتجاه
        # نتيجة الخروج: سلبية للاتجاه المعاكس
        
        return {
            "ok": True,
            "entry_score_bull": entry_score_bull,
            "entry_score_bear": entry_score_bear,
            "exit_score_bull": exit_score_bull,  # إشارة خروج من الشراء
            "exit_score_bear": exit_score_bear,  # إشارة خروج من البيع
            "current_candle": current_candle,
            "entry_reasons": entry_reasons,
            "exit_reasons": exit_reasons,
            "summary": {
                "strong_buy_entry": entry_score_bull >= FOOTPRINT_MIN_CONFIDENCE,
                "strong_sell_entry": entry_score_bear >= FOOTPRINT_MIN_CONFIDENCE,
                "buy_exit_signal": exit_score_bull >= abs(FOOTPRINT_EXIT_THRESHOLD),
                "sell_exit_signal": exit_score_bear >= abs(FOOTPRINT_EXIT_THRESHOLD)
            }
        }
        
    except Exception as e:
        return {"ok": False, "reason": f"خطأ في التحليل: {str(e)}", "entry_score": 0, "exit_score": 0}

def analyze_liquidity_pools(df, current_price):
    """تحليل تجمعات السيولة المخفية"""
    if len(df) < 50:
        return {}
    
    try:
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        volume = df['volume'].astype(float)
        
        # البحث عن مناطق السيولة (نقاط الدعم والمقاومة)
        lookback = min(100, len(df))
        recent_highs = high.tail(lookback)
        recent_lows = low.tail(lookback)
        
        # تحديد المستويات الرئيسية
        resistance_levels = find_significant_highs(recent_highs)
        support_levels = find_significant_lows(recent_lows)
        
        # تحليل القرب من مستويات السيولة
        buy_liquidity_above = False
        sell_liquidity_below = False
        
        for level in resistance_levels:
            if abs(current_price - level) / current_price <= 0.02:  # ضمن 2%
                sell_liquidity_below = True
                break
        
        for level in support_levels:
            if abs(current_price - level) / current_price <= 0.02:  # ضمن 2%
                buy_liquidity_above = True
                break
        
        return {
            "resistance_levels": resistance_levels[-3:],  # آخر 3 مستويات مقاومة
            "support_levels": support_levels[-3:],        # آخر 3 مستويات دعم
            "buy_liquidity_above": buy_liquidity_above,
            "sell_liquidity_below": sell_liquidity_below
        }
        
    except Exception as e:
        return {}

def find_significant_highs(series, window=5):
    """إيجاد القمم الهامة"""
    highs = []
    for i in range(window, len(series) - window):
        if (series.iloc[i] == series.iloc[i-window:i+window].max() and 
            series.iloc[i] > series.iloc[i-1] and 
            series.iloc[i] > series.iloc[i+1]):
            highs.append(series.iloc[i])
    return highs

def find_significant_lows(series, window=5):
    """إيجاد القيعان الهامة"""
    lows = []
    for i in range(window, len(series) - window):
        if (series.iloc[i] == series.iloc[i-window:i+window].min() and 
            series.iloc[i] < series.iloc[i-1] and 
            series.iloc[i] < series.iloc[i+1]):
            lows.append(series.iloc[i])
    return lows

# =================== DYNAMIC TRADE TYPE DETECTION ===================
class TradeTypeDetector:
    """كاشف نوع الصفقة الذكي"""
    
    @staticmethod
    def detect_trade_type(df, council_data, gz_data, current_price):
        """تحديد نوع الصفقة بدقة"""
        
        ind = council_data.get('ind', {})
        candles = council_data.get('candles', {})
        footprint = council_data.get('advanced_indicators', {}).get('footprint', {})
        
        # 1. تحليل قوة الاتجاه
        adx = ind.get('adx', 0)
        di_spread = ind.get('di_spread', 0)
        rsi = ind.get('rsi', 50)
        
        # 2. تحليل المناطق الذهبية
        is_golden_zone = gz_data and gz_data.get('ok') and gz_data.get('score', 0) >= 7.0
        
        # 3. تحليل Footprint
        footprint_strong = False
        if footprint.get('ok'):
            fp_entry_score = max(footprint.get('entry_score_bull', 0), footprint.get('entry_score_bear', 0))
            footprint_strong = fp_entry_score >= FOOTPRINT_MIN_CONFIDENCE * 1.5
        
        # 4. تحليل الشموع
        candle_score = max(candles.get('score_buy', 0), candles.get('score_sell', 0))
        
        # ======= قرار تحديد النوع =======
        
        # شرط الصفقة الذهبية الصاروخية
        golden_conditions = [
            is_golden_zone,
            footprint_strong,
            candle_score >= 3.0,
            adx >= 25,
            di_spread >= 8
        ]
        
        if sum(golden_conditions) >= 4:
            return 'GOLDEN_ROCKET', 'منطقة ذهبية مع إشارات قوية متعددة'
        
        # شرط ركوب الترند
        trend_conditions = [
            adx >= 22,
            di_spread >= 6,
            not (40 <= rsi <= 60),  # ليس في منطقة محايدة
            footprint.get('ok', False)
        ]
        
        if sum(trend_conditions) >= 3:
            return 'TREND_RIDING', 'اتجاه قوي مع تأكيد حجم'
        
        # شرط السكالب
        scalp_conditions = [
            adx < 20,  # سوق هادئ
            40 <= rsi <= 60,  # في منطقة محايدة
            candles.get('wick_up_big', False) or candles.get('wick_dn_big', False),  # فتائل كبيرة
            footprint.get('ok', False)  # تحليل Footprint متوفر
        ]
        
        if sum(scalp_conditions) >= 3:
            return 'SCALP', 'سوق هادئ مع فرص سكالب'
        
        # الإفتراضي
        return 'SCALP', 'النوع الإفتراضي (سكالب)'

# =================== PROFESSIONAL TRADE MANAGER ===================
class ProfessionalTradeManager:
    """مدير الصفقات الاحترافي مع نظام جني الأرباح المتعدد"""
    
    def __init__(self, exchange, symbol):
        self.exchange = exchange
        self.symbol = symbol
        self.active_trades = {}
        self.trade_history = []
        self.performance_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0,
            'largest_win': 0,
            'largest_loss': 0,
            'avg_win': 0,
            'avg_loss': 0
        }
    
    def open_trade(self, signal_data):
        """فتح صفقة جديدة بنظام إحترافي"""
        
        side = signal_data['side']
        entry_price = signal_data['entry_price']
        atr = signal_data['atr']
        trade_type = signal_data['trade_type']
        zone_strength = signal_data['zone_strength']
        confidence = signal_data['confidence']
        
        # حساب حجم المركز
        qty = self.calculate_position_size(entry_price, confidence, zone_strength)
        
        if qty <= 0:
            return None
        
        # حساب مستويات جني الأرباح الديناميكية
        tp_config = ProfitTakingSystem.calculate_dynamic_tp_levels(
            entry_price, atr, trade_type, zone_strength
        )
        
        # حساب وقف الخسارة الذكي
        stop_loss = self.calculate_smart_stop_loss(
            entry_price, atr, side, trade_type, zone_strength
        )
        
        # إنشاء الصفقة
        trade_id = f"{int(time.time())}_{side}_{random.randint(1000, 9999)}"
        
        trade = {
            'id': trade_id,
            'side': side,
            'entry_price': entry_price,
            'quantity': qty,
            'stop_loss': stop_loss,
            'take_profit_levels': tp_config['tp_levels_price'],
            'tp_fractions': tp_config['tp_fractions'],
            'trade_type': trade_type,
            'zone_strength': zone_strength,
            'confidence': confidence,
            'opened_at': datetime.now(),
            'status': 'OPEN',
            'current_pnl': 0,
            'current_pnl_pct': 0,
            'highest_pnl': 0,
            'highest_pnl_pct': 0,
            'tp_hit': [False] * len(tp_config['tp_levels_price']),
            'trail_activated': False,
            'trail_price': None,
            'breakeven_activated': False,
            'management_config': tp_config
        }
        
        # التنفيذ الفعلي
        if EXECUTE_ORDERS and not DRY_RUN and MODE_LIVE:
            try:
                params = exchange_specific_params(side, is_close=False)
                self.exchange.create_order(
                    self.symbol, "market", side, qty, None, params
                )
                self.active_trades[trade_id] = trade
                self.log_trade_opening(trade)
                return trade_id
            except Exception as e:
                log_e(f"فشل فتح الصفقة: {e}")
                return None
        else:
            # وضع المحاكاة
            self.active_trades[trade_id] = trade
            self.log_trade_opening(trade, simulated=True)
            return trade_id
    
    def calculate_position_size(self, entry_price, confidence, zone_strength):
        """حجم المركز الذكي بناءً على الثقة وقوة المنطقة"""
        
        # الحصول على الرصيد
        try:
            balance = balance_usdt() or 1000  # قيمة افتراضية للاختبار
        except:
            balance = 1000
        
        if balance <= 0:
            return 0
        
        # تحديد نسبة المخاطرة بناءً على الثقة وقوة المنطقة
        base_risk = RISK_ALLOC
        
        # تعديل المخاطرة حسب الثقة
        confidence_multiplier = confidence / 100.0
        risk_multiplier = min(1.5, max(0.3, confidence_multiplier))
        
        # تعديل المخاطرة حسب قوة المنطقة
        strength_map = {'VERY_STRONG': 1.5, 'STRONG': 1.2, 'MODERATE': 1.0, 'WEAK': 0.7, 'VERY_WEAK': 0.3}
        zone_multiplier = strength_map.get(zone_strength, 1.0)
        
        # المخاطرة النهائية
        final_risk = base_risk * risk_multiplier * zone_multiplier
        
        # حساب الحجم
        capital_to_risk = balance * final_risk
        position_value = capital_to_risk * LEVERAGE
        
        # في العقود الآجلة، الحجم = القيمة / السعر
        quantity = position_value / entry_price
        
        # التقريب حسب خطوة التداول
        return safe_qty(quantity)
    
    def calculate_smart_stop_loss(self, entry_price, atr, side, trade_type, zone_strength):
        """حساب وقف الخسارة الذكي"""
        
        # قاعدة أساسية: 1.5x ATR
        base_sl_atr_mult = 1.5
        
        # تعديل حسب نوع الصفقة
        if trade_type == 'GOLDEN_ROCKET':
            base_sl_atr_mult = 1.2  # وقف أقرب للصفقات الذهبية
        elif trade_type == 'SCALP':
            base_sl_atr_mult = 1.0  # وقف أقرب للسكالب
        
        # تعديل حسب قوة المنطقة
        strength_map = {'VERY_STRONG': 0.8, 'STRONG': 1.0, 'MODERATE': 1.2, 'WEAK': 1.5, 'VERY_WEAK': 2.0}
        zone_multiplier = strength_map.get(zone_strength, 1.0)
        
        # حساب وقف الخسارة النهائي
        sl_atr_distance = atr * base_sl_atr_mult * zone_multiplier
        
        if side.upper() in ['BUY', 'LONG']:
            return entry_price - sl_atr_distance
        else:
            return entry_price + sl_atr_distance
    
    def manage_trades(self, current_price, df):
        """إدارة جميع الصفقات النشطة"""
        
        for trade_id, trade in list(self.active_trades.items()):
            if trade['status'] != 'OPEN':
                continue
            
            # حساب الربح/الخسارة الحالي
            current_pnl, current_pnl_pct = self.calculate_current_pnl(trade, current_price)
            trade['current_pnl'] = current_pnl
            trade['current_pnl_pct'] = current_pnl_pct
            
            # تحديث أعلى ربح
            if current_pnl > trade['highest_pnl']:
                trade['highest_pnl'] = current_pnl
                trade['highest_pnl_pct'] = current_pnl_pct
            
            # تطبيق نظام جني الأرباح
            self.apply_profit_taking(trade_id, trade, current_price)
            
            # تطبيق وقف الخسارة المتحرك
            self.apply_trailing_stop(trade_id, trade, current_price, df)
            
            # التحقق من وقف الخسارة الأساسي
            if self.check_stop_loss(trade, current_price):
                self.close_trade(trade_id, "STOP_LOSS", current_price)
            
            # إغلاق الصفقة إذا أصبحت صغيرة جداً
            if trade['quantity'] < RESIDUAL_MIN_QTY:
                self.close_trade(trade_id, "DUST_POSITION", current_price)
    
    def apply_profit_taking(self, trade_id, trade, current_price):
        """تطبيق نظام جني الأرباح المتعدد المستويات"""
        
        side = trade['side']
        entry = trade['entry_price']
        tp_levels = trade['take_profit_levels']
        tp_fractions = trade['tp_fractions']
        
        # التحقق من كل مستوى جني أرباح
        for i, (tp_price, tp_hit) in enumerate(zip(tp_levels, trade['tp_hit'])):
            if tp_hit:
                continue  # تم جني الأرباح من هذا المستوى بالفعل
            
            # التحقق إذا وصل السعر لمستوى جني الأرباح
            hit_tp = False
            if side.upper() in ['BUY', 'LONG']:
                hit_tp = current_price >= tp_price
            else:
                hit_tp = current_price <= tp_price
            
            if hit_tp:
                # جني نسبة من الأرباح
                close_fraction = tp_fractions[i] if i < len(tp_fractions) else 0.5
                close_qty = trade['quantity'] * close_fraction
                
                if close_qty > 0:
                    # التنفيذ الفعلي
                    if EXECUTE_ORDERS and not DRY_RUN and MODE_LIVE:
                        close_side = 'sell' if side.upper() in ['BUY', 'LONG'] else 'buy'
                        try:
                            params = exchange_specific_params(close_side, is_close=True)
                            self.exchange.create_order(
                                self.symbol, "market", close_side, close_qty, None, params
                            )
                            
                            # تحديث الصفقة
                            trade['quantity'] -= close_qty
                            trade['tp_hit'][i] = True
                            
                            # تسجيل جني الأرباح
                            profit = (current_price - entry) * close_qty if side.upper() in ['BUY', 'LONG'] else (entry - current_price) * close_qty
                            trade['realized_profit'] = trade.get('realized_profit', 0) + profit
                            
                            log_g(f"✅ جني أرباح: {close_fraction*100:.0f}% من صفقة {trade_id} عند مستوى {i+1}")
                            
                            # إذا كان أول مستوى جني أرباح، ننقل وقف الخسارة لنقطة التعادل
                            if i == 0 and trade['management_config'].get('move_to_breakeven_after_tp1', False):
                                trade['stop_loss'] = entry
                                log_i(f"🛑 نقل وقف الخسارة لنقطة التعادل بعد جني الأرباح الأول")
                            
                        except Exception as e:
                            log_e(f"❌ فشل جني الأرباح: {e}")
                    else:
                        # وضع المحاكاة
                        trade['quantity'] -= close_qty
                        trade['tp_hit'][i] = True
                        log_i(f"DRY_RUN: جني {close_fraction*100:.0f}% من صفقة {trade_id}")
    
    def apply_trailing_stop(self, trade_id, trade, current_price, df):
        """تطبيق وقف الخسارة المتحرك الذكي"""
        
        config = trade['management_config']
        trail_start_pct = config['trail_start_pct']
        current_pnl_pct = trade['current_pnl_pct']
        
        # تفعيل التريل بعد تحقيق الربح المطلوب
        if not trade['trail_activated'] and current_pnl_pct >= trail_start_pct:
            trade['trail_activated'] = True
            log_i(f"🔄 تفعيل وقف الخسارة المتحرك لصفقة {trade_id}")
        
        # تحديث سعر التريل
        if trade['trail_activated']:
            # حساب ATR الحالي
            atr = compute_indicators(df).get('atr', 0.001)
            
            # حساب مسافة التريل
            trail_distance = atr * config['atr_trail_mult']
            
            # تحديث سعر التريل
            if trade['side'].upper() in ['BUY', 'LONG']:
                new_trail = current_price - trail_distance
                if trade['trail_price'] is None or new_trail > trade['trail_price']:
                    trade['trail_price'] = new_trail
            else:
                new_trail = current_price + trail_distance
                if trade['trail_price'] is None or new_trail < trade['trail_price']:
                    trade['trail_price'] = new_trail
            
            # التحقق إذا تم لمس التريل
            if trade['trail_price']:
                if (trade['side'].upper() in ['BUY', 'LONG'] and current_price <= trade['trail_price']) or \
                   (trade['side'].upper() not in ['BUY', 'LONG'] and current_price >= trade['trail_price']):
                    self.close_trade(trade_id, "TRAILING_STOP", current_price)
    
    def check_stop_loss(self, trade, current_price):
        """التحقق من وقف الخسارة الأساسي"""
        
        sl = trade['stop_loss']
        side = trade['side']
        
        if side.upper() in ['BUY', 'LONG']:
            return current_price <= sl
        else:
            return current_price >= sl
    
    def close_trade(self, trade_id, reason, current_price):
        """إغلاق صفقة كاملة"""
        
        trade = self.active_trades.get(trade_id)
        if not trade or trade['status'] != 'OPEN':
            return
        
        remaining_qty = trade['quantity']
        
        if remaining_qty > 0:
            # التنفيذ الفعلي
            if EXECUTE_ORDERS and not DRY_RUN and MODE_LIVE:
                close_side = 'sell' if trade['side'].upper() in ['BUY', 'LONG'] else 'buy'
                try:
                    params = exchange_specific_params(close_side, is_close=True)
                    self.exchange.create_order(
                        self.symbol, "market", close_side, remaining_qty, None, params
                    )
                    
                    # حساب الربح النهائي
                    final_pnl = (current_price - trade['entry_price']) * remaining_qty if trade['side'].upper() in ['BUY', 'LONG'] else (trade['entry_price'] - current_price) * remaining_qty
                    final_pnl += trade.get('realized_profit', 0)
                    
                    # تحديث الإحصائيات
                    self.update_statistics(final_pnl)
                    
                    # تسجيل الصفقة في التاريخ
                    trade['closed_at'] = datetime.now()
                    trade['close_price'] = current_price
                    trade['final_pnl'] = final_pnl
                    trade['close_reason'] = reason
                    trade['status'] = 'CLOSED'
                    
                    log_g(f"✅ إغلاق صفقة {trade_id}: {reason} | ربح: {final_pnl:.2f}")
                    
                except Exception as e:
                    log_e(f"❌ فشل إغلاق الصفقة: {e}")
            else:
                # وضع المحاكاة
                trade['closed_at'] = datetime.now()
                trade['close_price'] = current_price
                trade['status'] = 'CLOSED'
                log_i(f"DRY_RUN: إغلاق صفقة {trade_id}: {reason}")
        
        # إزالة من القائمة النشطة
        if trade_id in self.active_trades:
            self.trade_history.append(self.active_trades[trade_id])
            del self.active_trades[trade_id]
    
    def calculate_current_pnl(self, trade, current_price):
        """حساب الربح/الخسارة الحالي"""
        
        side = trade['side']
        entry = trade['entry_price']
        qty = trade['quantity']
        
        if side.upper() in ['BUY', 'LONG']:
            pnl = (current_price - entry) * qty
            pnl_pct = ((current_price - entry) / entry) * 100
        else:
            pnl = (entry - current_price) * qty
            pnl_pct = ((entry - current_price) / entry) * 100
        
        return pnl, pnl_pct
    
    def update_statistics(self, pnl):
        """تحديث إحصائيات الأداء"""
        
        self.performance_stats['total_trades'] += 1
        
        if pnl > 0:
            self.performance_stats['winning_trades'] += 1
            self.performance_stats['total_profit'] += pnl
            self.performance_stats['largest_win'] = max(self.performance_stats['largest_win'], pnl)
            
            # تحديث متوسط الربح
            if self.performance_stats['winning_trades'] > 0:
                self.performance_stats['avg_win'] = self.performance_stats['total_profit'] / self.performance_stats['winning_trades']
        else:
            self.performance_stats['losing_trades'] += 1
            self.performance_stats['largest_loss'] = min(self.performance_stats['largest_loss'], pnl)
            
            # تحديث متوسط الخسارة
            if self.performance_stats['losing_trades'] > 0:
                total_loss = abs(pnl) + abs(self.performance_stats.get('total_loss', 0))
                self.performance_stats['avg_loss'] = total_loss / self.performance_stats['losing_trades']
    
    def log_trade_opening(self, trade, simulated=False):
        """تسجيل فتح الصفقة"""
        
        mode = "SIMULATED" if simulated or DRY_RUN or not EXECUTE_ORDERS else "LIVE"
        
        log_banner(f"فتح صفقة {mode}")
        print(f"🎯 ID: {trade['id']}", flush=True)
        print(f"📈 الجانب: {trade['side']}", flush=True)
        print(f"💰 سعر الدخول: {trade['entry_price']:.6f}", flush=True)
        print(f"⚖️  الكمية: {trade['quantity']:.4f}", flush=True)
        print(f"🛑 وقف الخسارة: {trade['stop_loss']:.6f}", flush=True)
        print(f"🎯 مستويات جني الأرباح:", flush=True)
        for i, tp in enumerate(trade['take_profit_levels']):
            fraction = trade['tp_fractions'][i] if i < len(trade['tp_fractions']) else 0.5
            print(f"   المستوى {i+1}: {tp:.6f} ({fraction*100:.0f}%)", flush=True)
        print(f"🏷️  نوع الصفقة: {trade['trade_type']}", flush=True)
        print(f"💪 قوة المنطقة: {trade['zone_strength']}", flush=True)
        print(f"⭐ درجة الثقة: {trade['confidence']:.1f}%", flush=True)
        print(f"📊 الوصف: {trade['management_config']['description']}", flush=True)
        log_banner("")

    def get_performance_report(self):
        """تقرير أداء مفصل"""
        
        stats = self.performance_stats
        
        if stats['total_trades'] == 0:
            return "لا توجد صفقات بعد"
        
        win_rate = (stats['winning_trades'] / stats['total_trades']) * 100 if stats['total_trades'] > 0 else 0
        
        report = f"""
📊 تقرير أداء البوت الاحترافي:
═══════════════════════════════════════════════════════════
• إجمالي الصفقات: {stats['total_trades']}
• الصفقات الرابحة: {stats['winning_trades']} ({win_rate:.1f}%)
• الصفقات الخاسرة: {stats['losing_trades']}
• إجمالي الربح: ${stats['total_profit']:.2f}
• أكبر ربح: ${stats['largest_win']:.2f}
• أكبر خسارة: ${stats['largest_loss']:.2f}
• متوسط الربح: ${stats['avg_win']:.2f}
• متوسط الخسارة: ${stats['avg_loss']:.2f}
═══════════════════════════════════════════════════════════
"""
        
        # تحليل مفصل للصفقات الأخيرة
        if self.trade_history:
            recent_trades = self.trade_history[-5:]  # آخر 5 صفقات
            report += "\n📈 آخر 5 صفقات:\n"
            for trade in recent_trades:
                duration = (trade.get('closed_at', datetime.now()) - trade['opened_at']).total_seconds() / 60
                report += f"   • {trade['id']}: {trade['side']} | ربح: ${trade.get('final_pnl', 0):.2f} | المدة: {duration:.1f} دقيقة | السبب: {trade.get('close_reason', 'N/A')}\n"
        
        return report

# =================== ENHANCED COUNCIL WITH SMART DECISIONS ===================
class SmartTradingCouncil:
    """مجلس التداول الذكي المتقدم"""
    
    def __init__(self):
        self.member_weights = {
            'footprint': 4.0,      # أعلى وزن
            'golden_zone': 3.5,
            'trend': 3.0,
            'volume_momentum': 2.5,
            'candles': 2.0,
            'rsi': 1.5,
            'pivot_points': 1.0
        }
        
        self.decision_history = []
        self.learning_coefficients = {
            'footprint': 1.0,
            'golden_zone': 1.0,
            'trend': 1.0,
            'volume_momentum': 1.0,
            'candles': 1.0
        }
    
    def analyze_market(self, df):
        """تحليل السوق الشامل"""
        
        current_price = float(df['close'].iloc[-1])
        
        # جمع جميع التحليلات
        analyses = {
            'footprint': advanced_footprint_analysis(df, current_price),
            'golden_zone': golden_zone_check(df),
            'trend': dynamic_trend_indicator(df),
            'volume_momentum': enhanced_volume_momentum(df),
            'candles': compute_candles(df),
            'rsi': rsi_ma_context(df),
            'pivot_points': dynamic_pivot_points(df),
            'stoch_rsi': stochastic_rsi_enhanced(df)
        }
        
        # تحليل السيولة
        liquidity_analysis = analyze_liquidity_pools(df, current_price)
        
        # اتخاذ القرار النهائي
        decision = self.make_final_decision(analyses, current_price, liquidity_analysis)
        
        # تسجيل القرار للتعلم
        self.record_decision(decision, analyses)
        
        return decision
    
    def make_final_decision(self, analyses, current_price, liquidity_analysis):
        """اتخاذ القرار النهائي الذكي"""
        
        votes_buy = 0
        votes_sell = 0
        confidence_buy = 0.0
        confidence_sell = 0.0
        reasons = []
        
        # 1. تحليل Footprint (أعلى وزن)
        footprint = analyses['footprint']
        if footprint.get('ok'):
            fp_buy_score = footprint.get('entry_score_bull', 0)
            fp_sell_score = footprint.get('entry_score_bear', 0)
            
            if fp_buy_score >= FOOTPRINT_MIN_CONFIDENCE:
                weight = self.member_weights['footprint'] * self.learning_coefficients['footprint']
                votes_buy += weight
                confidence_buy += min(4.0, fp_buy_score)
                reasons.append(f"Footprint صاعد قوي (score: {fp_buy_score:.1f})")
            
            if fp_sell_score >= FOOTPRINT_MIN_CONFIDENCE:
                weight = self.member_weights['footprint'] * self.learning_coefficients['footprint']
                votes_sell += weight
                confidence_sell += min(4.0, fp_sell_score)
                reasons.append(f"Footprint هابط قوي (score: {fp_sell_score:.1f})")
        
        # 2. المناطق الذهبية
        gz = analyses['golden_zone']
        if gz and gz.get('ok'):
            weight = self.member_weights['golden_zone'] * self.learning_coefficients['golden_zone']
            
            if gz['zone']['type'] == 'golden_bottom' and gz['score'] >= 7.0:
                votes_buy += weight
                confidence_buy += min(3.0, gz['score'] / 2)
                reasons.append(f"منطقة ذهبية للشراء (score: {gz['score']:.1f})")
            
            elif gz['zone']['type'] == 'golden_top' and gz['score'] >= 7.0:
                votes_sell += weight
                confidence_sell += min(3.0, gz['score'] / 2)
                reasons.append(f"منطقة ذهبية للبيع (score: {gz['score']:.1f})")
        
        # 3. تحليل الاتجاه
        trend = analyses['trend']
        if trend['signal'] in ['strong_buy', 'buy']:
            weight = self.member_weights['trend'] * self.learning_coefficients['trend']
            votes_buy += weight
            confidence_buy += 2.0
            reasons.append(f"اتجاه صاعد ({trend['signal']})")
        
        if trend['signal'] in ['strong_sell', 'sell']:
            weight = self.member_weights['trend'] * self.learning_coefficients['trend']
            votes_sell += weight
            confidence_sell += 2.0
            reasons.append(f"اتجاه هابط ({trend['signal']})")
        
        # 4. الزخم الحجمي
        volume = analyses['volume_momentum']
        if volume['trend'] == 'bull' and volume['strength'] > 2.0:
            weight = self.member_weights['volume_momentum'] * self.learning_coefficients['volume_momentum']
            votes_buy += weight
            confidence_buy += min(2.0, volume['strength'] / 10)
            reasons.append(f"زخم حجمي صاعد (قوة: {volume['strength']:.1f})")
        
        if volume['trend'] == 'bear' and volume['strength'] > 2.0:
            weight = self.member_weights['volume_momentum'] * self.learning_coefficients['volume_momentum']
            votes_sell += weight
            confidence_sell += min(2.0, volume['strength'] / 10)
            reasons.append(f"زخم حجمي هابط (قوة: {volume['strength']:.1f})")
        
        # 5. الشموع اليابانية
        candles = analyses['candles']
        if candles['score_buy'] > 2.0:
            weight = self.member_weights['candles'] * self.learning_coefficients['candles']
            votes_buy += weight
            confidence_buy += min(1.5, candles['score_buy'])
            reasons.append(f"نمط شموع شراء (score: {candles['score_buy']:.1f})")
        
        if candles['score_sell'] > 2.0:
            weight = self.member_weights['candles'] * self.learning_coefficients['candles']
            votes_sell += weight
            confidence_sell += min(1.5, candles['score_sell'])
            reasons.append(f"نمط شموع بيع (score: {candles['score_sell']:.1f})")
        
        # 6. تحليل السيولة
        if liquidity_analysis:
            if liquidity_analysis.get('buy_liquidity_above'):
                votes_buy += 0.5
                confidence_buy += 0.5
                reasons.append("سيولة شراء قريبة")
            
            if liquidity_analysis.get('sell_liquidity_below'):
                votes_sell += 0.5
                confidence_sell += 0.5
                reasons.append("سيولة بيع قريبة")
        
        # ===== تحديد القرار النهائي =====
        
        # الحد الأدنى للثقة
        min_confidence = 8.0
        
        # تعزيز الثقة إذا كان هناك إجماع
        if len(reasons) >= 3:
            confidence_buy *= 1.2
            confidence_sell *= 1.2
        
        # قرار الشراء
        if votes_buy > votes_sell and confidence_buy >= min_confidence:
            # تحديد قوة المنطقة
            zone_strength = self.determine_zone_strength(analyses)
            
            # تحديد نوع الصفقة
            trade_type, trade_reason = TradeTypeDetector.detect_trade_type(
                pd.DataFrame(),  # نحتاج dataframe هنا
                {'ind': {}, 'candles': candles, 'advanced_indicators': {'footprint': footprint}},
                gz,
                current_price
            )
            
            decision = {
                'action': 'BUY',
                'confidence': confidence_buy,
                'votes_buy': votes_buy,
                'votes_sell': votes_sell,
                'reasons': reasons,
                'zone_strength': zone_strength,
                'trade_type': trade_type,
                'trade_reason': trade_reason,
                'timestamp': datetime.now()
            }
        
        # قرار البيع
        elif votes_sell > votes_buy and confidence_sell >= min_confidence:
            # تحديد قوة المنطقة
            zone_strength = self.determine_zone_strength(analyses)
            
            # تحديد نوع الصفقة
            trade_type, trade_reason = TradeTypeDetector.detect_trade_type(
                pd.DataFrame(),
                {'ind': {}, 'candles': candles, 'advanced_indicators': {'footprint': footprint}},
                gz,
                current_price
            )
            
            decision = {
                'action': 'SELL',
                'confidence': confidence_sell,
                'votes_buy': votes_buy,
                'votes_sell': votes_sell,
                'reasons': reasons,
                'zone_strength': zone_strength,
                'trade_type': trade_type,
                'trade_reason': trade_reason,
                'timestamp': datetime.now()
            }
        
        # لا قرار
        else:
            decision = {
                'action': 'HOLD',
                'confidence': max(confidence_buy, confidence_sell),
                'votes_buy': votes_buy,
                'votes_sell': votes_sell,
                'reasons': ["لا توجد إشارة قوية كافية"],
                'timestamp': datetime.now()
            }
        
        return decision
    
    def determine_zone_strength(self, analyses):
        """تحديد قوة المنطقة بناءً على جميع التحليلات"""
        
        strength_score = 0
        
        # Footprint
        footprint = analyses.get('footprint', {})
        if footprint.get('ok'):
            fp_score = max(footprint.get('entry_score_bull', 0), footprint.get('entry_score_bear', 0))
            if fp_score >= 3.0:
                strength_score += 2
            elif fp_score >= 2.0:
                strength_score += 1
        
        # Golden Zone
        gz = analyses.get('golden_zone', {})
        if gz and gz.get('ok'):
            if gz.get('score', 0) >= 8.0:
                strength_score += 2
            elif gz.get('score', 0) >= 6.0:
                strength_score += 1
        
        # Trend
        trend = analyses.get('trend', {})
        if trend.get('trend') in ['strong_bull', 'strong_bear']:
            strength_score += 2
        elif trend.get('trend') in ['bull', 'bear']:
            strength_score += 1
        
        # Volume Momentum
        volume = analyses.get('volume_momentum', {})
        if volume.get('strength', 0) > 3.0:
            strength_score += 1
        
        # تحويل النقاط إلى تصنيف
        if strength_score >= 5:
            return 'VERY_STRONG'
        elif strength_score >= 4:
            return 'STRONG'
        elif strength_score >= 3:
            return 'MODERATE'
        elif strength_score >= 2:
            return 'WEAK'
        else:
            return 'VERY_WEAK'
    
    def record_decision(self, decision, analyses):
        """تسجيل القرار للتعلم المستقبلي"""
        
        record = {
            'decision': decision,
            'analyses_summary': {
                'footprint_ok': analyses['footprint'].get('ok', False),
                'golden_zone_ok': analyses['golden_zone'].get('ok', False) if analyses['golden_zone'] else False,
                'trend_signal': analyses['trend'].get('signal', 'none'),
                'volume_strength': analyses['volume_momentum'].get('strength', 0),
                'candle_score': max(analyses['candles'].get('score_buy', 0), analyses['candles'].get('score_sell', 0))
            },
            'timestamp': datetime.now()
        }
        
        self.decision_history.append(record)
        
        # الاحتفاظ بأخر 100 قرار فقط
        if len(self.decision_history) > 100:
            self.decision_history.pop(0)
    
    def learn_from_results(self, trade_result):
        """التعلم من نتائج الصفقات السابقة"""
        
        # تحليل آخر 20 قرار أدت لصفقات
        recent_decisions = [d for d in self.decision_history[-20:] if d['decision']['action'] in ['BUY', 'SELL']]
        
        if len(recent_decisions) < 5:
            return  # لا توجد بيانات كافية للتعلم
        
        # تحليل فعالية كل مؤشر
        indicator_performance = {
            'footprint': {'success': 0, 'total': 0},
            'golden_zone': {'success': 0, 'total': 0},
            'trend': {'success': 0, 'total': 0},
            'volume_momentum': {'success': 0, 'total': 0},
            'candles': {'success': 0, 'total': 0}
        }
        
        for decision_record in recent_decisions:
            decision = decision_record['decision']
            analyses = decision_record['analyses_summary']
            
            # هنا يمكن إضافة تحليل الربح/الخسارة الفعلية
            # للمثال، سنفترض أن القرار كان ناجحاً إذا كانت الثقة عالية
            
            success = decision.get('confidence', 0) >= 10.0
            
            # تحديث أداء كل مؤشر
            if analyses['footprint_ok']:
                indicator_performance['footprint']['total'] += 1
                if success:
                    indicator_performance['footprint']['success'] += 1
            
            if analyses['golden_zone_ok']:
                indicator_performance['golden_zone']['total'] += 1
                if success:
                    indicator_performance['golden_zone']['success'] += 1
            
            if analyses['trend_signal'] != 'none':
                indicator_performance['trend']['total'] += 1
                if success:
                    indicator_performance['trend']['success'] += 1
            
            if analyses['volume_strength'] > 2.0:
                indicator_performance['volume_momentum']['total'] += 1
                if success:
                    indicator_performance['volume_momentum']['success'] += 1
            
            if analyses['candle_score'] > 2.0:
                indicator_performance['candles']['total'] += 1
                if success:
                    indicator_performance['candles']['success'] += 1
        
        # تحديث معاملات التعلم
        for indicator, perf in indicator_performance.items():
            if perf['total'] > 0:
                success_rate = perf['success'] / perf['total']
                # زيادة وزن المؤشرات الناجحة
                self.learning_coefficients[indicator] = min(1.5, max(0.5, success_rate))
        
        log_i(f"📊 تحديث معاملات التعلم: {self.learning_coefficients}")

# =================== SMART GOLDEN ZONE DETECTION ===================
def _ema_gz(series, n):
    """المتوسط المتحرك الأسي للمنطقة الذهبية"""
    return series.ewm(span=n, adjust=False).mean()

def _rsi_fallback_gz(close, n=14):
    """RSI بديل محسّن"""
    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    roll_up = up.ewm(span=n, adjust=False).mean()
    roll_down = down.ewm(span=n, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, 1e-12)
    rsi = 100 - (100/(1+rs))
    return rsi.fillna(50)

def _body_wicks_gz(h, l, o, c):
    """حساب الجسم والفتائل بدقة"""
    rng = max(1e-9, h - l)
    body = abs(c - o) / rng
    up_wick = (h - max(c, o)) / rng
    low_wick = (min(c, o) - l) / rng
    return body, up_wick, low_wick

def _displacement_gz(closes):
    """قياس اندفاع السعر"""
    if len(closes) < 22:
        return 0.0
    recent_std = closes.tail(20).std()
    return abs(closes.iloc[-1] - closes.iloc[-2]) / max(recent_std, 1e-9)

def _last_impulse_gz(df):
    """اكتشاف آخر موجة دافعة بدقة"""
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    
    # البحث عن القمة والقاع في آخر 120 شمعة
    lookback = min(120, len(df))
    recent_highs = h.tail(lookback)
    recent_lows = l.tail(lookback)
    
    hh_idx = recent_highs.idxmax()
    ll_idx = recent_lows.idxmin()
    
    hh = recent_highs.max()
    ll = recent_lows.min()
    
    # تحديد اتجاه الدافع
    if hh_idx < ll_idx:  # قمة ثم قاع => دافع هابط
        return ("down", hh_idx, ll_idx, hh, ll)
    else:  # قاع ثم قمة => دافع صاعد
        return ("up", ll_idx, hh_idx, ll, hh)

def golden_zone_check(df, ind=None, side_hint=None):
    """اكتشاف المناطق الذهبية بدقة محسنة"""
    if len(df) < 60:
        return {"ok": False, "score": 0.0, "zone": None, "reasons": ["short_df"]}
    
    try:
        # استخراج البيانات
        h = df['high'].astype(float)
        l = df['low'].astype(float)
        c = df['close'].astype(float)
        o = df['open'].astype(float)
        v = df['volume'].astype(float)
        
        # اكتشاف الدافع الأخير
        impulse_data = _last_impulse_gz(df)
        if not impulse_data:
            return {"ok": False, "score": 0.0, "zone": None, "reasons": ["no_clear_impulse"]}
            
        side, idx1, idx2, p1, p2 = impulse_data
        
        # حساب فيبوناتشي بناءً على اتجاه الدافع
        if side == "down":
            # دافع هابط: التصحيح الصاعد بين 0.618-0.786 من الهبوط
            swing_hi, swing_lo = p1, p2
            f618 = swing_lo + FIB_LOW * (swing_hi - swing_lo)
            f786 = swing_lo + FIB_HIGH * (swing_hi - swing_lo)
            zone_type = "golden_bottom"
        else:
            # دافع صاعد: التصحيح الهابط بين 0.618-0.786 من الصعود
            swing_lo, swing_hi = p1, p2
            f618 = swing_hi - FIB_HIGH * (swing_hi - swing_lo)
            f786 = swing_hi - FIB_LOW * (swing_hi - swing_lo)
            zone_type = "golden_top"
        
        last_close = float(c.iloc[-1])
        in_zone = (f618 <= last_close <= f786) if side == "down" else (f786 <= last_close <= f618)
        
        if not in_zone:
            return {"ok": False, "score": 0.0, "zone": None, "reasons": [f"price_not_in_zone {last_close:.6f} vs [{f618:.6f},{f786:.6f}]"]}
        
        # الشروط المساعدة
        current_high = float(h.iloc[-1])
        current_low = float(l.iloc[-1])
        current_open = float(o.iloc[-1])
        
        body, up_wick, low_wick = _body_wicks_gz(current_high, current_low, current_open, last_close)
        
        # حجم التداول
        vol_ma = v.rolling(VOL_MA_LEN).mean().iloc[-1]
        vol_ok = float(v.iloc[-1]) >= vol_ma * 0.9
        
        # RSI
        rsi_series = _rsi_fallback_gz(c, RSI_LEN_GZ)
        rsi_ma_series = _ema_gz(rsi_series, RSI_MA_LEN_GZ)
        rsi_last = float(rsi_series.iloc[-1])
        rsi_ma_last = float(rsi_ma_series.iloc[-1])
        
        # ADX من المؤشرات المحسنة
        adx = ind.get('adx', 0) if ind else 0
        
        # اندفاع السعر
        disp = _displacement_gz(c)
        
        # فتيلة مناسبة حسب الاتجاه
        if side == "down":  # نبحث عن فتيلة سفلية للشراء
            wick_ok = low_wick >= MIN_WICK_PCT
            rsi_ok = rsi_last > rsi_ma_last and rsi_last < 70
            candle_bullish = last_close > current_open
        else:  # نبحث عن فتيلة علوية للبيع
            wick_ok = up_wick >= MIN_WICK_PCT
            rsi_ok = rsi_last < rsi_ma_last and rsi_last > 30
            candle_bullish = last_close < current_open
        
        # حساب النقاط
        score = 0.0
        reasons = []
        
        # الشروط الأساسية
        if adx >= GZ_REQ_ADX:
            score += 2.0
            reasons.append(f"ADX_{adx:.1f}")
        
        if disp >= MIN_DISP:
            score += 1.5
            reasons.append(f"DISP_{disp:.2f}")
        
        if wick_ok:
            score += 1.5
            reasons.append("wick_ok")
        
        if vol_ok:
            score += 1.0
            reasons.append("vol_ok")
        
        if rsi_ok:
            score += 1.5
            reasons.append("rsi_ok")
        
        if candle_bullish:
            score += 0.5
            reasons.append("candle_confirm")
        
        # شرط المنطقة
        score += 2.0
        reasons.append("in_zone")
        
        # النتيجة النهائية
        ok = (score >= GZ_MIN_SCORE and in_zone and adx >= GZ_REQ_ADX)
        
        # تشخيص تفصيلي
        if LOG_ADDONS and in_zone:
            print(f"[GZ DEBUG] type={zone_type} zone={f618:.6f}-{f786:.6f} price={last_close:.6f} score={score:.1f} adx={adx:.1f} disp={disp:.2f} wick_ok={wick_ok} vol_ok={vol_ok} rsi_ok={rsi_ok}")
        
        return {
            "ok": ok,
            "score": round(score, 2),
            "zone": {
                "type": zone_type,
                "f618": f618,
                "f786": f786,
                "swing_high": swing_hi if side == "down" else swing_lo,
                "swing_low": swing_lo if side == "down" else swing_hi
            } if ok else None,
            "reasons": reasons
        }
        
    except Exception as e:
        log_w(f"golden_zone_check error: {e}")
        return {"ok": False, "score": 0.0, "zone": None, "reasons": [f"error: {str(e)}"]}

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

def price_now():
    try:
        t = with_retry(lambda: ex.fetch_ticker(SYMBOL))
        return t.get("last") or t.get("close")
    except Exception: return None

def balance_usdt():
    if not MODE_LIVE: return 1000.0
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

def compute_indicators(df: pd.DataFrame):
    if len(df) < max(ATR_LEN, RSI_LEN, ADX_LEN) + 2:
        return {"rsi":50.0,"plus_di":0.0,"minus_di":0.0,"dx":0.0,"adx":0.0,"atr":0.0}
    
    def wilder_ema(s: pd.Series, n: int): 
        return s.ewm(alpha=1/n, adjust=False).mean()
    
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
        "adx": float(adx.iloc[i]), "atr": float(atr.iloc[i]),
        "di_spread": abs(float(plus_di.iloc[i]) - float(minus_di.iloc[i]))
    }

# =================== ENHANCED MAIN LOOP ===================
def professional_trading_loop():
    """حلقة التداول الاحترافية الرئيسية"""
    
    # تهيئة الأنظمة
    council = SmartTradingCouncil()
    trade_manager = ProfessionalTradeManager(ex, SYMBOL)
    
    log_banner("🚀 بدء تشغيل مكنة الأرباح الاحترافية")
    print("📊 الأنظمة الجاهزة:", flush=True)
    print("   • مجلس التداول الذكي ✓", flush=True)
    print("   • نظام جني الأرباح المتعدد المستويات ✓", flush=True)
    print("   • مدير الصفقات الاحترافي ✓", flush=True)
    print("   • نظام التعلم الآلي ✓", flush=True)
    log_banner("")
    
    cycle_count = 0
    
    while True:
        try:
            cycle_count += 1
            
            print(f"\n{'='*60}", flush=True)
            print(f"🔄 الدورة #{cycle_count} - {datetime.now().strftime('%H:%M:%S')}", flush=True)
            print(f"{'='*60}", flush=True)
            
            # 1. جمع بيانات السوق
            df = fetch_ohlcv()
            current_price = price_now()
            
            if df is None or len(df) < 50 or current_price is None:
                print("⏳ انتظار بيانات السوق...", flush=True)
                time.sleep(BASE_SLEEP)
                continue
            
            # 2. إدارة الصفقات النشطة
            trade_manager.manage_trades(current_price, df)
            
            # 3. التحقق من الحد الأقصى للصفقات النشطة
            active_trade_count = len(trade_manager.active_trades)
            if active_trade_count >= 3:  # حد أقصى 3 صفقات في نفس الوقت
                print(f"⏸️  عدد الصفقات النشطة ({active_trade_count}) وصل الحد الأقصى", flush=True)
                time.sleep(BASE_SLEEP)
                continue
            
            # 4. تحليل السوق واتخاذ القرار
            print("🧠 مجلس التداول يجتمع...", flush=True)
            decision = council.analyze_market(df)
            
            # 5. عرض القرار
            print(f"📊 قرار المجلس: {decision['action']}", flush=True)
            print(f"⭐ درجة الثقة: {decision['confidence']:.1f}", flush=True)
            print(f"🗳️  الأصوات: شراء {decision['votes_buy']:.1f} | بيع {decision['votes_sell']:.1f}", flush=True)
            print("📝 الأسباب:", flush=True)
            for reason in decision.get('reasons', []):
                print(f"   • {reason}", flush=True)
            
            if decision.get('trade_type'):
                print(f"🏷️  نوع الصفقة: {decision['trade_type']} - {decision.get('trade_reason', '')}", flush=True)
            
            print(f"💪 قوة المنطقة: {decision.get('zone_strength', 'UNKNOWN')}", flush=True)
            
            # 6. تنفيذ القرار إذا كان قوياً
            if decision['action'] in ['BUY', 'SELL'] and decision['confidence'] >= 8.0:
                print(f"\n🎯 إشارة تداول قوية!", flush=True)
                
                # حساب المؤشرات الإضافية
                indicators = compute_indicators(df)
                atr = indicators.get('atr', 0.001)
                
                # إعداد بيانات الإشارة
                signal_data = {
                    'side': decision['action'].lower(),
                    'entry_price': current_price,
                    'atr': atr,
                    'trade_type': decision.get('trade_type', 'SCALP'),
                    'zone_strength': decision.get('zone_strength', 'MODERATE'),
                    'confidence': decision['confidence']
                }
                
                # فتح الصفقة
                trade_id = trade_manager.open_trade(signal_data)
                
                if trade_id:
                    print(f"✅ تم فتح الصفقة بنجاح: {trade_id}", flush=True)
                    
                    # تفعيل انتظار مؤقت
                    print("⏳ انتظار 30 ثانية قبل تحليل السوق مجدداً...", flush=True)
                    time.sleep(30)
            
            # 7. عرض إحصائيات الأداء كل 10 دورات
            if cycle_count % 10 == 0:
                performance_report = trade_manager.get_performance_report()
                print(performance_report, flush=True)
                
                # تحديث نظام التعلم
                council.learn_from_results(None)
            
            # 8. الانتظار قبل الدورة التالية
            sleep_time = NEAR_CLOSE_S if time_to_candle_close(df) <= 15 else BASE_SLEEP
            time.sleep(sleep_time)
            
        except Exception as e:
            print(f"❌ خطأ في الحلقة الرئيسية: {e}", flush=True)
            traceback.print_exc()
            time.sleep(BASE_SLEEP * 2)

# =================== API / KEEPALIVE ===================
app = Flask(__name__)

# تهيئة مدير الصفقات العالمي
global_trade_manager = None

@app.route("/")
def home():
    mode='LIVE' if MODE_LIVE else 'PAPER'
    return f"""
    <html>
        <head><title>SUI Professional Trading Bot</title></head>
        <body style="font-family: Arial, sans-serif; padding: 20px;">
            <h1>🚀 SUI Professional Trading Bot — Money Making Machine</h1>
            <p><strong>Exchange:</strong> {EXCHANGE_NAME.upper()}</p>
            <p><strong>Symbol:</strong> {SYMBOL} | <strong>Interval:</strong> {INTERVAL}</p>
            <p><strong>Mode:</strong> {mode} | <strong>Leverage:</strong> {LEVERAGE}x</p>
            <p><strong>Risk Allocation:</strong> {RISK_ALLOC*100}%</p>
            <p><strong>Version:</strong> {BOT_VERSION}</p>
            <hr>
            <p>Endpoints:</p>
            <ul>
                <li><a href="/metrics">/metrics</a> - Detailed metrics</li>
                <li><a href="/health">/health</a> - Health check</li>
                <li><a href="/performance">/performance</a> - Performance report</li>
                <li><a href="/trades">/trades</a> - Active trades</li>
            </ul>
        </body>
    </html>
    """

@app.route("/metrics")
def metrics():
    return jsonify({
        "exchange": EXCHANGE_NAME,
        "symbol": SYMBOL, 
        "interval": INTERVAL, 
        "mode": "live" if MODE_LIVE else "paper",
        "leverage": LEVERAGE, 
        "risk_alloc": RISK_ALLOC, 
        "price": price_now(),
        "balance": balance_usdt(),
        "bot_version": BOT_VERSION,
        "system_status": "RUNNING",
        "timestamp": datetime.utcnow().isoformat()
    })

@app.route("/health")
def health():
    try:
        # اختبار الاتصال بالبورصة
        price = price_now()
        df = fetch_ohlcv(limit=10)
        
        return jsonify({
            "ok": True, 
            "exchange_connected": price is not None,
            "data_available": df is not None and len(df) > 0,
            "price": price,
            "timestamp": datetime.utcnow().isoformat()
        }), 200
    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }), 500

@app.route("/performance")
def performance():
    if global_trade_manager:
        return jsonify({
            "performance_stats": global_trade_manager.performance_stats,
            "active_trades": len(global_trade_manager.active_trades),
            "total_trades_history": len(global_trade_manager.trade_history),
            "timestamp": datetime.utcnow().isoformat()
        })
    else:
        return jsonify({
            "message": "Trade manager not initialized",
            "timestamp": datetime.utcnow().isoformat()
        })

@app.route("/trades")
def trades():
    if global_trade_manager:
        active_trades = []
        for trade_id, trade in global_trade_manager.active_trades.items():
            active_trades.append({
                "id": trade_id,
                "side": trade['side'],
                "entry_price": trade['entry_price'],
                "quantity": trade['quantity'],
                "current_pnl": trade.get('current_pnl', 0),
                "current_pnl_pct": trade.get('current_pnl_pct', 0),
                "status": trade['status'],
                "opened_at": trade['opened_at'].isoformat() if isinstance(trade['opened_at'], datetime) else str(trade['opened_at'])
            })
        
        return jsonify({
            "active_trades": active_trades,
            "count": len(active_trades),
            "timestamp": datetime.utcnow().isoformat()
        })
    else:
        return jsonify({
            "active_trades": [],
            "count": 0,
            "timestamp": datetime.utcnow().isoformat()
        })

def keepalive_loop():
    url=(SELF_URL or "").strip().rstrip("/")
    if not url:
        log_w("keepalive disabled (SELF_URL not set)")
        return
    import requests
    sess=requests.Session(); sess.headers.update({"User-Agent":"sui-professional-bot/keepalive"})
    log_i(f"KEEPALIVE every 50s → {url}")
    while True:
        try: sess.get(url, timeout=8)
        except Exception: pass
        time.sleep(50)

# =================== MAIN EXECUTION ===================
if __name__ == "__main__":
    log_banner("🚀 SUI PROFESSIONAL TRADING BOT - MONEY MAKING MACHINE")
    
    # عرض إعدادات النظام
    print("⚙️ PROFESSIONAL SYSTEM CONFIGURATION", flush=True)
    print(f"🔧 EXCHANGE: {EXCHANGE_NAME.upper()} | SYMBOL: {SYMBOL} | TIMEFRAME: {INTERVAL}", flush=True)
    print(f"💰 LEVERAGE: {LEVERAGE}x | RISK: {RISK_ALLOC*100}%", flush=True)
    print(f"🎯 PROFIT SYSTEM: Multi-level TP for Golden Trades (3 levels)", flush=True)
    print(f"⚡ SCALP SYSTEM: Single TP for Scalp Trades (1 level)", flush=True)
    print(f"👣 ADVANCED FOOTPRINT: Active with smart analysis", flush=True)
    print(f"🧠 SMART COUNCIL: AI-powered decision making", flush=True)
    print(f"📊 PROFESSIONAL MANAGEMENT: Dynamic TP + Trail + Learning System", flush=True)
    print(f"🚀 EXECUTION: {'ACTIVE TRADING' if EXECUTE_ORDERS and not DRY_RUN else 'SIMULATION MODE'}", flush=True)
    
    if not EXECUTE_ORDERS:
        print("🟡 WARNING: EXECUTE_ORDERS=False - البوت في وضع التحليل فقط!", flush=True)
    if DRY_RUN:
        print("🟡 WARNING: DRY_RUN=True - البوت في وضع المحاكاة!", flush=True)
    
    log_banner("")
    
    # تهيئة مدير الصفقات العالمي
    global_trade_manager = ProfessionalTradeManager(ex, SYMBOL)
    
    # إعداد تسجيل الدخول
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('trading_bot.log'),
            logging.StreamHandler()
        ]
    )
    
    logging.info("SUI Professional Trading Bot starting...")
    
    # معالجة الإشارات
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGINT,  lambda *_: sys.exit(0))
    
    # بدء الخيوط
    import threading
    
    # خيط حلقة التداول الرئيسية
    trading_thread = threading.Thread(target=professional_trading_loop, daemon=True)
    trading_thread.start()
    
    # خيط Keepalive
    keepalive_thread = threading.Thread(target=keepalive_loop, daemon=True)
    keepalive_thread.start()
    
    # بدء خادم Flask
    print(f"\n🌐 Flask server starting on port {PORT}...", flush=True)
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
