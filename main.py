# -*- coding: utf-8 -*-
"""
SUI ULTRA PRO AI BOT - الإصدار الذكي المتقدم المتكامل
• مجلس الإدارة الفائق الذكي مع 15 استراتيجية متقدمة  
• نظام ركوب الترند الذكي المحترف لتحقيق أقصى ربح متتالي
• السكالب الفائق الذكي بأهداف متعددة محسوبة
• إدارة صفقات ذكية متكيفة مع قوة الترند
• نظام Footprint + Diagonal Order-Flow المتقدم
• Multi-Exchange Support: BingX & Bybit
• HQ Trading Intelligence Patch - مناطق ذهبية + SMC + OB/FVG
• SMART PROFIT AI - نظام جني الأرباح الذكي المتقدم
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

# ============================================
#  SMART PATCH — HQ Trading Intelligence Engine
# ============================================

# ---------- Z-SCORE بدون SciPy ----------
def simple_zscore(values, window=50):
    try:
        if len(values) < 5:
            return 0.0
        recent = values[-window:]
        mean = sum(recent) / len(recent)
        variance = sum((x - mean) ** 2 for x in recent) / len(recent)
        std = variance ** 0.5
        if std == 0:
            return 0.0
        return (recent[-1] - mean) / std
    except:
        return 0.0

# ---------- Smart Trend Context ----------
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
        
        # حساب الزخم
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

# ---------- SMC Liquidity Detection ----------
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
        
        # اكتشاف القمم والقيعان
        for i in range(lookback, len(highs) - lookback):
            if highs.iloc[i] == highs.iloc[i-lookback:i+lookback].max():
                self.swing_highs.append((i, highs.iloc[i]))
            if lows.iloc[i] == lows.iloc[i-lookback:i+lookback].min():
                self.swing_lows.append((i, lows.iloc[i]))
    
    def detect_liquidity_zones(self, current_price):
        zones = []
        # مناطق السيولة فوق السعر (لصفقات البيع)
        for _, high in self.swing_highs:
            if high > current_price * 1.01:  # فوق السعر ب 1%
                zones.append(("sell_liquidity", high))
        
        # مناطق السيولة تحت السعر (لصفقات الشراء)
        for _, low in self.swing_lows:
            if low < current_price * 0.99:  # تحت السعر ب 1%
                zones.append(("buy_liquidity", low))
                
        return zones

# ---------- Volume Confirmation ----------
def volume_is_strong(vol_list, window=20, threshold=1.4):
    if len(vol_list) < window:
        return False
    recent = vol_list[-window:]
    avg = sum(recent) / len(recent)
    return recent[-1] > avg * threshold

# ---------- OB (Order Block) Detection ----------
def detect_ob(candles):
    if len(candles) < 5:
        return None
    
    # تحويل DataFrame إلى قائمة
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
    
    # Bullish OB
    if b['close'] < b['open'] and c['close'] > c['open']:
        return ("bullish", b['open'], b['close'])
    
    # Bearish OB
    if b['close'] > b['open'] and c['close'] < c['open']:
        return ("bearish", b['open'], b['close'])
    
    return None

# ---------- FVG (Fair Value Gap) Detection ----------
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

    # Bullish FVG
    if a['high'] < c['low']:
        return ("bullish", a['high'], c['low'])

    # Bearish FVG
    if a['low'] > c['high']:
        return ("bearish", c['high'], a['low'])

    return None

# ---------- Zero Reversal Scalping ----------
class ZeroReversalScalper:
    def __init__(self):
        self.last_trade_time = 0
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.cooldown_until = 0
        
    def can_trade(self, current_time, min_interval=30):
        if current_time < self.cooldown_until:
            return False, f"Cooldown until {self.cooldown_until}"
        return current_time - self.last_trade_time >= min_interval, "Ready"
    
    def record_trade(self, current_time, is_win):
        self.last_trade_time = current_time
        if is_win:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            
        # Post-Big-Win Filter
        if self.consecutive_wins >= 3:
            self.cooldown_until = current_time + 300  # 5 minutes cooldown
            self.consecutive_wins = 0

# ---------- Signal Logger ----------
class SignalLogger:
    def __init__(self):
        self.missed_signals = deque(maxlen=50)
        self.entry_reasons = []
        
    def log_missed_signal(self, signal_type, price, reason):
        self.missed_signals.append({
            'timestamp': time.time(),
            'type': signal_type,
            'price': price,
            'reason': reason
        })
        
    def get_recent_missed(self, count=10):
        return list(self.missed_signals)[-count:]

# =============================
#  SMART PROFIT AI - نظام جني الأرباح الذكي
# =============================

def safe_float_series(df, col):
    """تحويل أي عمود Float بدون ما يكسر Pandas"""
    try:
        return pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    except:
        return df[col].astype(float)

def compute_momentum_indicators_safe(df):
    """نسخة آمنة تماماً من حساب الزخم"""
    try:
        if len(df) < 15:
            return {"rsi": 50.0, "high": 0.0, "low": 0.0, "close": 0.0, "volume": 0.0}
        
        high  = safe_float_series(df, "high")
        low   = safe_float_series(df, "low") 
        close = safe_float_series(df, "close")
        vol   = safe_float_series(df, "volume")

        # RSI آمن
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.rolling(14, min_periods=1).mean()
        avg_loss = loss.rolling(14, min_periods=1).mean().replace(0, 0.001)

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return {
            "rsi": float(rsi.iloc[-1]),
            "high": float(high.iloc[-1]),
            "low": float(low.iloc[-1]), 
            "close": float(close.iloc[-1]),
            "volume": float(vol.iloc[-1])
        }
    except Exception as e:
        log_w(f"Momentum indicators error: {e}")
        return {"rsi": 50.0, "high": 0.0, "low": 0.0, "close": 0.0, "volume": 0.0}

def smart_profit_ai(position_side, entry_price, current_price, trend_strength, vol_boost, mode="scalp"):
    """
    🧠 نظام جني الأرباح الذكي المتقدم
    - يحدد تلقائياً إذا كانت الصفقة سكالب أم ترند
    - يطبق استراتيجية خروج مخصصة لكل نوع
    - يركب الترند القوي لتحقيق أقصى ربح
    """
    
    if not all([entry_price, current_price]) or entry_price == 0:
        return "HOLD"
    
    profit_pct = ((current_price - entry_price) / entry_price) * 100
    if position_side.upper() in ["SELL", "SHORT"]:
        profit_pct = -profit_pct

    # تحديد نمط التداول تلقائياً
    if mode == "scalp" or trend_strength < 2:
        # 🔥 إستراتيجية السكالب السريع
        if profit_pct >= 0.45:
            return "TAKE_PROFIT_SCALP"
        elif profit_pct >= 0.25 and vol_boost:
            return "PARTIAL_PROFIT_25"
        elif profit_pct <= -0.35:
            return "STOP_LOSS_SCALP"
            
    elif 2 <= trend_strength < 4:
        # 📈 إستراتيجية الترند المتوسط
        if profit_pct >= 1.2:
            return "TAKE_PROFIT_PARTIAL_50"
        elif profit_pct >= 2.0:
            return "MOVE_STOP_BREAK_EVEN"
        elif profit_pct >= 3.0:
            return "TAKE_PROFIT_PARTIAL_30"
        elif profit_pct <= -1.5:
            return "STOP_LOSS_TREND"
            
    else:  # trend_strength >= 4
        # 🚀 إستراتيجية الترند القوي - ركوب الموجة
        if profit_pct >= 1.0 and not vol_boost:
            return "HOLD_WAIT_VOLUME"
        elif profit_pct >= 2.5:
            return "PARTIAL_PROFIT_20"
        elif profit_pct >= 4.0 and vol_boost:
            return "HOLD_TP_STRONG"
        elif profit_pct >= 6.0:
            return "FINAL_TP_STRONG"
        elif profit_pct >= 8.0:
            return "FULL_EXIT_MAX_PROFIT"
        elif profit_pct <= -2.0:
            return "STOP_LOSS_STRONG_TREND"

    return "HOLD"

def apply_smart_profit_strategy():
    """تطبيق إستراتيجية جني الأرباح على الصفقة الحالية"""
    if not STATE.get("open") or STATE["qty"] <= 0:
        return
        
    try:
        current_price = price_now()
        if not current_price:
            return
            
        # جمع بيانات السوق
        df = fetch_ohlcv(limit=50)
        momentum = compute_momentum_indicators_safe(df)
        volume_profile = compute_volume_profile(df)
        
        # حساب قوة الترند
        trend_strength = 0
        if safe_get(momentum, 'rsi', 50) > 60:
            trend_strength += 2
        if volume_profile.get('volume_spike'):
            trend_strength += 2
        if safe_get(STATE, 'pnl', 0) > 1.0:
            trend_strength += 1
            
        vol_boost = volume_profile.get('volume_spike', False)
        
        # استشارة الذكاء الاصطناعي لجني الأرباح
        decision = smart_profit_ai(
            STATE["side"],
            STATE["entry"], 
            current_price,
            trend_strength,
            vol_boost,
            STATE.get("mode", "scalp")
        )
        
        # تنفيذ القرار
        if decision != "HOLD":
            log_i(f"🧠 SMART PROFIT AI: {decision}")
            
            if "TAKE_PROFIT" in decision or "PARTIAL" in decision:
                # إغلاق جزئي
                close_percent = 0.3
                if "50" in decision:
                    close_percent = 0.5
                elif "25" in decision:
                    close_percent = 0.25
                elif "20" in decision:
                    close_percent = 0.2
                    
                close_qty = safe_qty(STATE["qty"] * close_percent)
                if close_qty > 0:
                    close_side = "sell" if STATE["side"] == "long" else "buy"
                    if MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN:
                        try:
                            params = exchange_specific_params(close_side, is_close=True)
                            ex.create_order(SYMBOL, "market", close_side, close_qty, None, params)
                            log_g(f"💰 SMART PARTIAL CLOSE: {close_percent*100}% | Decision: {decision}")
                            STATE["qty"] = safe_qty(STATE["qty"] - close_qty)
                        except Exception as e:
                            log_e(f"❌ Smart partial close failed: {e}")
                            
            elif "STOP_LOSS" in decision:
                close_market_strict(f"Smart Stop Loss: {decision}")
                
            elif "MOVE_STOP_BREAK_EVEN" in decision:
                STATE["breakeven"] = STATE["entry"]
                STATE["breakeven_armed"] = True
                log_i("🛡️ MOVED TO BREAKEVEN - Smart Profit AI")
                
            elif "FULL_EXIT" in decision:
                close_market_strict(f"Smart Full Exit: {decision}")
                
    except Exception as e:
        log_w(f"Smart profit strategy error: {e}")

# ---------- Initialize Global Objects ----------
trend_ctx = SmartTrendContext()
smc_detector = SMCDetector()
zero_scalper = ZeroReversalScalper()
signal_logger = SignalLogger()

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
LOG_LEGACY = False
LOG_ADDONS = True

# ==== Execution Switches ====
EXECUTE_ORDERS = True
SHADOW_MODE_DASHBOARD = False
DRY_RUN = False

# ==== Addon: Logging + Recovery Settings ====
BOT_VERSION = f"SUI ULTRA PRO AI v7.0 — {EXCHANGE_NAME.upper()} - SMART PROFIT AI"
print("🚀 Booting:", BOT_VERSION, flush=True)

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

# Indicators
RSI_LEN = 14
ADX_LEN = 14
ATR_LEN = 14

ENTRY_RF_ONLY = False
MAX_SPREAD_BPS = float(os.getenv("MAX_SPREAD_BPS", 6.0))

# Dynamic TP / trail - Optimized for SUI
TP1_PCT_BASE       = 0.45
TP1_CLOSE_FRAC     = 0.50
BREAKEVEN_AFTER    = 0.30
TRAIL_ACTIVATE_PCT = 1.20
ATR_TRAIL_MULT     = 1.8

TREND_TPS       = [0.50, 1.00, 1.80, 2.50, 3.50, 5.00, 7.00]
TREND_TP_FRACS  = [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.10]

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

# ==== Golden Zone Constants ====
FIB_LOW, FIB_HIGH = 0.618, 0.786
MIN_WICK_PCT = 0.35
VOL_MA_LEN = 20
RSI_LEN_GZ, RSI_MA_LEN_GZ = 14, 9
MIN_DISP = 0.8

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

MAX_TRADES_PER_HOUR = 8
COOLDOWN_SECS_AFTER_CLOSE = 45
ADX_GATE = 17

# ===== SUPER SCALP ENGINE =====
SCALP_MODE            = True
SCALP_EXECUTE         = True
SCALP_SIZE_FACTOR     = 0.35
SCALP_ADX_GATE        = 12.0
SCALP_MIN_SCORE       = 3.5
SCALP_IMB_THRESHOLD   = 1.00
SCALP_VOL_MA_FACTOR   = 1.20
SCALP_COOLDOWN_SEC    = 8
SCALP_RESPECT_WAIT    = False
SCALP_TP_SINGLE_PCT   = 0.35
SCALP_BE_AFTER_PCT    = 0.15
SCALP_ATR_TRAIL_MULT  = 1.0

# ===== SUPER COUNCIL ENHANCEMENTS =====
COUNCIL_AI_MODE = True
TREND_EARLY_DETECTION = True
MOMENTUM_ACCELERATION = True
VOLUME_CONFIRMATION = True
PRICE_ACTION_INTELLIGENCE = True

# أوزان التصويت الذكية
WEIGHT_ADX = 1.5
WEIGHT_RSI = 1.2
WEIGHT_MACD = 1.3
WEIGHT_VOLUME = 1.1
WEIGHT_FLOW = 1.4
WEIGHT_GOLDEN = 1.6
WEIGHT_CANDLES = 1.2
WEIGHT_MOMENTUM = 1.3
WEIGHT_FOOTPRINT = 1.5
WEIGHT_DIAGONAL = 1.4
WEIGHT_EARLY_TREND = 1.7
WEIGHT_BREAKOUT = 1.6

# ===== INTELLIGENT TREND MANAGEMENT =====
TREND_RIDING_AI = True
DYNAMIC_TP_ADJUSTMENT = True
ADAPTIVE_TRAILING = True
TREND_STRENGTH_ANALYSIS = True

# إعدادات ركوب الترند الذكية
TREND_FOLLOW_MULTIPLIER = 1.5
WEAK_TREND_EARLY_EXIT = True
STRONG_TREND_HOLD = True
TREND_REENTRY_STRATEGY = True

# ===== FLOW/FOOTPRINT Council Boost =====
FLOW_IMB_RATIO          = 1.6
FLOW_STACK_DEPTH        = 4
FLOW_ABSORB_PCTL        = 0.95
FLOW_ABSORB_MAX_TICKS   = 2
FP_WINDOW               = 3
FP_SCORE_BUY            = (2, 1.0)
FP_SCORE_SELL           = (2, 1.0)
FP_SCORE_ABSORB_PENALTY = (-1, -0.5)
DIAG_SCORE_BUY          = (2, 1.0)
DIAG_SCORE_SELL         = (2, 1.0)

# ===== PROFIT ACCUMULATION SYSTEM =====
COMPOUND_PROFIT_REINVEST = True
PROFIT_REINVEST_RATIO = 0.3  # 30% من الأرباح يعاد استثمارها
MIN_COMPOUND_BALANCE = 50.0  # الحد الأدنى للرصيد قبل البدء في المراكبة

# ===== ADVANCED TREND DETECTION =====
EARLY_TREND_DETECTION = True
TREND_CONFIRMATION_PERIOD = 3
BREAKOUT_CONFIRMATION = True
VOLUME_CONFIRMATION_MULTIPLIER = 1.2

# ===== SMART POSITION MANAGEMENT =====
ADAPTIVE_POSITION_SIZING = True
VOLATILITY_ADJUSTED_SIZE = True
DYNAMIC_LEVERAGE = False
MAX_LEVERAGE = 15

# =============== TRADE MODE CONFIG (SCALP vs TREND) ===============
TREND_ADX_MIN        = 22      # من أول هنا نعتبر إن فيه ترند محترم
TREND_DI_SPREAD_MIN  = 8       # فرق +DI/-DI عشان نعتبر الاتجاه واضح
CHOP_ADX_MAX         = 15      # تحت الرقم ده السوق تذبذب (chop)

RSI_TREND_PERSIST    = 3       # عدد الشمعات اللي RSI يمشي فيها فوق/تحت المتوسط عشان نعتبره ترند
RSI_NEUTRAL_LOW      = 45      # نطاق الرينج / التذبذب
RSI_NEUTRAL_HIGH     = 55

# إعدادات إدارة الصفقة بناءً على المود
SCALP_TP_PCT         = 0.35 / 100    # هدف سكالب محترم يغطي الرسوم
SCALP_BE_AFTER_PCT   = 0.25 / 100
SCALP_TRAIL_START_PCT= 0.30 / 100

TREND_TP1_PCT        = 0.80 / 100    # أول هدف في الترند
TREND_BE_AFTER_PCT   = 0.60 / 100
TREND_TRAIL_START_PCT= 1.00 / 100

# ===== SNAPSHOT & MARK SYSTEM =====
GREEN="🟢"; RED="🔴"
RESET="\x1b[0m"; BOLD="\x1b[1m"
FG_G="\x1b[32m"; FG_R="\x1b[31m"; FG_C="\x1b[36m"; FG_Y="\x1b[33m"; FG_M="\x1b[35m"

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
    
    log_i("🔄 Professional logging ready - File rotation + Werkzeug suppression")

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
    """يرجع float من آخر عنصر; يقبل Series/np.ndarray/list/float."""
    try:
        if isinstance(x, pd.Series): return float(x.iloc[-1])
        if isinstance(x, (list, tuple, np.ndarray)): return float(x[-1])
        if x is None: return float(default)
        return float(x)
    except Exception:
        return float(default)

def safe_get(ind: dict, key: str, default=0.0):
    """يقرأ مؤشر من dict ويحوّله scalar أخير."""
    if ind is None: 
        return float(default)
    val = ind.get(key, default)
    return last_scalar(val, default=default)

def _ind_brief(ind):
    if not ind: return "n/a"
    
    # استخراج قيم scalar بأمان
    adx = safe_get(ind, 'adx', 0)
    di_spread = safe_get(ind, 'di_spread', 0)
    rsi = safe_get(ind, 'rsi', 0)
    rsi_ma = safe_get(ind, 'rsi_ma', 0)
    atr = safe_get(ind, 'atr', 0)
    
    return (f"ADX={adx:.1f} DI={di_spread:.1f} | "
            f"RSI={rsi:.1f}/{rsi_ma:.1f} | "
            f"ATR={atr:.4f}")

def _council_brief(c):
    if not c: return "n/a"
    return f"B:{c.get('b',0)}/{_fmt(c.get('score_b',0),1)} | S:{c.get('s',0)}/{_fmt(c.get('score_s',0),1)}"

def _flow_brief(f):
    if not f: return "n/a"
    parts=[f"Δz={_fmt(f.get('delta_z','n/a'),2)}", f"CVD={_fmt(f.get('cvd_last','n/a'),0)}", f"trend={f.get('cvd_trend','?')}"]
    if f.get("spike"): parts.append("SPIKE")
    return " ".join(parts)

def print_position_snapshot(reason="OPEN", color=None):
    try:
        side   = STATE.get("side")
        open_f = STATE.get("open",False)
        qty    = STATE.get("qty"); px = STATE.get("entry")
        mode   = STATE.get("mode","trend")
        mode_why = STATE.get("mode_why", "")
        lev    = globals().get("LEVERAGE",0)
        tp1    = globals().get("TP1_PCT_BASE",0)
        be_a   = globals().get("BREAKEVEN_AFTER",0)
        trailA = globals().get("TRAIL_ACTIVATE_PCT",0)
        atrM   = globals().get("ATR_TRAIL_MULT",0)
        bal    = balance_usdt()
        spread = STATE.get("last_spread_bps")
        council= STATE.get("last_council")
        ind    = STATE.get("last_ind")
        flow   = STATE.get("last_flow")

        if color is None:
            icon = GREEN if side=="buy" else RED
            ccol = FG_G if side=="buy" else FG_R
        else:
            icon = GREEN if str(color).lower()=="green" else RED
            ccol = FG_G if icon==GREEN else FG_R

        # إضافة لون حسب النوع
        mode_color = FG_Y if mode == "scalp" else FG_M
        mode_icon = "⚡" if mode == "scalp" else "📈"
        
        log_i(f"{mode_color}{BOLD}{mode_icon} {reason} — {mode.upper()} POSITION | {mode_why}{RESET}")
        log_i(f"{BOLD}SIDE:{RESET} {side} | {BOLD}QTY:{RESET} {_fmt(qty)} | {BOLD}ENTRY:{RESET} {_fmt(px)} | "
              f"{BOLD}LEV:{RESET} {lev}× | {BOLD}MODE:{RESET} {mode} | {BOLD}OPEN:{RESET} {open_f}")
        log_i(f"{BOLD}TP1:{RESET} {_pct(tp1)} | {BOLD}BE@:{RESET} {_pct(be_a)} | "
              f"{BOLD}TRAIL:{RESET} act≥{_pct(trailA)}, ATR×{atrM} | {BOLD}SPREAD:{RESET} {_fmt(spread,2)} bps")
        log_i(f"{FG_C}IND:{RESET} {_ind_brief(ind)}")
        log_i(f"{FG_M}COUNCIL:{RESET} {_council_brief(council)}")
        log_i(f"{FG_Y}FLOW:{RESET} {_flow_brief(flow)}")
        log_i("—"*72)
    except Exception as e:
        log_w(f"SNAPSHOT ERR: {e}")

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

# =================== ADVANCED TREND DETECTION ===================
def detect_early_trend(df, ind):
    """اكتشاف مبكر للترند باستخدام تحليل متقدم"""
    try:
        if len(df) < 50:
            return {"trend": "neutral", "strength": 0.0, "confidence": 0.0}
        
        close = df['close'].astype(float)
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        volume = df['volume'].astype(float)
        
        # مؤشرات متقدمة للكشف المبكر
        ema_20 = close.ewm(span=20).mean()
        ema_50 = close.ewm(span=50).mean()
        sma_20 = close.rolling(20).mean()
        
        # اتجاه المتوسطات
        ema_trend = "bull" if ema_20.iloc[-1] > ema_50.iloc[-1] else "bear"
        price_vs_ema = "bull" if close.iloc[-1] > ema_20.iloc[-1] else "bear"
        
        # قوة الحركة
        momentum_5 = ((close.iloc[-1] - close.iloc[-5]) / close.iloc[-5]) * 100
        momentum_10 = ((close.iloc[-1] - close.iloc[-10]) / close.iloc[-10]) * 100
        
        # تحليل الحجم
        volume_ma = volume.rolling(20).mean()
        volume_spike = volume.iloc[-1] > volume_ma.iloc[-1] * 1.5
        
        # تحليل التقلب
        atr = safe_get(ind, 'atr', 0)
        recent_atr = (high - low).rolling(5).mean().iloc[-1]
        volatility_ratio = recent_atr / atr if atr > 0 else 1.0
        
        score_bull = 0.0
        score_bear = 0.0
        
        # تصويت الاتجاه الصاعد
        if ema_trend == "bull":
            score_bull += 2.0
        if price_vs_ema == "bull":
            score_bull += 1.5
        if momentum_5 > 0.5:
            score_bull += 1.0
        if momentum_10 > 1.0:
            score_bull += 1.5
        if volume_spike and close.iloc[-1] > close.iloc[-2]:
            score_bull += 1.5
        
        # تصويت الاتجاه الهابط
        if ema_trend == "bear":
            score_bear += 2.0
        if price_vs_ema == "bear":
            score_bear += 1.5
        if momentum_5 < -0.5:
            score_bear += 1.0
        if momentum_10 < -1.0:
            score_bear += 1.5
        if volume_spike and close.iloc[-1] < close.iloc[-2]:
            score_bear += 1.5
        
        # تحديد الاتجاه النهائي
        if score_bull > score_bear + 2.0:
            trend = "bull"
            strength = min(10.0, score_bull)
            confidence = min(1.0, strength / 8.0)
        elif score_bear > score_bull + 2.0:
            trend = "bear"
            strength = min(10.0, score_bear)
            confidence = min(1.0, strength / 8.0)
        else:
            trend = "neutral"
            strength = max(score_bull, score_bear)
            confidence = strength / 8.0
        
        return {
            "trend": trend,
            "strength": round(strength, 2),
            "confidence": round(confidence, 2),
            "momentum_5": momentum_5,
            "momentum_10": momentum_10,
            "volatility_ratio": volatility_ratio
        }
        
    except Exception as e:
        log_w(f"Early trend detection error: {e}")
        return {"trend": "neutral", "strength": 0.0, "confidence": 0.0}

def detect_breakout_opportunity(df, ind):
    """اكتشاف فرص الاختراق باستخدام تحليل متقدم"""
    try:
        if len(df) < 30:
            return {"breakout": False, "direction": "none", "strength": 0.0}
        
        close = df['close'].astype(float)
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        volume = df['volume'].astype(float)
        
        # مستويات المقاومة والدعم
        resistance = high.rolling(20).max()
        support = low.rolling(20).min()
        
        current_high = high.iloc[-1]
        current_low = low.iloc[-1]
        current_close = close.iloc[-1]
        
        # تحليل الاختراق
        breakout_up = current_close > resistance.iloc[-2] and current_high > resistance.iloc[-2]
        breakout_down = current_close < support.iloc[-2] and current_low < support.iloc[-2]
        
        # تأكيد الحجم
        volume_ma = volume.rolling(20).mean()
        volume_confirmation = volume.iloc[-1] > volume_ma.iloc[-1] * 1.2
        
        # قوة الاختراق
        strength = 0.0
        direction = "none"
        
        if breakout_up and volume_confirmation:
            direction = "up"
            # حساب قوة الاختراق
            breakout_power = (current_close - resistance.iloc[-2]) / resistance.iloc[-2] * 100
            strength = min(10.0, breakout_power * 10)
            strength += 2.0 if volume.iloc[-1] > volume_ma.iloc[-1] * 1.5 else 0.0
            
        elif breakout_down and volume_confirmation:
            direction = "down"
            # حساب قوة الاختراق
            breakout_power = (support.iloc[-2] - current_close) / support.iloc[-2] * 100
            strength = min(10.0, breakout_power * 10)
            strength += 2.0 if volume.iloc[-1] > volume_ma.iloc[-1] * 1.5 else 0.0
        
        return {
            "breakout": direction != "none",
            "direction": direction,
            "strength": round(strength, 2),
            "volume_confirmed": volume_confirmation
        }
        
    except Exception as e:
        log_w(f"Breakout detection error: {e}")
        return {"breakout": False, "direction": "none", "strength": 0.0}

# =================== FOOTPRINT & DIAGONAL FLOW SYSTEMS ===================
def analyze_footprint_fallback(df: pd.DataFrame, window: int = FP_WINDOW):
    """
    نسخة بديلة تعتمد على علاقة الحجم بالسعر عندما لا تتوفر بيانات Footprint مباشرة
    """
    try:
        sub = df.tail(window)
        votes_b = votes_s = 0
        score_b = score_s = 0.0
        tag = "balanced"

        for _, row in sub.iterrows():
            close = float(row["close"])
            open_ = float(row["open"])
            high = float(row["high"])
            low = float(row["low"])
            volume = float(row["volume"])
            
            candle_up = close > open_
            body_size = abs(close - open_)
            total_range = high - low
            
            if total_range <= 0:
                continue
                
            # نسبة الجسم إلى المدى (تشير إلى قوة الاتجاه)
            body_ratio = body_size / total_range
            
            # حجم التداول بالنسبة للمدى (كثافة التداول)
            volume_density = volume / total_range if total_range > 0 else 0
            
            # شمعة قوية صاعدة: جسم كبير + حجم عالي
            if candle_up and body_ratio > 0.6 and volume_density > np.percentile([v/(h-l) for v,h,l in zip(sub['volume'], sub['high'], sub['low']) if (h-l)>0], 70):
                vb, sb = FP_SCORE_BUY
                votes_b += vb; score_b += sb; tag = "aggressive_buy"
            
            # شمعة قوية هابطة: جسم كبير + حجم عالي  
            elif not candle_up and body_ratio > 0.6 and volume_density > np.percentile([v/(h-l) for v,h,l in zip(sub['volume'], sub['high'], sub['low']) if (h-l)>0], 70):
                vs, ss = FP_SCORE_SELL
                votes_s += vs; score_s += ss; tag = "aggressive_sell"
                
            # دوجي مع حجم عالي (امتصاص)
            elif body_ratio < 0.3 and volume_density > np.percentile([v/(h-l) for v,h,l in zip(sub['volume'], sub['high'], sub['low']) if (h-l)>0], 80):
                if close > open_:  # دوجي مع إغلاق أعلى (امتصاص بيع)
                    vs, ss = FP_SCORE_ABSORB_PENALTY
                    votes_s += vs; score_s += ss; tag = "absorb_bid"
                else:  # دوجي مع إغلاق أقل (امتصاص شراء)
                    vb, sb = FP_SCORE_ABSORB_PENALTY
                    votes_b += vb; score_b += sb; tag = "absorb_ask"

        return {"votes_b": votes_b, "votes_s": votes_s,
                "score_b": score_b, "score_s": score_s, "tag": tag}
    except Exception as e:
        return {"votes_b":0,"votes_s":0,"score_b":0.0,"score_s":0.0,"tag":f"err:{e}"}

def analyze_diagonal_flow(orderbook: dict, depth: int = FLOW_STACK_DEPTH, imb_ratio: float = FLOW_IMB_RATIO):
    try:
        bids = orderbook.get("bids", []) or []
        asks = orderbook.get("asks", []) or []
        n = min(len(bids), len(asks), depth)
        buy_strength = sell_strength = 0

        for i in range(n):
            b_qty = float(bids[i][1]); a_qty = float(asks[i][1])
            if b_qty <= 0 or a_qty <= 0: 
                continue
            r = b_qty / a_qty
            if r >= imb_ratio: buy_strength += 1
            elif r <= 1/imb_ratio: sell_strength += 1

        if buy_strength > sell_strength:
            bias = "buy"; votes, score = DIAG_SCORE_BUY
        elif sell_strength > buy_strength:
            bias = "sell"; votes, score = DIAG_SCORE_SELL
        else:
            bias = "neutral"; votes, score = 0, 0.0

        return {"bias": bias, "votes": votes, "score": score,
                "buy_strength": buy_strength, "sell_strength": sell_strength}
    except Exception as e:
        return {"bias":"neutral","votes":0,"score":0.0,"err":str(e)}

def council_boost_from_flow(df: pd.DataFrame, orderbook: dict):
    fp = analyze_footprint_fallback(df)
    dg = analyze_diagonal_flow(orderbook)

    # تجميع التصويت
    votes_b = fp["votes_b"] + (dg["votes"] if dg["bias"]=="buy" else 0)
    votes_s = fp["votes_s"] + (dg["votes"] if dg["bias"]=="sell" else 0)
    score_b = fp["score_b"] + (dg["score"] if dg["bias"]=="buy" else 0.0)
    score_s = fp["score_s"] + (dg["score"] if dg["bias"]=="sell" else 0.0)

    tag = f"FP:{fp['tag']} | DIAG:{dg['bias']}(B{dg.get('buy_strength',0)}/S{dg.get('sell_strength',0)})"
    return {"votes_b":votes_b, "votes_s":votes_s, "score_b":score_b, "score_s":score_s, "tag":tag}

# ========= Unified snapshot emitter =========
def emit_snapshots(exchange, symbol, df, balance_fn=None, pnl_fn=None):
    try:
        bm = bookmap_snapshot(exchange, symbol)
        flow = compute_flow_metrics(df)
        cv = council_votes_pro(df)
        mode = decide_strategy_mode(df)
        gz = golden_zone_check(df, {"adx": cv["ind"].get("adx", 0)}, "buy" if cv["b"]>=cv["s"] else "sell")

        bal = None; cpnl = None
        if callable(balance_fn):
            try: bal = balance_fn()
            except: bal = None
        if callable(pnl_fn):
            try: cpnl = pnl_fn()
            except: cpnl = None

        if bm.get("ok"):
            imb_tag = "🟢" if bm["imbalance"]>=IMBALANCE_ALERT else ("🔴" if bm["imbalance"]<=1/IMBALANCE_ALERT else "⚖️")
            bm_note = f"Bookmap: {imb_tag} Imb={bm['imbalance']:.2f} | Buy[{fmt_walls(bm['buy_walls'])}] | Sell[{fmt_walls(bm['sell_walls'])}]"
        else:
            bm_note = f"Bookmap: N/A ({bm.get('why')})"

        if flow.get("ok"):
            dtag = "🟢Buy" if flow["delta_last"]>0 else ("🔴Sell" if flow["delta_last"]<0 else "⚖️Flat")
            spk = " ⚡Spike" if flow["spike"] else ""
            fl_note = f"Flow: {dtag} Δ={flow['delta_last']:.0f} z={flow['delta_z']:.2f}{spk} | CVD {'↗️' if flow['cvd_trend']=='up' else '↘️'} {flow['cvd_last']:.0f}"
        else:
            fl_note = f"Flow: N/A ({flow.get('why')})"

        side_hint = "BUY" if cv["b"]>=cv["s"] else "SELL"
        dash = (f"DASH → hint-{side_hint} | Council BUY({cv['b']},{cv['score_b']:.1f}) "
                f"SELL({cv['s']},{cv['score_s']:.1f}) | "
                f"RSI={safe_get(cv['ind'],'rsi',0):.1f} ADX={safe_get(cv['ind'],'adx',0):.1f} "
                f"DI={safe_get(cv['ind'],'di_spread',0):.1f} | Confidence: {cv.get('confidence',0):.1f}")

        strat_icon = "⚡" if mode["mode"]=="scalp" else "📈" if mode["mode"]=="trend" else "ℹ️"
        strat = f"Strategy: {strat_icon} {mode['mode'].upper()}"

        bal_note = f"Balance={bal:.2f}" if bal is not None else ""
        pnl_note = f"CompoundPnL={cpnl:.6f}" if cpnl is not None else ""
        wallet = (" | ".join(x for x in [bal_note, pnl_note] if x)) or ""

        gz_note = ""
        if gz and gz.get("ok"):
            gz_note = f" | 🟡 {gz['zone']['type']} s={gz['score']:.1f}"

        if LOG_ADDONS:
            print(f"🧱 {bm_note}", flush=True)
            print(f"📦 {fl_note}", flush=True)
            print(f"📊 {dash}{gz_note}", flush=True)
            print(f"{strat}{(' | ' + wallet) if wallet else ''}", flush=True)
            
            gz_snap_note = ""
            if gz and gz.get("ok"):
                zone_type = gz["zone"]["type"]
                zone_score = gz["score"]
                gz_snap_note = f" | 🟡{zone_type} s={zone_score:.1f}"
            
            flow_z = flow['delta_z'] if flow and flow.get('ok') else 0.0
            bm_imb = bm['imbalance'] if bm and bm.get('ok') else 1.0
            
            print(f"🧠 SNAP | {side_hint} | votes={cv['b']}/{cv['s']} score={cv['score_b']:.1f}/{cv['score_s']:.1f} "
                  f"| ADX={safe_get(cv['ind'],'adx',0):.1f} DI={safe_get(cv['ind'],'di_spread',0):.1f} | "
                  f"z={flow_z:.2f} | imb={bm_imb:.2f}{gz_snap_note}", 
                  flush=True)
            
            print("✅ ADDONS LIVE", flush=True)

        return {"bm": bm, "flow": flow, "cv": cv, "mode": mode, "gz": gz, "wallet": wallet}
    except Exception as e:
        print(f"🟨 AddonLog error: {e}", flush=True)
        return {"bm": None, "flow": None, "cv": {"b":0,"s":0,"score_b":0.0,"score_s":0.0,"ind":{}},
                "mode": {"mode":"n/a"}, "gz": None, "wallet": ""}

# =================== ADVANCED INDICATORS ===================
def sma(series, n: int):
    return series.rolling(n, min_periods=1).mean()

def ema(series, n: int):
    return series.ewm(span=n, adjust=False).mean()

def compute_rsi(close, n: int = 14):
    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    roll_up = up.ewm(span=n, adjust=False).mean()
    roll_down = down.ewm(span=n, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, 1e-12)
    rsi = 100 - (100/(1+rs))
    return rsi.fillna(50)

def compute_macd(close, fast=12, slow=26, signal=9):
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd = ema_fast - ema_slow
    macd_signal = ema(macd, signal)
    macd_histogram = macd - macd_signal
    return macd, macd_signal, macd_histogram

def compute_bollinger_bands(close, n=20, k=2):
    sma_val = sma(close, n)
    std = close.rolling(n).std()
    upper = sma_val + (std * k)
    lower = sma_val - (std * k)
    return upper, sma_val, lower

def compute_stochastic(high, low, close, n=14, d=3):
    lowest_low = low.rolling(n).min()
    highest_high = high.rolling(n).max()
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_line = sma(k, d)
    return k, d_line

def compute_volume_profile(df, period=20):
    volume = df['volume'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    
    price_range = high - low
    volume_per_price = volume / (price_range.replace(0, 1e-12))
    
    return {
        'volume_ma': sma(volume, period),
        'volume_spike': volume > sma(volume, period) * 1.5,
        'volume_trend': 'up' if volume.iloc[-1] > volume.iloc[-2] else 'down'
    }

def compute_momentum_indicators(df):
    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    
    roc = ((close - close.shift(5)) / close.shift(5)) * 100
    price_accel = close.diff().diff()
    volatility = high - low
    
    return {
        'roc': roc.iloc[-1] if len(roc) > 0 else 0,
        'price_accel': price_accel.iloc[-1] if len(price_accel) > 0 else 0,
        'volatility': volatility.iloc[-1] if len(volatility) > 0 else 0,
        'volatility_ma': sma(volatility, 20).iloc[-1] if len(volatility) >= 20 else 0
    }

def compute_trend_strength(df, ind):
    close = df['close'].astype(float)
    adx = safe_get(ind, 'adx', 0)
    plus_di = safe_get(ind, 'plus_di', 0)
    minus_di = safe_get(ind, 'minus_di', 0)
    
    momentum_5 = ((close.iloc[-1] - close.iloc[-5]) / close.iloc[-5]) * 100 if len(close) >= 5 else 0
    momentum_10 = ((close.iloc[-1] - close.iloc[-10]) / close.iloc[-10]) * 100 if len(close) >= 10 else 0
    
    trend_consistency = 0
    if len(close) >= 10:
        up_days = sum(close.diff().tail(10) > 0)
        down_days = sum(close.diff().tail(10) < 0)
        trend_consistency = max(up_days, down_days) / 10.0
    
    if adx > 40 and abs(momentum_5) > 3.0 and trend_consistency > 0.7:
        strength = "very_strong"
        multiplier = 2.0
    elif adx > 30 and abs(momentum_5) > 2.0 and trend_consistency > 0.6:
        strength = "strong"
        multiplier = 1.5
    elif adx > 25 and abs(momentum_5) > 1.0:
        strength = "moderate"
        multiplier = 1.2
    elif adx > 20:
        strength = "weak"
        multiplier = 1.0
    else:
        strength = "no_trend"
        multiplier = 0.8
    
    direction = "up" if plus_di > minus_di else "down"
    
    return {
        "strength": strength,
        "direction": direction,
        "multiplier": multiplier,
        "adx": adx,
        "momentum_5": momentum_5,
        "momentum_10": momentum_10,
        "consistency": trend_consistency
    }

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

# =================== TRADE MODE CLASSIFICATION SYSTEM ===================
def _sma(series, n):
    """متوسط متحرك بسيط"""
    return series.rolling(n, min_periods=1).mean()

def _compute_rsi(close, n=14):
    """حساب RSI"""
    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    roll_up = up.ewm(span=n, adjust=False).mean()
    roll_down = down.ewm(span=n, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def rsi_trend_ctx(df, rsi_len=14, ma_len=9):
    """تحليل اتجاه RSI"""
    if len(df) < max(rsi_len, ma_len) + 2:
        return {"rsi": 50.0, "rsi_ma": 50.0, "trend": "none", "in_chop": True}

    rsi = _compute_rsi(df["close"].astype(float), rsi_len)
    rsi_ma = _sma(rsi, ma_len)

    above = (rsi > rsi_ma)
    below = (rsi < rsi_ma)
    
    # نتحقق من استمرارية الاتجاه
    bull = above.tail(RSI_TREND_PERSIST).all() if len(above) >= RSI_TREND_PERSIST else False
    bear = below.tail(RSI_TREND_PERSIST).all() if len(below) >= RSI_TREND_PERSIST else False

    trend = "bull" if bull else ("bear" if bear else "none")
    
    current_rsi = float(rsi.iloc[-1])
    in_chop = RSI_NEUTRAL_LOW <= current_rsi <= RSI_NEUTRAL_HIGH

    return {
        "rsi": current_rsi,
        "rsi_ma": float(rsi_ma.iloc[-1]),
        "trend": trend,
        "in_chop": in_chop,
    }

def classify_trade_mode(df, ind):
    """
    يقرر هل الصفقة دي SCALP ولا TREND قبل الدخول.
    يعتمد على: ADX / DI / RSI / تذبذب السوق.
    يرجّع dict: {mode: 'scalp'|'trend'|'chop', why: '...'}
    """
    adx = safe_get(ind, "adx", 0.0)
    plus_di = safe_get(ind, "plus_di", 0.0)
    minus_di = safe_get(ind, "minus_di", 0.0)

    di_spread = abs(plus_di - minus_di)

    rctx = rsi_trend_ctx(df)
    rsi_trend = rctx["trend"]
    in_chop = rctx["in_chop"]

    strong_trend = (
        adx >= TREND_ADX_MIN and
        di_spread >= TREND_DI_SPREAD_MIN
    ) or (
        rsi_trend in ("bull", "bear") and not in_chop
    )

    # 1) سوق تذبذب → سكالب بس / حذر
    if adx < CHOP_ADX_MAX or in_chop:
        return {
            "mode": "scalp",
            "why": f"chop_or_low_adx adx={adx:.1f} di_spread={di_spread:.1f} chop={in_chop}"
        }

    # 2) ترند قوي وواضح
    if strong_trend:
        return {
            "mode": "trend",
            "why": f"strong_trend adx={adx:.1f} di_spread={di_spread:.1f} rsi_trend={rsi_trend}"
        }

    # 3) منطقة وسطية → نعتبرها سكالب محسّن
    return {
        "mode": "scalp",
        "why": f"default_scalp adx={adx:.1f} di_spread={di_spread:.1f} rsi_trend={rsi_trend}"
    }

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

    rng1 = _rng(h1,l1); up = _upper_wick(h1,o1,c1); dn = _lower_wick(l1,o1,c1)
    wick_up_big = (up >= 1.2*_body(o1,c1)) and (up >= 0.4*rng1)
    wick_dn_big = (dn >= 1.2*_body(o1,c1)) and (dn >= 0.4*rng1)

    if is_doji:
        strength_b *= 0.8; strength_s *= 0.8

    return {
        "buy": strength_b>0, "sell": strength_s>0,
        "score_buy": round(strength_b,2), "score_sell": round(strength_s,2),
        "wick_up_big": bool(wick_up_big), "wick_dn_big": bool(wick_dn_big),
        "doji": bool(is_doji), "pattern": ",".join(tags) if tags else None
    }

# =================== SMART GOLDEN ZONE DETECTION ===================
def _ema_gz(series, n):
    return series.ewm(span=n, adjust=False).mean()

def _rsi_fallback_gz(close, n=14):
    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    roll_up = up.ewm(span=n, adjust=False).mean()
    roll_down = down.ewm(span=n, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, 1e-12)
    rsi = 100 - (100/(1+rs))
    return rsi.fillna(50)

def _body_wicks_gz(h, l, o, c):
    rng = max(1e-9, h - l)
    body = abs(c - o) / rng
    up_wick = (h - max(c, o)) / rng
    low_wick = (min(c, o) - l) / rng
    return body, up_wick, low_wick

def _displacement_gz(closes):
    if len(closes) < 22:
        return 0.0
    recent_std = closes.tail(20).std()
    return abs(closes.iloc[-1] - closes.iloc[-2]) / max(recent_std, 1e-9)

def _last_impulse_gz(df):
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    
    lookback = min(120, len(df))
    recent_highs = h.tail(lookback)
    recent_lows = l.tail(lookback)
    
    hh_idx = recent_highs.idxmax()
    ll_idx = recent_lows.idxmin()
    
    hh = recent_highs.max()
    ll = recent_lows.min()
    
    if hh_idx < ll_idx:
        return ("down", hh_idx, ll_idx, hh, ll)
    else:
        return ("up", ll_idx, hh_idx, ll, hh)

def golden_zone_check(df, ind=None, side_hint=None):
    if len(df) < 60:
        return {"ok": False, "score": 0.0, "zone": None, "reasons": ["short_df"]}
    
    try:
        h = df['high'].astype(float)
        l = df['low'].astype(float)
        c = df['close'].astype(float)
        o = df['open'].astype(float)
        v = df['volume'].astype(float)
        
        impulse_data = _last_impulse_gz(df)
        if not impulse_data:
            return {"ok": False, "score": 0.0, "zone": None, "reasons": ["no_clear_impulse"]}
            
        side, idx1, idx2, p1, p2 = impulse_data
        
        if side == "down":
            swing_hi, swing_lo = p1, p2
            f618 = swing_lo + FIB_LOW * (swing_hi - swing_lo)
            f786 = swing_lo + FIB_HIGH * (swing_hi - swing_lo)
            zone_type = "golden_bottom"
        else:
            swing_lo, swing_hi = p1, p2
            f618 = swing_hi - FIB_HIGH * (swing_hi - swing_lo)
            f786 = swing_hi - FIB_LOW * (swing_hi - swing_lo)
            zone_type = "golden_top"
        
        last_close = float(c.iloc[-1])
        in_zone = (f618 <= last_close <= f786) if side == "down" else (f786 <= last_close <= f618)
        
        if not in_zone:
            return {"ok": False, "score": 0.0, "zone": None, "reasons": [f"price_not_in_zone {last_close:.6f} vs [{f618:.6f},{f786:.6f}]"]}
        
        current_high = float(h.iloc[-1])
        current_low = float(l.iloc[-1])
        current_open = float(o.iloc[-1])
        
        body, up_wick, low_wick = _body_wicks_gz(current_high, current_low, current_open, last_close)
        
        vol_ma = v.rolling(VOL_MA_LEN).mean().iloc[-1]
        vol_ok = float(v.iloc[-1]) >= vol_ma * 0.9
        
        rsi_series = _rsi_fallback_gz(c, RSI_LEN_GZ)
        rsi_ma_series = _ema_gz(rsi_series, RSI_MA_LEN_GZ)
        rsi_last = float(rsi_series.iloc[-1])
        rsi_ma_last = float(rsi_ma_series.iloc[-1])
        
        adx = safe_get(ind, 'adx', 0) if ind else 0
        disp = _displacement_gz(c)
        
        if side == "down":
            wick_ok = low_wick >= MIN_WICK_PCT
            rsi_ok = rsi_last > rsi_ma_last and rsi_last < 70
            candle_bullish = last_close > current_open
        else:
            wick_ok = up_wick >= MIN_WICK_PCT
            rsi_ok = rsi_last < rsi_ma_last and rsi_last > 30
            candle_bullish = last_close < current_open
        
        score = 0.0
        reasons = []
        
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
        
        score += 2.0
        reasons.append("in_zone")
        
        ok = (score >= GZ_MIN_SCORE and in_zone and adx >= GZ_REQ_ADX)
        
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

def decide_strategy_mode(df, adx=None, di_plus=None, di_minus=None, rsi_ctx=None):
    if adx is None or di_plus is None or di_minus is None:
        ind = compute_indicators(df)
        adx = safe_get(ind, 'adx', 0)
        di_plus = safe_get(ind, 'plus_di', 0)
        di_minus = safe_get(ind, 'minus_di', 0)
    
    if rsi_ctx is None:
        rsi_ctx = rsi_ma_context(df)
    
    di_spread = abs(di_plus - di_minus)
    
    strong_trend = (
        (adx >= ADX_TREND_MIN and di_spread >= DI_SPREAD_TREND) or
        (rsi_ctx["trendZ"] in ("bull", "bear") and not rsi_ctx["in_chop"])
    )
    
    mode = "trend" if strong_trend else "scalp"
    why = "adx/di_trend" if adx >= ADX_TREND_MIN else ("rsi_trendZ" if rsi_ctx["trendZ"] != "none" else "scalp_default")
    
    return {"mode": mode, "why": why}

# =================== SUPER COUNCIL AI - ENHANCED VERSION ===================
def super_council_ai_enhanced(df):
    try:
        if len(df) < 50:
            return {"b": 0, "s": 0, "score_b": 0.0, "score_s": 0.0, "logs": [], "confidence": 0.0}
        
        ind = compute_indicators(df)
        
        # استخراج قيم scalar بأمان - إصلاح الخطأ الرئيسي
        adx = safe_get(ind, "adx", 0.0)
        plus_di = safe_get(ind, "plus_di", 0.0)
        minus_di = safe_get(ind, "minus_di", 0.0)
        di_spread = abs(plus_di - minus_di)
        rsi_val = safe_get(ind, "rsi", 50.0)
        atr_val = safe_get(ind, "atr", 0.0)
        
        rsi_ctx = rsi_ma_context(df)
        gz = golden_zone_check(df, ind)
        candles = compute_candles(df)
        flow = compute_flow_metrics(df)
        volume_profile = compute_volume_profile(df)
        momentum = compute_momentum_indicators(df)
        trend_strength = compute_trend_strength(df, ind)
        
        # الاكتشاف المبكر للترند والاختراق
        early_trend = detect_early_trend(df, ind)
        breakout = detect_breakout_opportunity(df, ind)
        
        # إصلاح: استخدام last_scalar بدلاً من الوصول المباشر
        macd, macd_signal, macd_hist = compute_macd(df['close'].astype(float))
        macd_current = last_scalar(macd)
        macd_signal_current = last_scalar(macd_signal) 
        macd_hist_current = last_scalar(macd_hist)
        
        macd_bullish = macd_current > macd_signal_current and macd_hist_current > 0
        macd_bearish = macd_current < macd_signal_current and macd_hist_current < 0
        
        bb_upper, bb_middle, bb_lower = compute_bollinger_bands(df['close'].astype(float))
        current_price = float(df['close'].iloc[-1])
        
        bb_upper_val = last_scalar(bb_upper)
        bb_lower_val = last_scalar(bb_lower)
        
        if bb_upper_val != bb_lower_val:
            bb_position = (current_price - bb_lower_val) / (bb_upper_val - bb_lower_val)
        else:
            bb_position = 0.5
        
        stoch_k, stoch_d = compute_stochastic(df['high'].astype(float), df['low'].astype(float), df['close'].astype(float))
        stoch_k_val = last_scalar(stoch_k)
        stoch_d_val = last_scalar(stoch_d)
        
        stoch_bullish = stoch_k_val > stoch_d_val and stoch_k_val < 80
        stoch_bearish = stoch_k_val < stoch_d_val and stoch_k_val > 20
        
        votes_b = 0; votes_s = 0
        score_b = 0.0; score_s = 0.0
        logs = []
        confidence_factors = []

        # ===== FLOW/FOOTPRINT BOOST =====
        try:
            current_orderbook = STATE.get("last_orderbook", {})
            if not current_orderbook:
                current_orderbook = ex.fetch_order_book(SYMBOL, limit=FLOW_STACK_DEPTH)
                STATE["last_orderbook"] = current_orderbook
            
            boost = council_boost_from_flow(df, current_orderbook)
            
            votes_b += boost["votes_b"]
            votes_s += boost["votes_s"]
            score_b += boost["score_b"] * WEIGHT_FOOTPRINT
            score_s += boost["score_s"] * WEIGHT_FOOTPRINT
            
            logs.append(f"🧭 FLOW-BOOST → {boost['tag']}  "
                       f"Δvotes: B+{boost['votes_b']} S+{boost['votes_s']} | "
                       f"Δscore: B+{boost['score_b']:.1f} S+{boost['score_s']:.1f}")
        except Exception as e:
            logs.append(f"🟨 FLOW-BOOST error: {e}")

        # ===== EARLY TREND DETECTION BOOST =====
        if EARLY_TREND_DETECTION and early_trend["trend"] != "neutral":
            trend_strength_early = early_trend["strength"]
            trend_confidence = early_trend["confidence"]
            
            if early_trend["trend"] == "bull" and trend_confidence > 0.6:
                early_score = WEIGHT_EARLY_TREND * trend_strength_early
                score_b += early_score
                votes_b += int(trend_strength_early)
                logs.append(f"🚀 اكتشاف مبكر لترند صاعد (قوة: {trend_strength_early:.1f})")
                confidence_factors.append(1.3)
                
            elif early_trend["trend"] == "bear" and trend_confidence > 0.6:
                early_score = WEIGHT_EARLY_TREND * trend_strength_early
                score_s += early_score
                votes_s += int(trend_strength_early)
                logs.append(f"💥 اكتشاف مبكر لترند هابط (قوة: {trend_strength_early:.1f})")
                confidence_factors.append(1.3)

        # ===== BREAKOUT DETECTION BOOST =====
        if BREAKOUT_CONFIRMATION and breakout["breakout"]:
            breakout_strength = breakout["strength"]
            
            if breakout["direction"] == "up" and breakout["volume_confirmed"]:
                breakout_score = WEIGHT_BREAKOUT * breakout_strength
                score_b += breakout_score
                votes_b += int(breakout_strength)
                logs.append(f"📈 اختراق صاعد قوي (قوة: {breakout_strength:.1f})")
                confidence_factors.append(1.4)
                
            elif breakout["direction"] == "down" and breakout["volume_confirmed"]:
                breakout_score = WEIGHT_BREAKOUT * breakout_strength
                score_s += breakout_score
                votes_s += int(breakout_strength)
                logs.append(f"📉 اختراق هابط قوي (قوة: {breakout_strength:.1f})")
                confidence_factors.append(1.4)

        # 1. تحليل الزخم المبكر
        if TREND_EARLY_DETECTION:
            momentum_accel = safe_get(momentum, 'price_accel', 0.0)
            momentum_roc = safe_get(momentum, 'roc', 0.0)
            
            if momentum_accel > 0 and momentum_roc > 0.5:
                score_b += WEIGHT_MOMENTUM * 1.5
                votes_b += 2
                logs.append("🚀 تسارع صاعد قوي")
                confidence_factors.append(1.2)
            
            if momentum_accel < 0 and momentum_roc < -0.5:
                score_s += WEIGHT_MOMENTUM * 1.5
                votes_s += 2
                logs.append("💥 تسارع هابط قوي")
                confidence_factors.append(1.2)

        # 2. تأكيد الحجم
        if VOLUME_CONFIRMATION:
            volume_spike = volume_profile.get('volume_spike', False)
            volume_trend = volume_profile.get('volume_trend', '')
            
            if volume_spike and volume_trend == 'up':
                if current_price > float(df['open'].iloc[-1]):
                    score_b += WEIGHT_VOLUME * 1.2
                    votes_b += 1
                    logs.append("📊 حجم صاعد مؤكد")
                else:
                    score_s += WEIGHT_VOLUME * 1.2
                    votes_s += 1
                    logs.append("📊 حجم هابط مؤكد")

        # 3. مؤشر الاتجاه المتقدم
        if adx > ADX_TREND_MIN:
            if plus_di > minus_di and di_spread > DI_SPREAD_TREND:
                score_b += WEIGHT_ADX * 2.0
                votes_b += 3
                logs.append(f"📈 ترند صاعد قوي (ADX: {adx:.1f})")
                confidence_factors.append(1.5)
            elif minus_di > plus_di and di_spread > DI_SPREAD_TREND:
                score_s += WEIGHT_ADX * 2.0
                votes_s += 3
                logs.append(f"📉 ترند هابط قوي (ADX: {adx:.1f})")
                confidence_factors.append(1.5)

        # 4. مؤشر RSI المتقدم
        rsi_cross = rsi_ctx.get("cross", "none")
        rsi_trendz = rsi_ctx.get("trendZ", "none")
        
        if rsi_cross == "bull" and rsi_val < 70:
            score_b += WEIGHT_RSI * 1.5
            votes_b += 2
            logs.append("🟢 RSI إيجابي قوي")
        elif rsi_cross == "bear" and rsi_val > 30:
            score_s += WEIGHT_RSI * 1.5
            votes_s += 2
            logs.append("🔴 RSI سلبي قوي")

        if rsi_trendz == "bull":
            score_b += WEIGHT_RSI * 2.0
            votes_b += 3
            logs.append("🚀 RSI ترند صاعد مستمر")
        elif rsi_trendz == "bear":
            score_s += WEIGHT_RSI * 2.0
            votes_s += 3
            logs.append("💥 RSI ترند هابط مستمر")

        # 5. المناطق الذهبية المحسنة
        if gz and gz.get("ok"):
            gz_score = gz.get("score", 0.0)
            zone_type = gz.get("zone", {}).get("type", "")
            
            if zone_type == 'golden_bottom' and gz_score >= 6.0:
                score_b += WEIGHT_GOLDEN * 2.5
                votes_b += 4
                logs.append(f"🏆 قاع ذهبي فائق (قوة: {gz_score:.1f})")
                confidence_factors.append(1.8)
            elif zone_type == 'golden_top' and gz_score >= 6.0:
                score_s += WEIGHT_GOLDEN * 2.5
                votes_s += 4
                logs.append(f"🏆 قمة ذهبية فائقة (قوة: {gz_score:.1f})")
                confidence_factors.append(1.8)

        # 6. تحليل الشموع اليابانية المتقدم
        candles_buy_score = candles.get("score_buy", 0.0)
        candles_sell_score = candles.get("score_sell", 0.0)
        
        if candles_buy_score > 0:
            enhanced_candle_score = min(3.0, candles_buy_score * 1.2)
            score_b += WEIGHT_CANDLES * enhanced_candle_score
            votes_b += int(enhanced_candle_score)
            logs.append(f"🕯️ شموع BUY قوية ({candles.get('pattern', '')}) +{enhanced_candle_score:.1f}")
        
        if candles_sell_score > 0:
            enhanced_candle_score = min(3.0, candles_sell_score * 1.2)
            score_s += WEIGHT_CANDLES * enhanced_candle_score
            votes_s += int(enhanced_candle_score)
            logs.append(f"🕯️ شموع SELL قوية ({candles.get('pattern', '')}) +{enhanced_candle_score:.1f}")

        # 7. تحليل التدفق والطلب المتقدم
        if flow.get("ok"):
            delta_z = flow.get("delta_z", 0.0)
            cvd_trend = flow.get("cvd_trend", "")
            
            if delta_z >= 2.0 and cvd_trend == "up":
                score_b += WEIGHT_FLOW * 1.8
                votes_b += 2
                logs.append(f"🌊 تدفق شرائي قوي (z: {delta_z:.2f})")
            elif delta_z <= -2.0 and cvd_trend == "down":
                score_s += WEIGHT_FLOW * 1.8
                votes_s += 2
                logs.append(f"🌊 تدفق بيعي قوي (z: {delta_z:.2f})")

        # 8. مؤشر MACD المتقدم
        if macd_bullish and macd_hist_current > last_scalar(macd_hist.shift(1) if hasattr(macd_hist, 'shift') else 0):
            score_b += WEIGHT_MACD * 1.5
            votes_b += 2
            logs.append("📈 MACD صاعد متسارع")
        elif macd_bearish and macd_hist_current < last_scalar(macd_hist.shift(1) if hasattr(macd_hist, 'shift') else 0):
            score_s += WEIGHT_MACD * 1.5
            votes_s += 2
            logs.append("📉 MACD هابط متسارع")

        # 9. بولنجر باندز لاكتشاف الانعكاسات
        if bb_position < 0.2 and current_price > bb_lower_val:
            score_b += 1.2
            votes_b += 1
            logs.append("🔄 ارتداد من نطاق بولنجر سفلي")
        elif bb_position > 0.8 and current_price < bb_upper_val:
            score_s += 1.2
            votes_s += 1
            logs.append("🔄 ارتداد من نطاق بولنجر علوي")

        # 10. ستوكاستيك للمدى القصير
        if stoch_bullish and stoch_k_val < 30:
            score_b += 1.0
            votes_b += 1
            logs.append("🎯 ستوكاستيك في منطقة شراء")
        elif stoch_bearish and stoch_k_val > 70:
            score_s += 1.0
            votes_s += 1
            logs.append("🎯 ستوكاستيك في منطقة بيع")

        # 11. قوة الترند
        trend_strength_val = trend_strength.get("strength", "")
        trend_direction = trend_strength.get("direction", "")
        trend_multiplier = trend_strength.get("multiplier", 1.0)
        
        if trend_strength_val in ["strong", "very_strong"]:
            if trend_direction == "up":
                score_b += trend_multiplier * 1.5
                votes_b += 2
                logs.append(f"💪 ترند صاعد {trend_strength_val} (مضاعف: {trend_multiplier})")
            else:
                score_s += trend_multiplier * 1.5
                votes_s += 2
                logs.append(f"💪 ترند هابط {trend_strength_val} (مضاعف: {trend_multiplier})")

        # تطبيق عوامل الثقة
        if confidence_factors:
            confidence_multiplier = sum(confidence_factors) / len(confidence_factors)
            score_b *= confidence_multiplier
            score_s *= confidence_multiplier

        # تخفيف في النطاق المحايد
        if rsi_ctx.get("in_chop", False):
            score_b *= 0.7
            score_s *= 0.7
            logs.append("⚖️ RSI محايد — تخفيض ثقة")

        # حارس ADX العام
        if adx < ADX_GATE:
            score_b *= 0.8
            score_s *= 0.8
            logs.append(f"🛡️ ADX Gate ({adx:.1f} < {ADX_GATE})")

        # حساب الثقة النهائية
        total_score = score_b + score_s
        confidence = min(1.0, total_score / 30.0) if total_score > 0 else 0.0

        # تحديث المؤشرات الإضافية
        ind.update({
            "rsi_ma": rsi_ctx.get("rsi_ma", 50.0),
            "rsi_trendz": rsi_trendz,
            "di_spread": di_spread,
            "gz": gz,
            "candle_buy_score": candles_buy_score,
            "candle_sell_score": candles_sell_score,
            "wick_up_big": candles.get("wick_up_big", False),
            "wick_dn_big": candles.get("wick_dn_big", False),
            "candle_tags": candles.get("pattern", ""),
            "macd_bullish": macd_bullish,
            "macd_bearish": macd_bearish,
            "bb_position": bb_position,
            "momentum": momentum,
            "volume_profile": volume_profile,
            "trend_strength": trend_strength,
            "early_trend": early_trend,
            "breakout": breakout
        })

        return {
            "b": votes_b, "s": votes_s,
            "score_b": round(score_b, 2), "score_s": round(score_s, 2),
            "logs": logs, "ind": ind, "gz": gz, "candles": candles,
            "confidence": round(confidence, 2),
            "momentum": momentum,
            "volume": volume_profile,
            "trend_strength": trend_strength,
            "early_trend": early_trend,
            "breakout": breakout
        }
    except Exception as e:
        log_w(f"super_council_ai_enhanced error: {e}")
        return {"b":0,"s":0,"score_b":0.0,"score_s":0.0,"logs":[],"ind":{},"confidence":0.0}

council_votes_pro_enhanced = super_council_ai_enhanced
council_votes_pro = super_council_ai_enhanced

# =================== SUPER SCALP AI - ENHANCED VERSION ===================
_last_scalp_ts = 0
_scalp_profit_total = 0.0

def detect_super_scalp_opportunity(df, ind, flow, volume_profile, momentum, spread_bps):
    try:
        if not SCALP_MODE or not SCALP_EXECUTE:
            return (None, "scalp_off")

        if spread_bps is not None and spread_bps > MAX_SPREAD_BPS:
            return (None, f"spread>{MAX_SPREAD_BPS}bps")

        current_price = float(df['close'].iloc[-1])
        volume_ok = volume_profile['volume_spike'] and volume_profile['volume_trend'] == 'up'
        momentum_ok = abs(momentum['roc']) > 0.3
        volatility_ok = momentum['volatility'] > momentum['volatility_ma'] * 0.8
        
        scalp_council = {
            'b': 0, 's': 0,
            'score_b': 0.0, 'score_s': 0.0
        }
        
        rsi = safe_get(ind, 'rsi', 50)
        if 30 <= rsi <= 45:
            scalp_council['score_b'] += 1.5
            scalp_council['b'] += 1
        elif 55 <= rsi <= 70:
            scalp_council['score_s'] += 1.5
            scalp_council['s'] += 1
        
        if flow and flow.get('ok'):
            if flow['delta_z'] > 1.5 and volume_ok:
                scalp_council['score_b'] += 2.0
                scalp_council['b'] += 2
            elif flow['delta_z'] < -1.5 and volume_ok:
                scalp_council['score_s'] += 2.0
                scalp_council['s'] += 2
        
        if momentum_ok and volatility_ok:
            if momentum['price_accel'] > 0 and momentum['roc'] > 0:
                scalp_council['score_b'] += 1.5
                scalp_council['b'] += 1
            elif momentum['price_accel'] < 0 and momentum['roc'] < 0:
                scalp_council['score_s'] += 1.5
                scalp_council['s'] += 1
        
        candles = compute_candles(df)
        if candles['score_buy'] > 1.0 and candles['wick_dn_big']:
            scalp_council['score_b'] += 1.2
            scalp_council['b'] += 1
        if candles['score_sell'] > 1.0 and candles['wick_up_big']:
            scalp_council['score_s'] += 1.2
            scalp_council['s'] += 1
        
        # إضافة الاكتشاف المبكر للسكالب
        early_trend = detect_early_trend(df, ind)
        if early_trend["trend"] == "bull" and early_trend["confidence"] > 0.6:
            scalp_council['score_b'] += 1.5
            scalp_council['b'] += 1
        elif early_trend["trend"] == "bear" and early_trend["confidence"] > 0.6:
            scalp_council['score_s'] += 1.5
            scalp_council['s'] += 1
        
        min_scalp_score = 4.0
        
        if scalp_council['score_b'] >= min_scalp_score and scalp_council['b'] > scalp_council['s']:
            reason = f"SCALP-BUY | score={scalp_council['score_b']:.1f} | vol={volume_ok} | mom={momentum_ok}"
            return ("buy", reason)
        
        if scalp_council['score_s'] >= min_scalp_score and scalp_council['s'] > scalp_council['b']:
            reason = f"SCALP-SELL | score={scalp_council['score_s']:.1f} | vol={volume_ok} | mom={momentum_ok}"
            return ("sell", reason)
        
        return (None, f"low_score_b={scalp_council['score_b']:.1f}_s={scalp_council['score_s']:.1f}")
        
    except Exception as e:
        return (None, f"scalp_err:{e}")

def execute_super_scalp(px_now, balance, df, ind, flow, volume_profile, momentum, spread_bps):
    global _last_scalp_ts, _scalp_profit_total
    
    if not SCALP_MODE or not SCALP_EXECUTE:
        return False
        
    if time.time() - _last_scalp_ts < SCALP_COOLDOWN_SEC:
        return False

    direction, reason = detect_super_scalp_opportunity(df, ind, flow, volume_profile, momentum, spread_bps)
    if direction is None:
        return False

    base_qty = compute_size(balance, px_now) * SCALP_SIZE_FACTOR
    volatility_factor = min(2.0, max(0.5, momentum['volatility'] / max(momentum['volatility_ma'], 1e-9)))
    smart_scalp_qty = base_qty * volatility_factor
    
    if smart_scalp_qty <= 0:
        log_w("SUPER SCALP: skip qty<=0")
        return False

    opened = open_market_enhanced(direction, smart_scalp_qty, px_now)
    if opened:
        _last_scalp_ts = time.time()
        STATE["mode"] = "super_scalp"
        
        STATE["scalp_multi_tp"] = True
        STATE["scalp_tp_levels"] = [0.15, 0.25, 0.35, 0.50]
        STATE["scalp_tp_weights"] = [0.3, 0.3, 0.25, 0.15]
        STATE["scalp_tp_achieved"] = [False, False, False, False]
        
        log_i(f"🔥 SUPER SCALP {direction.upper()} qty={smart_scalp_qty:.4f} px={px_now:.6f}")
        log_i(f"   Reason: {reason}")
        log_i(f"   Volatility Factor: {volatility_factor:.2f}")
        log_i(f"   Multi-TP: {STATE['scalp_tp_levels']}")
        
        try:
            print_position_snapshot(reason="SUPER_SCALP", color=("green" if direction=="buy" else "red"))
        except Exception as e:
            log_w(f"Snapshot error: {e}")
            
        return True
    
    return False

# =================== INTELLIGENT TREND RIDING SYSTEM - ENHANCED ===================
def manage_trend_ride_intelligently(df, ind, info, trend_strength):
    if not STATE["open"] or STATE["qty"] <= 0:
        return

    px = info["price"]
    entry = STATE["entry"]
    side = STATE["side"]
    qty = STATE["qty"]
    mode = STATE.get("mode", "trend")
    
    if mode != "trend":
        return
    
    pnl_pct = (px - entry) / entry * 100 * (1 if side == "long" else -1)
    STATE["pnl"] = pnl_pct
    
    if pnl_pct > STATE["highest_profit_pct"]:
        STATE["highest_profit_pct"] = pnl_pct

    current_trend_strength = compute_trend_strength(df, ind)
    trend_multiplier = current_trend_strength["multiplier"]
    
    # تعديل ديناميكي لأهداف الربح بناءً على قوة الترند
    dynamic_tp_levels = [tp * trend_multiplier for tp in TREND_TPS]
    dynamic_tp_fractions = [frac * (2.0 if trend_multiplier > 1.5 else 1.0) for frac in TREND_TP_FRACS]
    
    for i, (tp_level, tp_frac) in enumerate(zip(dynamic_tp_levels, dynamic_tp_fractions)):
        tp_key = f"tp_{i+1}_done"
        if not STATE.get(tp_key, False) and pnl_pct >= tp_level:
            close_qty = safe_qty(STATE["qty"] * tp_frac)
            if close_qty > 0:
                close_side = "sell" if STATE["side"] == "long" else "buy"
                if MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN:
                    try:
                        params = exchange_specific_params(close_side, is_close=True)
                        ex.create_order(SYMBOL, "market", close_side, close_qty, None, params)
                        log_g(f"🎯 TP{i+1} HIT: {tp_level:.2f}% | closed {tp_frac*100}% | Trend Strength: {current_trend_strength['strength']}")
                        STATE["profit_targets_achieved"] += 1
                    except Exception as e:
                        log_e(f"❌ TP{i+1} close failed: {e}")
                STATE["qty"] = safe_qty(STATE["qty"] - close_qty)
                STATE[tp_key] = True
                
                if current_trend_strength["strength"] in ["strong", "very_strong"] and i == len(dynamic_tp_levels) - 1:
                    log_i(f"💎 ترند قوي مستمر - الاحتفاظ بجزء من المركز للربح الإضافي")

    manage_intelligent_trailing_stop(px, side, ind, current_trend_strength)
    
    if TREND_REENTRY_STRATEGY and current_trend_strength["strength"] in ["strong", "very_strong"]:
        consider_trend_reentry(df, ind, px, side, current_trend_strength)

def manage_intelligent_trailing_stop(current_price, side, ind, trend_strength):
    if not STATE.get("trail_active", False):
        # تفعيل الوقف المتحرك عند تحقيق ربح معين
        if STATE.get("pnl", 0) >= TRAIL_ACTIVATE_PCT:
            STATE["trail_active"] = True
            STATE["breakeven_armed"] = True
            STATE["breakeven"] = STATE["entry"]
            log_i(f"🔄 Trail activated at {TRAIL_ACTIVATE_PCT}% profit")
        return
    
    atr = safe_get(ind, "atr", 0.0)
    pnl_pct = STATE.get("pnl", 0.0)
    
    # تكييف الوقف المتحرك مع قوة الترند
    if trend_strength["strength"] == "very_strong":
        trail_mult = ATR_TRAIL_MULT * 0.7
    elif trend_strength["strength"] == "strong":
        trail_mult = ATR_TRAIL_MULT * 0.8
    elif trend_strength["strength"] == "weak":
        trail_mult = ATR_TRAIL_MULT * 1.2
    else:
        trail_mult = ATR_TRAIL_MULT
    
    # تكييف إضافي بناءً على مستوى الربح
    if pnl_pct > 2.0:
        trail_mult *= 0.9
    elif pnl_pct > 1.0:
        trail_mult *= 0.95
    
    if side == "long":
        new_trail = current_price - (atr * trail_mult)
        if STATE.get("trail") is None or new_trail > STATE["trail"]:
            STATE["trail"] = new_trail
            if STATE["trail"] > STATE.get("entry", 0):
                log_i(f"🔼 وقف متحرك محدث: {STATE['trail']:.6f} (قوة الترند: {trend_strength['strength']})")
    else:
        new_trail = current_price + (atr * trail_mult)
        if STATE.get("trail") is None or new_trail < STATE["trail"]:
            STATE["trail"] = new_trail
            if STATE["trail"] < STATE.get("entry", float('inf')):
                log_i(f"🔽 وقف متحرك محدث: {STATE['trail']:.6f} (قوة الترند: {trend_strength['strength']})")
    
    if STATE.get("trail"):
        if (side == "long" and current_price <= STATE["trail"]) or (side == "short" and current_price >= STATE["trail"]):
            log_w(f"🛑 وقف متحرك: {current_price} vs trail {STATE['trail']}")
            close_market_strict("intelligent_trailing_stop")

def consider_trend_reentry(df, ind, current_price, current_side, trend_strength):
    if STATE["qty"] > FINAL_CHUNK_QTY * 2:
        return
    
    council_data = super_council_ai_enhanced(df)
    new_side = "buy" if council_data["score_b"] > council_data["score_s"] else "sell"
    
    if (new_side == current_side and 
        trend_strength["strength"] in ["strong", "very_strong"] and
        council_data["confidence"] > 0.7):
        
        reentry_qty = compute_size(balance_usdt(), current_price) * 0.3
        
        if reentry_qty > 0:
            log_i(f"🔄 إعادة دخول في الترند {current_side.upper()} | قوة: {trend_strength['strength']}")
            open_market_enhanced(new_side, reentry_qty, current_price)

# =================== EXECUTION MANAGER ===================
def execute_trade_decision(side, price, qty, mode, council_data, gz_data):
    if not EXECUTE_ORDERS or DRY_RUN:
        log_i(f"DRY_RUN: {side} {qty:.4f} @ {price:.6f} | mode={mode}")
        return True
    
    if qty <= 0:
        log_e("❌ كمية غير صالحة للتنفيذ")
        return False

    gz_note = ""
    if gz_data and gz_data.get("ok"):
        gz_note = f" | 🟡 {gz_data['zone']['type']} s={gz_data['score']:.1f}"
    
    votes = council_data
    print(f"🎯 EXECUTE: {side.upper()} {qty:.4f} @ {price:.6f} | "
          f"mode={mode} | votes={votes['b']}/{votes['s']} score={votes['score_b']:.1f}/{votes['score_s']:.1f}"
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
    if mode == "scalp":
        return {
            "tp1_pct": SCALP_TP_PCT,
            "be_activate_pct": SCALP_BE_AFTER_PCT,
            "trail_activate_pct": SCALP_TRAIL_START_PCT,
            "atr_trail_mult": SCALP_ATR_TRAIL_MULT,
            "close_aggression": "high"
        }
    else:
        return {
            "tp1_pct": TREND_TP1_PCT,
            "be_activate_pct": TREND_BE_AFTER_PCT,
            "trail_activate_pct": TREND_TRAIL_START_PCT,
            "atr_trail_mult": TREND_ATR_MULT,
            "close_aggression": "medium"
        }

# =================== ENHANCED TRADE EXECUTION ===================
def open_market_enhanced(side, qty, price):
    """نسخة محسنة من فتح الصفقة مع تصنيف الترند/السكالب"""
    if qty <= 0 or price is None:
        log_e("❌ invalid qty/price")
        return False

    # نجيب الداتا والمؤشرات قبل التنفيذ
    df = fetch_ohlcv(limit=200)
    ind = compute_indicators(df)

    # ✅ هنا نقرر: الصفقة دي ترند ولا سكالب؟
    mode_info = classify_trade_mode(df, ind)
    mode = mode_info["mode"]
    why_mode = mode_info["why"]

    log_i(f"🎛 TRADE MODE DECISION: {mode.upper()} | {why_mode}")

    # إعدادات الإدارة بناءً على المود
    management_config = setup_trade_management(mode)

    # تنفيذ الأوردر
    success = execute_trade_decision(side, price, qty, mode, 
                                   council_votes_pro_enhanced(df), 
                                   golden_zone_check(df, ind))

    if success:
        STATE.update({
            "open": True,
            "side": "long" if side.lower().startswith("b") else "short",
            "entry": float(price),
            "qty": float(qty),
            "pnl": 0.0,
            "bars": 0,
            "mode": mode,               # ✅ هنا البوت عرف الصفقة دي نوعها إيه
            "mode_why": why_mode,       # ✅ وليه اتخذ القرار ده
            "management": management_config,  # ✅ إعدادات مخصصة للنوع
            "opened_at": time.time(),
            "tp1_done": False,
            "trail_active": False,
            "breakeven_armed": False,
            "highest_profit_pct": 0.0,
            "profit_targets_achieved": 0
        })

        # حفظ السنابشوت
        save_state({
            "in_position": True,
            "side": "LONG" if side.upper().startswith("B") else "SHORT",
            "entry_price": price,
            "position_qty": qty,
            "leverage": LEVERAGE,
            "mode": mode,
            "mode_why": why_mode,
            "management": management_config,
            "opened_at": int(time.time())
        })

        log_g(f"✅ POSITION OPENED: {side.upper()} | mode={mode} | reason={why_mode}")
        print_position_snapshot(reason=f"OPEN - {mode.upper()}")
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
    global wait_for_next_signal_side
    wait_for_next_signal_side = "sell" if prev_side=="long" else ("buy" if prev_side=="short" else None)
    log_i(f"🛑 WAIT FOR NEXT SIGNAL: {wait_for_next_signal_side}")

def wait_gate_allow(df, info):
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
    global wait_for_next_signal_side
    prev_side = prev_side or STATE.get("side")
    STATE.update({
        "open": False, "side": None, "entry": None, "qty": 0.0,
        "pnl": 0.0, "bars": 0, "trail": None, "breakeven": None,
        "tp1_done": False, "highest_profit_pct": 0.0, "profit_targets_achieved": 0,
        "trail_tightened": False, "partial_taken": False
    })
    save_state({"in_position": False, "position_qty": 0})
    
    _arm_wait_after_close(prev_side)
    logging.info(f"AFTER_CLOSE waiting_for={wait_for_next_signal_side}")

# =================== ENHANCED TRADE MANAGEMENT ===================
def manage_after_entry_enhanced(df, ind, info):
    """إدارة محسنة للصفقات بناءً على نوعها"""
    if not STATE["open"] or STATE["qty"] <= 0:
        return

    px = info["price"]
    entry = STATE["entry"]
    side = STATE["side"]
    qty = STATE["qty"]
    mode = STATE.get("mode", "scalp")  # الإفتراضي سكالب

    pnl_pct = (px - entry) / entry * 100 * (1 if side == "long" else -1)
    STATE["pnl"] = pnl_pct

    if pnl_pct > STATE["highest_profit_pct"]:
        STATE["highest_profit_pct"] = pnl_pct

    # جلب إعدادات الإدارة من الـSTATE
    management = STATE.get("management", {})
    tp_target = management.get("tp1_pct", SCALP_TP_PCT) * 100
    be_after = management.get("be_activate_pct", SCALP_BE_AFTER_PCT) * 100
    trail_start = management.get("trail_activate_pct", SCALP_TRAIL_START_PCT) * 100

    # 1) جني ربح أولي
    if not STATE.get("tp1_done") and pnl_pct >= tp_target:
        close_qty = safe_qty(STATE["qty"] * 0.3)  # إغلاق 30% عند TP1
        if close_qty > 0:
            close_side = "sell" if STATE["side"] == "long" else "buy"
            if MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN:
                try:
                    params = exchange_specific_params(close_side, is_close=True)
                    ex.create_order(SYMBOL, "market", close_side, close_qty, None, params)
                    log_g(f"💰 TP1 HIT ({mode}) pnl={pnl_pct:.2f}% | closed 30%")
                    STATE["profit_targets_achieved"] += 1
                except Exception as e:
                    log_e(f"❌ TP1 close failed: {e}")
            STATE["qty"] = safe_qty(STATE["qty"] - close_qty)
            STATE["tp1_done"] = True

    # 2) تفعيل نقطة التعادل
    if not STATE.get("breakeven_armed") and pnl_pct >= be_after:
        STATE["breakeven_armed"] = True
        STATE["breakeven"] = entry
        log_i(f"🛡️ BE ARMED ({mode}) at {pnl_pct:.2f}%")

    # 3) تفعيل الوقف المتحرك
    if not STATE.get("trail_active") and pnl_pct >= trail_start:
        STATE["trail_active"] = True
        log_i(f"📈 TRAIL ACTIVE ({mode}) at {pnl_pct:.2f}%")

    # إدارة متقدمة بناءً على النوع
    if mode == "trend":
        trend_strength = compute_trend_strength(df, ind)
        manage_trend_ride_intelligently(df, ind, info, trend_strength)
    else:
        manage_scalp_trade(df, ind, info)

    # تحديث السجل
    STATE["bars"] += 1

manage_after_entry = manage_after_entry_enhanced

def manage_scalp_trade(df, ind, info):
    px = info["price"]
    entry = STATE["entry"]
    side = STATE["side"]
    qty = STATE["qty"]
    
    pnl_pct = (px - entry) / entry * 100 * (1 if side == "long" else -1)
    STATE["pnl"] = pnl_pct
    
    if pnl_pct > STATE["highest_profit_pct"]:
        STATE["highest_profit_pct"] = pnl_pct

    if STATE.get("scalp_multi_tp", False):
        for i, (tp_level, tp_weight) in enumerate(zip(STATE["scalp_tp_levels"], STATE["scalp_tp_weights"])):
            if not STATE["scalp_tp_achieved"][i] and pnl_pct >= tp_level:
                close_qty = safe_qty(STATE["qty"] * tp_weight)
                if close_qty > 0:
                    close_side = "sell" if STATE["side"] == "long" else "buy"
                    if MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN:
                        try:
                            params = exchange_specific_params(close_side, is_close=True)
                            ex.create_order(SYMBOL, "market", close_side, close_qty, None, params)
                            log_g(f"🎯 SCALP TP{i+1} HIT: {tp_level:.2f}% | closed {tp_weight*100}%")
                            STATE["profit_targets_achieved"] += 1
                        except Exception as e:
                            log_e(f"❌ SCALP TP{i+1} close failed: {e}")
                    STATE["qty"] = safe_qty(STATE["qty"] - close_qty)
                    STATE["scalp_tp_achieved"][i] = True

    manage_scalp_trailing_stop(px, side, ind)

def manage_scalp_trailing_stop(current_price, side, ind):
    if not STATE.get("trail_active", False):
        if STATE.get("pnl", 0) >= SCALP_BE_AFTER_PCT:
            STATE["trail_active"] = True
            STATE["breakeven_armed"] = True
            STATE["breakeven"] = STATE["entry"]
            log_i("SCALP: Breakeven armed & Trail activated")

    if STATE.get("trail_active"):
        atr = safe_get(ind, "atr", 0.0)
        trail_mult = SCALP_ATR_TRAIL_MULT
        
        if side == "long":
            new_trail = current_price - (atr * trail_mult)
            if STATE.get("trail") is None or new_trail > STATE["trail"]:
                STATE["trail"] = new_trail
        else:
            new_trail = current_price + (atr * trail_mult)
            if STATE.get("trail") is None or new_trail < STATE["trail"]:
                STATE["trail"] = new_trail

        if STATE.get("trail"):
            if (side == "long" and current_price <= STATE["trail"]) or (side == "short" and current_price >= STATE["trail"]):
                log_w(f"SCALP TRAIL STOP: {current_price} vs trail {STATE['trail']}")
                close_market_strict("scalp_trailing_stop")

def smart_exit_guard(state, df, ind, flow, bm, now_price, pnl_pct, mode, side, entry_price, gz=None):
    atr = safe_get(ind, 'atr', 0.0)
    adx = safe_get(ind, 'adx', 0.0)
    rsi = safe_get(ind, 'rsi', 50.0)
    rsi_ma = safe_get(ind, 'rsi_ma', 50.0)
    
    if len(df) >= 3:
        adx_slope = adx - safe_get(ind, 'adx_prev', adx)
    else:
        adx_slope = 0.0

    wick_signal = False
    if len(df) > 0:
        c = df.iloc[-1]
        wick_up = float(c['high']) - max(float(c['close']), float(c['open']))
        wick_down = min(float(c['close']), float(c['open'])) - float(c['low'])
        wick_signal = (wick_up >= WICK_ATR_MULT * atr) if side == "long" else (wick_down >= WICK_ATR_MULT * atr)

    rsi_cross_down = (rsi < rsi_ma) if side == "long" else (rsi > rsi_ma)
    adx_falling = (adx_slope < 0)
    cvd_down = (flow and flow.get('ok') and flow.get('cvd_trend') == 'down')
    evx_spike = False
    
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

    if state.get('tp1_done') and (gz and gz.get('ok')):
        opp = (gz['zone']['type']=='golden_top' and side=='long') or (gz['zone']['type']=='golden_bottom' and side=='short')
        if opp and gz.get('score',0) >= GOLDEN_REVERSAL_SCORE:
            return {
                "action": "close", 
                "why": "golden_reversal",
                "log": f"🔴 CLOSE STRONG | golden reversal after TP1 | score={gz['score']:.1f}"
            }

    tp1_target = TP1_SCALP_PCT if mode == 'scalp' else TP1_TREND_PCT
    if pnl_pct >= tp1_target and not state.get('tp1_done'):
        qty_pct = 0.35 if mode == 'scalp' else 0.25
        return {
            "action": "partial", 
            "why": f"TP1 hit {tp1_target*100:.2f}%",
            "qty_pct": qty_pct,
            "log": f"💰 TP1 جزئي {tp1_target*100:.2f}% | pnl={pnl_pct*100:.2f}% | mode={mode}"
        }

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

# ============================================
#  ENHANCED TRADE LOOP WITH SMART PATCH
# ============================================

def trade_loop_enhanced_with_smart_patch():
    global wait_for_next_signal_side, compound_pnl
    loop_i = 0
    
    # إحصائيات الأداء
    performance_stats = {
        'total_trades': 0,
        'winning_trades': 0,
        'total_profit': 0.0,
        'consecutive_wins': 0,
        'consecutive_losses': 0
    }
    
    while True:
        try:
            current_time = time.time()
            bal = balance_usdt()
            px = price_now()
            df = fetch_ohlcv()
            
            if df.empty:
                time.sleep(BASE_SLEEP)
                continue
                
            # ✅ إضافة نظام جني الأرباح الذكي
            if STATE.get("open") and px:
                apply_smart_profit_strategy()
                
            # تحديث جميع المحركات الذكية
            close_prices = df['close'].astype(float).tolist()
            volumes = df['volume'].astype(float).tolist()
            
            # تحديث السياق
            trend_ctx.update(close_prices[-1] if close_prices else 0)
            smc_detector.detect_swings(df)
            
            info = rf_signal_live(df)
            ind = compute_indicators(df)
            spread_bps = orderbook_spread_bps()
            
            # تحديث orderbook للـFlow Boost
            try:
                STATE["last_orderbook"] = ex.fetch_order_book(SYMBOL, limit=FLOW_STACK_DEPTH)
            except Exception as e:
                log_w(f"Orderbook update failed: {e}")
            
            snap = emit_snapshots(ex, SYMBOL, df,
                                balance_fn=lambda: float(bal) if bal else None,
                                pnl_fn=lambda: float(compound_pnl))
            
            if STATE["open"] and px:
                STATE["pnl"] = (px-STATE["entry"])*STATE["qty"] if STATE["side"]=="long" else (STATE["entry"]-px)*STATE["qty"]
            
            # ============================================
            #  SMART DECISION INTELLIGENCE BLOCK
            # ============================================
            
            entry_reasons = []
            allow_buy = False
            allow_sell = False
            
            close_price = float(df['close'].iloc[-1]) if len(df) > 0 else px
            
            # ---- Volume Confirmation ----
            vol_ok = volume_is_strong(volumes)
            
            # ---- OB / FVG Detection ----
            ob_signal = detect_ob(df)
            fvg_signal = detect_fvg(df)
            
            # ---- Golden Zones ----
            golden_data = golden_zone_check(df, ind)
            gb = golden_data.get("ok", False) and golden_data.get("zone", {}).get("type") == "golden_bottom"
            gt = golden_data.get("ok", False) and golden_data.get("zone", {}).get("type") == "golden_top"
            
            # ---- SMC Liquidity Analysis ----
            liquidity_zones = smc_detector.detect_liquidity_zones(close_price)
            buy_liquidity = any(zone[0] == "buy_liquidity" for zone in liquidity_zones)
            sell_liquidity = any(zone[0] == "sell_liquidity" for zone in liquidity_zones)
            
            # ---- ADX Gate ----
            adx_ok = safe_get(ind, "adx", 0) >= ADX_GATE
            
            # ---- Zero Reversal Scalping Check ----
            scalper_ready, scalper_reason = zero_scalper.can_trade(current_time)
            
            # ===== BUY CONDITIONS =====
            buy_conditions = []
            
            # Golden Bottom
            if gb and trend_ctx.trend != "down" and adx_ok:
                allow_buy = True
                buy_conditions.append("Golden Bottom")
            
            # Bullish FVG
            if fvg_signal and fvg_signal[0] == "bullish":
                allow_buy = True
                buy_conditions.append("Bullish FVG")
            
            # Bullish OB
            if ob_signal and ob_signal[0] == "bullish":
                allow_buy = True
                buy_conditions.append("Bullish OB")
            
            # Buy Liquidity
            if buy_liquidity and vol_ok:
                allow_buy = True
                buy_conditions.append("Buy Liquidity Zone")
            
            # ===== SELL CONDITIONS =====
            sell_conditions = []
            
            # Golden Top
            if gt and trend_ctx.trend != "up" and adx_ok:
                allow_sell = True
                sell_conditions.append("Golden Top")
            
            # Bearish FVG
            if fvg_signal and fvg_signal[0] == "bearish":
                allow_sell = True
                sell_conditions.append("Bearish FVG")
            
            # Bearish OB
            if ob_signal and ob_signal[0] == "bearish":
                allow_sell = True
                sell_conditions.append("Bearish OB")
            
            # Sell Liquidity
            if sell_liquidity and vol_ok:
                allow_sell = True
                sell_conditions.append("Sell Liquidity Zone")
            
            # ---- Volume Final Gate ----
            if not vol_ok:
                allow_buy = False
                allow_sell = False
                entry_reasons.append("Weak Volume - Blocked")
            else:
                entry_reasons.extend(buy_conditions)
                entry_reasons.extend(sell_conditions)
            
            # ---- Scalper Ready Check ----
            if not scalper_ready and SCALP_MODE:
                allow_buy = allow_buy and False
                allow_sell = allow_sell and False
                entry_reasons.append(f"Scalper Cooldown: {scalper_reason}")
            
            # ---- RF Signal Integration ----
            rf_buy = info.get("long", False)
            rf_sell = info.get("short", False)
            
            # ---- Missed Signals Logging ----
            if rf_buy and not allow_buy and not STATE["open"]:
                signal_logger.log_missed_signal("BUY", close_price, " | ".join(entry_reasons))
                
            if rf_sell and not allow_sell and not STATE["open"]:
                signal_logger.log_missed_signal("SELL", close_price, " | ".join(entry_reasons))
            
            # ============================================
            #  FINAL ENTRY EXECUTION LAYER
            # ============================================
            
            council_data = council_votes_pro_enhanced(df)
            final_signal = None
            
            if not STATE["open"]:
                # SUPER SCALP Execution
                if SCALP_MODE and SCALP_EXECUTE:
                    flow_ctx = compute_flow_metrics(df)
                    volume_profile = compute_volume_profile(df)
                    momentum = compute_momentum_indicators(df)
                    
                    if execute_super_scalp(px, bal, df, ind, flow_ctx, volume_profile, momentum, spread_bps):
                        continue
                
                # COUNCIL + SMART PATCH Decision
                if rf_buy and allow_buy:
                    final_signal = "buy"
                elif rf_sell and allow_sell:
                    final_signal = "sell"
                
                if final_signal and not STATE["open"]:
                    allow_wait, wait_reason = wait_gate_allow(df, info)
                    if not allow_wait:
                        log_i(f"⏳ Waiting: {wait_reason}")
                    else:
                        qty = compute_size(bal, px or info["price"])
                        if qty > 0:
                            ok = open_market(final_signal, qty, px or info["price"])
                            if ok:
                                wait_for_next_signal_side = None
                                log_i(f"🎯 SMART EXECUTION: {final_signal.upper()} | Reasons: {' | '.join(entry_reasons)}")
                                log_i(f"   Trend: {trend_ctx.trend} | Strength: {trend_ctx.strength:.2f}% | Volume: {'Strong' if vol_ok else 'Weak'}")
                                
                                # تسجيل تداول السكالب
                                if SCALP_MODE:
                                    zero_scalper.record_trade(current_time, True)
                        else:
                            log_w("❌ Quantity <= 0")
            
            # إدارة الصفقة المفتوحة
            if STATE["open"]:
                manage_after_entry_enhanced_with_smart_patch(df, ind, {
                    "price": px or info["price"], 
                    "bm": snap["bm"],
                    "flow": snap["flow"],
                    "trend_ctx": trend_ctx,
                    "vol_ok": vol_ok,
                    **info
                }, performance_stats)
            
            # Legacy Logging
            if LOG_LEGACY:
                pretty_snapshot(bal, {"price": px or info["price"], **info}, ind, spread_bps, " | ".join(entry_reasons), df)
            
            loop_i += 1
            sleep_s = NEAR_CLOSE_S if time_to_candle_close(df) <= 10 else BASE_SLEEP
            time.sleep(sleep_s)
            
        except Exception as e:
            log_e(f"Smart loop error: {e}\n{traceback.format_exc()}")
            time.sleep(BASE_SLEEP)

# استبدال الدورة الرئيسية
trade_loop = trade_loop_enhanced_with_smart_patch

# ============================================
#  SMART TRADE MANAGEMENT
# ============================================

def manage_after_entry_enhanced_with_smart_patch(df, ind, info, performance_stats):
    if not STATE["open"] or STATE["qty"] <= 0:
        return

    px = info["price"]
    entry = STATE["entry"]
    side = STATE["side"]
    qty = STATE["qty"]
    mode = STATE.get("mode", "trend")
    
    pnl_pct = (px - entry) / entry * 100 * (1 if side == "long" else -1)
    STATE["pnl"] = pnl_pct
    
    if pnl_pct > STATE["highest_profit_pct"]:
        STATE["highest_profit_pct"] = pnl_pct

    # ============================================
    #  SMART EXIT ENGINE
    # ============================================
    
    trend_ctx = info.get("trend_ctx", SmartTrendContext())
    vol_ok = info.get("vol_ok", False)
    
    # ---- حالة الترند القوي ----
    if trend_ctx.is_strong_trend() and mode == "trend":
        if not STATE.get("trail_tightened", False):
            # تشديد الوقف المتحرك في الترند القوي
            STATE["trail_tightened"] = True
            if "management" in STATE:
                STATE["management"]["atr_trail_mult"] *= 0.7  # تشديد الوقف
            log_i("📌 Strong Trend → Tightened Trail")
    
    # ---- كشف شمعة الانعكاس ----
    candles = compute_candles(df)
    reversal_candle = False
    if side == "long" and (candles.get("wick_up_big") or candles.get("score_sell", 0) > 2.0):
        reversal_candle = True
    elif side == "short" and (candles.get("wick_dn_big") or candles.get("score_buy", 0) > 2.0):
        reversal_candle = True
    
    if reversal_candle and pnl_pct > 0.5:
        # خروج جزئي عند شمعة انعكاس
        close_qty = safe_qty(STATE["qty"] * 0.3)
        if close_qty > 0:
            close_side = "sell" if STATE["side"] == "long" else "buy"
            if MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN:
                try:
                    params = exchange_specific_params(close_side, is_close=True)
                    ex.create_order(SYMBOL, "market", close_side, close_qty, None, params)
                    log_g(f"🕯️ Reversal Candle → Partial Exit 30% | PnL: {pnl_pct:.2f}%")
                    STATE["qty"] = safe_qty(STATE["qty"] - close_qty)
                    performance_stats['total_profit'] += (close_qty * abs(px - entry))
                except Exception as e:
                    log_e(f"❌ Reversal partial close failed: {e}")
    
    # ---- خروج عند ضعف الحجم ----
    if not vol_ok and pnl_pct > 0.3 and mode == "scalp":
        log_i("⛔ Weak Volume + Profit → Closing Position")
        close_market_strict("weak_volume_profit")
        performance_stats['total_trades'] += 1
        performance_stats['winning_trades'] += 1
        return
    
    # ---- حماية الأرباح الكبيرة ----
    if pnl_pct > 2.0 and mode == "trend":
        if not STATE.get("big_profit_protected", False):
            STATE["big_profit_protected"] = True
            # نقل الوقف إلى نقطة التعادل + ربح بسيط
            breakeven_plus = entry * (1.01 if side == "long" else 0.99)
            STATE["breakeven"] = breakeven_plus
            log_i(f"💰 Big Profit Protection → Breakeven+1%: {breakeven_plus:.6f}")
    
    # ---- إدارة الصفقات العادية ----
    if mode == "trend":
        trend_strength = compute_trend_strength(df, ind)
        manage_trend_ride_intelligently(df, ind, info, trend_strength)
    else:
        manage_scalp_trade(df, ind, info)

# =================== LOOP / LOG ===================
def pretty_snapshot(bal, info, ind, spread_bps, reason=None, df=None):
    if LOG_LEGACY:
        left_s = time_to_candle_close(df) if df is not None else 0
        print(colored("─"*100,"cyan"))
        print(colored(f"📊 {SYMBOL} {INTERVAL} • {EXCHANGE_NAME.upper()} • {'LIVE' if MODE_LIVE else 'PAPER'} • {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC","cyan"))
        print(colored("─"*100,"cyan"))
        print("📈 INDICATORS & RF")
        print(f"   💲 Price {fmt(info.get('price'))} | RF filt={fmt(info.get('filter'))}  hi={fmt(info.get('hi'))} lo={fmt(info.get('lo'))}")
        print(f"   🧮 RSI={fmt(safe_get(ind, 'rsi'))}  +DI={fmt(safe_get(ind, 'plus_di'))}  -DI={fmt(safe_get(ind, 'minus_di'))}  ADX={fmt(safe_get(ind, 'adx'))}  ATR={fmt(safe_get(ind, 'atr'))}")
        print(f"   🎯 ENTRY: SUPER COUNCIL AI + GOLDEN ENTRY + SUPER SCALP + SMART PROFIT AI |  spread_bps={fmt(spread_bps,2)}")
        print(f"   ⏱️ closes_in ≈ {left_s}s")
        print("\n🧭 POSITION")
        bal_line = f"Balance={fmt(bal,2)}  Risk={int(RISK_ALLOC*100)}%×{LEVERAGE}x  CompoundPnL={fmt(compound_pnl)}  Eq~{fmt((bal or 0)+compound_pnl,2)}"
        print(colored(f"   {bal_line}", "yellow"))
        if STATE["open"]:
            lamp='🟩 LONG' if STATE['side']=='long' else '🟥 SHORT'
            print(f"   {lamp}  Entry={fmt(STATE['entry'])}  Qty={fmt(STATE['qty'],4)}  Bars={STATE['bars']}  Trail={fmt(STATE['trail'])}  BE={fmt(STATE['breakeven'])}")
            print(f"   🎯 TP_done={STATE['profit_targets_achieved']}  HP={fmt(STATE['highest_profit_pct'],2)}%")
            print(f"   🎯 MODE={STATE.get('mode', 'trend')}  Multi-TP={STATE.get('scalp_multi_tp', False)}")
        else:
            print("   ⚪ FLAT")
            if wait_for_next_signal_side:
                print(colored(f"   ⏳ Waiting for opposite RF: {wait_for_next_signal_side.upper()}", "cyan"))
        if reason: print(colored(f"   ℹ️ reason: {reason}", "white"))
        print(colored("─"*100,"cyan"))

# =================== API / KEEPALIVE ===================
app = Flask(__name__)

@app.get("/mark/<color>")
def mark_position(color):
    color = color.lower()
    if color not in ["green", "red"]:
        return jsonify({"ok": False, "error": "Use /mark/green or /mark/red"}), 400
    
    print_position_snapshot(reason="MANUAL_MARK", color=color)
    return jsonify({"ok": True, "marked": color, "timestamp": datetime.utcnow().isoformat()})

@app.route("/")
def home():
    mode='LIVE' if MODE_LIVE else 'PAPER'
    return f"✅ SUI ULTRA PRO AI Bot — {EXCHANGE_NAME.upper()} — {SYMBOL} {INTERVAL} — {mode} — Super Council AI + Intelligent Trend Riding + Smart Profit AI"

@app.route("/metrics")
def metrics():
    return jsonify({
        "exchange": EXCHANGE_NAME,
        "symbol": SYMBOL, "interval": INTERVAL, "mode": "live" if MODE_LIVE else "paper",
        "leverage": LEVERAGE, "risk_alloc": RISK_ALLOC, "price": price_now(),
        "state": STATE, "compound_pnl": compound_pnl,
        "entry_mode": "SUPER_COUNCIL_AI_GOLDEN_SCALP_SMART_PROFIT", "wait_for_next_signal": wait_for_next_signal_side,
        "guards": {"max_spread_bps": MAX_SPREAD_BPS, "final_chunk_qty": FINAL_CHUNK_QTY},
        "scalp_mode": SCALP_MODE,
        "super_council_ai": COUNCIL_AI_MODE,
        "intelligent_trend_riding": TREND_RIDING_AI,
        "smart_profit_ai": True
    })

@app.route("/health")
def health():
    return jsonify({
        "ok": True, "exchange": EXCHANGE_NAME, "mode": "live" if MODE_LIVE else "paper",
        "open": STATE["open"], "side": STATE["side"], "qty": STATE["qty"],
        "compound_pnl": compound_pnl, "timestamp": datetime.utcnow().isoformat(),
        "entry_mode": "SUPER_COUNCIL_AI_GOLDEN_SCALP_SMART_PROFIT", "wait_for_next_signal": wait_for_next_signal_side,
        "scalp_mode": SCALP_MODE,
        "super_council_ai": COUNCIL_AI_MODE,
        "smart_profit_ai": True
    }), 200

# ============================================
#  API ENDPOINTS للإحصائيات الذكية
# ============================================

@app.route("/smart_stats")
def smart_stats():
    missed_signals = signal_logger.get_recent_missed(10)
    liquidity_zones = smc_detector.detect_liquidity_zones(price_now() or 0)
    
    return jsonify({
        "trend_context": {
            "trend": trend_ctx.trend,
            "strength": trend_ctx.strength,
            "momentum": trend_ctx.momentum
        },
        "liquidity_zones": liquidity_zones,
        "missed_signals": missed_signals,
        "scalper_status": {
            "consecutive_wins": zero_scalper.consecutive_wins,
            "consecutive_losses": zero_scalper.consecutive_losses,
            "cooldown_until": zero_scalper.cooldown_until
        },
        "smart_profit_ai": {
            "active": True,
            "version": "2.0",
            "features": ["scalp_profits", "trend_riding", "volume_analysis"]
        }
    })

@app.route("/market_context")
def market_context():
    df = fetch_ohlcv(limit=100)
    current_price = price_now()
    
    ob = detect_ob(df)
    fvg = detect_fvg(df)
    golden = golden_zone_check(df)
    liquidity = smc_detector.detect_liquidity_zones(current_price or 0)
    
    return jsonify({
        "order_block": ob,
        "fair_value_gap": fvg,
        "golden_zone": golden,
        "liquidity_zones": liquidity,
        "current_price": current_price,
        "timestamp": datetime.utcnow().isoformat()
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
    print(f"⚙️ EXECUTION ENVIRONMENT", flush=True)
    print(f"🔧 EXCHANGE: {EXCHANGE_NAME.upper()} | SYMBOL: {SYMBOL}", flush=True)
    print(f"🔧 EXECUTE_ORDERS: {EXECUTE_ORDERS} | DRY_RUN: {DRY_RUN}", flush=True)
    print(f"🎯 GOLDEN ENTRY: score={GOLDEN_ENTRY_SCORE} | ADX={GOLDEN_ENTRY_ADX}", flush=True)
    print(f"🚀 SMART PATCH: OB/FVG + SMC + Golden Zones + Volume Confirmation + SMART PROFIT AI", flush=True)
    print(f"🧠 SMART PROFIT AI: Scalp + Trend + Volume Analysis Activated", flush=True)

if __name__ == "__main__":
    verify_execution_environment()
    
    import threading
    threading.Thread(target=keepalive_loop, daemon=True).start()
    threading.Thread(target=trade_loop, daemon=True).start()
    
    log_i(f"🚀 SUI ULTRA PRO AI BOT STARTED - {BOT_VERSION}")
    log_i(f"🎯 SYMBOL: {SYMBOL} | INTERVAL: {INTERVAL} | LEVERAGE: {LEVERAGE}x")
    log_i(f"💡 SMART PATCH ACTIVATED: Golden Zones + SMC + OB/FVG + Zero Reversal Scalping + SMART PROFIT AI")
    
    app.run(host="0.0.0.0", port=PORT, debug=False)
