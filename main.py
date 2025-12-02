# -*- coding: utf-8 -*-
"""
ULTRA PRO AI BOT - الإصدار المتكامل الذكي المحسن
• نظام كشف وتصنيف مناطق ضرب الستوبات (Stop Hunt Zones)
• تمييز FVG الحقيقي من الوهمي + كشف مصائد السيولة  
• مجلس الإدارة الفائق الذكي مع 20 استراتيجية متقدمة
• نظام ركوب الترند الذكي المحترف + RF الحقيقي
• إدارة صفقات ذكية متكيفة مع قوة الترند + Edge Algo
• Multi-Exchange Support: BingX & Bybit
• SMART PROFIT AI - نظام جني الأرباح الذكي المتقدم
• STOP HUNT DETECTION - كشف واستغلال مناطق ضرب الستوبات
• BOX REJECTION ENGINE + SMC CONTEXT + GOLDEN ZONES
• TRAP MODE - استغلال مناطق ضرب الستوبات بذكاء
• STOP-HUNT PREDICTION ENGINE - توقع مناطق ضرب الستوبات القادمة
• TRADE PROFILE SYSTEM - 3 أنواع صفقات + TP/SL ديناميكي
• TRAP OVERRIDE ENGINE - دخول قسري في فرص الستوب هانت
• EQUITY TRACKING - تتبع الربح التراكمي والرصيد
• WEB SERVICE - واجهة ويب للرصد والإدارة
• ULTRA PANEL - نظام لوج محترف بالشكل المطلوب
• ADX+ATR FILTER - فلتر ذكي لمنع الدخول في ترند مجنون
• RSI Wilder's RMA + ADX Wilder's RMA - متطابق مع TradingView
"""

import os
import time
import math
import random
import traceback
import logging
import json
import pandas as pd
import numpy as np
import ccxt
from datetime import datetime
from decimal import Decimal, ROUND_DOWN
from collections import deque
from typing import Literal, Dict, Any, Optional, Tuple
from flask import Flask, jsonify
import threading
import sys
import signal
from termcolor import colored

Side = Literal["BUY", "SELL"]

# ============================================
#  CONFIGURATION
# ============================================

# Exchange Configuration
EXCHANGE_NAME = os.getenv("EXCHANGE", "bingx").lower()
API_KEY = os.getenv("BINGX_API_KEY" if EXCHANGE_NAME == "bingx" else "BYBIT_API_KEY", "")
API_SECRET = os.getenv("BINGX_API_SECRET" if EXCHANGE_NAME == "bingx" else "BYBIT_API_SECRET", "")

# Trading Configuration
SYMBOL = os.getenv("SYMBOL", "SUI/USDT:USDT")
INTERVAL = os.getenv("INTERVAL", "15m")
LEVERAGE = 10
RISK_ALLOC = 0.60
POSITION_MODE = os.getenv("POSITION_MODE", "oneway")

# Mode Configuration
MODE_LIVE = bool(API_KEY and API_SECRET)
EXECUTE_ORDERS = True
DRY_RUN = False
LOG_LEVEL = "INFO"

# Web Service Configuration
PORT = int(os.getenv("PORT", "5000"))

# Bot Version
BOT_VERSION = f"ULTRA PRO AI v12.0 - WEB SERVICE EDITION - {EXCHANGE_NAME.upper()}"

print(f"🚀 Booting: {BOT_VERSION}", flush=True)

# ============================================
#  PROFIT PROFILES DEFINITION
# ============================================

PROFIT_PROFILES = {
    "SCALP_STRICT": {
        "tp_levels_rr": [1.0],        # TP واحد عند 1R
        "tp_fracs":     [1.0],        # يقفل كل الكمية
        "hard_sl_rr":   -0.6,         # ستوب ثابت -0.6R
        "be_after_tp":  True,         # مفيش معنى هنا لكن نخليه True
        "trail_start_rr": None,       # بدون تريل
        "trail_atr_mult": None,
    },
    "MID_TREND": {
        "tp_levels_rr": [1.0, 2.0],   # TP1=1R, TP2=2R
        "tp_fracs":     [0.6, 0.4],   # 60% ثم 40%
        "hard_sl_rr":   -0.7,
        "be_after_tp":  True,         # بعد TP1 انقل BE
        "trail_start_rr": 1.8,        # فعّل تريل بعد ما تعدي 1.8R
        "trail_atr_mult": 1.0,        # تريل ATR خفيف
    },
    "FULL_TREND": {
        "tp_levels_rr": [0.8, 1.8, 3.0],
        "tp_fracs":     [0.3, 0.3, 0.4],
        "hard_sl_rr":   -0.8,
        "be_after_tp":  True,         # بعد TP1
        "trail_start_rr": 1.5,        # تريل بدري شوية
        "trail_atr_mult": 1.5,        # تريل أوسع لركوب الترند
    },
    "TRAP_TREND": {
        # Stop-Hunt مع الترند: ناخد ربح محترم بس مش نطمع قوي
        "tp_levels_rr": [1.2, 2.0],
        "tp_fracs":     [0.7, 0.3],
        "hard_sl_rr":   -0.7,
        "be_after_tp":  True,
        "trail_start_rr": 2.0,
        "trail_atr_mult": 1.2,
    },
}

def select_profit_profile(trade_mode, analysis):
    """اختيار الـ Profile المناسب بناءً على تحليل السوق"""
    rr = float(analysis.get("edge_rr", 1.0))
    if analysis.get("edge_setup") and analysis["edge_setup"].get("valid"):
        rr = float(analysis["edge_setup"].get("rr1", 1.0))
    
    adx = float(analysis.get("trend", {}).get("adx", 0.0))
    conf = float(analysis.get("confidence", 0.0))
    stop_q = float(analysis.get("stop_hunt_trap_quality", 0.0))
    golden = analysis.get("golden_zone", {}).get("type")

    # 1) صفقات Trap مع ترند + Stop-Hunt قوي
    if trade_mode == "TRAP" and stop_q >= 3.0:
        return "TRAP_TREND"

    # 2) Golden / Trend قوي / RR عالي ⇒ ترند كامل
    if golden in ("golden_bottom", "golden_top") or adx >= 28 or rr >= 2.0 or conf >= 7.0:
        return "FULL_TREND"

    # 3) صفقات عادية RR متوسط
    if rr >= 1.3 and (18 <= adx <= 28 or conf >= 5.0):
        return "MID_TREND"

    # 4) الباقي ⇒ SCALP_STRICT
    return "SCALP_STRICT"

# ============================================
#  COLORED LOGGING SYSTEM
# ============================================

class ColorLogger:
    """نظام اللوج الملوّن المحترف"""
    
    @staticmethod
    def info(msg: str):
        print(colored(msg, "cyan"))

    @staticmethod
    def success(msg: str):
        print(colored(msg, "green"))

    @staticmethod
    def warning(msg: str):
        print(colored(msg, "yellow"))

    @staticmethod
    def error(msg: str):
        print(colored(msg, "red"))

    @staticmethod
    def critical(msg: str):
        print(colored(msg, "magenta", attrs=["bold"]))

log_i = ColorLogger.info
log_g = ColorLogger.success
log_w = ColorLogger.warning
log_e = ColorLogger.error
log_r = ColorLogger.critical

def log_equity_snapshot(balance_usdt: float, compound_pnl: float):
    """لوج موحَّد يوضح الرصيد والربح التراكمي"""
    log_i(
        f"💼 BALANCE SNAPSHOT | "
        f"Balance: {balance_usdt:.2f} USDT  | "
        f"👑 CumPnL: {compound_pnl:.2f} USDT"
    )

# ============================================
#  ULTRA PANEL SYSTEM
# ============================================

def log_ultra_panel(analysis: dict, state: dict):
    """
    يطبع بلوك لوج كامل في كل تيك: Bookmap / Flow / Council / Strategy / SMC / SNAP / Footprint...
    analysis: dict راجع من مجلس الإدارة
    state:   حالة البوت (رصيد، compound_pnl، وضع الصفقة...)
    """
    a = analysis or {}

    # قيم افتراضية عشان ما يضربش لو حاجة ناقصة
    trend     = a.get("trend", {})
    smc_ctx   = a.get("smc_ctx", {})
    fvg_ctx   = a.get("fvg_analysis", {})
    edge      = a.get("edge_setup", {})
    rf_ctx    = a.get("rf", {})
    stop_hunt = a.get("predicted_stop_hunt", {})

    balance        = state.get("balance", 0.0)
    compound_pnl   = state.get("compound_pnl", 0.0)
    mode           = "LIVE" if MODE_LIVE else "PAPER"

    # 1) Bookmap / Imbalance
    log_i(
        f"📊 Bookmap: "
        f"Imb={rf_ctx.get('imbalance', 0):.2f} | "
        f"Buy[{rf_ctx.get('buy_wall', 0)}] | "
        f"Sell[{rf_ctx.get('sell_wall', 0)}]"
    )

    # 2) Flow (OBI / Delta / CVD)
    flow_side = "NEUTRAL"
    if a.get("score_buy", 0) > a.get("score_sell", 0) + 2:
        flow_side = "BUY"
    elif a.get("score_sell", 0) > a.get("score_buy", 0) + 2:
        flow_side = "SELL"
    
    log_i(
        f"🌊 Flow: {flow_side} "
        f"Δ={a.get('score_buy', 0)-a.get('score_sell', 0):.1f} | "
        f"Conf={a.get('confidence', 0):.2f}"
    )

    # 3) Council summary (BUY/SELL hint) مع RSI + ADX Wilder
    hint_side = "NEUTRAL"
    if a.get("score_buy", 0) > a.get("score_sell", 0):
        hint_side = "BUY"
    elif a.get("score_sell", 0) > a.get("score_buy", 0):
        hint_side = "SELL"
    
    # عرض RSI وADX Wilder معاً (متطابق مع TradingView)
    log_i(
        f"📌 DASH → hint-{hint_side} | "
        f"Council BUY({a.get('score_buy',0):.1f}) "
        f"SELL({a.get('score_sell',0):.1f}) | "
        f"RSI={a.get('rsi',0):.1f} "
        f"RSI_MA={a.get('rsi_ma',0):.1f} "
        f"Zone={a.get('rsi_zone','?')} | "
        f"ADX={trend.get('adx', 0):.1f} "
        f"DI+={trend.get('di_plus', 0):.1f} DI-={trend.get('di_minus', 0):.1f}"
    )

    # 4) Strategy + Balance
    strategy_label = "SCALP"
    if edge and edge.get("grade"):
        strategy_label = edge.get("grade", "MID").upper()
    
    log_i(
        f"⚡ Strategy: {strategy_label} | "
        f"Balance={balance:.2f} | CompoundPnL={compound_pnl:.4f} | Mode={mode}"
    )

    # 5) SMC BEST
    smc_label = "order_block_entry"
    if smc_ctx.get("supply_box"):
        smc_label = "supply_box"
    elif smc_ctx.get("demand_box"):
        smc_label = "demand_box"
    
    log_i(
        f"🧱 SMC BEST: {smc_label} "
        f"{hint_side} "
        f"({a.get('confidence',0):.1f})"
    )

    # 6) SNAP votes (Panel التصويت)
    votes_total = a.get("score_buy", 0) + a.get("score_sell", 0)
    votes_side = "?" if votes_total == 0 else ("BUY" if a.get("score_buy", 0) > a.get("score_sell", 0) else "SELL")
    
    log_i(
        f"🎯 SNAP | {votes_side} | "
        f"votes={max(a.get('score_buy',0), a.get('score_sell',0)):.0f}/{votes_total:.0f} "
        f"score={a.get('confidence',0):.1f} | "
        f"ADX={trend.get('adx',0):.1f} "
        f"DI={trend.get('di_plus',0)-trend.get('di_minus',0):.1f}"
    )

    # 7) Footprint / Volume delta
    volume_ctx = a.get("volume_analysis", {})
    log_i(
        f"🦶 FOOTPRINT | Δ={volume_ctx.get('delta',0):.0f} | "
        f"Spike={volume_ctx.get('spike', False)} | "
        f"AbsBull={volume_ctx.get('abs_bull', False)} | "
        f"AbsBear={volume_ctx.get('abs_bear', False)}"
    )

    # 8) SMC addons / FVG / Golden
    golden = a.get("golden_zone", {})
    log_i(
        f"🧠 ENHANCED SMC ADDONS | "
        f"FVG_real={fvg_ctx.get('real',False) if fvg_ctx else False} | "
        f"Golden={golden.get('type', 'None')} "
        f"| Trap={a.get('stop_hunt_trap_side', 'None')} "
        f"Q={a.get('stop_hunt_trap_quality',0):.1f}"
    )

# ============================================
#  BOOT BANNER SYSTEM
# ============================================

def log_banner():
    """طباعة بانر بداية التشغيل المحترف"""
    mode = "LIVE" if MODE_LIVE else "PAPER"
    if DRY_RUN:
        mode += " (DRY RUN)"
    
    print("\n" + "="*80)
    print(colored(" ULTRA PRO AI TRADING ENGINE — STARTUP ", "cyan", attrs=["bold"]))
    print("="*80)

    print(colored(f" MODE           : {mode}", "yellow"))
    print(colored(f" SYMBOL         : {SYMBOL}", "yellow"))
    print(colored(f" INTERVAL       : {INTERVAL}", "yellow"))
    print(colored(f" LEVERAGE       : {LEVERAGE}x", "yellow"))
    print(colored(f" RISK           : {int(RISK_ALLOC*100)}%", "yellow"))
    print(colored(f" EXCHANGE       : {EXCHANGE_NAME.upper()}", "yellow"))

    print(colored("\n ADVANCED FEATURES:", "green"))
    print(colored("  • RF Real Engine", "yellow"))
    print(colored("  • EdgeAlgo Smart RR Zones", "yellow"))
    print(colored("  • SMC: Supply/Demand + OB + Breaker + BOS", "yellow"))
    print(colored("  • Box Rejection Engine", "yellow"))
    print(colored("  • Advanced FVG Detection", "yellow"))
    print(colored("  • Golden Zones (Top/Bottom)", "yellow"))
    print(colored("  • Stop-Hunt Prediction Engine", "yellow"))
    print(colored("  • Trap Mode & Liquidity Sweep", "yellow"))
    print(colored("  • Smart Profit AI (TP1/TP2/TP3)", "yellow"))
    print(colored("  • Dynamic Stop-Burn + Breakeven", "yellow"))
    print(colored("  • Trend Mode + Momentum Scanner", "yellow"))
    print(colored("  • Equity Tracking + Compound PnL", "yellow"))
    print(colored("  • Web Service + Health Metrics", "yellow"))
    print(colored("  • ULTRA PANEL - Professional Logging System", "yellow"))
    print(colored("  • ADX+ATR FILTER - Smart Trend Filter", "yellow"))
    print(colored("  • RSI Wilder + ADX Wilder - متطابق مع TradingView", "yellow"))

    print("="*80)
    print(colored("🚀 INITIALIZING ULTRA PRO AI ENGINE...", "cyan", attrs=["bold"]))
    print("="*80)
    print()

# ============================================
#  KEEPALIVE SYSTEM
# ============================================

def keepalive_loop():
    """Loop لتثبيت العملية ومنع Render من قتل البوت."""
    log_i("🔄 KeepAlive loop started (50s intervals)")
    while True:
        try:
            time.sleep(50)
            log_i("💓 KeepAlive pulse - Bot is running...")
        except Exception as e:
            log_w(f"⚠️ KeepAlive error: {e}")

def setup_signal_handlers():
    """إعداد معالجات الإشارات للإغلاق الآمن"""
    def signal_handler(signum, frame):
        log_i(f"🛑 Received signal {signum} - Shutting down gracefully...")
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

# ============================================
#  EXCHANGE MANAGER
# ============================================

class ExchangeManager:
    """مدير البورصة الموحّد"""
    
    def __init__(self):
        self.exchange = None
        self.initialized = False
        self.setup_exchange()
    
    def setup_exchange(self):
        """إعداد الاتصال بالبورصة"""
        try:
            config = {
                "apiKey": API_KEY,
                "secret": API_SECRET,
                "enableRateLimit": True,
                "timeout": 30000,
                "options": {"defaultType": "swap"}
            }
            
            if EXCHANGE_NAME == "bybit":
                self.exchange = ccxt.bybit(config)
            else:
                self.exchange = ccxt.bingx(config)
            
            self.exchange.load_markets()
            self.initialized = True
            log_g(f"✅ Exchange {EXCHANGE_NAME.upper()} initialized successfully")
            
        except Exception as e:
            log_e(f"❌ Failed to initialize exchange: {e}")
            self.initialized = False
    
    def fetch_ohlcv(self, limit=100):
        """جلب بيانات OHLCV"""
        try:
            data = self.exchange.fetch_ohlcv(SYMBOL, timeframe=INTERVAL, limit=limit)
            df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
            return df
        except Exception as e:
            log_e(f"❌ Failed to fetch OHLCV: {e}")
            return pd.DataFrame()
    
    def get_current_price(self):
        """الحصول على السعر الحالي"""
        try:
            ticker = self.exchange.fetch_ticker(SYMBOL)
            return ticker.get('last', ticker.get('close'))
        except Exception as e:
            log_e(f"❌ Failed to get current_price: {e}")
            return None
    
    def get_balance(self):
        """الحصول على الرصيد"""
        if not MODE_LIVE:
            return 1000.0
            
        try:
            balance = self.exchange.fetch_balance()
            usdt_balance = balance.get('USDT', {}).get('free', 0.0)
            return float(usdt_balance)
        except Exception as e:
            log_e(f"❌ Failed to get balance: {e}")
            return 0.0
    
    def execute_order(self, side, quantity, price):
        """تنفيذ أمر تداول"""
        if DRY_RUN or not EXECUTE_ORDERS:
            log_i(f"🔹 DRY RUN: {side.upper()} {quantity:.4f} @ {price:.6f}")
            return True
            
        try:
            if MODE_LIVE and self.initialized:
                params = {}
                if EXCHANGE_NAME == "bybit":
                    params = {"positionSide": "Long" if side == "buy" else "Short"}
                else:
                    params = {"positionSide": "LONG" if side == "buy" else "SHORT"}
                
                order = self.exchange.create_order(
                    SYMBOL,
                    'market',
                    side,
                    quantity,
                    None,
                    params
                )
                log_g(f"✅ Order Executed: {side.upper()} {quantity:.4f} @ {price:.6f}")
                return True
        except Exception as e:
            log_e(f"❌ Order execution failed: {e}")
            
        return False

# ============================================
#  STATE MANAGEMENT
# ============================================

class StateManager:
    """مدير حالة البوت"""
    
    def __init__(self):
        self.state = {
            "open": False,
            "side": None,
            "entry": None,
            "qty": 0.0,
            "pnl": 0.0,
            "bars": 0,
            "mode": "scalp",
            "tp_profile": "SCALP_SMALL",
            "highest_profit_pct": 0.0,
            "profit_targets_achieved": 0,
            "opened_at": None,
            "last_signal": None,
            "sl": None,
            "tp1": None,
            "tp2": None,
            "tp3": None,
            "tp_mode": None,
            "trade_type": "normal",
            "tp1_hit": False,
            "tp2_hit": False,
            "compound_pnl": 0.0,
            "total_trades": 0,
            "trade_profile": "MID_TREND",
            "dynamic_sl": None,
            "high_water": None,
            "tp_levels": [],
            "entry_price": None,
            "edge_setup": None,
            "balance": 0.0,
            "mode_live": MODE_LIVE,
            "profit_profile": "SCALP_STRICT",
            "profit_engine_active": False
        }
        self.state_file = "bot_state.json"
        self.load_state()
    
    def get(self, key, default=None):
        """محاكاة دالة get الخاصة بالـ dict"""
        return self.state.get(key, default)
    
    def setdefault(self, key, default=None):
        """محاكاة دالة setdefault الخاصة بالـ dict"""
        if key not in self.state:
            self.state[key] = default
        return self.state[key]
    
    def load_state(self):
        """تحميل حالة البوت"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, "r", encoding="utf-8") as f:
                    saved_state = json.load(f)
                    self.state.update(saved_state)
                log_i("🔹 Bot state loaded successfully")
        except Exception as e:
            log_w(f"⚠️ Failed to load state: {e}")
    
    def save_state(self):
        """حفظ حالة البوت"""
        try:
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            log_w(f"⚠️ Failed to save state: {e}")
    
    def update(self, **kwargs):
        """تحديث حالة البوت"""
        self.state.update(kwargs)
        self.save_state()
    
    def reset(self):
        """إعادة تعيين حالة البوت"""
        self.state.update({
            "open": False,
            "side": None,
            "entry": None,
            "qty": 0.0,
            "pnl": 0.0,
            "bars": 0,
            "highest_profit_pct": 0.0,
            "profit_targets_achieved": 0,
            "opened_at": None,
            "sl": None,
            "tp1": None,
            "tp2": None,
            "tp3": None,
            "tp_mode": None,
            "trade_type": "normal",
            "tp1_hit": False,
            "tp2_hit": False,
            "dynamic_sl": None,
            "high_water": None,
            "tp_levels": [],
            "entry_price": None,
            "edge_setup": None,
            "profit_profile": "SCALP_STRICT",
            "profit_engine_active": False
        })
        self.save_state()
    
    def __getitem__(self, key):
        return self.state.get(key)
    
    def __setitem__(self, key, value):
        self.state[key] = value
        self.save_state()

# ============================================
#  TREND ANALYSIS ENGINE WITH ADX WILDER + RSI
# ============================================

class TrendAnalyzer:
    """محرك تحليل الاتجاه مع ADX Wilder + RSI + ATR (متطابق مع TradingView)"""
    
    def __init__(self):
        self.fast_ma = deque(maxlen=20)
        self.slow_ma = deque(maxlen=50)
        self.trend = "flat"
        self.strength = 0.0
        self.momentum = 0.0
        self.adx = 0.0
        self.di_plus = 0.0
        self.di_minus = 0.0
        self.atr = 0.0
        self.atr_mult = 1.0
        self.rsi = 50.0
        self.rsi_ma = 50.0
        self.rsi_zone = "neutral"
        
    def update(self, df):
        """تحديث تحليل الاتجاه"""
        if len(df) < 14:
            return
            
        close_prices = df['close'].astype(float)
        current_close = close_prices.iloc[-1]
        
        self.fast_ma.append(current_close)
        self.slow_ma.append(current_close)
        
        if len(self.slow_ma) < 10:
            return
            
        fast_avg = sum(self.fast_ma) / len(self.fast_ma)
        slow_avg = sum(self.slow_ma) / len(self.slow_ma)
        
        delta = fast_avg - slow_avg
        self.strength = abs(delta) / slow_avg * 100 if slow_avg != 0 else 0
        
        if len(close_prices) >= 5:
            recent = close_prices.tail(5).values
            self.momentum = (recent[-1] - recent[0]) / recent[0] * 100 if recent[0] != 0 else 0
            
        # حساب ADX Wilder + RSI + ATR
        self._calculate_adx_wilder(df)
        self._calculate_rsi(df)
        self._calculate_atr(df)
            
        if delta > 0 and self.strength > 0.1:
            self.trend = "up"
        elif delta < 0 and self.strength > 0.1:
            self.trend = "down" 
        else:
            self.trend = "flat"
            
    def _calculate_adx_wilder(self, df):
        """حساب ADX باستخدام Wilder's RMA (متطابق مع TradingView)"""
        try:
            if len(df) < 14:
                self.adx = 0.0
                self.di_plus = 0.0
                self.di_minus = 0.0
                return
                
            high = pd.Series(df['high'], dtype=float)
            low = pd.Series(df['low'], dtype=float)
            close = pd.Series(df['close'], dtype=float)
            
            # حساب +DM و -DM
            up_move = high.diff()
            down_move = -low.diff()
            
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
            
            # حساب True Range
            tr1 = high - low
            tr2 = (high - close.shift(1)).abs()
            tr3 = (low - close.shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # دالة RMA (Wilder)
            def rma(series, length):
                return series.ewm(alpha=1/length, adjust=False).mean()
            
            length = 14
            tr_rma = rma(tr, length)
            plus_dm_rma = rma(pd.Series(plus_dm), length)
            minus_dm_rma = rma(pd.Series(minus_dm), length)
            
            plus_di = 100 * (plus_dm_rma / tr_rma)
            minus_di = 100 * (minus_dm_rma / tr_rma)
            
            dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
            adx = rma(dx, length)
            
            self.adx = float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else 0.0
            self.di_plus = float(plus_di.iloc[-1]) if not pd.isna(plus_di.iloc[-1]) else 0.0
            self.di_minus = float(minus_di.iloc[-1]) if not pd.isna(minus_di.iloc[-1]) else 0.0
            
        except Exception as e:
            log_w(f"⚠️ ADX Wilder calculation error: {e}")
            self.adx = 0.0
            self.di_plus = 0.0
            self.di_minus = 0.0
    
    def _calculate_rsi(self, df, period=14, ma_period=9):
        """حساب RSI"""
        try:
            close = pd.Series(df['close'], dtype=float)
            
            delta = close.diff()
            gain = np.where(delta > 0, delta, 0.0)
            loss = np.where(delta < 0, -delta, 0.0)
            
            gain_ma = pd.Series(gain).rolling(period).mean()
            loss_ma = pd.Series(loss).rolling(period).mean()
            
            rs = gain_ma / loss_ma.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            
            self.rsi = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
            self.rsi_ma = float(rsi.rolling(ma_period).mean().iloc[-1]) if not pd.isna(rsi.rolling(ma_period).mean().iloc[-1]) else 50.0
            
            if self.rsi > 70:
                self.rsi_zone = "overbought"
            elif self.rsi < 30:
                self.rsi_zone = "oversold"
            else:
                self.rsi_zone = "neutral"
                
        except Exception as e:
            log_w(f"⚠️ RSI calculation error: {e}")
            self.rsi = 50.0
            self.rsi_ma = 50.0
            self.rsi_zone = "neutral"
    
    def _calculate_atr(self, df, period=14):
        """حساب ATR"""
        try:
            high = df['high'].astype(float)
            low = df['low'].astype(float)
            close = df['close'].astype(float)
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(period).mean()
            
            self.atr = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.0
            
            # حساب ATR Multiplier
            if len(tr) >= 20:
                atr_base = tr.rolling(window=20).mean().iloc[-1]
            else:
                atr_base = self.atr
                
            self.atr_mult = self.atr / atr_base if atr_base > 0 else 1.0
            
        except Exception as e:
            log_w(f"⚠️ ATR calculation error: {e}")
            self.atr = 0.0
            self.atr_mult = 1.0
            
    def is_strong_trend(self):
        """التحقق من قوة الاتجاه"""
        return self.strength > 0.3 and abs(self.momentum) > 0.5
    
    def get_trend_info(self):
        """الحصول على معلومات الاتجاه"""
        return {
            "direction": self.trend,
            "strength": self.strength,
            "momentum": self.momentum,
            "adx": self.adx,
            "di_plus": self.di_plus,
            "di_minus": self.di_minus,
            "atr": self.atr,
            "atr_mult": self.atr_mult,
            "is_strong": self.is_strong_trend(),
            "rsi": self.rsi,
            "rsi_ma": self.rsi_ma,
            "rsi_zone": self.rsi_zone
        }
    
    def analyze_stop_hunt_context(self, df, stop_hunt_zone):
        """
        تحليل سياق الستوب هانت باستخدام ADX و ATR
        
        Returns:
            {
                "trend_context": "flat"/"moderate"/"strong"/"extreme",
                "adx_slope": float,
                "atr_multiplier": float,
                "wick_ratio": float,
                "valid_for_trap": bool,
                "reason": str,
                "allowed_side": "BUY"/"SELL"/None
            }
        """
        if len(df) < 20 or not stop_hunt_zone:
            return {"valid_for_trap": False, "reason": "insufficient_data"}
        
        try:
            current_price = float(df['close'].iloc[-1])
            
            # تحليل ADX
            adx_slope = self._calculate_adx_slope(df)
            
            # تحليل ATR
            atr_mult = self.atr_mult
            
            # تحليل الشمعة
            last_candle = df.iloc[-1]
            candle_high = float(last_candle['high'])
            candle_low = float(last_candle['low'])
            candle_close = float(last_candle['close'])
            candle_open = float(last_candle['open'])
            
            candle_range = candle_high - candle_low
            body_size = abs(candle_close - candle_open)
            wick_size = candle_range - body_size
            wick_ratio = wick_size / candle_range if candle_range > 0 else 0
            
            # تحديد سياق الترند
            trend_context = "flat"
            if self.adx < 20:
                trend_context = "flat"
            elif self.adx < 35:
                trend_context = "moderate"
            elif self.adx < 50:
                trend_context = "strong"
            else:
                trend_context = "extreme"
            
            # تحليل الجانب المسموح للـ Trap
            allowed_side = None
            
            if trend_context == "extreme":
                # ترند مجنون - فقط مع الاتجاه
                if self.trend == "down":
                    allowed_side = "SELL"
                elif self.trend == "up":
                    allowed_side = "BUY"
                valid_for_trap = stop_hunt_zone.get("type") == f"{allowed_side.lower()}_stop_hunt"
                reason = f"extreme_trend_{self.trend}_only"
                
            elif trend_context == "strong":
                # ترند قوي - الأفضل مع الاتجاه، لكن ممكن ضد الاتجاه بحذر
                if self.trend == "down":
                    allowed_side = "SELL"  # الأفضل
                    valid_for_trap = True
                    reason = "strong_downtrend"
                elif self.trend == "up":
                    allowed_side = "BUY"   # الأفضل
                    valid_for_trap = True
                    reason = "strong_uptrend"
                else:
                    valid_for_trap = atr_mult >= 1.3 and wick_ratio >= 0.6
                    reason = "strong_range_trap"
                    
            elif trend_context == "moderate":
                # ترند معقول - Trap مسموح في كلا الاتجاهين
                valid_for_trap = atr_mult >= 1.3 and wick_ratio >= 0.6
                allowed_side = "BUY" if stop_hunt_zone.get("type") == "buy_stop_hunt" else "SELL"
                reason = "moderate_trend_trap"
                
            else:  # flat
                # سوق فلات - Trap ضعيف
                valid_for_trap = atr_mult >= 1.5 and wick_ratio >= 0.7
                allowed_side = "BUY" if stop_hunt_zone.get("type") == "buy_stop_hunt" else "SELL"
                reason = "flat_market_trap"
            
            # شروط إضافية للـ ATR
            if atr_mult < 1.2:
                valid_for_trap = False
                reason = "low_atr_multiplier"
            
            if atr_mult > 2.5 and adx_slope > 0:
                valid_for_trap = False
                reason = "breakout_continuation"
            
            return {
                "trend_context": trend_context,
                "adx_slope": adx_slope,
                "atr_multiplier": atr_mult,
                "wick_ratio": wick_ratio,
                "valid_for_trap": valid_for_trap,
                "reason": reason,
                "allowed_side": allowed_side,
                "adx": self.adx,
                "trend": self.trend
            }
            
        except Exception as e:
            log_w(f"⚠️ Stop hunt context analysis error: {e}")
            return {"valid_for_trap": False, "reason": f"error: {e}"}
    
    def _calculate_adx_slope(self, df, lookback=3):
        """حساب ميل ADX"""
        try:
            if len(df) < 14 + lookback:
                return 0.0
            
            # حساب ADX مبسط للـ lookback الأخيرة
            if lookback == 0:
                return 0.0
                
            # استخدام طريقة مبسطة لحساب ميل ADX
            current_adx = self.adx
            
            # حساب ADX مبسط للفترات السابقة
            if len(df) >= 15:
                # تقدير ADX السابق بناءً على الاتجاه الحالي
                prev_adx = current_adx * 0.95  # تقدير بسيط
                return current_adx - prev_adx
            return 0.0
            
        except:
            return 0.0

# ============================================
#  STOP HUNT DETECTION ENGINE WITH ADX+ATR FILTER
# ============================================

class StopHuntDetector:
    """محرك كشف مناطق ضرب الستوبات مع ADX+ATR فلتر"""
    
    def __init__(self):
        self.swing_highs = deque(maxlen=10)
        self.swing_lows = deque(maxlen=10)
        self.liquidity_zones = []
        self.recent_stop_hunts = deque(maxlen=5)
        self.trend_analyzer = TrendAnalyzer()
        
    def detect_swings(self, df, lookback=20):
        """كشف القمم والقيعان"""
        if len(df) < lookback * 2:
            return
            
        highs = df['high'].astype(float)
        lows = df['low'].astype(float)
        
        for i in range(lookback, len(highs) - lookback):
            if highs.iloc[i] == highs.iloc[i-lookback:i+lookback].max():
                self.swing_highs.append((i, highs.iloc[i]))
            if lows.iloc[i] == lows.iloc[i-lookback:i+lookback].min():
                self.swing_lows.append((i, lows.iloc[i]))
    
    def detect_liquidity_zones(self, current_price):
        """كشف مناطق السيولة"""
        zones = []
        for _, high in self.swing_highs:
            if high > current_price * 1.01:
                zones.append(("sell_liquidity", high))
        for _, low in self.swing_lows:
            if low < current_price * 0.99:
                zones.append(("buy_liquidity", low))
        return zones
    
    def detect_stop_hunt_zones(self, df):
        """كشف مناطق ضرب الستوبات مع فلتر ADX+ATR"""
        if len(df) < 10:
            return []
            
        self.trend_analyzer.update(df)
        trend_info = self.trend_analyzer.get_trend_info()
        
        stop_hunt_zones = []
        highs = df['high'].astype(float)
        lows = df['low'].astype(float)
        closes = df['close'].astype(float)
        volumes = df['volume'].astype(float)
        
        for i in range(5, len(df)-1):
            # كشف Stop Hunt صاعد (شرائي)
            if (lows.iloc[i] < lows.iloc[i-1] and
                closes.iloc[i] > lows.iloc[i-1] and
                volumes.iloc[i] > volumes.iloc[i-1:i-4:-1].mean() * 1.5):
                
                zone = {
                    "type": "buy_stop_hunt",
                    "level": lows.iloc[i-1],
                    "high": highs.iloc[i],
                    "index": i,
                    "strength": 3.0,
                    "adx_context": self._analyze_candle_context(df, i, "buy")
                }
                
                # تطبيق فلتر ADX+ATR
                if self._validate_stop_hunt_with_adx_atr(zone, trend_info):
                    stop_hunt_zones.append(zone)
            
            # كشف Stop Hunt هابط (بيعي)
            if (highs.iloc[i] > highs.iloc[i-1] and
                closes.iloc[i] < highs.iloc[i-1] and
                volumes.iloc[i] > volumes.iloc[i-1:i-4:-1].mean() * 1.5):
                
                zone = {
                    "type": "sell_stop_hunt", 
                    "level": highs.iloc[i-1],
                    "low": lows.iloc[i],
                    "index": i,
                    "strength": 3.0,
                    "adx_context": self._analyze_candle_context(df, i, "sell")
                }
                
                if self._validate_stop_hunt_with_adx_atr(zone, trend_info):
                    stop_hunt_zones.append(zone)
                
        self.recent_stop_hunts.extend(stop_hunt_zones[-3:])
        return stop_hunt_zones[-3:]
    
    def _analyze_candle_context(self, df, index, zone_type):
        """تحليل سياق الشمعة باستخدام ADX و ATR"""
        try:
            if index < 3 or index >= len(df):
                return {"valid": False, "reason": "invalid_index"}
            
            candle = df.iloc[index]
            prev_candle = df.iloc[index-1]
            
            candle_high = float(candle['high'])
            candle_low = float(candle['low'])
            candle_close = float(candle['close'])
            candle_open = float(candle['open'])
            
            prev_high = float(prev_candle['high'])
            prev_low = float(prev_candle['low'])
            
            candle_range = candle_high - candle_low
            body_size = abs(candle_close - candle_open)
            wick_size = candle_range - body_size
            wick_ratio = wick_size / candle_range if candle_range > 0 else 0
            
            # حساب ATR للشمعة
            tr1 = candle_high - candle_low
            tr2 = abs(candle_high - float(prev_candle['close']))
            tr3 = abs(candle_low - float(prev_candle['close']))
            tr = max(tr1, tr2, tr3)
            
            # حساب ATR الأساسي
            if len(df) >= 20:
                atr_values = []
                for j in range(max(0, index-19), index+1):
                    if j >= len(df):
                        continue
                    h = float(df['high'].iloc[j])
                    l = float(df['low'].iloc[j])
                    pc = float(df['close'].iloc[j-1]) if j > 0 else float(df['open'].iloc[j])
                    atr_tr = max(h-l, abs(h-pc), abs(l-pc))
                    atr_values.append(atr_tr)
                
                atr_base = sum(atr_values) / len(atr_values) if atr_values else tr
                atr_mult = tr / atr_base if atr_base > 0 else 1.0
            else:
                atr_mult = 1.0
            
            # تحليل الذيل
            if zone_type == "buy":
                lower_wick = min(candle_close, candle_open) - candle_low
                lower_wick_ratio = lower_wick / candle_range if candle_range > 0 else 0
                has_long_lower_wick = lower_wick_ratio >= 0.6
            else:  # sell
                upper_wick = candle_high - max(candle_close, candle_open)
                upper_wick_ratio = upper_wick / candle_range if candle_range > 0 else 0
                has_long_upper_wick = upper_wick_ratio >= 0.6
            
            valid_stop_hunt = (
                atr_mult >= 1.3 and
                wick_ratio >= 0.6 and
                ((zone_type == "buy" and has_long_lower_wick) or
                 (zone_type == "sell" and has_long_upper_wick))
            )
            
            return {
                "valid": valid_stop_hunt,
                "atr_mult": atr_mult,
                "wick_ratio": wick_ratio,
                "candle_range": candle_range,
                "reason": "valid_stop_hunt" if valid_stop_hunt else "weak_candle_structure"
            }
            
        except Exception as e:
            return {"valid": False, "reason": f"error: {e}"}
    
    def _validate_stop_hunt_with_adx_atr(self, zone, trend_info):
        """التحقق من صحة الستوب هانت باستخدام ADX و ATR"""
        try:
            adx = trend_info.get("adx", 0)
            atr_mult = trend_info.get("atr_mult", 1.0)
            trend = trend_info.get("direction", "flat")
            zone_type = zone.get("type", "")
            
            # قاعدة: ممنوع Trap عكسي في ترند مجنون
            if adx > 50:  # ترند وحشي
                # فقط Trap مع الاتجاه مسموح
                if trend == "down" and zone_type == "sell_stop_hunt":
                    return True  # SELL مع الترند الهابط
                elif trend == "up" and zone_type == "buy_stop_hunt":
                    return True  # BUY مع الترند الصاعد
                else:
                    return False  # Trap عكسي ممنوع
            
            # شروط ATR
            if atr_mult < 1.2:
                return False  # حركة ضعيفة
            
            if atr_mult > 2.5 and adx > 35:
                return False  # Breakout مستمر
            
            # شروط ADX
            if adx < 20:
                # سوق فلات - يحتاج شروط أقوى
                candle_context = zone.get("adx_context", {})
                return candle_context.get("valid", False) and atr_mult >= 1.5
            
            return True
            
        except Exception as e:
            log_w(f"⚠️ ADX/ATR validation error: {e}")
            return True  # في حالة الخطأ، نرجع True عشان ما نخسر فرص
    
    def get_active_stop_hunt_zones(self, current_price, df):
        """الحصول على مناطق الستوب هانت النشطة مع تحليل ADX"""
        active_zones = []
        for zone in self.recent_stop_hunts:
            if zone["type"] == "buy_stop_hunt" and current_price > zone["level"] * 0.995:
                # تحليل ADX للـ Trap
                trend_context = self.trend_analyzer.analyze_stop_hunt_context(df, zone)
                zone["trend_context"] = trend_context
                active_zones.append(zone)
            elif zone["type"] == "sell_stop_hunt" and current_price < zone["level"] * 1.005:
                trend_context = self.trend_analyzer.analyze_stop_hunt_context(df, zone)
                zone["trend_context"] = trend_context
                active_zones.append(zone)
        return active_zones

# ============================================
#  PROFIT ENGINE - نظام جني الأرباح الذكي
# ============================================

class ProfitEngine:
    """محرك جني الأرباح والوقف المتحرك الذكي"""
    
    def __init__(self, exchange, state):
        self.exchange = exchange
        self.state = state
        
        self.profile_name = None
        self.profile_cfg = None
        self.side = None
        self.entry_price = None
        self.atr_entry = None
        
        self.tp_levels = []     # [(price, frac, label)]
        self.tp_hit = set()
        
        self.sl_price = None
        self.trail_active = False
        self.trail_price = None
    
    def init_trade(self, side, entry_price, atr_value, trade_mode, analysis):
        """تهيئة الصفقة مع تحديد الـ Profile المناسب"""
        self.side = side  # "long" / "short"
        self.entry_price = float(entry_price)
        self.atr_entry = float(atr_value)
        
        # اختيار الـ Profile المناسب
        self.profile_name = select_profit_profile(trade_mode, analysis)
        self.profile_cfg = PROFIT_PROFILES[self.profile_name]
        
        direction = 1 if side == "long" else -1
        
        # 1) ستوب مبدئي (قائم على R)
        hard_sl_rr = self.profile_cfg["hard_sl_rr"]
        sl_dist = abs(hard_sl_rr) * self.atr_entry
        if side == "long":
            self.sl_price = self.entry_price - sl_dist
        else:
            self.sl_price = self.entry_price + sl_dist
        
        # 2) حساب مستويات TP بالسعر
        self.tp_levels = []
        self.tp_hit = set()
        for i, (rr, frac) in enumerate(zip(self.profile_cfg["tp_levels_rr"],
                                           self.profile_cfg["tp_fracs"])):
            dist = rr * self.atr_entry
            price = self.entry_price + direction * dist
            label = f"TP{i+1}_{self.profile_name}"
            self.tp_levels.append((price, frac, label))
        
        self.trail_active = False
        self.trail_price = None
        
        # تحديث state
        self.state["profit_profile"] = self.profile_name
        self.state["profit_engine_active"] = True
        
        log_i(
            f"🎯 PROFIT PLAN [{self.profile_name}] | "
            f"side={side} | entry={self.entry_price:.6f} | "
            f"ATR={self.atr_entry:.6f} | SL={self.sl_price:.6f} | "
            f"TPs={[(round(p,6), f'{f*100:.0f}%') for p,f,_ in self.tp_levels]}"
        )
    
    def calculate_atr(self, df, period=14):
        """حساب ATR من الـ DataFrame"""
        if len(df) < period:
            return self.atr_entry if self.atr_entry else 0.01 * self.entry_price
        
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        close = df['close'].astype(float)
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]
        
        return float(atr) if not pd.isna(atr) else self.atr_entry
    
    def on_tick(self, df):
        """تحديث وإدارة الصفقة في كل تيك"""
        if not self.state["open"]:
            return False
            
        price = self.exchange.get_current_price()
        if not price:
            return False
            
        price = float(price)
        side = self.state["side"]
        qty = self.state["qty"]
        direction = 1 if side == "long" else -1
        
        # حساب ATR الحالي و R
        atr_now = self.calculate_atr(df)
        R_now = (price - self.entry_price) * direction / max(atr_now, 1e-8)
        
        # ===== 1) وقف خسارة ثابت / متحرك (تنفيذ لو اتضرب) =====
        if self.sl_price is not None:
            if (side == "long" and price <= self.sl_price) or \
               (side == "short" and price >= self.sl_price):
                
                log_r(
                    f"🛑 HARD SL HIT | profile={self.profile_name} | "
                    f"side={side} | qty={qty:.4f} | exit={price:.6f} | R={R_now:.2f}"
                )
                
                # تنفيذ إغلاق كامل
                close_side = "sell" if side == "long" else "buy"
                if self.exchange.execute_order(close_side, qty, price):
                    self.state["open"] = False
                    self.state["profit_engine_active"] = False
                    return True
                return False
        
        # ===== 2) تنفيذ TP الجزئية =====
        for idx, (tp_price, frac, label) in enumerate(self.tp_levels):
            if idx in self.tp_hit:
                continue
            
            hit = (direction == 1 and price >= tp_price) or \
                  (direction == -1 and price <= tp_price)
            
            if hit:
                close_qty = qty * frac
                close_side = "sell" if side == "long" else "buy"
                
                if self.exchange.execute_order(close_side, close_qty, price):
                    self.tp_hit.add(idx)
                    self.state["qty"] -= close_qty
                    qty = self.state["qty"]
                    
                    # حساب الربح المحقق
                    if side == "long":
                        realized_pnl = (price - self.entry_price) * close_qty
                    else:
                        realized_pnl = (self.entry_price - price) * close_qty
                    
                    # تحديث الربح التراكمي
                    self.state["compound_pnl"] = self.state.get("compound_pnl", 0.0) + realized_pnl
                    
                    log_g(
                        f"✅ {label} HIT | price={price:.6f} | "
                        f"closed={close_qty:.4f} | remain={qty:.4f} | "
                        f"R≈{R_now:.2f} | PnL={realized_pnl:.3f} USDT"
                    )
                    
                    # بعد أول TP → Breakeven لو مفعّل
                    if self.profile_cfg["be_after_tp"] and len(self.tp_hit) == 1:
                        if side == "long":
                            self.sl_price = self.entry_price * 1.0001  # +0.01%
                        else:
                            self.sl_price = self.entry_price * 0.9999  # -0.01%
                        log_w(f"⚖ Breakeven set at {self.sl_price:.6f}")
        
        # ===== 3) تفعيل / تحديث التريل =====
        start_rr = self.profile_cfg["trail_start_rr"]
        atr_mult = self.profile_cfg["trail_atr_mult"]
        
        if start_rr and atr_mult and self.state["qty"] > 0:
            if (not self.trail_active) and R_now >= start_rr:
                self.trail_active = True
                # أول تريل
                dist = atr_mult * atr_now
                if side == "long":
                    self.trail_price = price - dist
                else:
                    self.trail_price = price + dist
                # خلي الستوب يساوي التريل
                self.sl_price = self.trail_price
                log_w(
                    f"🧷 TRAIL ACTIVATED | profile={self.profile_name} | "
                    f"trail={self.trail_price:.6f} | R≈{R_now:.2f}"
                )
            
            if self.trail_active:
                dist = atr_mult * atr_now
                if side == "long":
                    new_trail = price - dist
                    if new_trail > self.trail_price:
                        self.trail_price = new_trail
                        self.sl_price = self.trail_price
                else:
                    new_trail = price + dist
                    if new_trail < self.trail_price:
                        self.trail_price = new_trail
                        self.sl_price = self.trail_price
        
        # ===== 4) لو كل TPs اتنفذت ومفيش تريل إضافي ⇒ قفل صارم =====
        if len(self.tp_hit) == len(self.tp_levels) and not self.trail_active:
            if self.state["qty"] > 0:
                close_side = "sell" if side == "long" else "buy"
                if self.exchange.execute_order(close_side, self.state["qty"], price):
                    log_g(
                        f"💰 FINAL STRICT CLOSE | profile={self.profile_name} | "
                        f"side={side} | qty={self.state['qty']:.4f} | exit={price:.6f}"
                    )
                    self.state["open"] = False
                    self.state["profit_engine_active"] = False
                    return True
        
        return False
    
    def get_status(self):
        """الحصول على حالة الـ Profit Engine"""
        return {
            "profile": self.profile_name,
            "entry_price": self.entry_price,
            "sl_price": self.sl_price,
            "tp_levels": [(p, f, l) for p, f, l in self.tp_levels],
            "tp_hit": list(self.tp_hit),
            "trail_active": self.trail_active,
            "trail_price": self.trail_price
        }

# ============================================
#  SMART POSITION MANAGER WITH PROFIT ENGINE
# ============================================

class SmartPositionManager:
    """مدير المراكز الذكي المتكامل مع Profit Engine"""
    
    def __init__(self, exchange_manager, state_manager):
        self.exchange = exchange_manager
        self.state = state_manager
        self.profit_engine = ProfitEngine(exchange_manager, state_manager)
    
    def calculate_position_size(self, balance, price):
        """حساب حجم المركز"""
        if balance <= 0 or price <= 0:
            return 0.0
        
        capital = balance * RISK_ALLOC
        notional = capital * LEVERAGE
        size = notional / price
        
        log_i(f"🔹 Position Size: Balance={balance:.2f}, Capital={capital:.2f}, Size={size:.4f}")
        return round(size, 4)
    
    def open_position(self, side, df, analysis):
        """فتح مركز جديد مع Profit Engine"""
        if self.state["open"]:
            log_w("⚠️ Position already open")
            return False
            
        current_price = self.exchange.get_current_price()
        balance = self.exchange.get_balance()
        
        if not current_price or balance <= 10:
            log_w("⚠️ Insufficient balance or invalid price")
            return False
            
        position_size = self.calculate_position_size(balance, current_price)
        
        if position_size <= 0:
            log_w("⚠️ Invalid position size")
            return False
            
        # تحديد نوع الصفقة (TRAP / GOLDEN / NORMAL)
        trade_type = "normal"
        trade_mode = "SCALP"
        
        if analysis.get("stop_hunt_trap_side") and analysis.get("stop_hunt_trap_quality", 0) >= 3.0:
            trade_type = "trap"
            trade_mode = "TRAP"
        elif analysis.get("golden_zone", {}).get("valid"):
            trade_type = "golden"
            trade_mode = "GOLDEN"
        elif "PREDICTIVE STOP-HUNT" in analysis.get("signals", []):
            trade_type = "predictive"
            trade_mode = "PREDICTIVE"
        
        # تنفيذ الأمر
        if self.exchange.execute_order(side.lower(), position_size, current_price):
            # تحديث حالة البوت
            self.state.update({
                "open": True,
                "side": side.lower(),
                "entry": current_price,
                "qty": position_size,
                "pnl": 0.0,
                "bars": 0,
                "highest_profit_pct": 0.0,
                "profit_targets_achieved": 0,
                "opened_at": time.time(),
                "last_signal": side,
                "trade_type": trade_type,
                "trade_profile": "SCALP_STRICT",  # سيتم تحديثه بواسطة Profit Engine
                "edge_setup": analysis.get("edge_setup"),
                "entry_price": current_price,
                "tp1_hit": False,
                "tp2_hit": False
            })
            
            log_g(f"✅ New Position Opened: {side.upper()} | Size: {position_size:.4f} | Entry: {current_price:.6f} | Type: {trade_type.upper()}")
            
            # تهيئة Profit Engine
            atr_value = analysis.get("trend", {}).get("atr", current_price * 0.01)
            self.profit_engine.init_trade(side.lower(), current_price, atr_value, trade_mode, analysis)
            
            # لوج الرصيد بعد فتح الصفقة
            balance_now = self.exchange.get_balance()
            log_equity_snapshot(balance_now, self.state.get("compound_pnl", 0.0))
            
            return True
            
        return False
    
    def manage_position(self, df):
        """إدارة المركز المفتوح مع Profit Engine"""
        if not self.state["open"]:
            return
            
        # استخدام Profit Engine لإدارة الصفقة
        closed = self.profit_engine.on_tick(df)
        
        if not closed:
            # تحديث الربح/الخسارة
            current_price = self.exchange.get_current_price()
            if current_price:
                entry_price = self.state["entry_price"]
                side = self.state["side"]
                
                if side == "long":
                    pnl_pct = (current_price - entry_price) / entry_price * 100
                else:
                    pnl_pct = (entry_price - current_price) / entry_price * 100
                    
                self.state["pnl"] = pnl_pct
                
                # تحديث أعلى ربح
                if pnl_pct > self.state["highest_profit_pct"]:
                    self.state["highest_profit_pct"] = pnl_pct
                
                self.state["bars"] += 1
    
    def close_position(self, reason=""):
        """إغلاق المركز الحالي"""
        if not self.state["open"]:
            return False
            
        side = "sell" if self.state["side"] == "long" else "buy"
        current_price = self.exchange.get_current_price()
        
        if current_price and self.exchange.execute_order(side, self.state["qty"], current_price):
            # حساب الربح النهائي
            entry_price = self.state["entry_price"]
            if self.state["side"] == "long":
                realized_pnl = (current_price - entry_price) * self.state["qty"]
            else:
                realized_pnl = (entry_price - current_price) * self.state["qty"]

            self.state["total_trades"] = self.state.get("total_trades", 0) + 1
            self.state["compound_pnl"] = self.state.get("compound_pnl", 0.0) + realized_pnl

            log_g(
                f"💰 TRADE CLOSED | side={self.state['side']} | "
                f"qty={self.state['qty']:.4f} | pnl={realized_pnl:.3f} USDT | "
                f"🔄 trade#{self.state['total_trades']} | Reason: {reason}"
            )

            balance_after = self.exchange.get_balance()
            log_equity_snapshot(balance_after, self.state["compound_pnl"])

            self.state.reset()
            return True
            
        log_e(f"❌ Failed to close position: {reason}")
        return False

# ============================================
#  ULTRA COUNCIL AI - نظام التصويت الذكي المتكامل
# ============================================

class UltraCouncilAI:
    """مجلس الإدارة الذكي المتكامل مع جميع المحركات"""
    
    def __init__(self):
        # المحركات الأساسية
        self.stop_hunt_detector = StopHuntDetector()
        self.trend_analyzer = TrendAnalyzer()
        
        # المحركات المتقدمة
        self.edge_algo = EdgeAlgoEngine()
        self.smc_ctx_engine = self  # سأستخدم نفس الكلاس للسياق
        self.golden_engine = GoldenZoneEngine()
        self.sh_predictor = StopHuntPredictor()
        
        # معايير القرار
        self.min_confidence = 0.6
        self.min_score = 8

    def _empty_analysis(self):
        """تحليل فارغ عند الخطأ"""
        return {
            "score_buy": 0.0,
            "score_sell": 0.0,
            "confidence": 0.0,
            "signals": [],
            "trend": {
                "direction": "flat",
                "strength": 0.0,
                "momentum": 0.0,
                "adx": 0.0,
                "di_plus": 0.0,
                "di_minus": 0.0,
                "atr": 0.0,
                "is_strong": False,
                "rsi": 50.0,
                "rsi_ma": 50.0,
                "rsi_zone": "neutral"
            },
            "stop_hunt_zones": 0,
            "smc_ctx": {},
            "edge_setup": None,
            "stop_hunt_trap_side": None,
            "stop_hunt_trap_quality": 0.0,
            "golden_zone": {"type": None, "valid": False},
            "predicted_stop_hunt": {},
            "volume_analysis": {},
            "rsi": 50.0,
            "rsi_ma": 50.0,
            "rsi_zone": "neutral"
        }

    def build_context(self, df, current_price, stop_hunt_info, fvg_ctx, liquidity_zones):
        """بناء سياق SMC مبسط"""
        ctx = {
            "supply_box": False,
            "demand_box": False,
            "liquidity_sweep": False,
            "fake_break": False,
            "stop_hunt_zone": False,
        }

        high = df["high"].astype(float)
        low = df["low"].astype(float)
        lookback = 20
        recent_high = high.tail(lookback).max()
        recent_low = low.tail(lookback).min()

        if current_price >= recent_high * 0.995:
            ctx["supply_box"] = True
        if current_price <= recent_low * 1.005:
            ctx["demand_box"] = True

        if stop_hunt_info.get("active_count", 0) > 0:
            ctx["stop_hunt_zone"] = True

        for z_type, level in liquidity_zones:
            diff_pct = abs(current_price - level) / current_price
            if diff_pct < 0.002:
                ctx["liquidity_sweep"] = True

        return ctx

    def analyze_market(self, df):
        """تحليل السوق الشامل المتكامل مع RSI + ADX Wilder"""
        if len(df) < 20:
            return self._empty_analysis()

        try:
            current_price = float(df['close'].iloc[-1])
            signals = []
            score_buy = 0
            score_sell = 0
            
            # تحديث المحركات الأساسية
            self.trend_analyzer.update(df)
            trend_info = self.trend_analyzer.get_trend_info()
            
            # استخراج قيم RSI من trend_info
            rsi = trend_info.get("rsi", 50.0)
            rsi_ma = trend_info.get("rsi_ma", 50.0)
            rsi_zone = trend_info.get("rsi_zone", "neutral")
            
            # إضافة إشارات RSI
            if rsi_zone == "overbought":
                signals.append("⚠️ RSI Overbought")
                score_sell += 1.0
            elif rsi_zone == "oversold":
                signals.append("⚠️ RSI Oversold")
                score_buy += 1.0
            
            if rsi > 60:
                signals.append(f"📈 RSI Bullish ({rsi:.1f})")
            elif rsi < 40:
                signals.append(f"📉 RSI Bearish ({rsi:.1f})")
            
            # 1. الستوب هانت والسيولة
            self.stop_hunt_detector.detect_swings(df)
            stop_hunt_zones = self.stop_hunt_detector.detect_stop_hunt_zones(df)
            active_zones = self.stop_hunt_detector.get_active_stop_hunt_zones(current_price, df)
            active_count = len(active_zones)

            # 2. تحليل Trap Mode مع ADX+ATR
            trap_side = None
            trap_quality = 0.0

            for zone in active_zones:
                trend_context = zone.get("trend_context", {})
                
                if zone["type"] == "buy_stop_hunt" and trend_context.get("valid_for_trap", False):
                    allowed_side = trend_context.get("allowed_side")
                    if allowed_side in ["BUY", None]:  # مسموح أو غير محدد
                        trap_side = "BUY"
                        trap_quality = max(trap_quality, zone["strength"] + trend_context.get("adx", 0)/50)
                        signals.append(f"🧨 TRAP_BUY_ZONE @ {zone['level']:.6f} | ADX={trend_context.get('adx',0):.1f}")
                
                if zone["type"] == "sell_stop_hunt" and trend_context.get("valid_for_trap", False):
                    allowed_side = trend_context.get("allowed_side")
                    if allowed_side in ["SELL", None]:
                        trap_side = "SELL"
                        trap_quality = max(trap_quality, zone["strength"] + trend_context.get("adx", 0)/50)
                        signals.append(f"🧨 TRAP_SELL_ZONE @ {zone['level']:.6f} | ADX={trend_context.get('adx',0):.1f}")

            # 3. الاتجاه والزخم
            if trend_info["direction"] == "up":
                score_buy += 1.0
                signals.append("📈 Uptrend")
            elif trend_info["direction"] == "down":
                score_sell += 1.0
                signals.append("📉 Downtrend")
                
            if trend_info["is_strong"]:
                if trend_info["direction"] == "up":
                    score_buy += 2.0
                    signals.append("💪 Strong Uptrend")
                else:
                    score_sell += 2.0
                    signals.append("💪 Strong Downtrend")
                    
            if trend_info["momentum"] > 0.5:
                score_buy += 1.0
                signals.append("🚀 Positive Momentum")
            elif trend_info["momentum"] < -0.5:
                score_sell += 1.0
                signals.append("💥 Negative Momentum")

            # 4. Edge Algo Setup
            edge_side = None
            if score_buy > score_sell:
                edge_side = "BUY"
            elif score_sell > score_buy:
                edge_side = "SELL"

            # بناء سياق SMC
            smc_ctx = self.build_context(
                df, current_price, 
                {"active_count": active_count},
                {},
                self.stop_hunt_detector.detect_liquidity_zones(current_price)
            )

            edge_setup = None
            if edge_side:
                edge_setup = self.edge_algo.compute_setup(df, edge_side, trend_info, smc_ctx)
                if edge_setup.get("valid"):
                    signals.append(
                        f"🧠 EdgeAlgo {edge_setup['grade'].upper()} | "
                        f"RR1={edge_setup['rr1']:.2f} RR2={edge_setup['rr2']:.2f} RR3={edge_setup['rr3']:.2f}"
                    )
                    if edge_setup["grade"] == "strong":
                        if edge_side == "BUY":
                            score_buy += 2.0
                        else:
                            score_sell += 2.0
                    elif edge_setup["grade"] == "mid":
                        if edge_side == "BUY":
                            score_buy += 1.0
                        else:
                            score_sell += 1.0

            # 5. Golden Zones
            golden = self.golden_engine.compute(df)
            if golden["valid"]:
                if golden["type"] == "golden_bottom":
                    score_buy += 2
                    signals.append("🟢 Golden Bottom Zone")
                elif golden["type"] == "golden_top":
                    score_sell += 2
                    signals.append("🔴 Golden Top Zone")

            # 6. التنبؤ بالستوب هانت
            predicted_sh = self.sh_predictor.predict(df)
            if predicted_sh.get("up_target"):
                signals.append(f"🎯 Predicted Stop-Hunt UP @ {predicted_sh['up_target']:.6f}")
                # تنبؤ بضرب استوبات فوق ثم هبوط
                score_sell += 1.5

            if predicted_sh.get("down_target"):
                signals.append(f"🎯 Predicted Stop-Hunt DOWN @ {predicted_sh['down_target']:.6f}")
                # تنبؤ بضرب استوبات تحت ثم صعود
                score_buy += 1.5

            # الثقة النهائية
            total_score = score_buy + score_sell
            confidence = min(1.0, total_score / 20.0)
            
            return {
                "score_buy": round(score_buy, 2),
                "score_sell": round(score_sell, 2),
                "confidence": round(confidence, 2),
                "signals": signals,
                "trend": trend_info,
                "stop_hunt_zones": active_count,
                "smc_ctx": smc_ctx,
                "edge_setup": edge_setup,
                "edge_rr": edge_setup["rr1"] if edge_setup and edge_setup.get("valid") else 1.0,
                "stop_hunt_trap_side": trap_side,
                "stop_hunt_trap_quality": trap_quality,
                "golden_zone": golden,
                "predicted_stop_hunt": predicted_sh,
                "volume_analysis": {
                    "delta": 0,
                    "spike": False,
                    "abs_bull": False,
                    "abs_bear": False
                },
                # إضافة RSI إلى التحليل
                "rsi": rsi,
                "rsi_ma": rsi_ma,
                "rsi_zone": rsi_zone
            }
            
        except Exception as e:
            log_e(f"❌ Ultra market analysis error: {e}")
            return self._empty_analysis()

    def should_enter_trade(self, df):
        """تحديد ما إذا كان يجب الدخول في صفقة"""
        analysis = self.analyze_market(df)

        if analysis is None:
            return None, "NO_ANALYSIS", analysis

        trap_side = analysis.get("stop_hunt_trap_side")
        trap_q = analysis.get("stop_hunt_trap_quality", 0.0)
        predicted = analysis.get("predicted_stop_hunt", {})
        smc_ctx = analysis.get("smc_ctx", {})
        trend = analysis.get("trend", {})
        rsi = analysis.get("rsi", 50.0)
        rsi_zone = analysis.get("rsi_zone", "neutral")

        # 1) TRAP OVERRIDE MODE – دخول قسري لو الفرصة خبيثة جدًا
        if trap_side and trap_q >= 2.5:
            log_w("🧨 TRAP OVERRIDE MODE ACTIVATED")

            sweep = smc_ctx.get("liquidity_sweep", False)
            stop_hunt = smc_ctx.get("stop_hunt_zone", False)

            if sweep or stop_hunt:
                entry_signal = trap_side.lower()
                reason = (
                    f"TRAP_OVERRIDE | StopHunt={trap_q:.1f} "
                    f"| sweep={sweep} | stop_hunt={stop_hunt} | ADX={trend.get('adx',0):.1f} | RSI={rsi:.1f}"
                )
                return entry_signal, reason, analysis

        # 2) لو الثقة قليلة جرّب Trap Mode قبل الرفض
        if analysis.get("confidence", 0.0) < self.min_confidence:
            if trap_side and trap_q >= 3.0:
                entry_signal = trap_side.lower()
                reason = f"TRAP MODE {trap_side} | Stop-Hunt Exploit | Q={trap_q:.1f} | RSI={rsi:.1f}"
                return entry_signal, reason, analysis

            return None, "Low confidence", analysis

        # 3) توقع ضرب الاستوبات (Predictive Stop-Hunt)
        trend_dir = trend.get("direction", "flat")

        # لو في هدف ستوب هانت فوق والسوق ترنده هابط → بيع خبيث
        if predicted.get("up_target") and trend_dir == "down":
            if analysis.get("score_sell", 0) >= self.min_score - 3:
                return "sell", "PREDICTIVE STOP-HUNT SELL", analysis

        # لو في هدف ستوب هانت تحت والسوق ترنده صاعد → شراء خبيث
        if predicted.get("down_target") and trend_dir == "up":
            if analysis.get("score_buy", 0) >= self.min_score - 3:
                return "buy", "PREDICTIVE STOP-HUNT BUY", analysis

        # 4) Golden Zone Override
        entry_signal = None
        reason = ""
        golden = analysis.get("golden_zone", {})

        if golden.get("valid"):
            if golden.get("type") == "golden_bottom" and analysis.get("score_buy", 0) >= self.min_score - 2:
                entry_signal = "buy"
                reason = (
                    f"ULTRA BUY | Golden Override | "
                    f"Score: {analysis['score_buy']} | Conf: {analysis['confidence']} | RSI: {rsi:.1f}"
                )
            elif golden.get("type") == "golden_top" and analysis.get("score_sell", 0) >= self.min_score - 2:
                entry_signal = "sell"
                reason = (
                    f"ULTRA SELL | Golden Override | "
                    f"Score: {analysis['score_sell']} | Conf: {analysis['confidence']} | RSI: {rsi:.1f}"
                )

        # 5) القرار العادي لو مفيش Override
        if entry_signal is None:
            if analysis.get("score_buy", 0) >= self.min_score and analysis["score_buy"] > analysis["score_sell"]:
                entry_signal = "buy"
                reason = (
                    f"ULTRA BUY | Score: {analysis['score_buy']} "
                    f"| Confidence: {analysis['confidence']} | RSI: {rsi:.1f}"
                )
            elif analysis.get("score_sell", 0) >= self.min_score and analysis["score_sell"] > analysis["score_buy"]:
                entry_signal = "sell"
                reason = (
                    f"ULTRA SELL | Score: {analysis['score_sell']} "
                    f"| Confidence: {analysis['confidence']} | RSI: {rsi:.1f}"
                )
            else:
                reason = (
                    f"No clear signal | Buy: {analysis.get('score_buy', 0)} "
                    f"| Sell: {analysis.get('score_sell', 0)} | RSI: {rsi:.1f}"
                )

        return entry_signal, reason, analysis

# ============================================
#  SUPPORTING CLASSES
# ============================================

class StopHuntPredictor:
    """محرك توقع مناطق ضرب الستوبات القادمة"""
    def __init__(self):
        self.liq_threshold = 0.003
        self.cluster_lookback = 15
        self.min_cluster = 2

    def predict(self, df):
        """التنبؤ بمناطق ضرب الستوبات القادمة"""
        if len(df) < 30:
            return {"up_target": None, "down_target": None}

        highs = df["high"].astype(float).values
        lows = df["low"].astype(float).values

        recent_highs = highs[-self.cluster_lookback:]
        sorted_highs = sorted(recent_highs, reverse=True)

        up_target = None
        if len(sorted_highs) >= 2 and sorted_highs[0] - sorted_highs[1] <= sorted_highs[0] * self.liq_threshold:
            up_target = sorted_highs[0]

        recent_lows = lows[-self.cluster_lookback:]
        sorted_lows = sorted(recent_lows)

        down_target = None
        if len(sorted_lows) >= 2 and sorted_lows[1] - sorted_lows[0] <= sorted_lows[0] * self.liq_threshold:
            down_target = sorted_lows[0]

        return {"up_target": up_target, "down_target": down_target}

class GoldenZoneEngine:
    """محرك القاع/القمة الذهبية"""
    def compute(self, df):
        if len(df) < 40:
            return {"type": None, "valid": False}

        high = df["high"].astype(float).values
        low = df["low"].astype(float).values

        swing_high = max(high[-30:])
        swing_low = min(low[-30:])

        f618 = swing_low + 0.618 * (swing_high - swing_low)
        f786 = swing_low + 0.786 * (swing_high - swing_low)

        price = df["close"].iloc[-1]

        if f618 <= price <= f786:
            return {"type": "golden_bottom", "valid": True, "zone": (f618, f786)}
        elif f618 >= price >= f786:
            return {"type": "golden_top", "valid": True, "zone": (f786, f618)}

        return {"type": None, "valid": False}

# ============================================
#  EDGE ALGO ENGINE (RR Zones + SL/TP1/2/3)
# ============================================

class EdgeAlgoEngine:
    """
    مود جديد يحسب Setup كامل:
    - entry_zone
    - stop_loss
    - TP1/TP2/TP3
    - strength / نوع (weak/mid/strong)
    """

    def __init__(self):
        self.last_setup: Optional[Dict[str, Any]] = None

    def compute_setup(
        self,
        df: pd.DataFrame,
        side: Side,
        trend_info: Dict[str, Any],
        smc_ctx: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        يحاول يبني صفقة احترافية من:
        - box / demand / supply
        - stop واضح
        - RR 1:1, 1:2, 1:3
        """
        if len(df) < 30:
            return {"valid": False, "reason": "not_enough_data"}

        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        price = close.iloc[-1]

        lookback = 15
        recent_high = high.tail(lookback).max()
        recent_low = low.tail(lookback).min()

        # Supply/Demand مبسطة
        if side == "BUY":
            entry = price
            sl = recent_low * 0.998
            rr_unit = entry - sl
            if rr_unit <= 0:
                return {"valid": False, "reason": "invalid_rr_buy"}

            tp1 = entry + rr_unit * 1.0
            tp2 = entry + rr_unit * 2.0
            tp3 = entry + rr_unit * 3.0
        else:
            entry = price
            sl = recent_high * 1.002
            rr_unit = sl - entry
            if rr_unit <= 0:
                return {"valid": False, "reason": "invalid_rr_sell"}

            tp1 = entry - rr_unit * 1.0
            tp2 = entry - rr_unit * 2.0
            tp3 = entry - rr_unit * 3.0

        strength_score = 0.0
        tags = []

        if side == "BUY" and trend_info.get("direction") == "up":
            strength_score += 2.0
            tags.append("trend_up")
        if side == "SELL" and trend_info.get("direction") == "down":
            strength_score += 2.0
            tags.append("trend_down")

        if smc_ctx.get("demand_box") and side == "BUY":
            strength_score += 2.0
            tags.append("demand_box")
        if smc_ctx.get("supply_box") and side == "SELL":
            strength_score += 2.0
            tags.append("supply_box")

        if smc_ctx.get("liquidity_sweep"):
            strength_score += 1.0
            tags.append("liq_sweep")

        if smc_ctx.get("stop_hunt_zone"):
            strength_score += 1.0
            tags.append("stop_hunt")

        if trend_info.get("is_strong"):
            strength_score += 1.0
            tags.append("strong_trend")

        if strength_score >= 5:
            grade = "strong"
        elif strength_score >= 3:
            grade = "mid"
        else:
            grade = "weak"

        setup = {
            "valid": True,
            "side": side,
            "entry": entry,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "tp3": tp3,
            "rr1": abs((tp1 - entry) / (entry - sl)) if (entry - sl) != 0 else 1.0,
            "rr2": abs((tp2 - entry) / (entry - sl)) if (entry - sl) != 0 else 2.0,
            "rr3": abs((tp3 - entry) / (entry - sl)) if (entry - sl) != 0 else 3.0,
            "strength_score": strength_score,
            "grade": grade,
            "tags": tags,
        }
        self.last_setup = setup
        return setup

# ============================================
#  ULTRA PRO AI BOT - الإصدار المتكامل النهائي
# ============================================

class UltraProAIBot:
    """البوت الرئيسي المتكامل مع جميع الميزات"""
    
    def __init__(self):
        self.exchange = ExchangeManager()
        self.state = StateManager()
        self.position_manager = SmartPositionManager(self.exchange, self.state)
        self.council = UltraCouncilAI()
        self.running = False
        
    def start(self):
        """بدء تشغيل البوت"""
        log_g("🚀 Starting ULTRA PRO AI Trading Bot - WEB SERVICE EDITION...")
        log_g(f"🔹 Exchange: {EXCHANGE_NAME.upper()}")
        log_g(f"🔹 Symbol: {SYMBOL}")
        log_g(f"🔹 Timeframe: {INTERVAL}")
        log_g(f"🔹 Leverage: {LEVERAGE}x")
        log_g(f"🔹 Risk Allocation: {RISK_ALLOC*100}%")
        log_g(f"🔹 Mode: {'LIVE' if MODE_LIVE else 'PAPER'} {'(DRY RUN)' if DRY_RUN else ''}")
        log_g(f"🔹 Web Service: http://0.0.0.0:{PORT}")
        log_g("🔹 FEATURES: RF Real + EdgeAlgo + SMC + Golden Zones + Trap Mode + Stop-Hunt Prediction + SMART PROFIT ENGINE + Web Service + ULTRA PANEL + ADX+ATR FILTER + RSI Wilder + ADX Wilder")
        
        balance_now = self.exchange.get_balance()
        log_equity_snapshot(balance_now, self.state["compound_pnl"])
        
        self.running = True
    
    def stop(self):
        """إيقاف البوت"""
        self.running = False
        log_i("🛑 Bot stopped by user")
    
    def trade_loop(self):
        """حلقة التداول الرئيسية"""
        consecutive_errors = 0
        max_errors = 5

        while self.running:
            try:
                df = self.exchange.fetch_ohlcv(limit=100)
                if df.empty:
                    time.sleep(5)
                    continue

                current_price = self.exchange.get_current_price()
                balance = self.exchange.get_balance()

                if not current_price:
                    time.sleep(5)
                    continue

                # تحديث state بالرصيد
                self.state["balance"] = balance
                self.state.setdefault("compound_pnl", 0.0)
                self.state["mode_live"] = MODE_LIVE

                # Snapshot للرصيد كل دورة
                log_equity_snapshot(balance, self.state.get("compound_pnl", 0.0))

                if not self.state["open"]:
                    self._handle_trading_decision(df, current_price, balance)
                else:
                    self.position_manager.manage_position(df)

                consecutive_errors = 0
                time.sleep(10)

            except KeyboardInterrupt:
                self.stop()
                break
            except Exception as e:
                consecutive_errors += 1
                log_e(f"❌ Main loop error: {e}")
                traceback.print_exc()

                if consecutive_errors >= max_errors:
                    log_r("🔴 Too many consecutive errors - restarting loop")
                    time.sleep(60)
                    consecutive_errors = 0
                else:
                    time.sleep(5)

    def _handle_trading_decision(self, df, current_price, balance):
        """معالجة قرار التداول المتكامل"""
        if balance <= 10:
            return
            
        # تحليل السوق عبر مجلس الإدارة المتكامل
        decision, reason, analysis = self.council.should_enter_trade(df)
        
        # تسجيل الـ Ultra Panel
        log_ultra_panel(analysis, self.state)
        
        if analysis.get("signals"):
            log_i(f"🔍 ULTRA Analysis: {', '.join(analysis['signals'][:3])}...")
        
        if decision:
            log_i(f"🎯 ULTRA Decision: {reason}")

            # عرض تفاصيل Edge Algo
            edge_setup = analysis.get("edge_setup")
            if edge_setup and edge_setup.get("valid"):
                log_i(
                    f"🧠 EDGE SETUP | {edge_setup['side']} | "
                    f"Entry: {edge_setup['entry']:.6f} | "
                    f"SL: {edge_setup['sl']:.6f} | "
                    f"TP1: {edge_setup['tp1']:.6f} | "
                    f"TP2: {edge_setup['tp2']:.6f} | "
                    f"TP3: {edge_setup['tp3']:.6f} | "
                    f"Grade: {edge_setup['grade']} | "
                    f"Tags: {edge_setup['tags']}"
                )

            # فتح المركز
            if self.position_manager.open_position(decision.upper(), df, analysis):
                log_g(f"💰 ULTRA Position opened successfully | Signals: {len(analysis['signals'])}")
                
                # عرض تفاصيل Profit Profile
                profile = self.state.get("profit_profile", "SCALP_STRICT")
                log_i(f"📊 PROFIT PROFILE ACTIVATED: {profile} | سيتم إدارة الصفقة تلقائياً")
            else:
                log_e("❌ Failed to open ULTRA position")
        else:
            if analysis.get("confidence", 0) > 0.3:
                log_i(f"⏳ ULTRA Waiting for better opportunity: {reason}")

    def get_status(self):
        """الحصول على حالة البوت"""
        status = {
            "running": self.running,
            "exchange": EXCHANGE_NAME,
            "symbol": SYMBOL,
            "balance": self.exchange.get_balance(),
            "position": self.state.state,
            "version": BOT_VERSION
        }
        
        # إضافة معلومات Profit Engine
        if self.state["open"] and self.state.get("profit_engine_active"):
            status["profit_engine"] = self.position_manager.profit_engine.get_status()
        
        return status

# ============================================
#  WEB SERVICE
# ============================================

app = Flask(__name__)
bot = None

def create_app(bot_instance):
    """إنشاء تطبيق Flask"""
    app = Flask(__name__)
    
    @app.route("/")
    def home():
        return "OK - ULTRA PRO AI BOT LIVE"
    
    @app.route("/health")
    def health():
        return jsonify({
            "status": "ok",
            "mode": "LIVE" if MODE_LIVE else "PAPER",
            "symbol": SYMBOL,
            "exchange": EXCHANGE_NAME,
            "version": BOT_VERSION
        })
    
    @app.route("/metrics")
    def metrics():
        if not bot_instance:
            return jsonify({"error": "Bot not initialized"})
        
        status = bot_instance.get_status()
        return jsonify({
            "status": "running" if bot_instance.running else "stopped",
            "exchange": status["exchange"],
            "symbol": status["symbol"],
            "balance": status["balance"],
            "position_open": status["position"]["open"],
            "position_side": status["position"]["side"],
            "position_pnl": status["position"]["pnl"],
            "compound_pnl": status["position"].get("compound_pnl", 0),
            "total_trades": status["position"].get("total_trades", 0),
            "trade_profile": status["position"].get("trade_profile", "N/A"),
            "profit_profile": status["position"].get("profit_profile", "N/A"),
            "profit_engine_active": status["position"].get("profit_engine_active", False),
            "version": status["version"],
            "timestamp": datetime.now().isoformat()
        })
    
    @app.route("/stop")
    def stop_bot():
        if bot_instance:
            bot_instance.stop()
            return jsonify({"status": "stopping"})
        return jsonify({"error": "Bot not running"})
    
    @app.route("/start")
    def start_bot():
        if bot_instance and not bot_instance.running:
            bot_instance.start()
            return jsonify({"status": "starting"})
        return jsonify({"error": "Bot already running or not initialized"})
    
    return app

# ============================================
#  START APPLICATION
# ============================================

def main():
    """الدالة الرئيسية لتشغيل التطبيق"""
    global bot
    
    try:
        # طباعة البانر المحترف
        log_banner()
        
        # إعداد معالجات الإشارات
        setup_signal_handlers()
        
        # تشغيل KeepAlive loop
        threading.Thread(target=keepalive_loop, daemon=True).start()
        
        # إنشاء البوت
        bot = UltraProAIBot()
        
        # تشغيل البوت
        bot.start()
        
        # تشغيل حلقة التداول في خيط منفصل
        trade_thread = threading.Thread(target=bot.trade_loop, daemon=True)
        trade_thread.start()
        
        log_g(f"🌐 Web Service starting on port {PORT}...")
        
        # إنشاء وتشغيل Flask
        app_instance = create_app(bot)
        app_instance.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
        
    except KeyboardInterrupt:
        log_i("🛑 Application stopped by user")
    except Exception as e:
        log_e(f"🔴 Fatal error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
