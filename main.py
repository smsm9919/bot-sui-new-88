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

# Bot Version
BOT_VERSION = f"ULTRA PRO AI v10.0 - MASTER EDITION - {EXCHANGE_NAME.upper()}"

print(f"🚀 Booting: {BOT_VERSION}", flush=True)

# ============================================
#  LOGGING SYSTEM
# ============================================

class ColorLogger:
    """نظام التسجيل الملوّن المحترف"""
    
    COLORS = {
        'INFO': '\033[94m',      # أزرق
        'SUCCESS': '\033[92m',   # أخضر
        'WARNING': '\033[93m',   # أصفر
        'ERROR': '\033[91m',     # أحمر
        'CRITICAL': '\033[95m',  # بنفسجي
        'RESET': '\033[0m'       # إعادة الضبط
    }
    
    @staticmethod
    def log(level, message):
        color = ColorLogger.COLORS.get(level, ColorLogger.COLORS['RESET'])
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{color}{timestamp} | {level} | {message}{ColorLogger.COLORS['RESET']}", flush=True)
    
    @staticmethod
    def info(msg): ColorLogger.log('INFO', msg)
    @staticmethod
    def success(msg): ColorLogger.log('SUCCESS', msg)
    @staticmethod
    def warning(msg): ColorLogger.log('WARNING', msg)
    @staticmethod
    def error(msg): ColorLogger.log('ERROR', msg)
    @staticmethod
    def critical(msg): ColorLogger.log('CRITICAL', msg)

# إختصار الدوال
log_i = ColorLogger.info
log_g = ColorLogger.success
log_w = ColorLogger.warning
log_e = ColorLogger.error
log_r = ColorLogger.critical

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
            
            # تحميل الأسواق
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
            log_e(f"❌ Failed to get current price: {e}")
            return None
    
    def get_balance(self):
        """الحصول على الرصيد"""
        if not MODE_LIVE:
            return 100.0
            
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
                # إعدادات خاصة بالبورصة
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
                    None,  # السعر غير مطلوب للأوامر السوقية
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
            "trade_type": "normal",  # normal, trap, golden, predictive
            "tp1_hit": False,
            "tp2_hit": False
        }
        self.state_file = "bot_state.json"
        self.load_state()
    
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
            "tp2_hit": False
        })
        self.save_state()
    
    def __getitem__(self, key):
        return self.state.get(key)
    
    def __setitem__(self, key, value):
        self.state[key] = value
        self.save_state()

# ============================================
#  STOP HUNT DETECTION ENGINE
# ============================================

class StopHuntDetector:
    """محرك كشف مناطق ضرب الستوبات"""
    
    def __init__(self):
        self.swing_highs = deque(maxlen=10)
        self.swing_lows = deque(maxlen=10)
        self.liquidity_zones = []
        self.recent_stop_hunts = deque(maxlen=5)
        
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
        """كشف مناطق ضرب الستوبات"""
        if len(df) < 10:
            return []
            
        stop_hunt_zones = []
        highs = df['high'].astype(float)
        lows = df['low'].astype(float)
        closes = df['close'].astype(float)
        volumes = df['volume'].astype(float)
        
        for i in range(5, len(df)-1):
            # كشف Stop Hunt صاعد (شرائي)
            if (lows.iloc[i] < lows.iloc[i-1] and  # كسر قاع
                closes.iloc[i] > lows.iloc[i-1] and  # إغلاق فوق القاع
                volumes.iloc[i] > volumes.iloc[i-1:i-4:-1].mean() * 1.5):  # حجم مرتفع
                
                stop_hunt_zones.append({
                    "type": "buy_stop_hunt",
                    "level": lows.iloc[i-1],
                    "high": highs.iloc[i],
                    "index": i,
                    "strength": 3.0
                })
            
            # كشف Stop Hunt هابط (بيعي)
            if (highs.iloc[i] > highs.iloc[i-1] and  # كسر قمة
                closes.iloc[i] < highs.iloc[i-1] and  # إغلاق تحت القمة
                volumes.iloc[i] > volumes.iloc[i-1:i-4:-1].mean() * 1.5):  # حجم مرتفع
                
                stop_hunt_zones.append({
                    "type": "sell_stop_hunt", 
                    "level": highs.iloc[i-1],
                    "low": lows.iloc[i],
                    "index": i,
                    "strength": 3.0
                })
                
        self.recent_stop_hunts.extend(stop_hunt_zones[-3:])
        return stop_hunt_zones[-3:]
    
    def get_active_stop_hunt_zones(self, current_price):
        """الحصول على مناطق الستوب هانت النشطة"""
        active_zones = []
        for zone in self.recent_stop_hunts:
            if zone["type"] == "buy_stop_hunt" and current_price > zone["level"] * 0.995:
                active_zones.append(zone)
            elif zone["type"] == "sell_stop_hunt" and current_price < zone["level"] * 1.005:
                active_zones.append(zone)
        return active_zones

# ============================================
#  STOP-HUNT PREDICTION ENGINE
# ============================================

class StopHuntPredictor:
    """محرك توقع مناطق ضرب الستوبات القادمة"""

    def __init__(self):
        self.liq_threshold = 0.003   # 0.3% = منطقة محتملة للسيولة
        self.cluster_lookback = 15
        self.min_cluster = 2

    def predict(self, df):
        """التنبؤ بمناطق ضرب الستوبات القادمة"""
        if len(df) < 30:
            return {"up_target": None, "down_target": None}

        highs = df["high"].astype(float).values
        lows = df["low"].astype(float).values

        # 1) تجميع قمم متقاربة = سيولة فوق
        recent_highs = highs[-self.cluster_lookback:]
        sorted_highs = sorted(recent_highs, reverse=True)

        up_target = None
        if len(sorted_highs) >= 2 and sorted_highs[0] - sorted_highs[1] <= sorted_highs[0] * self.liq_threshold:
            up_target = sorted_highs[0]

        # 2) تجميع قيعان متقاربة = سيولة تحت
        recent_lows = lows[-self.cluster_lookback:]
        sorted_lows = sorted(recent_lows)

        down_target = None
        if len(sorted_lows) >= 2 and sorted_lows[1] - sorted_lows[0] <= sorted_lows[0] * self.liq_threshold:
            down_target = sorted_lows[0]

        return {
            "up_target": up_target,
            "down_target": down_target
        }

# ============================================
#  FVG DETECTION ENGINE
# ============================================

class FVGDetector:
    """محرك كشف فجوات القيمة العادلة"""
    
    def __init__(self):
        self.valid_fvg_threshold = 0.3  # 30% من المدى الحديث
        self.volume_threshold = 1.2     # زيادة 20% في الحجم
        
    def detect_fvg(self, df):
        """كشف فجوات القيمة العادلة (FVG)"""
        if len(df) < 4:
            return None
            
        try:
            # تحويل البيانات
            candles = []
            for i in range(len(df)):
                candles.append({
                    'open': float(df['open'].iloc[i]),
                    'high': float(df['high'].iloc[i]),
                    'low': float(df['low'].iloc[i]),
                    'close': float(df['close'].iloc[i])
                })
            
            if len(candles) < 4:
                return None
                
            a = candles[-4]  # الشمعة الدافعة
            b = candles[-3]  # شمعة الفجوة
            c = candles[-2]  # شمعة التأكيد

            # Bullish FVG
            if a['high'] < c['low']:
                return {
                    "type": "bullish",
                    "low": a['high'],
                    "high": c['low'],
                    "mid": (a['high'] + c['low']) / 2,
                    "strength": self.calculate_fvg_strength(df, "bullish")
                }

            # Bearish FVG  
            if a['low'] > c['high']:
                return {
                    "type": "bearish",
                    "low": c['high'],
                    "high": a['low'],
                    "mid": (c['high'] + a['low']) / 2,
                    "strength": self.calculate_fvg_strength(df, "bearish")
                }

        except Exception as e:
            log_w(f"⚠️ FVG detection error: {e}")
            
        return None
    
    def calculate_fvg_strength(self, df, fvg_type):
        """حساب قوة FVG"""
        try:
            highs = df["high"].astype(float).values
            lows = df["low"].astype(float).values
            volumes = df["volume"].astype(float).values
            
            # حساب المدى الحديث
            recent_range = max(highs[-5:]) - min(lows[-5:])
            
            # حساب حجم FVG
            if fvg_type == "bullish":
                fvg_low = highs[-4]
                fvg_high = lows[-2]
            else:
                fvg_low = highs[-2]
                fvg_high = lows[-4]
                
            fvg_range = fvg_high - fvg_low
            
            # فجوة واضحة
            displacement_ok = fvg_range >= self.valid_fvg_threshold * recent_range
            
            # حجم مرتفع
            volume_ma = df["volume"].rolling(20).mean().iloc[-2]
            volume_ok = volumes[-2] > volume_ma * self.volume_threshold if volume_ma > 0 else False
            
            # حساب القوة النهائية
            strength = 0.0
            if displacement_ok:
                strength += 2.0
            if volume_ok:
                strength += 1.0
                
            return min(strength, 3.0)
            
        except Exception as e:
            log_w(f"⚠️ FVG strength calculation error: {e}")
            return 1.0
    
    def classify_fvg_context(self, df, fvg_signal):
        """تصنيف FVG حقيقي vs وهمي"""
        if not fvg_signal or len(df) < 30:
            return {"real": False, "stop_hunt": False, "reason": "no_fvg"}
            
        try:
            closes = df["close"].astype(float).values
            highs = df["high"].astype(float).values  
            lows = df["low"].astype(float).values
            
            last_close = closes[-1]
            zone_mid = fvg_signal["mid"]
            
            # تحليل اختراق المنطقة
            touched_zone = (lows[-1] <= fvg_signal["high"] and highs[-1] >= fvg_signal["low"])
            
            if fvg_signal["type"] == "bullish":
                respected = touched_zone and last_close > zone_mid
                invalidated = touched_zone and last_close < fvg_signal["low"]
            else:
                respected = touched_zone and last_close < zone_mid
                invalidated = touched_zone and last_close > fvg_signal["high"]
            
            # كشف Stop Hunt داخل FVG
            last_high = highs[-1]
            last_low = lows[-1] 
            last_body = abs(closes[-1] - df["open"].astype(float).values[-1])
            last_range = max(last_high - last_low, 1e-9)
            
            upper_wick = last_high - max(closes[-1], df["open"].astype(float).values[-1])
            lower_wick = min(closes[-1], df["open"].astype(float).values[-1]) - last_low
            
            stop_hunt = False
            if fvg_signal["type"] == "bullish":
                if last_low < fvg_signal["low"] and closes[-1] > fvg_signal["low"] and lower_wick > 0.6 * last_range:
                    stop_hunt = True
            else:
                if last_high > fvg_signal["high"] and closes[-1] < fvg_signal["high"] and upper_wick > 0.6 * last_range:
                    stop_hunt = True
                    
            real_fvg = fvg_signal["strength"] >= 2.0 and respected and not invalidated
            
            return {
                "real": real_fvg,
                "stop_hunt": stop_hunt,
                "type": fvg_signal["type"],
                "reason": "real_fvg" if real_fvg else "fake_fvg",
                "strength": fvg_signal["strength"]
            }
            
        except Exception as e:
            log_w(f"⚠️ FVG classification error: {e}")
            return {"real": False, "stop_hunt": False, "reason": f"error: {e}"}

# ============================================
#  TREND ANALYSIS ENGINE
# ============================================

class TrendAnalyzer:
    """محرك تحليل الاتجاه"""
    
    def __init__(self):
        self.fast_ma = deque(maxlen=20)
        self.slow_ma = deque(maxlen=50)
        self.trend = "flat"
        self.strength = 0.0
        self.momentum = 0.0
        
    def update(self, df):
        """تحديث تحليل الاتجاه"""
        if len(df) < 10:
            return
            
        close_prices = df['close'].astype(float)
        current_close = close_prices.iloc[-1]
        
        # تحديث المتوسطات المتحركة
        self.fast_ma.append(current_close)
        self.slow_ma.append(current_close)
        
        if len(self.slow_ma) < 10:
            return
            
        # حساب المتوسطات
        fast_avg = sum(self.fast_ma) / len(self.fast_ma)
        slow_avg = sum(self.slow_ma) / len(self.slow_ma)
        
        # حساب قوة الاتجاه
        delta = fast_avg - slow_avg
        self.strength = abs(delta) / slow_avg * 100 if slow_avg != 0 else 0
        
        # حساب الزخم
        if len(close_prices) >= 5:
            recent = close_prices.tail(5).values
            self.momentum = (recent[-1] - recent[0]) / recent[0] * 100 if recent[0] != 0 else 0
            
        # تحديد الاتجاه
        if delta > 0 and self.strength > 0.1:
            self.trend = "up"
        elif delta < 0 and self.strength > 0.1:
            self.trend = "down" 
        else:
            self.trend = "flat"
            
    def is_strong_trend(self):
        """التحقق من قوة الاتجاه"""
        return self.strength > 0.3 and abs(self.momentum) > 0.5
    
    def get_trend_info(self):
        """الحصول على معلومات الاتجاه"""
        return {
            "direction": self.trend,
            "strength": self.strength,
            "momentum": self.momentum,
            "is_strong": self.is_strong_trend()
        }

# ============================================
#  RANGE FILTER ENGINE (RF الحقيقي)
# ============================================

class RangeFilterEngine:
    """محرك RF الحقيقي كمؤشر مساعد"""

    def __init__(self, period: int = 20, qty: float = 3.5):
        self.period = period
        self.qty = qty

    def compute(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        يحسب:
        df['rf_filt'], df['rf_dir'], df['rf_buy_signal'], df['rf_sell_signal']
        """
        if df.empty or len(df) < self.period + 5:
            return {
                "filt": None,
                "dir": 0,
                "buy_signal": False,
                "sell_signal": False,
            }

        close = df["close"].astype(float)

        # avrng = EMA(|close - close.shift(1)|, n)
        diff = close.diff().abs()
        avrng = diff.ewm(span=self.period, adjust=False).mean()

        # AC = EMA(avrng, wper) * qty حيث wper = 2*n - 1
        wper = 2 * self.period - 1
        ac = avrng.ewm(span=wper, adjust=False).mean() * self.qty

        filt = [close.iloc[0]]
        rf_dir = [0]
        buy_sig = [0]
        sell_sig = [0]

        for i in range(1, len(close)):
            c = close.iloc[i]
            prev_filt = filt[-1]
            thr = ac.iloc[i]

            # منطق Range Filter
            if c - prev_filt > thr:
                new_filt = c - thr
            elif prev_filt - c > thr:
                new_filt = c + thr
            else:
                new_filt = prev_filt

            # الاتجاه
            if new_filt > prev_filt:
                d = 1
            elif new_filt < prev_filt:
                d = -1
            else:
                d = rf_dir[-1] if rf_dir[-1] != 0 else 0

            # إشارات تقاطع الاتجاه
            bs = 1 if d == 1 and rf_dir[-1] == -1 else 0
            ss = 1 if d == -1 and rf_dir[-1] == 1 else 0

            filt.append(new_filt)
            rf_dir.append(d)
            buy_sig.append(bs)
            sell_sig.append(ss)

        df["rf_filt"] = pd.Series(filt, index=df.index)
        df["rf_dir"] = pd.Series(rf_dir, index=df.index)
        df["rf_buy_signal"] = pd.Series(buy_sig, index=df.index)
        df["rf_sell_signal"] = pd.Series(sell_sig, index=df.index)

        last_idx = df.index[-1]
        return {
            "filt": float(df.loc[last_idx, "rf_filt"]),
            "dir": int(df.loc[last_idx, "rf_dir"]),
            "buy_signal": bool(df.loc[last_idx, "rf_buy_signal"] == 1),
            "sell_signal": bool(df.loc[last_idx, "rf_sell_signal"] == 1),
        }

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
            # stop تحت قاع recent + قليل buffer
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

        # تقييم القوة (weak/mid/strong)
        strength_score = 0.0
        tags = []

        if side == "BUY" and trend_info["direction"] == "up":
            strength_score += 2.0
            tags.append("trend_up")
        if side == "SELL" and trend_info["direction"] == "down":
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
            "rr1": abs((tp1 - entry) / (entry - sl)),
            "rr2": abs((tp2 - entry) / (entry - sl)),
            "rr3": abs((tp3 - entry) / (entry - sl)),
            "strength_score": strength_score,
            "grade": grade,
            "tags": tags,
        }
        self.last_setup = setup
        return setup

# ============================================
#  SMC CONTEXT ENGINE
# ============================================

class SMCContextEngine:
    """
    يربط بين:
    - مناطق السيولة + stop hunts
    - فجوات القيمة (حقيقية / فيك)
    - supply/demand مبسطة
    """

    def build_context(
        self,
        df: pd.DataFrame,
        current_price: float,
        stop_hunt_info: Dict[str, Any],
        fvg_ctx: Dict[str, Any],
        liquidity_zones: list,
    ) -> Dict[str, Any]:
        ctx = {
            "supply_box": False,
            "demand_box": False,
            "liquidity_sweep": False,
            "fake_break": False,
            "spring": False,
            "stop_hunt_zone": False,
            "fvg_valid": fvg_ctx.get("real", False),
            "fvg_type": fvg_ctx.get("type"),
            "golden_zone": None,
        }

        # Supply / Demand مبسطة
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        lookback = 20
        recent_high = high.tail(lookback).max()
        recent_low = low.tail(lookback).min()

        if current_price >= recent_high * 0.995:
            ctx["supply_box"] = True
        if current_price <= recent_low * 1.005:
            ctx["demand_box"] = True

        # stop hunt active؟
        if stop_hunt_info.get("active_count", 0) > 0:
            ctx["stop_hunt_zone"] = True

        # liquidity sweep مبسطة
        for z_type, level in liquidity_zones:
            diff_pct = abs(current_price - level) / current_price
            if diff_pct < 0.002:
                ctx["liquidity_sweep"] = True

        # FVG فيك = تحذير
        if fvg_ctx and not fvg_ctx.get("real", False) and fvg_ctx.get("reason", "").startswith("fake"):
            ctx["fake_break"] = True

        return ctx

# ============================================
#  BOX REJECTION ENGINE
# ============================================

class BoxRejectionEngine:
    """محرك رفض البوكس مع VWAP"""
    
    def __init__(self):
        self.box_quality_threshold = 4.0
        
    def detect_supply_demand_zones(self, df, lookback=20):
        """كشف مناطق العرض والطلب المبسطة"""
        if len(df) < lookback:
            return {"supply_zone": None, "demand_zone": None}
            
        highs = df['high'].astype(float)
        lows = df['low'].astype(float)
        current_price = float(df['close'].iloc[-1])
        
        recent_high = highs.tail(lookback).max()
        recent_low = lows.tail(lookback).min()
        
        supply_zone = recent_high if current_price >= recent_high * 0.995 else None
        demand_zone = recent_low if current_price <= recent_low * 1.005 else None
        
        return {
            "supply_zone": supply_zone,
            "demand_zone": demand_zone,
            "in_supply": supply_zone is not None,
            "in_demand": demand_zone is not None
        }
    
    def analyze_rejection(self, df, current_price, side):
        """تحليل رفض البوكس"""
        zones = self.detect_supply_demand_zones(df)
        
        if side == "BUY" and zones["in_demand"]:
            # تحليل رفض للشراء من منطقة طلب
            last_candle = df.iloc[-1]
            low = float(last_candle['low'])
            close = float(last_candle['close']) 
            open_price = float(last_candle['open'])
            
            # شمعة رفض مع ذيل سفلي طويل
            body = abs(close - open_price)
            lower_wick = min(close, open_price) - low
            total_range = max(float(last_candle['high']) - low, 0.001)
            
            if lower_wick > body and lower_wick > 0.4 * total_range:
                return {
                    "valid": True,
                    "type": "demand_rejection",
                    "strength": min(2.0, lower_wick / total_range * 3),
                    "zone": zones["demand_zone"]
                }
                
        elif side == "SELL" and zones["in_supply"]:
            # تحليل رفض للبيع من منطقة عرض
            last_candle = df.iloc[-1]
            high = float(last_candle['high'])
            close = float(last_candle['close'])
            open_price = float(last_candle['open'])
            
            # شمعة رفض مع ذيل علوي طويل
            body = abs(close - open_price)
            upper_wick = high - max(close, open_price)
            total_range = max(high - float(last_candle['low']), 0.001)
            
            if upper_wick > body and upper_wick > 0.4 * total_range:
                return {
                    "valid": True, 
                    "type": "supply_rejection",
                    "strength": min(2.0, upper_wick / total_range * 3),
                    "zone": zones["supply_zone"]
                }
        
        return {"valid": False}

# ============================================
#  ADVANCED FVG DETECTION
# ============================================

class AdvancedFVGDetector:
    """FVG متقدم مع تصنيف حقيقي/فيك + كشف ستوب هانت"""
    
    def __init__(self):
        self.basic_detector = FVGDetector()
        
    def detect_advanced_fvg(self, df):
        """كشف FVG متقدم مع تحليل السياق"""
        basic_fvg = self.basic_detector.detect_fvg(df)
        if not basic_fvg:
            return None
            
        # تصنيف متقدم
        classification = self.basic_detector.classify_fvg_context(df, basic_fvg)
        
        return {
            **basic_fvg,
            "classification": classification,
            "is_real_fvg": classification["real"],
            "has_stop_hunt": classification["stop_hunt"],
            "trading_zone": self.analyze_trading_zone(df, basic_fvg)
        }
    
    def analyze_trading_zone(self, df, fvg_signal):
        """تحليل منطقة التداول حول FVG"""
        if not fvg_signal:
            return "neutral"
            
        current_price = float(df['close'].iloc[-1])
        zone_low = fvg_signal.get('low', current_price)
        zone_high = fvg_signal.get('high', current_price)
        zone_mid = (zone_low + zone_high) / 2
        
        # تحديد الموقع الحالي بالنسبة للمنطقة
        if current_price > zone_high:
            return "above_zone"
        elif current_price < zone_low:
            return "below_zone" 
        elif current_price > zone_mid:
            return "upper_zone"
        else:
            return "lower_zone"

# ============================================
#  GOLDEN ZONE ENGINE
# ============================================

class GoldenZoneEngine:
    """محرك القاع/القمة الذهبية"""

    def compute(self, df):
        if len(df) < 40:
            return {"type": None, "valid": False}

        high = df["high"].astype(float).values
        low = df["low"].astype(float).values

        swing_high = max(high[-30:])
        swing_low = min(low[-30:])

        # مستويات فيبو
        f618 = swing_low + 0.618 * (swing_high - swing_low)
        f786 = swing_low + 0.786 * (swing_high - swing_low)

        price = df["close"].iloc[-1]

        if f618 <= price <= f786:
            return {"type": "golden_bottom", "valid": True, "zone": (f618, f786)}
        elif f618 >= price >= f786:
            return {"type": "golden_top", "valid": True, "zone": (f786, f618)}

        return {"type": None, "valid": False}

# ============================================
#  ULTRA COUNCIL AI - نظام التصويت الذكي المتكامل
# ============================================

class UltraCouncilAI:
    """مجلس الإدارة الذكي المتكامل مع جميع المحركات"""
    
    def __init__(self):
        # المحركات الأساسية
        self.stop_hunt_detector = StopHuntDetector()
        self.fvg_detector = FVGDetector()
        self.trend_analyzer = TrendAnalyzer()
        
        # المحركات المتقدمة
        self.rf_engine = RangeFilterEngine(period=20, qty=3.5)
        self.edge_algo = EdgeAlgoEngine()
        self.smc_ctx_engine = SMCContextEngine()
        self.box_rejection_engine = BoxRejectionEngine()
        self.advanced_fvg = AdvancedFVGDetector()
        self.golden_engine = GoldenZoneEngine()
        self.sh_predictor = StopHuntPredictor()  # المحرك الجديد للتنبؤ بالستوب هانت
        
        # معايير القرار
        self.min_confidence = 0.6
        self.min_score = 8

    def analyze_market(self, df):
        """تحليل السوق الشامل المتكامل"""
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
            
            # 1. RF الحقيقي
            rf_state = self.rf_engine.compute(df)
            if rf_state["buy_signal"] and current_price > (rf_state["filt"] or current_price):
                score_buy += 1.5
                signals.append("📡 RF BUY Signal")
            if rf_state["sell_signal"] and current_price < (rf_state["filt"] or current_price):
                score_sell += 1.5
                signals.append("📡 RF SELL Signal")

            # 2. الستوب هانت والسيولة
            self.stop_hunt_detector.detect_swings(df)
            stop_hunt_zones = self.stop_hunt_detector.detect_stop_hunt_zones(df)
            active_zones = self.stop_hunt_detector.get_active_stop_hunt_zones(current_price)
            active_count = len(active_zones)

            for zone in active_zones:
                if zone["type"] == "buy_stop_hunt":
                    score_buy += zone["strength"]
                    signals.append(f"🔄 Buy Stop Hunt (Strength: {zone['strength']})")
                elif zone["type"] == "sell_stop_hunt":
                    score_sell += zone["strength"]
                    signals.append(f"🔄 Sell Stop Hunt (Strength: {zone['strength']})")

            # منطق TRAP MODE (استغلال ضرب الاستوبات)
            trap_side = None
            trap_quality = 0.0

            for zone in active_zones:
                # Buy Stop Hunt = تصفية Longs ساذجة / ضرب استوبات تحت القاع
                if zone["type"] == "buy_stop_hunt" and trend_info["direction"] == "up":
                    # ندور على BUY عكسي بعد الرجوع فوق المستوى
                    trap_side = "BUY"
                    trap_quality = max(trap_quality, zone["strength"] + 1.0)
                    signals.append(f"🧨 TRAP_LONG_ZONE @ {zone['level']:.6f}")
                
                # Sell Stop Hunt = تصفية Shorts / ضرب استوبات فوق القمة
                if zone["type"] == "sell_stop_hunt" and trend_info["direction"] == "down":
                    trap_side = "SELL"
                    trap_quality = max(trap_quality, zone["strength"] + 1.0)
                    signals.append(f"🧨 TRAP_SHORT_ZONE @ {zone['level']:.6f}")

            # 3. FVG المتقدم
            fvg_advanced = self.advanced_fvg.detect_advanced_fvg(df)
            if fvg_advanced and fvg_advanced["is_real_fvg"]:
                if fvg_advanced["type"] == "bullish":
                    score_buy += fvg_advanced["strength"]
                    signals.append(f"🎯 Real Bullish FVG (Strength: {fvg_advanced['strength']})")
                else:
                    score_sell += fvg_advanced["strength"]
                    signals.append(f"🎯 Real Bearish FVG (Strength: {fvg_advanced['strength']})")

            if fvg_advanced and fvg_advanced["has_stop_hunt"]:
                if fvg_advanced["type"] == "bullish":
                    score_buy += 2.0
                    signals.append("🎯 Bullish FVG with Stop Hunt")
                else:
                    score_sell += 2.0
                    signals.append("🎯 Bearish FVG with Stop Hunt")

            # 4. رفض البوكس
            buy_rejection = self.box_rejection_engine.analyze_rejection(df, current_price, "BUY")
            sell_rejection = self.box_rejection_engine.analyze_rejection(df, current_price, "SELL")
            
            if buy_rejection["valid"]:
                score_buy += buy_rejection["strength"]
                signals.append(f"📦 {buy_rejection['type']} (Strength: {buy_rejection['strength']})")
                
            if sell_rejection["valid"]:
                score_sell += sell_rejection["strength"]
                signals.append(f"📦 {sell_rejection['type']} (Strength: {sell_rejection['strength']})")

            # 5. السيولة
            liquidity_zones = self.stop_hunt_detector.detect_liquidity_zones(current_price)
            for zone_type, level in liquidity_zones:
                price_diff_pct = abs(current_price - level) / current_price
                if price_diff_pct < 0.005:
                    if zone_type == "buy_liquidity":
                        score_buy += 2.0
                        signals.append("💧 Buy Liquidity Zone")
                    elif zone_type == "sell_liquidity":
                        score_sell += 2.0
                        signals.append("💧 Sell Liquidity Zone")

            # 6. Golden Zones
            golden = self.golden_engine.compute(df)
            if golden["valid"]:
                if golden["type"] == "golden_bottom":
                    score_buy += 2
                    signals.append("🟢 Golden Bottom Zone")
                elif golden["type"] == "golden_top":
                    score_sell += 2
                    signals.append("🔴 Golden Top Zone")

            # 7. الاتجاه والزخم
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

            # 8. Edge Algo Setup
            edge_side = None
            if score_buy > score_sell:
                edge_side = "BUY"
            elif score_sell > score_buy:
                edge_side = "SELL"

            # بناء سياق SMC
            smc_ctx = self.smc_ctx_engine.build_context(
                df, current_price, 
                {"active_count": active_count},
                fvg_advanced.get("classification", {}) if fvg_advanced else {},
                liquidity_zones
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

            # 9. التنبؤ بالستوب هانت (المحرك الجديد)
            predicted_sh = self.sh_predictor.predict(df)
            if predicted_sh.get("up_target"):
                signals.append(f"🎯 Predicted Stop-Hunt UP @ {predicted_sh['up_target']:.6f}")
                score_sell += 1.5   # لأن السوق هيروح يضرب فوق ثم يهبط

            if predicted_sh.get("down_target"):
                signals.append(f"🎯 Predicted Stop-Hunt DOWN @ {predicted_sh['down_target']:.6f}")
                score_buy += 1.5    # لأن السوق هيروح يضرب تحت ثم يصعد

            # الثقة النهائية
            total_score = score_buy + score_sell
            confidence = min(1.0, total_score / 20.0)
            
            return {
                "score_buy": round(score_buy, 2),
                "score_sell": round(score_sell, 2),
                "confidence": round(confidence, 2),
                "signals": signals,
                "trend": trend_info,
                "fvg_analysis": fvg_advanced,
                "stop_hunt_zones": active_count,
                "rf": rf_state,
                "smc_ctx": smc_ctx,
                "edge_setup": edge_setup,
                "box_rejection": {
                    "buy": buy_rejection,
                    "sell": sell_rejection
                },
                "stop_hunt_trap_side": trap_side,
                "stop_hunt_trap_quality": trap_quality,
                "golden_zone": golden,
                "predicted_stop_hunt": predicted_sh  # إضافة التوقعات
            }
            
        except Exception as e:
            log_e(f"❌ Ultra market analysis error: {e}")
            return self._empty_analysis()

    def _empty_analysis(self):
        """تحليل فارغ عند الخطأ"""
        return {
            "score_buy": 0, "score_sell": 0, "confidence": 0, 
            "signals": [], "rf": {}, "edge_setup": None,
            "trend": {"direction": "flat", "strength": 0, "momentum": 0, "is_strong": False},
            "fvg_analysis": None, "stop_hunt_zones": 0, "smc_ctx": {}, 
            "box_rejection": {"buy": {"valid": False}, "sell": {"valid": False}},
            "stop_hunt_trap_side": None, "stop_hunt_trap_quality": 0,
            "golden_zone": {"valid": False},
            "predicted_stop_hunt": {"up_target": None, "down_target": None}
        }

    def should_enter_trade(self, df):
        """تحديد ما إذا كان يجب الدخول في صفقة"""
        analysis = self.analyze_market(df)
        
        # أولاً: لو الثقة قليلة، نجرب TRAP MODE قبل ما نرفض
        if analysis["confidence"] < self.min_confidence:
            trap_side = analysis.get("stop_hunt_trap_side")
            trap_q = analysis.get("stop_hunt_trap_quality", 0.0)

            # لو في منطقة Trap قوية (ضرب استوبات واضح + ترند معاه)
            if trap_side and trap_q >= 3.0:
                entry_signal = trap_side.lower()   # "buy" أو "sell"
                reason = f"TRAP MODE {trap_side} | Stop-Hunt Exploit | Q={trap_q:.1f}"
                return entry_signal, reason, analysis

            # مافيش Trap محترم -> فعلاً Low confidence
            return None, "Low confidence", analysis
        
        entry_signal = None
        reason = ""
        
        # التوقع الخبيث لضرب الاستوبات (المحرك الجديد)
        pred = analysis.get("predicted_stop_hunt", {})

        # لو في target فوق + السعر تحت الهدف + ترند هابط = SELL خبيث
        if pred.get("up_target") and analysis["trend"]["direction"] == "down":
            if analysis["score_sell"] >= self.min_score - 3:
                return "sell", "PREDICTIVE STOP-HUNT SELL", analysis

        # لو في target تحت + السعر فوق الهدف + ترند صاعد = BUY خبيث
        if pred.get("down_target") and analysis["trend"]["direction"] == "up":
            if analysis["score_buy"] >= self.min_score - 3:
                return "buy", "PREDICTIVE STOP-HUNT BUY", analysis

        # Golden Zone Override
        golden = analysis.get("golden_zone", {})
        if golden.get("valid"):
            if golden["type"] == "golden_bottom" and analysis["score_buy"] >= self.min_score - 2:
                entry_signal = "buy"
                reason = f"ULTRA BUY | Golden Override | Score: {analysis['score_buy']} | Confidence: {analysis['confidence']}"
            elif golden["type"] == "golden_top" and analysis["score_sell"] >= self.min_score - 2:
                entry_signal = "sell"
                reason = f"ULTRA SELL | Golden Override | Score: {analysis['score_sell']} | Confidence: {analysis['confidence']}"
        
        if analysis["score_buy"] >= self.min_score and analysis["score_buy"] > analysis["score_sell"]:
            entry_signal = "buy"
            reason = f"ULTRA BUY | Score: {analysis['score_buy']} | Confidence: {analysis['confidence']}"
            
        elif analysis["score_sell"] >= self.min_score and analysis["score_sell"] > analysis["score_buy"]:
            entry_signal = "sell"
            reason = f"ULTRA SELL | Score: {analysis['score_sell']} | Confidence: {analysis['confidence']}"
            
        else:
            reason = f"No clear signal | Buy: {analysis['score_buy']} | Sell: {analysis['score_sell']}"
            
        return entry_signal, reason, analysis

# ============================================
#  SMART POSITION MANAGER
# ============================================

class SmartPositionManager:
    """مدير المراكز الذكي المتكامل"""
    
    def __init__(self, exchange_manager, state_manager):
        self.exchange = exchange_manager
        self.state = state_manager
        self.council = UltraCouncilAI()
        
    def calculate_position_size(self, balance, price):
        """حساب حجم المركز"""
        if balance <= 0 or price <= 0:
            return 0.0
        
        capital = balance * RISK_ALLOC
        notional = capital * LEVERAGE
        size = notional / price
        
        log_i(f"🔹 Position Size: Balance={balance:.2f}, Capital={capital:.2f}, Size={size:.4f}")
        return round(size, 4)
    
    def open_position(self, side, df):
        """فتح مركز جديد"""
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
            
        # تحليل السوق لتحديد نوع الصفقة
        analysis = self.council.analyze_market(df)
        trade_type = "normal"
        
        # تحديد نوع الصفقة
        if analysis.get("stop_hunt_trap_side") and analysis.get("stop_hunt_trap_quality", 0) >= 3.0:
            trade_type = "trap"
        elif analysis.get("golden_zone", {}).get("valid"):
            trade_type = "golden"
        elif "PREDICTIVE STOP-HUNT" in analysis.get("signals", []):
            trade_type = "predictive"
            
        # تنفيذ الأمر
        if self.exchange.execute_order(side, position_size, current_price):
            # حفظ بيانات EdgeAlgo إن وجدت
            edge_setup = analysis.get("edge_setup")
            if edge_setup and edge_setup.get("valid"):
                self.state.update({
                    "sl": edge_setup["sl"],
                    "tp1": edge_setup["tp1"],
                    "tp2": edge_setup["tp2"],
                    "tp3": edge_setup["tp3"],
                    "tp_mode": edge_setup["grade"]
                })
            else:
                self.state.update({
                    "sl": None,
                    "tp1": None,
                    "tp2": None,
                    "tp3": None,
                    "tp_mode": None
                })
                
            self.state.update({
                "open": True,
                "side": side,
                "entry": current_price,
                "qty": position_size,
                "pnl": 0.0,
                "bars": 0,
                "highest_profit_pct": 0.0,
                "profit_targets_achieved": 0,
                "opened_at": time.time(),
                "last_signal": side,
                "trade_type": trade_type,
                "tp1_hit": False,
                "tp2_hit": False
            })
            
            log_g(f"✅ New Position Opened: {side.upper()} | Size: {position_size:.4f} | Entry: {current_price:.6f} | Type: {trade_type.upper()}")
            return True
            
        return False
    
    def manage_position(self, df):
        """إدارة المركز المفتوح"""
        if not self.state["open"]:
            return
            
        current_price = self.exchange.get_current_price()
        if not current_price:
            return
            
        entry_price = self.state["entry"]
        side = self.state["side"]
        trade_type = self.state.get("trade_type", "normal")
        
        # حساب الربح/الخسارة
        if side == "long":
            pnl_pct = (current_price - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - current_price) / entry_price * 100
            
        self.state["pnl"] = pnl_pct
        
        # تحديث أعلى ربح
        if pnl_pct > self.state["highest_profit_pct"]:
            self.state["highest_profit_pct"] = pnl_pct
            
        # تحليل السوق الحالي
        analysis = self.council.analyze_market(df)
        
        # 1. إدارة SL/TP من EdgeAlgo
        sl = self.state["sl"]
        if sl:
            if side == "long" and current_price <= sl:
                return self.close_position("HIT_SL_EDGE")
            if side == "short" and current_price >= sl:
                return self.close_position("HIT_SL_EDGE")

        # إدارة TP بناءً على نوع الصفقة
        self._manage_take_profits(current_price, side, trade_type, pnl_pct)

        # 2. حماية متقدمة بناءً على نوع الصفقة
        exit_reason = self._get_advanced_exit_reason(pnl_pct, analysis, side, trade_type)
        
        if exit_reason:
            self.close_position(exit_reason)
        else:
            self.state["bars"] += 1
    
    def _manage_take_profits(self, current_price, side, trade_type, pnl_pct):
        """إدارة مستويات جني الأرباح"""
        tp1 = self.state["tp1"]
        tp2 = self.state["tp2"] 
        tp3 = self.state["tp3"]
        
        if not tp1:
            return
            
        # TP1
        if not self.state["tp1_hit"]:
            if (side == "long" and current_price >= tp1) or (side == "short" and current_price <= tp1):
                self.state["tp1_hit"] = True
                log_g("🎯 TP1 HIT")
                # في صفقات TRAP نغلق جزء عند TP1
                if trade_type == "trap" and pnl_pct >= 1.5:
                    self._partial_close(50, "TRAP_TP1_PARTIAL")
        
        # TP2
        if not self.state["tp2_hit"] and self.state["tp1_hit"]:
            if (side == "long" and current_price >= tp2) or (side == "short" and current_price <= tp2):
                self.state["tp2_hit"] = True
                log_g("🔥 TP2 HIT")
                # في صفقات GOLDEN نغلق جزء عند TP2
                if trade_type == "golden" and pnl_pct >= 3.0:
                    self._partial_close(30, "GOLDEN_TP2_PARTIAL")
        
        # TP3
        if self.state["tp1_hit"] and self.state["tp2_hit"] and tp3:
            if (side == "long" and current_price >= tp3) or (side == "short" and current_price <= tp3):
                self.close_position("TP3_FINAL")
    
    def _partial_close(self, percentage, reason):
        """إغلاق جزء من المركز"""
        try:
            current_qty = self.state["qty"]
            close_qty = current_qty * (percentage / 100.0)
            side = "sell" if self.state["side"] == "long" else "buy"
            current_price = self.exchange.get_current_price()
            
            if self.exchange.execute_order(side, close_qty, current_price):
                new_qty = current_qty - close_qty
                self.state["qty"] = new_qty
                log_g(f"✅ Partial Close: {percentage}% | Reason: {reason} | New Qty: {new_qty:.4f}")
                return True
        except Exception as e:
            log_e(f"❌ Partial close failed: {e}")
        return False
    
    def _get_advanced_exit_reason(self, pnl_pct, analysis, current_side, trade_type):
        """تحديد سبب الخروج المتقدم بناءً على نوع الصفقة"""
        rf_state = analysis.get("rf", {})
        smc_ctx = analysis.get("smc_ctx", {})

        # حماية RF + SMC للجميع
        bad_zone = False
        reasons = []

        if current_side == "long" and smc_ctx.get("supply_box"):
            bad_zone = True
            reasons.append("supply_box")
        if current_side == "short" and smc_ctx.get("demand_box"):
            bad_zone = True
            reasons.append("demand_box")
        if smc_ctx.get("liquidity_sweep"):
            bad_zone = True
            reasons.append("liquidity_sweep")
        if smc_ctx.get("fake_break"):
            bad_zone = True
            reasons.append("fake_fvg")

        # RF Flip ضد الصفقة
        if current_side == "long" and rf_state.get("sell_signal"):
            bad_zone = True
            reasons.append("rf_flip_sell")
        if current_side == "short" and rf_state.get("buy_signal"):
            bad_zone = True
            reasons.append("rf_flip_buy")

        if bad_zone:
            return f"SMART_EXIT_PROTECT | {'+'.join(reasons)}"

        # استراتيجيات الخروج الخاصة بكل نوع صفقة
        if trade_type == "trap":
            # صفقات TRAP نخرج بسرعة عند تحقيق ربح معقول
            if pnl_pct >= 2.5 and analysis["confidence"] < 0.4:
                return "TRAP_QUICK_PROFIT"
            if pnl_pct <= -1.5:
                return "TRAP_STOP_LOSS"
                
        elif trade_type == "golden":
            # صفقات GOLDEN نعطيها مساحة أكثر
            if pnl_pct >= 4.0 and analysis["confidence"] < 0.3:
                return "GOLDEN_TARGET_REACHED"
            if pnl_pct <= -2.5:
                return "GOLDEN_STOP_LOSS"
                
        elif trade_type == "predictive":
            # صفقات PREDICTIVE نخرج عند تحقق التنبؤ أو تحقيق ربح جيد
            if pnl_pct >= 2.0 and analysis["confidence"] < 0.4:
                return "PREDICTIVE_TARGET_REACHED"
            if pnl_pct <= -1.0:
                return "PREDICTIVE_STOP_LOSS"
                
        else:  # normal
            # صفقات عادية
            if pnl_pct >= 1.5 and analysis["confidence"] < 0.3:
                return "Target Profit Reached"
            if pnl_pct <= -2.0:
                return "Stop Loss"

        # خروج عند انعكاس الإشارات القوي
        if current_side == "long" and analysis["score_sell"] > analysis["score_buy"] + 5:
            return "Strong Sell Signal Reversal"
        elif current_side == "short" and analysis["score_buy"] > analysis["score_sell"] + 5:
            return "Strong Buy Signal Reversal"
            
        # خروج عند ضعف الثقة لفترة طويلة
        if self.state["bars"] > 20 and analysis["confidence"] < 0.2 and pnl_pct > 0:
            return "Low Confidence Exit"
            
        return None
    
    def close_position(self, reason=""):
        """إغلاق المركز الحالي"""
        if not self.state["open"]:
            return
            
        side = "sell" if self.state["side"] == "long" else "buy"
        current_price = self.exchange.get_current_price()
        
        if current_price and self.exchange.execute_order(side, self.state["qty"], current_price):
            log_g(f"✅ Position Closed: {reason} | PnL: {self.state['pnl']:.2f}% | Type: {self.state.get('trade_type', 'normal').upper()}")
            self.state.reset()
            return True
            
        log_e(f"❌ Failed to close position: {reason}")
        return False

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
        log_g("🚀 Starting ULTRA PRO AI Trading Bot - MASTER EDITION...")
        log_g(f"🔹 Exchange: {EXCHANGE_NAME.upper()}")
        log_g(f"🔹 Symbol: {SYMBOL}")
        log_g(f"🔹 Timeframe: {INTERVAL}")
        log_g(f"🔹 Leverage: {LEVERAGE}x")
        log_g(f"🔹 Risk Allocation: {RISK_ALLOC*100}%")
        log_g(f"🔹 Mode: {'LIVE' if MODE_LIVE else 'PAPER'} {'(DRY RUN)' if DRY_RUN else ''}")
        log_g("🔹 FEATURES: RF Real + EdgeAlgo + SMC + Box Rejection + Advanced FVG + Golden Zones + Trap Mode + Stop-Hunt Prediction")
        
        self.running = True
        self._main_loop()
    
    def stop(self):
        """إيقاف البوت"""
        self.running = False
        log_i("🛑 Bot stopped by user")
    
    def _main_loop(self):
        """الحلقة الرئيسية للتداول المتكامل"""
        consecutive_errors = 0
        max_errors = 5
        
        while self.running:
            try:
                # جلب بيانات السوق
                df = self.exchange.fetch_ohlcv(limit=100)
                if df.empty:
                    time.sleep(5)
                    continue
                
                # تحديث السعر والرصيد
                current_price = self.exchange.get_current_price()
                balance = self.exchange.get_balance()
                
                if not current_price:
                    time.sleep(5)
                    continue
                
                # اتخاذ قرار التداول المتكامل
                if not self.state["open"]:
                    self._handle_trading_decision(df, current_price, balance)
                else:
                    self.position_manager.manage_position(df)
                
                # إعادة تعيين عداد الأخطاء
                consecutive_errors = 0
                time.sleep(10)  # انتظار 10 ثواني بين الدورات
                
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
        
        # تسجيل التحليل المتكامل
        if analysis["signals"]:
            log_i(f"🔍 ULTRA Analysis: {', '.join(analysis['signals'])}")
        
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
            if self.position_manager.open_position(decision, df):
                log_g(f"💰 ULTRA Position opened successfully | Signals: {len(analysis['signals'])}")
            else:
                log_e("❌ Failed to open ULTRA position")
        else:
            # تسجيل حالة عدم التداول
            if analysis["confidence"] > 0.3:
                log_i(f"⏳ ULTRA Waiting for better opportunity: {reason}")

    def get_status(self):
        """الحصول على حالة البوت"""
        status = {
            "running": self.running,
            "exchange": EXCHANGE_NAME,
            "symbol": SYMBOL,
            "balance": self.exchange.get_balance(),
            "position": self.state.state
        }
        return status

# ============================================
#  START APPLICATION
# ============================================

def main():
    """الدالة الرئيسية لتشغيل التطبيق"""
    try:
        # إنشاء وتشغيل البوت
        bot = UltraProAIBot()
        
        # تشغيل البوت
        bot.start()
        
    except KeyboardInterrupt:
        log_i("🛑 Application stopped by user")
    except Exception as e:
        log_e(f"🔴 Fatal error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
