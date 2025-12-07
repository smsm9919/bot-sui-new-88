# -*- coding: utf-8 -*-
"""
ULTRA PRO AI BOT - الإصدار المتكامل الذكي المحسن
• نظام كشف وتصنيف مناطق ضرب الستوبات (Stop Hunt Zones)
• تمييز FVG الحقيقي من الوهمي + كشف مصائد السيولة  
• مجلس الإدارة الفائق الذكي مع 20 استراتيجية متقدمة
• نظام ركود الترند الذكي المحترف + RF الحقيقي
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
• AUTO-RECOVERY SYSTEM - استعادة الصفقات بعد إعادة التشغيل
• EMA HOLD ENGINE - نظام ركوب الترند الذكي مع Hold-TP
• EMA CROSS CLASSIFIER - نظام تصنيف تقاطع EMA القوي/الضعيف
• TREND BIRTH ENGINE - نظام كشف ولادة الترند مع دخول OB/FVG
• SMART COOLDOWN SYSTEM - نظام كول داون 30 دقيقة مع استثناء الترند القوي
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
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
from collections import deque
from typing import Literal, Dict, Any, Optional, Tuple
from flask import Flask, jsonify
import threading
import sys
import signal
from termcolor import colored

Side = Literal["BUY", "SELL"]

# =========================
# SMART COOLDOWN SYSTEM FUNCTIONS
# =========================

def is_strong_trend_opportunity(signal: str, analysis: dict) -> bool:
    """
    استثناء للكول داون:
    يسمح فقط لو:
      - إشارة BUY مع ترند صاعد قوي
      - إشارة SELL مع ترند هابط قوي
      - ثقة عالية + Score قوي
    """
    if signal not in ("buy", "sell"):
        return False

    trend = analysis.get("trend", {}) or {}
    direction = trend.get("direction")   # "up" / "down" / None
    is_strong = trend.get("is_strong", False)
    adx = trend.get("adx", 0.0)

    # لازم يكون ترند قوي (ADX > 25 أو is_strong = True)
    if not is_strong and adx < 25:
        return False

    # لازم الدخول يكون مع الترند مش عكسه
    if signal == "buy" and direction != "up":
        return False
    if signal == "sell" and direction != "down":
        return False

    # ثقة مجلس الإدارة
    confidence = float(analysis.get("confidence", 0.0))
    score_buy = float(analysis.get("score_buy", 0.0))
    score_sell = float(analysis.get("score_sell", 0.0))
    signal_score = max(score_buy, score_sell)

    # عتبات الاستثناء
    if confidence < 0.85:
        return False
    if signal_score < 8.5:
        return False

    # شرط RR محترم
    expected_rr = float(analysis.get("edge_rr", 2.0))
    if expected_rr < 1.8:
        return False

    # شرط EMA Cross قوي
    ema_cross = analysis.get("ema_cross", {})
    if ema_cross.get("strength") != "strong":
        return False

    # شرط Golden Zone يعطي أولوية إضافية
    golden_zone = analysis.get("golden_zone", {})
    if golden_zone.get("valid"):
        return True

    return True

def is_cooldown_active(state: dict):
    """
    يرجّع:
      active (bool), minutes_left (float)
    """
    cd_until_str = state.get("cooldown_until")
    if not cd_until_str:
        return False, 0.0

    try:
        cd_until = datetime.fromisoformat(cd_until_str)
    except Exception:
        state["cooldown_until"] = None
        return False, 0.0

    now = datetime.utcnow()
    if now >= cd_until:
        state["cooldown_until"] = None
        return False, 0.0

    mins_left = (cd_until - now).total_seconds() / 60.0
    return True, mins_left

# =========================
# TREND BIRTH ENGINE
# =========================

def trend_birth_engine(df, smc_ctx, rsi_ctx, ema_ctx, flow_ctx):
    """
    Trend Birth + OB/FVG confirmation
    يعمل على:
     1. Liquidity Sweep
     2. Structure Shift (BOS / CHoCH)
     3. Momentum Flip (RSI / Flow)
     4. Retest ذكي على Order Block أو FVG
    """
    result = {
        "active": False,
        "side": None,
        "confidence": 0.0,
        "entry_zone": None,   # OB / FVG
        "reasons": []
    }

    # ===== 1) Liquidity Sweep =====
    liq_grab = smc_ctx.get("liq_grab", {})
    sweep_up = liq_grab.get("grab_up", False)
    sweep_down = liq_grab.get("grab_down", False)

    # ===== 2) Structure Shift =====
    bos = smc_ctx.get("bos")
    choch = smc_ctx.get("choch")

    # ===== 3) Momentum Flip =====
    rsi_flip_up = rsi_ctx.get("trendZ") == "bull"
    rsi_flip_down = rsi_ctx.get("trendZ") == "bear"
    flow_side = flow_ctx.get("flow_side", "NEUTRAL")

    # ===== 4) Order Block / FVG =====
    ob = None
    fvg = None
    
    # محاولة استخراج OB من smc_ctx
    if smc_ctx.get("demand_box"):
        recent_low = df["low"].astype(float).iloc[-20:].min()
        ob = {"type": "bull", "zone": (recent_low * 0.995, recent_low * 1.005)}
    elif smc_ctx.get("supply_box"):
        recent_high = df["high"].astype(float).iloc[-20:].max()
        ob = {"type": "bear", "zone": (recent_high * 0.995, recent_high * 1.005)}
    
    # محاولة استخراج FVG من smc_ctx
    fvg_ctx = smc_ctx.get("fvg", {})
    if fvg_ctx:
        bull_fvg = fvg_ctx.get("bull")
        bear_fvg = fvg_ctx.get("bear")
        if bull_fvg:
            fvg = {"type": "bull", "zone": (bull_fvg.get("lower", 0), bull_fvg.get("upper", 0))}
        elif bear_fvg:
            fvg = {"type": "bear", "zone": (bear_fvg.get("lower", 0), bear_fvg.get("upper", 0))}

    price = float(df["close"].iloc[-1])

    # ========= BUY =========
    if sweep_down and (bos == "up" or choch == "up") and rsi_flip_up:
        if flow_side in ("BUY", "STRONG_BUY"):
            if ob and ob["type"] == "bull" and ob["zone"][0] <= price <= ob["zone"][1]:
                result.update({
                    "active": True,
                    "side": "BUY",
                    "confidence": 0.85,
                    "entry_zone": "OB",
                    "reasons": ["liquidity_sweep", "structure_shift", "momentum_flip", "order_block"]
                })
            elif fvg and fvg["type"] == "bull" and fvg["zone"][0] <= price <= fvg["zone"][1]:
                result.update({
                    "active": True,
                    "side": "BUY",
                    "confidence": 0.80,
                    "entry_zone": "FVG",
                    "reasons": ["liquidity_sweep", "structure_shift", "momentum_flip", "fvg"]
                })

    # ========= SELL =========
    if sweep_up and (bos == "down" or choch == "down") and rsi_flip_down:
        if flow_side in ("SELL", "STRONG_SELL"):
            if ob and ob["type"] == "bear" and ob["zone"][0] <= price <= ob["zone"][1]:
                result.update({
                    "active": True,
                    "side": "SELL",
                    "confidence": 0.85,
                    "entry_zone": "OB",
                    "reasons": ["liquidity_sweep", "structure_shift", "momentum_flip", "order_block"]
                })
            elif fvg and fvg["type"] == "bear" and fvg["zone"][0] <= price <= fvg["zone"][1]:
                result.update({
                    "active": True,
                    "side": "SELL",
                    "confidence": 0.80,
                    "entry_zone": "FVG",
                    "reasons": ["liquidity_sweep", "structure_shift", "momentum_flip", "fvg"]
                })

    return result

# =========================
# INDICATORS ENGINE (BOT GAMED)
# =========================

RSI_LEN = 14
ADX_LEN = 14
ATR_LEN = 14

def wilder_ema(s: pd.Series, n: int) -> pd.Series:
    """Wilder EMA (RMA) نفس المستخدم في بوت جامد"""
    return s.ewm(alpha=1/n, adjust=False).mean()

def compute_indicators(df: pd.DataFrame) -> dict:
    """
    نفس compute_indicators في bot.gamed.py
    يرجّع قيم RSI / ATR / ADX / DI+/DI- على آخر شمعة.
    """
    if len(df) < max(ATR_LEN, RSI_LEN, ADX_LEN) + 2:
        return {
            "rsi": 50.0,
            "plus_di": 0.0,
            "minus_di": 0.0,
            "dx": 0.0,
            "adx": 0.0,
            "atr": 0.0,
        }

    c = df["close"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)

    # True Range + ATR (Wilder)
    tr = pd.concat([
        (h - l).abs(),
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = wilder_ema(tr, ATR_LEN)

    # RSI (Wilder)
    delta = c.diff()
    up = delta.clip(lower=0.0)
    dn = (-delta).clip(lower=0.0)
    rs = wilder_ema(up, RSI_LEN) / wilder_ema(dn, RSI_LEN).replace(0, 1e-12)
    rsi = 100 - (100 / (1 + rs))

    # +DI / -DI / ADX (Wilder)
    up_move = h.diff()
    down_move = l.shift(1) - l

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    plus_di = 100 * (wilder_ema(plus_dm, ADX_LEN) / atr.replace(0, 1e-12))
    minus_di = 100 * (wilder_ema(minus_dm, ADX_LEN) / atr.replace(0, 1e-12))

    dx = (100 * (plus_di - minus_di).abs() /
          (plus_di + minus_di).replace(0, 1e-12)).fillna(0.0)
    adx = wilder_ema(dx, ADX_LEN)

    i = len(df) - 1
    return {
        "rsi": float(rsi.iloc[i]),
        "plus_di": float(plus_di.iloc[i]),
        "minus_di": float(minus_di.iloc[i]),
        "dx": float(dx.iloc[i]),
        "adx": float(adx.iloc[i]),
        "atr": float(atr.iloc[i]),
    }

# =========================
# RANGE FILTER REAL (RF) — PINE EXACT
# =========================

def compute_range_filter(df: pd.DataFrame, period: int = 20, qty: float = 3.5) -> dict:
    """
    تحويل سكريبت Pine Range Filter (DW) إلى Python
    يرجّع:
      - rf_filt, rf_dir
      - rf_buy_signal, rf_sell_signal
      - hi_band, lo_band
    ويضيف الأعمدة دي في df أيضًا.
    """
    src = df["close"].astype(float).copy()

    if len(src) < period + 2:
        # df صغير → رجّع قيم افتراضية
        df["rf_filt"] = src
        df["rf_hi"] = src
        df["rf_lo"] = src
        df["rf_dir"] = 0
        df["rf_buy_signal"] = False
        df["rf_sell_signal"] = False
        return {
            "filt": float(src.iloc[-1]),
            "hi_band": float(src.iloc[-1]),
            "lo_band": float(src.iloc[-1]),
            "dir": 0,
            "buy_signal": False,
            "sell_signal": False,
        }

    # ===== rng_size من Pine =====
    diff = (src - src.shift(1)).abs()
    avrng = diff.ewm(span=period, adjust=False).mean()
    wper = (period * 2) - 1
    ac = avrng.ewm(span=wper, adjust=False).mean() * qty  # AC في Pine

    # ===== rng_filt array logic =====
    filt_vals = []
    hi_vals = []
    lo_vals = []

    # أول قيمة
    first_x = float(src.iloc[0])
    first_r = float(ac.iloc[0])
    cur_filt = first_x
    filt_vals.append(cur_filt)
    hi_vals.append(cur_filt + first_r)
    lo_vals.append(cur_filt - first_r)

    for i in range(1, len(src)):
        x = float(src.iloc[i])
        r = float(ac.iloc[i])
        prev = cur_filt

        # نفس منطق:
        # if x - r > rfilt[1] → rfilt[0] = x - r
        if x - r > prev:
            cur_filt = x - r
        # if x + r < rfilt[1] → rfilt[0] = x + r
        elif x + r < prev:
            cur_filt = x + r
        # else يبقى كما هو

        filt_vals.append(cur_filt)
        hi_vals.append(cur_filt + r)
        lo_vals.append(cur_filt - r)

    rf_filt = pd.Series(filt_vals, index=df.index)
    hi_band = pd.Series(hi_vals, index=df.index)
    lo_band = pd.Series(lo_vals, index=df.index)

    # ===== Direction + Signals من Pine =====
    fdir = [0] * len(src)
    cond_ini = [0] * len(src)
    long_sig = [False] * len(src)
    short_sig = [False] * len(src)

    for i in range(1, len(src)):
        # fdir := filt > filt[1] ? 1 : filt < filt[1] ? -1 : fdir
        if rf_filt.iloc[i] > rf_filt.iloc[i - 1]:
            fdir[i] = 1
        elif rf_filt.iloc[i] < rf_filt.iloc[i - 1]:
            fdir[i] = -1
        else:
            fdir[i] = fdir[i - 1]

        upward = fdir[i] == 1
        downward = fdir[i] == -1

        # longCond / shortCond من Pine بالظبط
        longCond = (
            (src.iloc[i] > rf_filt.iloc[i] and src.iloc[i] > src.iloc[i - 1] and upward)
            or (src.iloc[i] > rf_filt.iloc[i] and src.iloc[i] < src.iloc[i - 1] and upward)
        )
        shortCond = (
            (src.iloc[i] < rf_filt.iloc[i] and src.iloc[i] < src.iloc[i - 1] and downward)
            or (src.iloc[i] < rf_filt.iloc[i] and src.iloc[i] > src.iloc[i - 1] and downward)
        )

        # CondIni := long ? 1 : short ? -1 : CondIni[1]
        if longCond:
            cond_ini[i] = 1
        elif shortCond:
            cond_ini[i] = -1
        else:
            cond_ini[i] = cond_ini[i - 1]

        # longCondition = longCond and CondIni[1] == -1
        if longCond and cond_ini[i - 1] == -1:
            long_sig[i] = True
        # shortCondition = shortCond and CondIni[1] == 1
        if shortCond and cond_ini[i - 1] == 1:
            short_sig[i] = True

    rf_dir = pd.Series(fdir, index=df.index)
    buy_series = pd.Series(long_sig, index=df.index)
    sell_series = pd.Series(short_sig, index=df.index)

    # الحق الأعمدة في df لاستخدامها لاحقاً لو حبّينا
    df["rf_filt"] = rf_filt
    df["rf_hi"] = hi_band
    df["rf_lo"] = lo_band
    df["rf_dir"] = rf_dir
    df["rf_buy_signal"] = buy_series
    df["rf_sell_signal"] = sell_series

    return {
        "filt": float(rf_filt.iloc[-1]),
        "hi_band": float(hi_band.iloc[-1]),
        "lo_band": float(lo_band.iloc[-1]),
        "dir": int(rf_dir.iloc[-1]),
        "buy_signal": bool(buy_series.iloc[-1]),
        "sell_signal": bool(sell_series.iloc[-1]),
    }

# =========================
# VWAP ENGINE (SESSION VWAP)
# =========================

def compute_vwap(df: pd.DataFrame) -> float:
    """
    VWAP الكلاسيكي:
    sum(price * volume) / sum(volume) من بداية البيانات حتى آخر شمعة.
    """
    if "close" not in df.columns or "volume" not in df.columns or len(df) == 0:
        return 0.0

    close = df["close"].astype(float)
    vol = df["volume"].astype(float)

    pv = close * vol
    cum_pv = pv.cumsum()
    cum_vol = vol.cumsum().replace(0, np.nan)

    vwap = cum_pv / cum_vol
    df["vwap"] = vwap

    return float(vwap.iloc[-1])

# =========================
# EMA TREND / HOLD ENGINE
# =========================

def compute_ema_trend(df, fast_len=20, slow_len=50, stair_window=20):
    """
    قراءة قوة الاتجاه من EMA + سلوك السعر حولها (درج الترند).
    ترجع:
      dir: "up" / "down" / "neutral"
      strength: "strong" / "mid" / "weak"
      stair_state: "holding" / "broken"
    """
    if len(df) < slow_len + 5:
        return {
            "ema_fast": None,
            "ema_slow": None,
            "dir": "neutral",
            "strength": "weak",
            "stair_state": "broken",
            "score": 0.0,
        }

    closes = df["close"].astype(float)
    lows   = df["low"].astype(float)
    highs  = df["high"].astype(float)

    ema_fast_series = closes.ewm(span=fast_len, adjust=False).mean()
    ema_slow_series = closes.ewm(span=slow_len, adjust=False).mean()

    ema_fast = float(ema_fast_series.iloc[-1])
    ema_slow = float(ema_slow_series.iloc[-1])
    price    = float(closes.iloc[-1])

    # اتجاه EMA
    if ema_fast > ema_slow * 1.0005:
        ema_dir = "up"
    elif ema_fast < ema_slow * 0.9995:
        ema_dir = "down"
    else:
        ema_dir = "neutral"

    # درج الترند: هل التصحيحات تحترم EMA السريعة / البطيئة؟
    w = min(stair_window, len(df) - 1)
    recent_lows  = lows.iloc[-w:]
    recent_highs = highs.iloc[-w:]

    stair_state = "holding"
    # لو السعر كسر أسفل EMA البطيئة بوضوح أكثر من مرة → كسر درج
    if ema_dir == "up":
        if (recent_lows < ema_slow * 0.997).sum() >= 2:
            stair_state = "broken"
    elif ema_dir == "down":
        if (recent_highs > ema_slow * 1.003).sum() >= 2:
            stair_state = "broken"
    else:
        stair_state = "broken"

    # سكورنغ بسيط لقوة الاتجاه
    score = 0.0

    # علاقة السعر بالـ EMA
    if ema_dir == "up":
        if price > ema_fast: score += 1.0
        if price > ema_slow: score += 1.0
    elif ema_dir == "down":
        if price < ema_fast: score += 1.0
        if price < ema_slow: score += 1.0

    # احترام الدرَج
    if stair_state == "holding":
        score += 0.5
    else:
        score -= 0.5

    # فرق EMA
    spread = abs(ema_fast - ema_slow) / max(ema_slow, 1e-9) * 10000  # بالنِّقاط
    if spread >= 8:
        score += 0.5
    if spread >= 15:
        score += 0.5

    # تحويل score إلى strength نصي
    if score >= 2.0:
        strength = "strong"
    elif score >= 0.8:
        strength = "mid"
    else:
        strength = "weak"

    return {
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
        "dir": ema_dir,
        "strength": strength,      # strong / mid / weak
        "stair_state": stair_state,
        "score": float(score),
    }

# ================= EMA CROSS CLASSIFIER =================

def classify_ema_crossover(df: pd.DataFrame, fast: int = 20, slow: int = 50, lookback: int = 4) -> dict:
    """
    تصنيف تقاطع EMA إلى:
      - strong_bull / weak_bull
      - strong_bear / weak_bear
      - none
    """
    try:
        if len(df) < slow + lookback + 2:
            return {
                "label": "none",
                "side": None,
                "strength": None,
                "fast": None,
                "slow": None,
            }

        closes = df["close"].astype(float)
        ema_fast = closes.ewm(span=fast, adjust=False).mean()
        ema_slow = closes.ewm(span=slow, adjust=False).mean()

        f_prev, f_now = float(ema_fast.iloc[-2]), float(ema_fast.iloc[-1])
        s_prev, s_now = float(ema_slow.iloc[-2]), float(ema_slow.iloc[-1])

        crossed_up = (f_prev <= s_prev) and (f_now > s_now)
        crossed_down = (f_prev >= s_prev) and (f_now < s_now)

        if not (crossed_up or crossed_down):
            return {
                "label": "none",
                "side": None,
                "strength": None,
                "fast": f_now,
                "slow": s_now,
            }

        side = "bull" if crossed_up else "bear"

        # زاوية الحركة: EMA السريع لازم يتحرك أسرع من البطيء
        f_old = float(ema_fast.iloc[-1 - lookback])
        s_old = float(ema_slow.iloc[-1 - lookback])
        fast_slope = abs(f_now - f_old)
        slow_slope = abs(s_now - s_old)
        slope_ratio = fast_slope / (slow_slope + 1e-9)

        price = float(closes.iloc[-1])

        # موضع السعر بعد التقاطع
        if side == "bull":
            price_ok = (price > f_now) and (price > s_now)
        else:
            price_ok = (price < f_now) and (price < s_now)

        is_strong = (slope_ratio >= 1.3) and price_ok

        strength = "strong" if is_strong else "weak"
        label = f"{strength}_{side}"

        return {
            "label": label,
            "side": side,          # bull / bear
            "strength": strength,  # strong / weak
            "fast": f_now,
            "slow": s_now,
        }
    except Exception:
        return {
            "label": "none",
            "side": None,
            "strength": None,
            "fast": None,
            "slow": None,
        }

# =========================
# ULTRA MARKET STRUCTURE ENGINE
# =========================

class UltraMarketStructureEngine:
    """
    تبسيط علمي لمؤشر Ultra Market Structure:
    - Internal / External structure (آخر قمم وقيعان + BOS / CHoCH)
    - FVG (Bull / Bear) + فلتر حجم gap
    - Premium / Discount zones بناءً على SMA200 + انحراف
    - Liquidity Grab (كسرة وهمية فوق قمة أو تحت قاع)
    """

    def __init__(
        self,
        int_lookback: int = 20,
        ext_lookback: int = 200,
        fvg_threshold_mult: float = 1.0,
        premium_mult_inner: float = 2.0,
        premium_mult_outer: float = 3.0,
    ):
        self.int_lookback = int_lookback
        self.ext_lookback = ext_lookback
        self.fvg_threshold_mult = fvg_threshold_mult
        self.prem_inner = premium_mult_inner
        self.prem_outer = premium_mult_outer

    def _detect_swings(self, df: pd.DataFrame, window: int = 3):
        """
        اكتشاف swing highs/lows البسيطة (internal).
        """
        h = df["high"].astype(float)
        l = df["low"].astype(float)

        swing_high_idx = []
        swing_low_idx = []

        for i in range(window, len(df) - window):
            hi = h.iloc[i]
            lo = l.iloc[i]

            if hi == h.iloc[i - window : i + window + 1].max():
                swing_high_idx.append(i)

            if lo == l.iloc[i - window : i + window + 1].min():
                swing_low_idx.append(i)

        return swing_high_idx, swing_low_idx

    def _last_swing_levels(self, df: pd.DataFrame, lookback: int):
        """
        استخراج آخر قمة وآخر قاع خلال نطاق lookback.
        """
        sub = df.iloc[-lookback:]
        high = sub["high"].astype(float)
        low = sub["low"].astype(float)

        last_high_idx = high.idxmax()
        last_low_idx = low.idxmin()

        return (
            float(df.loc[last_high_idx, "high"]),
            int(df.index.get_loc(last_high_idx)),
            float(df.loc[last_low_idx, "low"]),
            int(df.index.get_loc(last_low_idx)),
        )

    def _detect_bos_choch(self, df: pd.DataFrame, lookback: int = 50):
        """
        BOS / CHoCH بسيط:
        - BOS UP: إغلاق فوق آخر قمة مهمة.
        - BOS DOWN: إغلاق تحت آخر قاع مهم.
        """
        if len(df) < lookback + 5:
            return None, None

        close = df["close"].astype(float)
        last_high, last_high_pos, last_low, last_low_pos = self._last_swing_levels(df, lookback)

        bos = None
        choch = None

        # BOS UP
        if close.iloc[-1] > last_high and close.iloc[-2] <= last_high:
            bos = "up"
        # BOS DOWN
        if close.iloc[-1] < last_low and close.iloc[-2] >= last_low:
            bos = "down"

        # CHoCH = BOS عكس الاتجاه السابق البسيط
        if bos is not None:
            choch = bos

        return bos, choch

    def _detect_fvg(self, df: pd.DataFrame, max_lookback: int = 40):
        """
        كشف أقرب FVG بسيط خلال آخر max_lookback شمعة.
        """
        if len(df) < 5:
            return None

        h = df["high"].astype(float)
        l = df["low"].astype(float)
        c = df["close"].astype(float)

        # ATR بسيط للفلتر
        tr1 = (h - l).abs()
        tr2 = (h - c.shift(1)).abs()
        tr3 = (l - c.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14, min_periods=5).mean()
        atr_val = float(atr.iloc[-1]) if not np.isnan(atr.iloc[-1]) else 0.0
        if atr_val <= 0:
            atr_val = (h.iloc[-1] - l.iloc[-1]) or 1e-6

        start_idx = max(2, len(df) - max_lookback)
        bull_fvg = None
        bear_fvg = None

        for i in range(start_idx, len(df)):
            # Bullish FVG: المنطقة بين high[i-2] و low[i]
            if l.iloc[i] > h.iloc[i - 2]:
                gap = l.iloc[i] - h.iloc[i - 2]
                if gap >= self.fvg_threshold_mult * (0.5 * atr_val):
                    bull_fvg = {
                        "type": "bull",
                        "index": int(i),
                        "upper": float(l.iloc[i]),
                        "lower": float(h.iloc[i - 2]),
                        "size": float(gap),
                    }

            # Bearish FVG: المنطقة بين low[i-2] و high[i]
            if h.iloc[i] < l.iloc[i - 2]:
                gap = l.iloc[i - 2] - h.iloc[i]
                if gap >= self.fvg_threshold_mult * (0.5 * atr_val):
                    bear_fvg = {
                        "type": "bear",
                        "index": int(i),
                        "upper": float(l.iloc[i - 2]),
                        "lower": float(h.iloc[i]),
                        "size": float(gap),
                    }

        current_price = float(df["close"].iloc[-1])
        fvg_ctx = {
            "bull_near": False,
            "bear_near": False,
            "bull": bull_fvg,
            "bear": bear_fvg,
        }

        if bull_fvg is not None:
            mid = 0.5 * (bull_fvg["upper"] + bull_fvg["lower"])
            if abs(current_price - mid) <= atr_val:
                fvg_ctx["bull_near"] = True

        if bear_fvg is not None:
            mid = 0.5 * (bear_fvg["upper"] + bear_fvg["lower"])
            if abs(current_price - mid) <= atr_val:
                fvg_ctx["bear_near"] = True

        return fvg_ctx

    def _premium_discount(self, df: pd.DataFrame):
        """
        Premium / Discount بناءً على SMA200 + انحراف قياسي.
        """
        c = df["close"].astype(float)
        if len(c) < 210:
            return {
                "zone": "mid",
                "basis": float(c.iloc[-1]),
                "upper": float(c.iloc[-1]),
                "lower": float(c.iloc[-1]),
            }

        basis = c.rolling(window=200).mean()
        std = c.rolling(window=200).std()

        b = float(basis.iloc[-1])
        s = float(std.iloc[-1])
        if np.isnan(b) or np.isnan(s) or s == 0:
            b = float(c.iloc[-1])
            s = (c.max() - c.min()) / 10 or 1e-6

        upper_outer = b + self.prem_outer * s
        lower_outer = b - self.prem_outer * s

        price = float(c.iloc[-1])

        zone = "mid"
        if price > upper_outer:
            zone = "ultra_premium"
        elif price > b + self.prem_inner * s:
            zone = "premium"
        elif price < lower_outer:
            zone = "ultra_discount"
        elif price < b - self.prem_inner * s:
            zone = "discount"

        return {
            "zone": zone,
            "basis": b,
            "upper": upper_outer,
            "lower": lower_outer,
        }

    def _detect_liquidity_grab(self, df: pd.DataFrame, lookback: int = 20):
        """
        Liquidity Grab بسيط:
        - شمعة عملت ذيل فوق آخر قمة ثم أغلقت تحتها → grab up.
        - أو تحت آخر قاع ثم أغلقت فوقه → grab down.
        """
        if len(df) < lookback + 3:
            return {"grab_up": False, "grab_down": False}

        sub = df.iloc[-lookback:]
        high = sub["high"].astype(float)
        low = sub["low"].astype(float)
        close = sub["close"].astype(float)

        last_high = float(high.max())
        last_low = float(low.min())

        # آخر شمعة
        h_last = float(df["high"].iloc[-1])
        l_last = float(df["low"].iloc[-1])
        c_last = float(df["close"].iloc[-1])

        grab_up = h_last > last_high and c_last < last_high
        grab_down = l_last < last_low and c_last > last_low

        return {
            "grab_up": bool(grab_up),
            "grab_down": bool(grab_down),
        }

    def analyze(self, df: pd.DataFrame) -> dict:
        """
        يرجّع سياق كامل لـ Ultra Market Structure:
        - bias (bull/bear/neutral)
        - bos / choch
        - fvg context
        - premium/discount zone
        - liquidity grab flags
        """
        if df is None or len(df) < 30:
            return {
                "bias": "neutral",
                "bos": None,
                "choch": None,
                "fvg": None,
                "premium_discount": None,
                "liq_grab": {"grab_up": False, "grab_down": False},
            }

        bos_int, choch_int = self._detect_bos_choch(df, lookback=self.int_lookback)
        fvg_ctx = self._detect_fvg(df, max_lookback=40)
        prem_ctx = self._premium_discount(df)
        liq_ctx = self._detect_liquidity_grab(df, lookback=self.int_lookback)

        # bias بسيط:
        bias = "neutral"
        if bos_int == "up":
            bias = "bull"
        elif bos_int == "down":
            bias = "bear"

        return {
            "bias": bias,
            "bos": bos_int,
            "choch": choch_int,
            "fvg": fvg_ctx,
            "premium_discount": prem_ctx,
            "liq_grab": liq_ctx,
        }

# =========================
# ORDER FLOW / BOOKMAP ENGINE
# =========================

class OrderFlowEngine:
    """
    محرك OrderFlow / Footprint / Bookmap-Lite:
    - يستخدم fetch_trades لحساب Delta / CVD / Buy/Sell Volume
    - يستخدم fetch_orderbook لحساب Buy/Sell Walls + Imbalance
    """
    def __init__(self, exchange_manager: "ExchangeManager"):
        self.ex = exchange_manager

    def _compute_flow_from_trades(self, trades) -> dict:
        if not trades:
            return {
                "buy_volume": 0.0,
                "sell_volume": 0.0,
                "delta": 0.0,
                "cvd": 0.0,
                "flow_side": "NEUTRAL",
            }

        buy_vol = 0.0
        sell_vol = 0.0
        cvd = 0.0

        for t in trades:
            try:
                side = t.get("side")
                amount = float(t.get("amount", 0.0))
                if not amount:
                    continue

                if side == "buy":
                    buy_vol += amount
                    cvd += amount
                elif side == "sell":
                    sell_vol += amount
                    cvd -= amount
                else:
                    pass
            except Exception:
                continue

        delta = buy_vol - sell_vol
        if buy_vol > sell_vol * 1.3:
            flow_side = "BUY"
        elif sell_vol > buy_vol * 1.3:
            flow_side = "SELL"
        else:
            flow_side = "NEUTRAL"

        return {
            "buy_volume": buy_vol,
            "sell_volume": sell_vol,
            "delta": delta,
            "cvd": cvd,
            "flow_side": flow_side,
        }

    def _compute_bookmap_from_ob(self, orderbook, current_price: float) -> dict:
        bids = orderbook.get("bids", []) or []
        asks = orderbook.get("asks", []) or []

        if not bids and not asks:
            return {
                "book_imbalance": 0.0,
                "buy_wall": False,
                "sell_wall": False,
                "wall_side": None,
                "wall_distance": None,
            }

        near_buy_vol = 0.0
        near_sell_vol = 0.0
        max_buy_level = None
        max_buy_vol = 0.0
        max_sell_level = None
        max_sell_vol = 0.0

        for price, vol in bids:
            price = float(price); vol = float(vol)
            if current_price and price >= current_price * 0.99:
                near_buy_vol += vol
                if vol > max_buy_vol:
                    max_buy_vol = vol
                    max_buy_level = price

        for price, vol in asks:
            price = float(price); vol = float(vol)
            if current_price and price <= current_price * 1.01:
                near_sell_vol += vol
                if vol > max_sell_vol:
                    max_sell_vol = vol
                    max_sell_level = price

        if (near_buy_vol + near_sell_vol) > 0:
            book_imb = (near_buy_vol - near_sell_vol) / (near_buy_vol + near_sell_vol)
        else:
            book_imb = 0.0

        buy_wall = max_buy_vol > 0 and max_buy_vol >= near_sell_vol * 1.5
        sell_wall = max_sell_vol > 0 and max_sell_vol >= near_buy_vol * 1.5

        wall_side = None
        wall_distance = None
        if buy_wall and max_buy_level:
            wall_side = "BUY"
            wall_distance = (current_price - max_buy_level) / current_price if current_price else None
        elif sell_wall and max_sell_level:
            wall_side = "SELL"
            wall_distance = (max_sell_level - current_price) / current_price if current_price else None

        return {
            "book_imbalance": book_imb,
            "buy_wall": buy_wall,
            "sell_wall": sell_wall,
            "wall_side": wall_side,
            "wall_distance": wall_distance,
        }

    def compute(self, current_price: float) -> dict:
        """يرجع سياق OrderFlow + Bookmap"""
        try:
            trades = self.ex.fetch_trades(limit=200)
            ob = self.ex.fetch_orderbook(depth=50)

            flow_ctx = self._compute_flow_from_trades(trades)
            book_ctx = self._compute_bookmap_from_ob(ob, current_price)

            ctx = {**flow_ctx, **book_ctx}
            return ctx
        except Exception as e:
            log_w(f"⚠️ OrderFlowEngine error: {e}")
            return {
                "buy_volume": 0.0,
                "sell_volume": 0.0,
                "delta": 0.0,
                "cvd": 0.0,
                "flow_side": "NEUTRAL",
                "book_imbalance": 0.0,
                "buy_wall": False,
                "sell_wall": False,
                "wall_side": None,
                "wall_distance": None,
            }

# ============================================
#  CVD DIVERGENCE ENGINE
# ============================================

def compute_cvd_from_ohlcv(df: pd.DataFrame) -> pd.Series:
    """
    CVD تقريبي من OHLCV:
    - لو الشمعة خضراء → volume محسوب شراء
    - لو حمرا → volume بيع
    """
    if df is None or len(df) == 0:
        return pd.Series(dtype=float)

    close = df["close"].astype(float)
    open_ = df["open"].astype(float)
    vol = df["volume"].astype(float)

    buy_vol = vol.where(close > open_, 0.0)
    sell_vol = vol.where(close < open_, 0.0)
    delta = buy_vol - sell_vol
    cvd = delta.cumsum()
    return cvd

def detect_cvd_divergence(df: pd.DataFrame, lookback: int = 5) -> Optional[str]:
    """
    يكشف عن:
    - bearish divergence  → "bearish"
    - bullish divergence  → "bullish"
    - otherwise           → None
    """
    if df is None or len(df) < lookback + 2:
        return None

    df = df.copy()
    df["cvd"] = compute_cvd_from_ohlcv(df)

    last = len(df) - 1

    # Higher High in price + Lower High in CVD → bearish
    price_hh = float(df["high"].iloc[last]) > float(df["high"].iloc[last-lookback:last].max())
    cvd_lh = float(df["cvd"].iloc[last]) < float(df["cvd"].iloc[last-lookback:last].max())

    # Lower Low in price + Higher Low in CVD → bullish
    price_ll = float(df["low"].iloc[last]) < float(df["low"].iloc[last-lookback:last].min())
    cvd_hl = float(df["cvd"].iloc[last]) > float(df["cvd"].iloc[last-lookback:last].min())

    if price_hh and cvd_lh:
        return "bearish"
    if price_ll and cvd_hl:
        return "bullish"
    return None

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
BOT_VERSION = f"ULTRA PRO AI v13.0 - TREND BIRTH EDITION - {EXCHANGE_NAME.upper()} - AUTO-RECOVERY + SMART COOLDOWN ENABLED"

print(f"🚀 Booting: {BOT_VERSION}", flush=True)

# ============================================
#  PROFIT PROFILES DEFINITION
# ============================================

PROFIT_PROFILES = {
    "SCALP_STRICT": {
        "tp_levels_rr": [1.0],
        "tp_fracs":     [1.0],
        "hard_sl_rr":   -0.6,
        "be_after_tp":  True,
        "trail_start_rr": None,
        "trail_atr_mult": None,
    },
    "MID_TREND": {
        "tp_levels_rr": [1.0, 2.0],
        "tp_fracs":     [0.6, 0.4],
        "hard_sl_rr":   -0.7,
        "be_after_tp":  True,
        "trail_start_rr": 1.8,
        "trail_atr_mult": 1.0,
    },
    "FULL_TREND": {
        "tp_levels_rr": [0.8, 1.8, 3.0],
        "tp_fracs":     [0.3, 0.3, 0.4],
        "hard_sl_rr":   -0.8,
        "be_after_tp":  True,
        "trail_start_rr": 2.0,
        "trail_atr_mult": 1.5,
    },
    "TRAP_TREND": {
        "tp_levels_rr": [1.2, 2.0],
        "tp_fracs":     [0.7, 0.3],
        "hard_sl_rr":   -0.7,
        "be_after_tp":  True,
        "trail_start_rr": 2.0,
        "trail_atr_mult": 1.2,
    },
}

# ============================================
#  TREND POWER BUCKET HELPER
# ============================================

def _trend_power_bucket(analysis):
    """
    تصنيف قوة الترند:
      weak  → رجّح سكالب
      mid   → 2TP
      strong→ 3TP / ترند كامل
    """
    trend = (analysis.get("trend") or {})
    adx = float(trend.get("adx", 0.0))
    strength_num = float(trend.get("strength", 0.0))
    di_plus = float(trend.get("di_plus", 0.0))
    di_minus = float(trend.get("di_minus", 0.0))
    conf = float(analysis.get("confidence", 0.0))

    ema = (analysis.get("ema_trend") or {})
    ema_dir = ema.get("dir", "neutral")
    ema_strength = ema.get("strength", "weak")
    stair_state = ema.get("stair_state", "broken")
    
    ema_cross = analysis.get("ema_cross", {})
    ema_cross_label = ema_cross.get("label", "none")
    ema_cross_strength = ema_cross.get("strength")

    di_spread = abs(di_plus - di_minus)

    score = 0.0

    # ADX + DI
    if adx >= 18:
        score += 1.0
    if adx >= 25:
        score += 1.0
    if di_spread >= 10:
        score += 0.5
    if di_spread >= 18:
        score += 0.5

    if strength_num >= 6:
        score += 1.0
    if strength_num >= 8:
        score += 0.5

    if conf >= 6:
        score += 0.5
    if conf >= 8:
        score += 0.5

    if ema_dir in ("up", "down"):
        score += 0.5
    if ema_strength == "strong":
        score += 1.0
    elif ema_strength == "mid":
        score += 0.5

    if stair_state == "holding":
        score += 0.5
    elif stair_state == "broken":
        score -= 0.5
        
    if ema_cross_strength == "strong":
        score += 1.5
    elif ema_cross_strength == "weak":
        score += 0.5

    if score >= 3.0:
        return "strong"
    if score >= 1.5:
        return "mid"
    return "weak"

def select_profit_profile(trade_mode, analysis):
    """
    اختيار الـ Profile المناسب
    """
    rr = float(analysis.get("edge_rr", 1.0))
    edge = analysis.get("edge_setup") or {}
    if edge.get("valid"):
        rr = float(edge.get("rr1", rr))

    adx = float(analysis.get("trend", {}).get("adx", 0.0))
    conf = float(analysis.get("confidence", 0.0))
    stop_q = float(analysis.get("stop_hunt_trap_quality", 0.0))
    golden = analysis.get("golden_zone", {}).get("type")
    trend_bucket = _trend_power_bucket(analysis)
    
    ema_cross = analysis.get("ema_cross", {})
    ema_cross_label = ema_cross.get("label", "none")
    ema_cross_strength = ema_cross.get("strength")

    # Trend Birth Override: استخدم FULL_TREND دائماً
    if analysis.get("trend_birth", {}).get("active"):
        return "FULL_TREND"

    edge_tp = None
    if edge.get("valid"):
        edge_tp = edge.get("tp_profile")

    # 1) TRAP قوي → TRAP_TREND
    if trade_mode == "TRAP" and stop_q >= 3.0 and adx >= 20:
        return "TRAP_TREND"

    # 2) Golden Zone مؤكدة مع ترند محترم → FULL_TREND
    if golden in ("golden_bottom", "golden_top") and (adx >= 20 or conf >= 6.0):
        return "FULL_TREND"

    # 3) Edge Algo يقترح خطة TP جاهزة
    if edge_tp == "TREND_3TP":
        return "FULL_TREND"
    if edge_tp == "MID_2TP":
        return "MID_TREND"
    if edge_tp == "SCALP_STRICT":
        base_profile = "SCALP_STRICT"
    else:
        base_profile = None

    # 4) تقاطع EMA قوي → FULL_TREND أو MID_TREND بناءً على قوة التقاطع
    if ema_cross_strength == "strong":
        if "bull" in ema_cross_label and rr >= 1.5:
            return "FULL_TREND"
        elif "bear" in ema_cross_label and rr >= 1.5:
            return "FULL_TREND"
        elif rr >= 1.2:
            return "MID_TREND"

    # 5) ترند قوي + RR محترم → FULL_TREND
    if trend_bucket == "strong" and (rr >= 1.5 or adx >= 28 or conf >= 7.0):
        return "FULL_TREND"

    # 6) ترند متوسط → MID_TREND
    if trend_bucket == "mid" and (rr >= 1.2 or adx >= 20 or conf >= 5.0):
        return "MID_TREND"

    # 7) لو Edge قال سكالب صريح → SCALP_STRICT
    if base_profile == "SCALP_STRICT":
        return "SCALP_STRICT"

    # 8) fallback
    if rr >= 1.3 and (18 <= adx <= 28 or conf >= 5.0):
        return "MID_TREND"

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
    يطبع بلوك لوج كامل في كل تيك
    """
    a = analysis or {}

    trend     = a.get("trend", {})
    adx = trend.get("adx", 0.0)
    smc_ctx   = a.get("smc_ctx", {})
    fvg_ctx   = a.get("fvg_analysis", {})
    edge      = a.get("edge_setup", {})
    rf_ctx    = a.get("rf", {})
    of_ctx    = a.get("orderflow", {}) or {}
    ultra_ms  = a.get("ultra_ms", {}) or {}
    ms_bias   = ultra_ms.get("bias", "neutral")
    ms_zone   = (ultra_ms.get("premium_discount") or {}).get("zone", "mid")

    balance        = state.get("balance", 0.0)
    compound_pnl   = state.get("compound_pnl", 0.0)
    mode           = "LIVE" if MODE_LIVE else "PAPER"

    # 1) Bookmap / OrderBook Imbalance
    log_i(
        f"📊 Bookmap: "
        f"Imb={of_ctx.get('book_imbalance', 0.0):.2f} | "
        f"BuyWall[{of_ctx.get('buy_wall', False)}] | "
        f"SellWall[{of_ctx.get('sell_wall', False)}]"
    )

    # 2) Flow (Delta / CVD)
    flow_side = of_ctx.get("flow_side", "NEUTRAL")
    delta_val = of_ctx.get("delta", 0.0)
    cvd_val   = of_ctx.get("cvd", 0.0)

    log_i(
        f"🌊 Flow: {flow_side} "
        f"Δ={delta_val:.4f} | "
        f"CVD={cvd_val:.4f} | "
        f"Conf={a.get('confidence', 0):.2f}"
    )

    # 3) RF REAL + VWAP
    log_i(
        f"📡 RF: dir={rf_ctx.get('dir', 0)} | "
        f"filt={rf_ctx.get('filt', 0):.4f} | "
        f"BUY={rf_ctx.get('buy_signal', False)} "
        f"SELL={rf_ctx.get('sell_signal', False)} | "
        f"VWAP={a.get('vwap', 0.0):.4f}"
    )

    # 4) Ultra Market Structure
    log_i(
        f"🏛 UltraMS: bias={ms_bias} | zone={ms_zone} | "
        f"FVG bull_near={ (ultra_ms.get('fvg') or {}).get('bull_near', False) } "
        f"bear_near={ (ultra_ms.get('fvg') or {}).get('bear_near', False) }"
    )

    # 5) Council summary
    hint_side = "NEUTRAL"
    if a.get("score_buy", 0) > a.get("score_sell", 0):
        hint_side = "BUY"
    elif a.get("score_sell", 0) > a.get("score_buy", 0):
        hint_side = "SELL"
    
    log_i(
        f"📌 DASH → hint-{hint_side} | "
        f"Council BUY({a.get('score_buy',0):.1f}) "
        f"SELL({a.get('score_sell',0):.1f}) | "
        f"RSI={trend.get('rsi', 0):.1f} | "
        f"ADX={adx:.1f} "
        f"DI+={trend.get('di_plus', 0):.1f} DI-={trend.get('di_minus', 0):.1f}"
    )

    # 6) Strategy + Balance
    strategy_label = "SCALP"
    if edge and edge.get("grade"):
        strategy_label = edge.get("grade", "MID").upper()
    
    log_i(
        f"⚡ Strategy: {strategy_label} | "
        f"Balance={balance:.2f} | CompoundPnL={compound_pnl:.4f} | Mode={mode}"
    )

    # 7) SMC BEST
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

    # 8) SNAP votes
    votes_total = a.get("score_buy", 0) + a.get("score_sell", 0)
    votes_side = "?" if votes_total == 0 else ("BUY" if a.get("score_buy", 0) > a.get("score_sell", 0) else "SELL")
    
    log_i(
        f"🎯 SNAP | {votes_side} | "
        f"votes={max(a.get('score_buy',0), a.get('score_sell',0)):.0f}/{votes_total:.0f} "
        f"score={a.get('confidence',0):.1f} | "
        f"ADX={adx:.1f} "
        f"DI={trend.get('di_plus',0)-trend.get('di_minus',0):.1f}"
    )

    # 9) Footprint / Volume delta
    volume_ctx = a.get("volume_analysis", {})
    log_i(
        f"🦶 FOOTPRINT | Δ={volume_ctx.get('delta',0):.0f} | "
        f"Spike={volume_ctx.get('spike', False)} | "
        f"AbsBull={volume_ctx.get('abs_bull', False)} | "
        f"AbsBear={volume_ctx.get('abs_bear', False)}"
    )

    # 10) SMC addons / FVG / Golden
    golden = a.get("golden_zone", {})
    ema_trend = a.get("ema_trend", {})
    profile = state.get("profit_profile", "N/A")
    trend_bucket = _trend_power_bucket(a)
    
    log_i(
        f"🧠 ENHANCED SMC ADDONS | "
        f"FVG_real={fvg_ctx.get('real',False) if fvg_ctx else False} | "
        f"Golden={golden.get('type', 'None')} "
        f"| Trap={a.get('stop_hunt_trap_side', 'None')} "
        f"Q={a.get('stop_hunt_trap_quality',0):.1f}"
    )
    
    # 11) PROFIT PROFILE + EMA TREND INFO
    log_i(
        f"📊 PROFIT_PROFILE={profile} | "
        f"trend={trend_bucket} | rr={a.get('edge_rr', 0):.1f} | "
        f"adx={adx:.1f} | conf={a.get('confidence', 0):.1f} | "
        f"EMA={ema_trend.get('dir', 'neutral')}/{ema_trend.get('strength', 'weak')}"
    )
    
    # 12) EMA CROSS INFO
    ema_cross = a.get("ema_cross", {})
    log_i(
        f"📊 EMA CROSS: {ema_cross.get('label', 'none')} | "
        f"side={ema_cross.get('side', 'none')} | "
        f"strength={ema_cross.get('strength', 'none')}"
    )

    # 13) TREND BIRTH INFO
    trend_birth = a.get("trend_birth", {})
    if trend_birth.get("active"):
        log_i(
            f"🔥 TREND BIRTH: {trend_birth['side']} @ {trend_birth['entry_zone']} | "
            f"conf={trend_birth['confidence']:.2f} | "
            f"reasons={', '.join(trend_birth['reasons'])}"
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
    print(colored("  • VWAP Engine - Fair Value Axis", "yellow"))
    print(colored("  • Ultra Market Structure Engine", "yellow"))
    print(colored("  • AUTO-RECOVERY SYSTEM - استعادة الصفقات بعد الإعادة", "yellow"))
    print(colored("  • EMA HOLD ENGINE - نظام ركوب الترند الذكي مع Hold-TP", "yellow"))
    print(colored("  • EMA CROSS CLASSIFIER - Strong/Weak Crossover Detection", "yellow"))
    print(colored("  • TREND BIRTH ENGINE - نظام كشف ولادة الترند مع دخول OB/FVG", "yellow", attrs=["bold"]))
    print(colored("  • SMART COOLDOWN SYSTEM - 30 دقيقة مع استثناء الترند القوي", "yellow", attrs=["bold"]))

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
    
    def fetch_trades(self, limit: int = 200):
        """جلب آخر الصفقات من البورصة"""
        if not self.initialized:
            return []
        try:
            trades = self.exchange.fetch_trades(SYMBOL, limit=limit)
            return trades or []
        except Exception as e:
            log_w(f"⚠️ Failed to fetch trades for orderflow: {e}")
            return []

    def fetch_orderbook(self, depth: int = 50):
        """جلب الـ OrderBook"""
        if not self.initialized:
            return {"bids": [], "asks": []}
        try:
            ob = self.exchange.fetch_order_book(SYMBOL, limit=depth)
            return ob or {"bids": [], "asks": []}
        except Exception as e:
            log_w(f"⚠️ Failed to fetch order book for bookmap: {e}")
            return {"bids": [], "asks": []}
    
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

    def get_open_position(self):
        """
        قراءة المركز المفتوح فعليًا من البورصة.
        """
        if not MODE_LIVE or not self.initialized:
            return None

        try:
            positions = []
            if hasattr(self.exchange, "fetch_positions"):
                positions = self.exchange.fetch_positions([SYMBOL])
            elif hasattr(self.exchange, "fetchPositions"):
                positions = self.exchange.fetchPositions([SYMBOL])
            else:
                return None

            if not positions:
                return None

            for p in positions:
                try:
                    sym = p.get("symbol") or p.get("info", {}).get("symbol")
                    if sym != SYMBOL:
                        continue

                    amt = p.get("contracts")
                    if amt is None:
                        amt = p.get("contractSize")
                    if amt is None:
                        amt = p.get("positionAmt")

                    amt = float(amt or 0.0)
                    if amt == 0:
                        continue

                    raw_side = (p.get("side") or "").lower()
                    if raw_side in ("long", "buy"):
                        side = "long"
                    elif raw_side in ("short", "sell"):
                        side = "short"
                    else:
                        side = "long" if amt > 0 else "short"

                    entry = (
                        p.get("entryPrice")
                        or p.get("avgEntryPrice")
                        or p.get("info", {}).get("entry_price")
                        or p.get("info", {}).get("avgEntryPrice")
                    )

                    try:
                        entry_price = float(entry) if entry is not None else float(self.get_current_price() or 0.0)
                    except Exception:
                        entry_price = float(self.get_current_price() or 0.0)

                    return {
                        "side": side,
                        "qty": abs(amt),
                        "entry_price": entry_price,
                    }
                except Exception:
                    continue

            return None

        except Exception as e:
            log_w(f"⚠️ Failed to fetch open position from exchange: {e}")
            return None

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
            "profit_engine_active": False,
            "cooldown_until": None,
            "last_close_reason": None,
            "last_close_time": None,
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
    
    def update(self, *args, **kwargs):
        """
        تحديث حالة البوت
        """
        if args:
            if len(args) > 1:
                raise TypeError(f"update expected at most 1 argument, got {len(args)}")
            other = args[0]
            if isinstance(other, dict):
                self.state.update(other)
            else:
                try:
                    self.state.update(dict(other))
                except (TypeError, ValueError):
                    raise TypeError(f"cannot convert {type(other)} to dict")
        if kwargs:
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
            "profit_engine_active": False,
            "cooldown_until": None,
            "last_close_reason": None,
            "last_close_time": None,
        })
        self.save_state()
    
    def __getitem__(self, key):
        return self.state.get(key)
    
    def __setitem__(self, key, value):
        self.state[key] = value
        self.save_state()

# ============================================
#  TREND ANALYSIS ENGINE WITH ADX + ATR
# ============================================

class TrendAnalyzer:
    """محرك تحليل الاتجاه مع ADX + ATR"""
    
    def __init__(self):
        self.fast_ma = deque(maxlen=20)
        self.slow_ma = deque(maxlen=50)
        self.trend = "flat"
        self.strength = 0.0
        self.momentum = 0.0

        self.rsi = 50.0
        self.adx = 0.0
        self.di_plus = 0.0
        self.di_minus = 0.0
        self.atr = 0.0
        self.atr_mult = 1.0
        
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
            
        self._calculate_adx_atr(df)
            
        if delta > 0 and self.strength > 0.1:
            self.trend = "up"
        elif delta < 0 and self.strength > 0.1:
            self.trend = "down" 
        else:
            self.trend = "flat"
            
    def _calculate_adx_atr(self, df):
        """حساب ADX / DI / ATR / RSI"""
        try:
            ind = compute_indicators(df)

            self.rsi      = ind["rsi"]
            self.adx      = ind["adx"]
            self.di_plus  = ind["plus_di"]
            self.di_minus = ind["minus_di"]
            self.atr      = ind["atr"]

            high = df["high"].astype(float)
            low  = df["low"].astype(float)
            close = df["close"].astype(float)

            tr = pd.concat([
                (high - low).abs(),
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs()
            ], axis=1).max(axis=1)

            if len(tr) >= 20:
                atr_base = tr.rolling(window=20).mean().iloc[-1]
            else:
                atr_base = self.atr

            self.atr_mult = self.atr / atr_base if atr_base and atr_base > 0 else 1.0

        except Exception as e:
            log_w(f"⚠️ ADX/ATR calculation error: {e}")
            self.rsi = 50.0
            self.adx = 0.0
            self.di_plus = 0.0
            self.di_minus = 0.0
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
            "rsi": self.rsi,
            "adx": self.adx,
            "di_plus": self.di_plus,
            "di_minus": self.di_minus,
            "atr": self.atr,
            "atr_mult": self.atr_mult,
            "is_strong": self.is_strong_trend()
        }
    
    def analyze_stop_hunt_context(self, df, stop_hunt_zone):
        """
        تحليل سياق الستوب هانت باستخدام ADX و ATR
        """
        if len(df) < 20 or not stop_hunt_zone:
            return {"valid_for_trap": False, "reason": "insufficient_data"}
        
        try:
            current_price = float(df['close'].iloc[-1])
            
            adx_slope = self._calculate_adx_slope(df)
            atr_mult = self.atr_mult
            
            last_candle = df.iloc[-1]
            candle_high = float(last_candle['high'])
            candle_low = float(last_candle['low'])
            candle_close = float(last_candle['close'])
            candle_open = float(last_candle['open'])
            
            candle_range = candle_high - candle_low
            body_size = abs(candle_close - candle_open)
            wick_size = candle_range - body_size
            wick_ratio = wick_size / candle_range if candle_range > 0 else 0
            
            trend_context = "flat"
            if self.adx < 20:
                trend_context = "flat"
            elif self.adx < 35:
                trend_context = "moderate"
            elif self.adx < 50:
                trend_context = "strong"
            else:
                trend_context = "extreme"
            
            allowed_side = None
            
            if trend_context == "extreme":
                if self.trend == "down":
                    allowed_side = "SELL"
                elif self.trend == "up":
                    allowed_side = "BUY"
                else:
                    allowed_side = None

                if allowed_side:
                    valid_for_trap = (
                        stop_hunt_zone.get("type") == f"{allowed_side.lower()}_stop_hunt"
                    )
                else:
                    valid_for_trap = False

                reason = f"extreme_trend_{self.trend}_only"
            
            elif trend_context == "strong":
                if self.trend == "down":
                    allowed_side = "SELL"
                    valid_for_trap = True
                    reason = "strong_downtrend"
                elif self.trend == "up":
                    allowed_side = "BUY"
                    valid_for_trap = True
                    reason = "strong_uptrend"
                else:
                    valid_for_trap = atr_mult >= 1.3 and wick_ratio >= 0.6
                    reason = "strong_range_trap"
                    
            elif trend_context == "moderate":
                valid_for_trap = atr_mult >= 1.3 and wick_ratio >= 0.6
                allowed_side = "BUY" if stop_hunt_zone.get("type") == "buy_stop_hunt" else "SELL"
                reason = "moderate_trend_trap"
                
            else:
                valid_for_trap = atr_mult >= 1.5 and wick_ratio >= 0.7
                allowed_side = "BUY" if stop_hunt_zone.get("type") == "buy_stop_hunt" else "SELL"
                reason = "flat_market_trap"
            
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
            
            if lookback == 0:
                return 0.0
                
            current_adx = self.adx
            
            if len(df) >= 15:
                prev_adx = current_adx * 0.95
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
            
            tr1 = candle_high - candle_low
            tr2 = abs(candle_high - float(prev_candle['close']))
            tr3 = abs(candle_low - float(prev_candle['close']))
            tr = max(tr1, tr2, tr3)
            
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
            
            if zone_type == "buy":
                lower_wick = min(candle_close, candle_open) - candle_low
                lower_wick_ratio = lower_wick / candle_range if candle_range > 0 else 0
                has_long_lower_wick = lower_wick_ratio >= 0.6
            else:
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
            
            if adx > 50:
                if trend == "down" and zone_type == "sell_stop_hunt":
                    return True
                elif trend == "up" and zone_type == "buy_stop_hunt":
                    return True
                else:
                    return False
            
            if atr_mult < 1.2:
                return False
            
            if atr_mult > 2.5 and adx > 35:
                return False
            
            if adx < 20:
                candle_context = zone.get("adx_context", {})
                return candle_context.get("valid", False) and atr_mult >= 1.5
            
            return True
            
        except Exception as e:
            log_w(f"⚠️ ADX/ATR validation error: {e}")
            return True
    
    def get_active_stop_hunt_zones(self, current_price, df):
        """الحصول على مناطق الستوب هانت النشطة مع تحليل ADX"""
        active_zones = []
        for zone in self.recent_stop_hunts:
            if zone["type"] == "buy_stop_hunt" and current_price > zone["level"] * 0.995:
                trend_context = self.trend_analyzer.analyze_stop_hunt_context(df, zone)
                zone["trend_context"] = trend_context
                active_zones.append(zone)
            elif zone["type"] == "sell_stop_hunt" and current_price < zone["level"] * 1.005:
                trend_context = self.trend_analyzer.analyze_stop_hunt_context(df, zone)
                zone["trend_context"] = trend_context
                active_zones.append(zone)
        return active_zones

# ============================================
#  PROFIT ENGINE - نظام جني الأرباح الذكي مع Hold-TP
# ============================================

class ProfitEngine:
    """محرك جني الأرباح والوقف المتحرك الذكي مع Hold-TP"""
    
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
        
        # Trend Birth Override: استخدم FULL_TREND دائماً
        if analysis.get("trend_birth", {}).get("active"):
            self.profile_name = "FULL_TREND"
            log_i(f"🔥 TREND BIRTH MODE: Using FULL_TREND profile with Hold-TP")
        else:
            self.profile_name = select_profit_profile(trade_mode, analysis)
        
        self.profile_cfg = PROFIT_PROFILES[self.profile_name]
        
        direction = 1 if side == "long" else -1

        hard_sl_rr = self.profile_cfg["hard_sl_rr"]
        sl_dist = abs(hard_sl_rr) * self.atr_entry
        if side == "long":
            self.sl_price = self.entry_price - sl_dist
        else:
            self.sl_price = self.entry_price + sl_dist

        trend = (analysis.get("trend") or {})
        adx = float(trend.get("adx", 0.0))
        conf = float(analysis.get("confidence", 0.0))
        edge = analysis.get("edge_setup") or {}
        edge_rr = float(analysis.get("edge_rr", 1.0))
        if edge.get("valid"):
            edge_rr = float(edge.get("rr1", edge_rr))

        ema = (analysis.get("ema_trend") or {})
        ema_dir = ema.get("dir", "neutral")
        ema_strength = ema.get("strength", "weak")

        base_rrs = list(self.profile_cfg["tp_levels_rr"])
        boost_mult = 1.0

        same_dir_ema = (
            (ema_dir == "up" and side == "long") or
            (ema_dir == "down" and side == "short")
        )

        if self.profile_name in ("FULL_TREND", "TRAP_TREND"):
            if adx >= 32 and conf >= 7.5 and same_dir_ema and ema_strength == "strong":
                boost_mult = 1.40
            elif adx >= 28 and conf >= 6.5 and same_dir_ema:
                boost_mult = 1.30
            elif adx >= 24 and conf >= 6.0:
                boost_mult = 1.15
        elif self.profile_name == "MID_TREND":
            if adx >= 24 and conf >= 6.0:
                boost_mult = 1.15
            elif adx >= 20 and conf >= 5.0:
                boost_mult = 1.05
        elif self.profile_name == "SCALP_STRICT":
            if conf < 4.0 or adx < 15:
                boost_mult = 0.85

        if edge_rr >= 2.0 and self.profile_name in ("FULL_TREND", "TRAP_TREND"):
            boost_mult = max(boost_mult, 1.20)

        boosted_rrs = []
        for i, rr in enumerate(base_rrs):
            rr_boosted = rr * boost_mult
            if i == 0:
                rr_boosted = max(0.6, rr_boosted)
            boosted_rrs.append(rr_boosted)

        self.tp_levels = []
        self.tp_hit = set()
        for i, (rr, frac) in enumerate(zip(boosted_rrs, self.profile_cfg["tp_fracs"])):
            dist = rr * self.atr_entry
            price = self.entry_price + direction * dist
            label = f"TP{i+1}_{self.profile_name}"
            self.tp_levels.append((price, frac, label))
        
        self.trail_active = False
        self.trail_price = None
        
        self.state["profit_profile"] = self.profile_name
        self.state["profit_engine_active"] = True
        
        try:
            tp_preview = [(round(p, 6), f"{f*100:.0f}%") for p, f, _ in self.tp_levels]
        except Exception:
            tp_preview = []

        boost_note = ""
        if abs(boost_mult - 1.0) > 0.01:
            boost_note = f" | TP_boost×{boost_mult:.2f}"

        log_i(
            f"🎯 PROFIT PLAN [{self.profile_name}] | "
            f"side={side} | entry={self.entry_price:.6f} | "
            f"ATR={self.atr_entry:.6f} | SL={self.sl_price:.6f} | "
            f"TPs={tp_preview}{boost_note}"
        )

        trend_bucket = _trend_power_bucket(analysis)
        log_i(
            f"📊 PROFILE_SELECTION | trend={trend_bucket} | "
            f"adx={adx:.1f} | conf={conf:.1f} | ema={ema_dir}/{ema_strength} | "
            f"edge_rr={edge_rr:.1f}"
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
        
        atr_now = self.calculate_atr(df)
        R_now = (price - self.entry_price) * direction / max(atr_now, 1e-8)
        
        # ===== 1) وقف خسارة ثابت / متحرك =====
        if self.sl_price is not None:
            if (side == "long" and price <= self.sl_price) or \
               (side == "short" and price >= self.sl_price):
                
                log_r(
                    f"🛑 HARD SL HIT | profile={self.profile_name} | "
                    f"side={side} | qty={qty:.4f} | exit={price:.6f} | R={R_now:.2f}"
                )
                
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

            hit = False
            if side == "long" and price >= tp_price:
                hit = True
            elif side == "short" and price <= tp_price:
                hit = True

            if hit:
                close_qty = qty * frac

                will_trail_now = (
                    (not self.trail_active)
                    and bool(self.profile_cfg.get("trail_start_rr"))
                    and R_now >= float(self.profile_cfg.get("trail_start_rr", 0.0))
                )

                close_side = "sell" if side == "long" else "buy"
                if self.exchange.execute_order(close_side, close_qty, price):
                    self.tp_hit.add(idx)
                    self.state["qty"] -= close_qty
                    qty = self.state["qty"]

                    if side == "long":
                        realized_pnl = (price - self.entry_price) * close_qty
                    else:
                        realized_pnl = (self.entry_price - price) * close_qty

                    self.state["compound_pnl"] = self.state.get("compound_pnl", 0.0) + realized_pnl

                    profile = self.profile_name or self.state.get("profit_profile", "N/A")
                    closed_pct = frac * 100.0
                    trail_tag = " | trail ON" if will_trail_now else ""

                    log_g(
                        f"✅ TP{idx+1} HIT ({profile}) | closed={closed_pct:.0f}% "
                        f"| R≈{R_now:.2f} | qty={close_qty:.4f} | remain={qty:.4f} "
                        f"| PnL={realized_pnl:.3f} USDT{trail_tag}"
                    )

                    if self.profile_cfg["be_after_tp"] and len(self.tp_hit) == 1:
                        if side == "long":
                            self.sl_price = self.entry_price * 1.0001
                        else:
                            self.sl_price = self.entry_price * 0.9999
                        log_w(f"⚖ Breakeven set at {self.sl_price:.6f}")
        
        # ===== 3) تفعيل / تحديث التريل =====
        start_rr = self.profile_cfg["trail_start_rr"]
        atr_mult = self.profile_cfg["trail_atr_mult"]
        
        if start_rr and atr_mult and self.state["qty"] > 0:
            if (not self.trail_active) and R_now >= start_rr:
                self.trail_active = True
                dist = atr_mult * atr_now
                if side == "long":
                    self.trail_price = price - dist
                else:
                    self.trail_price = price + dist
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
        
        # ===== 4) لو كل TPs اتنفذت ⇒ قفل صارم =====
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
#  SMART POSITION MANAGER WITH PROFIT ENGINE AND AUTO-RECOVERY
# ============================================

class SmartPositionManager:
    """مدير المراكز الذكي المتكامل مع Profit Engine ونظام استعادة الصفقات"""
    
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
            
        exchange_side = "buy" if side.upper() == "BUY" else "sell"
        pos_side = "long" if exchange_side == "buy" else "short"

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

        # Trend Birth Override
        if analysis.get("trend_birth", {}).get("active"):
            trade_type = "trend_birth"
            trade_mode = "FULL_TREND"

        if self.exchange.execute_order(exchange_side, position_size, current_price):
            self.state.update({
                "open": True,
                "side": pos_side,
                "entry": current_price,
                "qty": position_size,
                "pnl": 0.0,
                "bars": 0,
                "highest_profit_pct": 0.0,
                "profit_targets_achieved": 0,
                "opened_at": time.time(),
                "last_signal": pos_side,
                "trade_type": trade_type,
                "trade_profile": "SCALP_STRICT",
                "edge_setup": analysis.get("edge_setup"),
                "entry_price": current_price,
                "tp1_hit": False,
                "tp2_hit": False,
            })

            log_g(
                f"✅ New Position Opened: {pos_side.upper()} | "
                f"Size={position_size:.4f} | Entry: {current_price:.6f} | "
                f"Type: {trade_type.upper()}"
            )

            atr_value = analysis.get("trend", {}).get("atr", current_price * 0.01)
            self.profit_engine.init_trade(pos_side, current_price, atr_value, trade_mode, analysis)

            balance_now = self.exchange.get_balance()
            log_equity_snapshot(balance_now, self.state.get("compound_pnl", 0.0))
            return True

        return False
    
    def sync_with_exchange(self, df):
        """
        مزامنة حالة البوت مع المركز الفعلي على البورصة.
        """
        if not MODE_LIVE:
            return

        pos = self.exchange.get_open_position()
        state_open = bool(self.state["open"])

        if not pos:
            if state_open:
                log_w("⚠️ State says position OPEN but exchange has NO position → resetting state.")
                self.state.reset()
            return

        side = pos["side"]
        qty = float(pos["qty"])
        entry_price = float(pos["entry_price"])

        self.profit_engine.entry_price = entry_price
        self.profit_engine.atr_entry = entry_price * 0.01
        atr_value = self.profit_engine.calculate_atr(df)

        recovered_analysis = {
            "trend": {"atr": atr_value},
            "confidence": 0.5,
            "edge_setup": self.state.get("edge_setup"),
            "golden_zone": {"type": None, "valid": False},
            "stop_hunt_trap_side": None,
            "stop_hunt_trap_quality": 0.0,
            "signals": ["RECOVERED_FROM_EXCHANGE"],
        }
        trade_mode = "SCALP"

        if not state_open:
            self.state.update({
                "open": True,
                "side": side,
                "entry": entry_price,
                "qty": qty,
                "pnl": 0.0,
                "bars": 0,
                "highest_profit_pct": 0.0,
                "profit_targets_achieved": 0,
                "opened_at": time.time(),
                "last_signal": side,
                "trade_type": self.state.get("trade_type", "recovered"),
                "trade_profile": self.state.get("trade_profile", "SCALP_STRICT"),
                "entry_price": entry_price,
                "tp1_hit": False,
                "tp2_hit": False,
                "profit_engine_active": False,
            })

            self.profit_engine.init_trade(side, entry_price, atr_value, trade_mode, recovered_analysis)

            log_g(
                f"♻️ Re-attached to existing exchange position | "
                f"side={side.upper()} | qty={qty:.4f} | entry={entry_price:.6f}"
            )
            return

        if self.state["open"] and not self.state.get("profit_engine_active", False):
            if not self.state.get("entry_price"):
                self.state["entry_price"] = entry_price
            if not self.state.get("qty"):
                self.state["qty"] = qty
            if not self.state.get("side"):
                self.state["side"] = side

            side_state = self.state["side"]
            entry_state = float(self.state["entry_price"])

            self.profit_engine.init_trade(side_state, entry_state, atr_value, trade_mode, recovered_analysis)
            log_g(
                f"♻️ Profit Engine re-initialized for existing position "
                f"| side={side_state.upper()} | qty={self.state['qty']:.4f} | entry={entry_state:.6f}"
            )
            return

        mismatch = False
        try:
            state_side = (self.state.get("side") or "").lower()
            state_qty = float(self.state.get("qty", 0.0))
            state_entry = float(self.state.get("entry_price", entry_price))

            if state_side not in ("long", "short"):
                mismatch = True
            if abs(state_qty - qty) > 1e-6:
                mismatch = True
        except Exception:
            mismatch = True

        if mismatch:
            log_w(
                "⚠️ State/Exchange position mismatch → resyncing.\n"
                f"    state: side={self.state.get('side')} qty={self.state.get('qty')} entry={self.state.get('entry_price')}\n"
                f"    exch : side={side} qty={qty} entry={entry_price}"
            )

            self.state.update({
                "open": True,
                "side": side,
                "entry": entry_price,
                "entry_price": entry_price,
                "qty": qty,
            })

            self.profit_engine.init_trade(side, entry_price, atr_value, trade_mode, recovered_analysis)
            log_g(
                f"♻️ State re-synced to exchange position | "
                f"side={side.upper()} | qty={qty:.4f} | entry={entry_price:.6f}"
            )
    
    def manage_position(self, df):
        """إدارة المركز المفتوح مع Profit Engine"""
        if not self.state["open"]:
            return
            
        side = self.state["side"]
        entry = self.state["entry"]
        qty = self.state["qty"]
        sl = self.state.get("sl")
        tp1 = self.state.get("tp1")
        tp2 = self.state.get("tp2")
        tp3 = self.state.get("tp3")
        profile = self.state.get("profit_profile", "SCALP_STRICT")

        log_i(
            f"🧩 TRADE MANAGEMENT | side={side} | qty={qty:.4f} | entry={entry} | "
            f"SL={sl} | TP1={tp1} | TP2={tp2} | TP3={tp3} | profile={profile}"
        )
        
        closed = self.profit_engine.on_tick(df)
        
        if not closed:
            current_price = self.exchange.get_current_price()
            if current_price:
                entry_price = self.state["entry_price"]
                side = self.state["side"]
                
                if side == "long":
                    pnl_pct = (current_price - entry_price) / entry_price * 100
                else:
                    pnl_pct = (entry_price - current_price) / entry_price * 100
                    
                self.state["pnl"] = pnl_pct
                
                if pnl_pct > self.state["highest_profit_pct"]:
                    self.state["highest_profit_pct"] = pnl_pct
                
                self.state["bars"] += 1
    
    def close_position(self, reason=""):
        """إغلاق المركز الحالي مع تفعيل الكول داون"""
        if not self.state["open"]:
            return False
            
        side = "sell" if self.state["side"] == "long" else "buy"
        current_price = self.exchange.get_current_price()
        
        if current_price and self.exchange.execute_order(side, self.state["qty"], current_price):
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

            # تفعيل الكول داون لمدة 30 دقيقة
            cooldown_until = datetime.utcnow() + timedelta(minutes=30)
            self.state["cooldown_until"] = cooldown_until.isoformat()
            self.state["last_close_reason"] = reason
            self.state["last_close_time"] = datetime.now().isoformat()

            self.state.reset()
            return True
            
        log_e(f"❌ Failed to close position: {reason}")
        return False

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
        يحاول يبني صفقة احترافية
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
#  ULTRA COUNCIL AI - نظام التصويت الذكي المتكامل
# ============================================

class UltraCouncilAI:
    """مجلس الإدارة الذكي المتكامل مع جميع المحركات + OrderFlow/Bookmap"""
    
    def __init__(self, exchange_manager: "ExchangeManager" = None):
        self.exchange_manager = exchange_manager
        
        self.stop_hunt_detector = StopHuntDetector()
        self.trend_analyzer = TrendAnalyzer()
        
        self.orderflow_engine = OrderFlowEngine(exchange_manager) if exchange_manager else None
        
        self.ultra_ms = UltraMarketStructureEngine()

        self.edge_algo = EdgeAlgoEngine()
        self.smc_ctx_engine = self
        self.golden_engine = GoldenZoneEngine()
        self.sh_predictor = StopHuntPredictor()
        
        self.min_confidence = 0.6
        self.min_score = 7

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
                "rsi": 50.0,
                "adx": 0.0,
                "di_plus": 0.0,
                "di_minus": 0.0,
                "atr": 0.0,
                "is_strong": False,
            },
            "stop_hunt_zones": 0,
            "smc_ctx": {},
            "edge_setup": None,
            "stop_hunt_trap_side": None,
            "stop_hunt_trap_quality": 0.0,
            "golden_zone": {"type": None, "valid": False},
            "predicted_stop_hunt": {},
            "volume_analysis": {},
            "rf": {
                "filt": 0.0,
                "hi_band": 0.0,
                "lo_band": 0.0,
                "dir": 0,
                "buy_signal": False,
                "sell_signal": False,
            },
            "vwap": 0.0,
            "ultra_ms": {
                "bias": "neutral",
                "bos": None,
                "choch": None,
                "fvg": None,
                "premium_discount": None,
                "liq_grab": {"grab_up": False, "grab_down": False},
            },
            "orderflow": {
                "buy_volume": 0.0,
                "sell_volume": 0.0,
                "delta": 0.0,
                "cvd": 0.0,
                "flow_side": "NEUTRAL",
                "book_imbalance": 0.0,
                "buy_wall": False,
                "sell_wall": False,
                "wall_side": None,
                "wall_distance": None,
            },
            "cvd_divergence": {
                "signal": None,
            },
            "ema_trend": {
                "dir": "neutral",
                "strength": "weak",
                "stair_state": "broken",
                "score": 0.0,
            },
            "ema_cross": {
                "label": "none",
                "side": None,
                "strength": None,
                "fast": None,
                "slow": None,
            },
            "trend_birth": {
                "active": False,
                "side": None,
                "confidence": 0.0,
                "entry_zone": None,
                "reasons": []
            },
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
        """تحليل السوق الشامل المتكامل مع Trend Birth Engine"""
        if len(df) < 20:
            return self._empty_analysis()

        try:
            current_price = float(df['close'].iloc[-1])
            signals = []
            score_buy = 0
            score_sell = 0
            
            self.trend_analyzer.update(df)
            trend_info = self.trend_analyzer.get_trend_info()
            trend_dir = trend_info.get("direction", "flat")
            
            rf_ctx = compute_range_filter(df, period=20, qty=3.5)
            vwap_value = compute_vwap(df)
            
            ultra_ms_ctx = self.ultra_ms.analyze(df)
            
            ema_trend = compute_ema_trend(df)
            
            orderflow_ctx = {}
            if self.orderflow_engine is not None:
                orderflow_ctx = self.orderflow_engine.compute(current_price)
                flow_side = orderflow_ctx.get("flow_side", "NEUTRAL")
                
                if flow_side == "BUY":
                    score_buy += 1.5
                    signals.append("🌊 OrderFlow BUY Pressure")
                elif flow_side == "SELL":
                    score_sell += 1.5
                    signals.append("🌊 OrderFlow SELL Pressure")
                
                wall_side = orderflow_ctx.get("wall_side")
                if wall_side == "BUY":
                    score_buy += 0.5
                    signals.append("🧱 Buy Wall Support")
                elif wall_side == "SELL":
                    score_sell += 0.5
                    signals.append("🧱 Sell Wall Resistance")
            
            cvd_sig = detect_cvd_divergence(df)
            analysis_result = self._empty_analysis()
            analysis_result["cvd_divergence"]["signal"] = cvd_sig
            
            if cvd_sig == "bullish":
                score_buy += 1.5
                signals.append("📈 CVD BULLISH DIVERGENCE")
            elif cvd_sig == "bearish":
                score_sell += 1.5
                signals.append("📉 CVD BEARISH DIVERGENCE")
            
            # ===== TREND BIRTH ENGINE =====
            smc_ctx = self.build_context(
                df, current_price, 
                {"active_count": 0},
                ultra_ms_ctx.get("fvg", {}),
                self.stop_hunt_detector.detect_liquidity_zones(current_price)
            )
            
            smc_ctx.update({
                "liq_grab": ultra_ms_ctx.get("liq_grab", {}),
                "bos": ultra_ms_ctx.get("bos"),
                "choch": ultra_ms_ctx.get("choch"),
                "fvg": ultra_ms_ctx.get("fvg", {})
            })
            
            rsi_value = trend_info.get("rsi", 50)
            rsi_ctx = {"trendZ": "bull" if rsi_value > 50 else "bear"}
            
            ema_cross = classify_ema_crossover(df)
            
            trend_birth = trend_birth_engine(
                df,
                smc_ctx=smc_ctx,
                rsi_ctx=rsi_ctx,
                ema_ctx=ema_cross,
                flow_ctx=orderflow_ctx
            )
            
            analysis_result["trend_birth"] = trend_birth

            if trend_birth["active"]:
                analysis_result["confidence"] = max(analysis_result.get("confidence", 0), trend_birth["confidence"])
                if trend_birth["side"] == "BUY":
                    score_buy += 3.0
                else:
                    score_sell += 3.0
                signals.append(
                    f"🔥 TREND BIRTH {trend_birth['side']} @ {trend_birth['entry_zone']}"
                )
            
            # ===== RF REAL CONTRIBUTION =====
            if rf_ctx.get("buy_signal") and current_price > rf_ctx.get("filt", current_price):
                score_buy += 1.5
                signals.append("📗 RF BUY Signal")

            if rf_ctx.get("sell_signal") and current_price < rf_ctx.get("filt", current_price):
                score_sell += 1.5
                signals.append("📕 RF SELL Signal")

            # ===== VWAP CONTRIBUTION =====
            if vwap_value:
                dist = (current_price - vwap_value) / vwap_value

                if dist > 0 and trend_dir == "up":
                    score_buy += 1.0
                    signals.append("⚖️ Above VWAP in Uptrend")
                elif dist < 0 and trend_dir == "down":
                    score_sell += 1.0
                    signals.append("⚖️ Below VWAP in Downtrend")

                if abs(dist) > 0.01:
                    if dist > 0 and score_buy < score_sell:
                        score_sell *= 0.9
                        signals.append("⚠️ SELL far above VWAP (risk)")
                    elif dist < 0 and score_sell < score_buy:
                        score_buy *= 0.9
                        signals.append("⚠️ BUY far below VWAP (risk)")

            # ===== ULTRA MARKET STRUCTURE CONTRIBUTION =====
            ms_bias = ultra_ms_ctx.get("bias", "neutral")
            ms_fvg = ultra_ms_ctx.get("fvg") or {}
            ms_prem = ultra_ms_ctx.get("premium_discount") or {}
            liq_ctx = ultra_ms_ctx.get("liq_grab") or {}

            if ms_bias == "bull":
                score_buy += 2.0
                signals.append("🏛 UltraMS Bull BOS")
            elif ms_bias == "bear":
                score_sell += 2.0
                signals.append("🏛 UltraMS Bear BOS")

            if ms_fvg:
                if ms_fvg.get("bull_near"):
                    score_buy += 1.5
                    signals.append("🟩 Bull FVG Near")
                if ms_fvg.get("bear_near"):
                    score_sell += 1.5
                    signals.append("🟥 Bear FVG Near")

            zone = ms_prem.get("zone", "mid")
            if zone in ("discount", "ultra_discount") and ms_bias == "bull":
                score_buy += 1.0
                signals.append("💚 Discount + Bull Bias")
            if zone in ("premium", "ultra_premium") and ms_bias == "bear":
                score_sell += 1.0
                signals.append("❤️ Premium + Bear Bias")

            if liq_ctx.get("grab_up"):
                score_sell += 1.0
                signals.append("💦 Liquidity Grab UP")
            if liq_ctx.get("grab_down"):
                score_buy += 1.0
                signals.append("💦 Liquidity Grab DOWN")

            # ===== EMA TREND CONTRIBUTION =====
            if ema_trend["dir"] == "up":
                score_buy += 1.0
                signals.append(f"📈 EMA Uptrend ({ema_trend['strength']})")
            elif ema_trend["dir"] == "down":
                score_sell += 1.0
                signals.append(f"📉 EMA Downtrend ({ema_trend['strength']})")
            
            if ema_trend["stair_state"] == "holding":
                if ema_trend["dir"] == "up":
                    score_buy += 0.5
                    signals.append("🧱 EMA Stair Holding (Bullish)")
                elif ema_trend["dir"] == "down":
                    score_sell += 0.5
                    signals.append("🧱 EMA Stair Holding (Bearish)")

            # الستوب هانت والسيولة
            self.stop_hunt_detector.detect_swings(df)
            stop_hunt_zones = self.stop_hunt_detector.detect_stop_hunt_zones(df)
            active_zones = self.stop_hunt_detector.get_active_stop_hunt_zones(current_price, df)
            active_count = len(active_zones)

            # تحليل Trap Mode مع ADX+ATR
            trap_side = None
            trap_quality = 0.0

            for zone in active_zones:
                trend_context = zone.get("trend_context", {})
                
                if zone["type"] == "buy_stop_hunt" and trend_context.get("valid_for_trap", False):
                    allowed_side = trend_context.get("allowed_side")
                    if allowed_side in ["BUY", None]:
                        trap_side = "BUY"
                        trap_quality = max(trap_quality, zone["strength"] + trend_context.get("adx", 0)/50)
                        signals.append(f"🧨 TRAP_BUY_ZONE @ {zone['level']:.6f} | ADX={trend_context.get('adx',0):.1f}")
                
                if zone["type"] == "sell_stop_hunt" and trend_context.get("valid_for_trap", False):
                    allowed_side = trend_context.get("allowed_side")
                    if allowed_side in ["SELL", None]:
                        trap_side = "SELL"
                        trap_quality = max(trap_quality, zone["strength"] + trend_context.get("adx", 0)/50)
                        signals.append(f"🧨 TRAP_SELL_ZONE @ {zone['level']:.6f} | ADX={trend_context.get('adx',0):.1f}")

            # الاتجاه والزخم
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

            # ===== EMA CROSS CONTRIBUTION =====
            ema_cross = classify_ema_crossover(df)

            if ema_cross["label"] != "none":
                side = ema_cross["side"]
                strength = ema_cross["strength"]

                if side == "bull":
                    if strength == "strong":
                        score_buy += 1.5
                        signals.append("📈 EMA Strong Bullish Cross")
                    else:
                        score_buy += 0.5
                        signals.append("📈 EMA Weak Bullish Cross")
                elif side == "bear":
                    if strength == "strong":
                        score_sell += 1.5
                        signals.append("📉 EMA Strong Bearish Cross")
                    else:
                        score_sell += 0.5
                        signals.append("📉 EMA Weak Bearish Cross")
            else:
                ema_cross = {
                    "label": "none",
                    "side": None,
                    "strength": None,
                    "fast": None,
                    "slow": None,
                }

            # Edge Algo Setup
            edge_side = None
            if score_buy > score_sell:
                edge_side = "BUY"
            elif score_sell > score_buy:
                edge_side = "SELL"

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

            # Golden Zones
            golden = self.golden_engine.compute(df)
            if golden["valid"]:
                if golden["type"] == "golden_bottom":
                    score_buy += 2
                    signals.append("🟢 Golden Bottom Zone")
                elif golden["type"] == "golden_top":
                    score_sell += 2
                    signals.append("🔴 Golden Top Zone")

            # التنبؤ بالستوب هانت
            predicted_sh = self.sh_predictor.predict(df)
            if predicted_sh.get("up_target"):
                signals.append(f"🎯 Predicted Stop-Hunt UP @ {predicted_sh['up_target']:.6f}")
                score_sell += 1.5

            if predicted_sh.get("down_target"):
                signals.append(f"🎯 Predicted Stop-Hunt DOWN @ {predicted_sh['down_target']:.6f}")
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
                "rf": rf_ctx,
                "vwap": vwap_value,
                "ultra_ms": ultra_ms_ctx,
                "volume_analysis": {
                    "delta": 0,
                    "spike": False,
                    "abs_bull": False,
                    "abs_bear": False
                },
                "orderflow": orderflow_ctx,
                "cvd_divergence": analysis_result["cvd_divergence"],
                "ema_trend": ema_trend,
                "ema_cross": ema_cross,
                "trend_birth": trend_birth,
            }
            
        except Exception as e:
            log_e(f"❌ Ultra market analysis error: {e}")
            return self._empty_analysis()

    def should_enter_trade(self, df):
        """تحديد ما إذا كان يجب الدخول في صفقة مع أولوية Trend Birth"""
        analysis = self.analyze_market(df)

        if analysis is None:
            return None, "NO_ANALYSIS", analysis

        # 🔥 TREND BIRTH OVERRIDE - أولوية قصوى
        tb = analysis.get("trend_birth", {})
        if tb.get("active") and tb.get("confidence", 0) >= 0.80:
            entry_signal = "buy" if tb["side"] == "BUY" else "sell"
            reason = f"TREND_BIRTH_{tb['side']}_{tb['entry_zone']}"
            return entry_signal, reason, analysis

        trap_side = analysis.get("stop_hunt_trap_side")
        trap_q = analysis.get("stop_hunt_trap_quality", 0.0)
        predicted = analysis.get("predicted_stop_hunt", {})
        smc_ctx = analysis.get("smc_ctx", {})
        trend = analysis.get("trend", {})
        
        cvd_sig = analysis["cvd_divergence"]["signal"]
        of = analysis["orderflow"]
        flow_side = of.get("flow_side", "NEUTRAL")
        vwap_price = analysis.get("vwap", 0.0)
        rf_info = analysis["rf"]
        current_price = float(df["close"].iloc[-1])
        confidence = analysis.get("confidence", 0.0)

        # 1) TRAP OVERRIDE MODE
        if trap_side and trap_q >= 2.5:
            log_w("🧨 TRAP OVERRIDE MODE ACTIVATED")

            sweep = smc_ctx.get("liquidity_sweep", False)
            stop_hunt = smc_ctx.get("stop_hunt_zone", False)

            if sweep or stop_hunt:
                entry_signal = trap_side.lower()
                reason = (
                    f"TRAP_OVERRIDE | StopHunt={trap_q:.1f} "
                    f"| sweep={sweep} | stop_hunt={stop_hunt} | ADX={trend.get('adx',0):.1f}"
                )
                return entry_signal, reason, analysis

        # 2) لو الثقة قليلة جرّب Trap Mode قبل الرفض
        if analysis.get("confidence", 0.0) < self.min_confidence:
            if trap_side and trap_q >= 3.0:
                entry_signal = trap_side.lower()
                reason = f"TRAP MODE {trap_side} | Stop-Hunt Exploit | Q={trap_q:.1f}"
                return entry_signal, reason, analysis

            return None, "Low confidence", analysis

        # 3) توقع ضرب الاستوبات
        trend_dir = trend.get("direction", "flat")

        if predicted.get("up_target") and trend_dir == "down":
            if analysis.get("score_sell", 0) >= self.min_score - 3:
                sniper_ok = True
                if cvd_sig != "bearish":
                    sniper_ok = False
                if not (rf_info.get("sell_signal") or rf_info.get("dir", 0) <= 0):
                    sniper_ok = False
                if vwap_price and current_price >= vwap_price:
                    sniper_ok = False
                if flow_side not in ("SELL", "STRONG_SELL"):
                    sniper_ok = False
                
                if sniper_ok:
                    return "sell", "SNIPER_SELL_CVD_DIVERGENCE", analysis
                else:
                    return None, "sell_rejected_sniper_filter", analysis

        if predicted.get("down_target") and trend_dir == "up":
            if analysis.get("score_buy", 0) >= self.min_score - 3:
                sniper_ok = True
                if cvd_sig != "bullish":
                    sniper_ok = False
                if not (rf_info.get("buy_signal") or rf_info.get("dir", 0) >= 0):
                    sniper_ok = False
                if vwap_price and current_price <= vwap_price:
                    sniper_ok = False
                if flow_side not in ("BUY", "STRONG_BUY"):
                    sniper_ok = False
                
                if sniper_ok:
                    return "buy", "SNIPER_BUY_CVD_DIVERGENCE", analysis
                else:
                    return None, "buy_rejected_sniper_filter", analysis

        # 4) Golden Zone Override
        entry_signal = None
        reason = ""
        golden = analysis.get("golden_zone", {})

        if golden.get("valid"):
            zone_type  = golden.get("type")
            score_buy  = float(analysis.get("score_buy", 0) or 0.0)
            score_sell = float(analysis.get("score_sell", 0) or 0.0)

            min_golden_score = max(self.min_score - 3, 5)

            rf_dir    = rf_info.get("dir", 0)
            buy_sig   = rf_info.get("buy_signal", False)
            sell_sig  = rf_info.get("sell_signal", False)
            vwap_ok_up   = (not vwap_price) or (current_price >= vwap_price * 0.995)
            vwap_ok_down = (not vwap_price) or (current_price <= vwap_price * 1.005)

            if zone_type == "golden_bottom" and score_buy >= min_golden_score:
                rf_ok   = (rf_dir >= 0) or buy_sig
                flow_ok = flow_side not in ("SELL", "STRONG_SELL")
                cvd_ok  = (cvd_sig != "bearish")

                if rf_ok and flow_ok and vwap_ok_up and cvd_ok:
                    entry_signal = "buy"
                    reason = (
                        f"GOLDEN BOTTOM SNIPER | "
                        f"Score={score_buy:.1f} | Conf={confidence:.1f}"
                    )

            elif zone_type == "golden_top" and score_sell >= min_golden_score:
                rf_ok   = (rf_dir <= 0) or sell_sig
                flow_ok = flow_side not in ("BUY", "STRONG_BUY")
                cvd_ok  = (cvd_sig != "bullish")

                if rf_ok and flow_ok and vwap_ok_down and cvd_ok:
                    entry_signal = "sell"
                    reason = (
                        f"GOLDEN TOP SNIPER | "
                        f"Score={score_sell:.1f} | Conf={confidence:.1f}"
                    )

        if entry_signal:
            return entry_signal, reason, analysis

        # 5) القرار العادي مع CVD Filter
        if entry_signal is None:
            if analysis.get("score_buy", 0) >= self.min_score and analysis["score_buy"] > analysis["score_sell"]:
                sniper_ok = True
                if cvd_sig != "bullish":
                    sniper_ok = False
                if not (rf_info.get("buy_signal") or rf_info.get("dir", 0) >= 0):
                    sniper_ok = False
                if vwap_price and current_price <= vwap_price:
                    sniper_ok = False
                if flow_side not in ("BUY", "STRONG_BUY"):
                    sniper_ok = False
                
                if sniper_ok:
                    entry_signal = "buy"
                    reason = (
                        f"ULTRA BUY | Score: {analysis['score_buy']} "
                        f"| Confidence: {analysis['confidence']}"
                    )
                else:
                    return None, "buy_rejected_sniper_filter", analysis
                    
            elif analysis.get("score_sell", 0) >= self.min_score and analysis["score_sell"] > analysis["score_buy"]:
                sniper_ok = True
                if cvd_sig != "bearish":
                    sniper_ok = False
                if not (rf_info.get("sell_signal") or rf_info.get("dir", 0) <= 0):
                    sniper_ok = False
                if vwap_price and current_price >= vwap_price:
                    sniper_ok = False
                if flow_side not in ("SELL", "STRONG_SELL"):
                    sniper_ok = False
                
                if sniper_ok:
                    entry_signal = "sell"
                    reason = (
                        f"ULTRA SELL | Score: {analysis['score_sell']} "
                        f"| Confidence: {analysis['confidence']}"
                    )
                else:
                    return None, "sell_rejected_sniper_filter", analysis
            else:
                reason = (
                    f"No clear signal | Buy: {analysis.get('score_buy', 0)} "
                    f"| Sell: {analysis.get('score_sell', 0)}"
                )

        return entry_signal, reason, analysis

# ============================================
#  ULTRA PRO AI BOT - الإصدار المتكامل النهائي مع نظام الاستعادة
# ============================================

class UltraProAIBot:
    """البوت الرئيسي المتكامل مع جميع الميزات"""
    
    def __init__(self):
        self.exchange = ExchangeManager()
        self.state = StateManager()
        self.position_manager = SmartPositionManager(self.exchange, self.state)
        self.council = UltraCouncilAI(self.exchange)
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
        log_g("🔹 FEATURES: RF Real + EdgeAlgo + SMC + Golden Zones + Trap Mode + Stop-Hunt Prediction + SMART PROFIT ENGINE + Web Service + ULTRA PANEL + ADX+ATR FILTER + VWAP + Ultra Market Structure + AUTO-RECOVERY SYSTEM + EMA HOLD ENGINE + EMA CROSS CLASSIFIER + TREND BIRTH ENGINE + SMART COOLDOWN SYSTEM")
        
        balance_now = self.exchange.get_balance()
        log_equity_snapshot(balance_now, self.state["compound_pnl"])
        
        self.running = True
    
    def stop(self):
        """إيقاف البوت"""
        self.running = False
        log_i("🛑 Bot stopped by user")
    
    def trade_loop(self):
        """حلقة التداول الرئيسية مع نظام استعادة الصفقات"""
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

                self.state["balance"] = balance
                self.state.setdefault("compound_pnl", 0.0)
                self.state["mode_live"] = MODE_LIVE

                log_equity_snapshot(balance, self.state.get("compound_pnl", 0.0))

                self.position_manager.sync_with_exchange(df)

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
        """معالجة قرار التداول المتكامل مع نظام الكول داون الذكي"""
        if balance <= 10:
            return

        decision, reason, analysis = self.council.should_enter_trade(df)
        
        log_ultra_panel(analysis, self.state)
        
        if analysis.get("signals"):
            log_i(f"🔍 ULTRA Analysis: {', '.join(analysis['signals'][:3])}...")
        
        if decision:
            # ===== SMART COOLDOWN CHECK =====
            cooldown_active, mins_left = is_cooldown_active(self.state)
            
            if cooldown_active:
                # استثناء: Trend Birth Engine يكسر الكول داون دائماً
                tb = analysis.get("trend_birth", {})
                if tb.get("active") and tb.get("confidence", 0) >= 0.80:
                    log_w(
                        f"⚡ TREND BIRTH OVERRIDE COOLDOWN: {tb['side']} @ {tb['entry_zone']} | "
                        f"confidence={tb['confidence']:.2f} | "
                        f"minutes_left={mins_left:.1f}"
                    )
                # استثناء: ترند قوي يكسر الكول داون
                elif is_strong_trend_opportunity(decision, analysis):
                    log_w(
                        f"⚡ TREND STRENGTH OVERRIDE COOLDOWN: {decision.upper()} | "
                        f"conf={analysis.get('confidence'):.2f} | "
                        f"score_b={analysis.get('score_buy')} "
                        f"score_s={analysis.get('score_sell')} | "
                        f"minutes_left={mins_left:.1f}"
                    )
                else:
                    log_i(
                        f"⏸ COOLDOWN ACTIVE | minutes_left={mins_left:.1f} | "
                        f"last_close={self.state.get('last_close_reason')}"
                    )
                    return

            log_i(f"🎯 ULTRA Decision: {reason}")

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

            tb = analysis.get("trend_birth", {})
            if tb.get("active"):
                analysis["trade_mode"] = "FULL_TREND"
                log_i(f"🔥 TREND BIRTH MODE: {tb['side']} @ {tb['entry_zone']} → FULL_TREND profile")

            if self.position_manager.open_position(decision.upper(), df, analysis):
                log_g(f"💰 ULTRA Position opened successfully | Signals: {len(analysis['signals'])}")
                
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
        
        if self.state["open"] and self.state.get("profit_engine_active"):
            status["profit_engine"] = self.position_manager.profit_engine.get_status()
        
        cooldown_active, mins_left = is_cooldown_active(self.state)
        status["cooldown"] = {
            "active": cooldown_active,
            "minutes_left": mins_left,
            "until": self.state.get("cooldown_until")
        }
        
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
            "cooldown_active": status["cooldown"]["active"],
            "cooldown_minutes_left": round(status["cooldown"]["minutes_left"], 1),
            "last_close_reason": status["position"].get("last_close_reason", None),
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
    
    @app.route("/reset_cooldown")
    def reset_cooldown():
        if bot_instance:
            bot_instance.state["cooldown_until"] = None
            return jsonify({"status": "cooldown_reset"})
        return jsonify({"error": "Bot not running"})
    
    return app

# ============================================
#  START APPLICATION
# ============================================

def main():
    """الدالة الرئيسية لتشغيل التطبيق"""
    global bot
    
    try:
        log_banner()
        
        setup_signal_handlers()
        
        threading.Thread(target=keepalive_loop, daemon=True).start()
        
        bot = UltraProAIBot()
        
        bot.start()
        
        trade_thread = threading.Thread(target=bot.trade_loop, daemon=True)
        trade_thread.start()
        
        log_g(f"🌐 Web Service starting on port {PORT}...")
        
        app_instance = create_app(bot)
        app_instance.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
        
    except KeyboardInterrupt:
        log_i("🛑 Application stopped by user")
    except Exception as e:
        log_e(f"🔴 Fatal error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
