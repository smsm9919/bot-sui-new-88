# path: app-run-host-4-0-board-dobj-flashy.py
# -*- coding: utf-8 -*-
"""
SUI SMART RF BOT (15m)
- Entry: True RF Buy/Sell (last closed bar, no repaint)
- Council-managed exits: Scalp(1 TP) / Trend(2 TPs) / Strong(3 TPs + trail)
- Hard SL = 0.50% always; enforce min RR
- Sizing = 60% of quote balance * 10x leverage (fixed)
- Threads: keepalive_loop + trade_loop
- Flask API: /, /health, /last
"""

from __future__ import annotations

import os
import math
import time
import threading
import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from flask import Flask, jsonify

try:
    import ccxt  # type: ignore
except Exception:
    ccxt = None  # يسمح بالتشغيل في وضع معاينة بدون ccxt

# ───────────── إعدادات عامة ─────────────
BOT_VERSION = "v1.0-smart-rf-15m"
PORT = int(os.getenv("PORT", "6000"))
DRY_RUN = os.getenv("CONFIRM_LIVE", "").upper() != "YES"  # لماذا: قفل حي صريح

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("SUI-RF")

# ثابتين حسب طلبك
LEVERAGE: int = 10
RISK_ALLOC: float = 0.60

# ───────────── أنواع وإعدادات ─────────────
class Side(Enum):
    LONG = auto()
    SHORT = auto()

class SignalClass(Enum):
    SCALP = auto()
    TREND = auto()
    STRONG_TREND = auto()

@dataclass(frozen=True)
class BotConfig:
    exchange: str = os.getenv("EXCHANGE", "bingx")
    symbol: str = os.getenv("SYMBOL", "SUI/USDT")
    timeframe: str = "15m"
    fetch_limit: int = 600

    rf_period: int = 20
    rf_qty: float = 3.5

    adx_period: int = 14
    atr_period: int = 14
    vwap_lookback: int = 288
    smc_pivot_width: int = 3
    smc_lookback: int = 50

    min_council_for_entry: int = 3
    adx_trend_threshold: float = 18.0
    vwap_extension_threshold: float = 1.2

    hard_sl_pct: float = 0.005
    min_rr: float = 1.2
    trail_atr_mult: float = 2.0

    tps_scalp: Tuple[float, ...] = (0.010,)                  # 1.0%
    tps_trend: Tuple[float, ...] = (0.008, 0.016)            # 0.8%,1.6%
    tps_strong: Tuple[float, ...] = (0.008, 0.016, 0.024)    # 0.8%,1.6%,2.4%
    splits_trend: Tuple[float, ...] = (0.5, 0.5)
    splits_strong: Tuple[float, ...] = (0.34, 0.33, 0.33)

    otc_enabled: bool = True
    otc_min_vol_sma_ratio: float = 0.6
    otc_max_spread_to_atr: float = 0.6

@dataclass(frozen=True)
class OrderLeg:
    kind: Literal["entry", "tp", "sl", "trail"]
    price: float
    size_pct: float
    reduce_only: bool

@dataclass(frozen=True)
class OrderPlan:
    side: Side
    entry_price: float
    legs: List[OrderLeg]
    meta: Dict[str, float]

# ───────────── مؤشرات (EMA سلوك Pine) ─────────────
def _ema(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1: return x.copy()
    return pd.Series(x).ewm(span=n, adjust=False).mean().to_numpy()

def _atr(df: pd.DataFrame, n: int) -> np.ndarray:
    h,l,c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h-l, np.maximum(np.abs(h-pc), np.abs(l-pc)))
    return _ema(tr, n)

def _adx(df: pd.DataFrame, n: int) -> np.ndarray:
    h,l,c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    up = h - np.r_[h[0], h[:-1]]
    dn = np.r_[l[0], l[:-1]] - l
    plus_dm  = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = _atr(df, 1); tr_s = _ema(tr, n)
    plus_di  = 100.0 * (_ema(plus_dm, n)  / np.maximum(tr_s, 1e-12))
    minus_di = 100.0 * (_ema(minus_dm, n) / np.maximum(tr_s, 1e-12))
    dx = 100.0 * np.abs(plus_di - minus_di) / np.maximum(plus_di + minus_di, 1e-12)
    return _ema(dx, n)

def _vwap_window(df: pd.DataFrame, lookback: int) -> np.ndarray:
    p = df["close"].to_numpy()
    v = df["volume"].to_numpy() if "volume" in df else np.ones_like(p)
    pv = p * v; cpv = np.cumsum(pv); cv = np.cumsum(v)
    out = cpv / np.maximum(cv, 1e-12)
    if lookback <= 0 or lookback >= len(p): return out
    cpv_pad = np.r_[0.0, cpv]; cv_pad = np.r_[0.0, cv]
    win_pv = cpv_pad[lookback:] - cpv_pad[:-lookback]
    win_v  = cv_pad[lookback:] - cv_pad[:-lookback]
    vw = win_pv / np.maximum(win_v, 1e-12)
    return np.r_[np.full(lookback-1, np.nan), vw]

# ───────────── RF الحقيقي (Pine مطابق) ─────────────
def _range_size(x: np.ndarray, qty: float, n: int) -> np.ndarray:
    dx = np.abs(x - np.r_[x[0], x[:-1]])
    avrng = _ema(dx, n); wper = max(1, 2*n - 1)
    return _ema(avrng, wper) * float(qty)

def _range_filter(x: np.ndarray, r: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x); out[0] = x[0]
    for i in range(1, len(x)):
        prev = out[i-1]; up = x[i] - r[i]; dn = x[i] + r[i]
        val = prev
        if up > prev: val = up
        if dn < prev: val = dn
        out[i] = val
    return out

@dataclass(frozen=True)
class RFSignals:
    filt: np.ndarray
    hi_band: np.ndarray
    lo_band: np.ndarray
    long_label: np.ndarray
    short_label: np.ndarray
    fdir: np.ndarray

def compute_rf_signals(df: pd.DataFrame, n: int, qty: float) -> RFSignals:
    x = df["close"].to_numpy()
    r = _range_size(x, qty, n)
    filt = _range_filter(x, r)
    hi, lo = filt + r, filt - r
    fdir = np.zeros_like(x, dtype=float)
    for i in range(1, len(x)):
        fdir[i] = 1 if filt[i] > filt[i-1] else (-1 if filt[i] < filt[i-1] else fdir[i-1])
    up, dn = (fdir == 1), (fdir == -1)
    xp = np.r_[x[0], x[:-1]]
    longCond  = ((x > filt) & (x > xp) & up) | ((x > filt) & (x < xp) & up)
    shortCond = ((x < filt) & (x < xp) & dn) | ((x < filt) & (x > xp) & dn)
    cond = np.zeros_like(x, dtype=int)
    for i in range(1, len(x)):
        cond[i] = 1 if longCond[i] else (-1 if shortCond[i] else cond[i-1])
    long_label  = longCond  & (np.r_[0, cond[:-1]] == -1)
    short_label = shortCond & (np.r_[0, cond[:-1]] ==  1)
    return RFSignals(filt=filt, hi_band=hi, lo_band=lo,
                     long_label=long_label, short_label=short_label, fdir=fdir)

# ───────────── SMC + Council + OTC ─────────────
def _pivots(series: np.ndarray, width: int) -> Tuple[List[int], List[int]]:
    n = len(series); highs, lows = [], []
    for i in range(width, n - width):
        s = series[i]
        if s == np.max(series[i - width:i + width + 1]): highs.append(i)
        if s == np.min(series[i - width:i + width + 1]): lows.append(i)
    return highs, lows

def _smc_structure(df: pd.DataFrame, width: int, lookback: int) -> Tuple[bool, bool]:
    h, l = df["high"].to_numpy(), df["low"].to_numpy()
    hi_idx, lo_idx = _pivots(h, width)[0], _pivots(l, width)[1]
    rhi = [i for i in hi_idx if i >= max(0, len(h)-lookback)]
    rlo = [i for i in lo_idx if i >= max(0, len(l)-lookback)]
    up = dn = False
    if len(rhi) >= 2 and len(rlo) >= 2:
        hh = h[rhi[-1]] > h[rhi[-2]]
        hl = l[rlo[-1]] > l[rlo[-2]]
        up = hh and hl
        ll = l[rlo[-1]] < l[rlo[-2]]
        lh = h[rhi[-1]] < h[rhi[-2]]
        dn = ll and lh
    return up, dn

@dataclass(frozen=True)
class CouncilVotes:
    rf_dir: int
    vwap_side: int
    adx_trend: int
    smc_trend: int
    strength: float
    votes: int

def _council(df: pd.DataFrame, cfg: BotConfig, rf: RFSignals) -> Tuple[np.ndarray, CouncilVotes, np.ndarray, np.ndarray]:
    c = df["close"].to_numpy()
    adx = _adx(df, cfg.adx_period)
    atr = _atr(df, cfg.atr_period)
    vwap = _vwap_window(df, cfg.vwap_lookback)
    i = len(c) - 1
    rf_dir = 1 if rf.filt[i] > rf.filt[i-1] else (-1 if rf.filt[i] < rf.filt[i-1] else 0)
    vwap_side = 1 if c[i] > vwap[i] else -1
    adx_tr = 1 if (adx[i] >= cfg.adx_trend_threshold and c[i] > rf.filt[i]) else (-1 if (adx[i] >= cfg.adx_trend_threshold and c[i] < rf.filt[i]) else 0)
    smc_up, smc_dn = _smc_structure(df, cfg.smc_pivot_width, cfg.smc_lookback)
    smc_tr = 1 if smc_up else (-1 if smc_dn else 0)
    raw = [rf_dir, vwap_side, adx_tr, smc_tr]
    votes = int(sum(1 for v in raw if v != 0 and (v > 0 if rf_dir > 0 else v < 0)))
    rf_slope = (rf.filt[i] - rf.filt[i-1]) / max(abs(rf.filt[i]), 1e-8)
    vwap_dev = abs((c[i] - vwap[i]) / max(atr[i], 1e-8))
    strength = max(0.0, 0.5*(adx[i]/25.0) + 0.3*min(vwap_dev/cfg.vwap_extension_threshold, 1.5) + 0.2*abs(rf_slope))
    return atr, CouncilVotes(rf_dir, vwap_side, adx_tr, smc_tr, strength, votes), adx, vwap

def _otc_ok(df: pd.DataFrame, cfg: BotConfig, atr: np.ndarray) -> bool:
    if not cfg.otc_enabled: return True
    i = len(df) - 1
    vol = df["volume"].to_numpy() if "volume" in df else np.ones(len(df))
    vol_sma = pd.Series(vol).rolling(20).mean().to_numpy()
    if np.isnan(vol_sma[i]): return False
    vol_ok = vol[i] >= cfg.otc_min_vol_sma_ratio * vol_sma[i]
    spread = df["high"].iat[i] - df["low"].iat[i]
    atr_i = atr[i] if i < len(atr) else np.nan
    if atr_i <= 0 or np.isnan(atr_i): return False
    spread_ok = (spread / atr_i) <= cfg.otc_max_spread_to_atr
    return bool(vol_ok and spread_ok)

# ───────────── تخطيط دخول/خروج ─────────────
def _signal_class(v: CouncilVotes) -> SignalClass:
    if v.votes >= 3 and v.strength >= 1.3: return SignalClass.STRONG_TREND
    if v.votes >= 2 and v.strength >= 0.9: return SignalClass.TREND
    return SignalClass.SCALP

def _ensure_rr(entry: float, sl: float, tp_prices: List[float], min_rr: float) -> List[float]:
    risk = abs(entry - sl)
    def rr(tp: float) -> float: return abs(tp - entry) / max(risk, 1e-12)
    out = []
    for tp in tp_prices:
        if rr(tp) < min_rr:
            sign = 1 if tp > entry else -1
            tp = entry + sign * (risk * min_rr)
        out.append(tp)
    return out

def _build_tp_ladder(side: Side, entry: float, tps: Tuple[float, ...], splits: Tuple[float, ...]) -> List[OrderLeg]:
    legs: List[OrderLeg] = []
    for pct, sz in zip(tps, splits):
        price = entry * (1.0 + pct if side == Side.LONG else 1.0 - pct)
        legs.append(OrderLeg(kind="tp", price=float(price), size_pct=float(sz), reduce_only=True))
    return legs

def _trail_leg(atr_val: float, mult: float) -> OrderLeg:
    return OrderLeg(kind="trail", price=float(mult * atr_val), size_pct=1.0, reduce_only=True)

def plan_orders(df: pd.DataFrame, cfg: BotConfig) -> Optional[OrderPlan]:
    need = max(200, cfg.rf_period * 4)
    if len(df) < need: return None
    rf = compute_rf_signals(df, cfg.rf_period, cfg.rf_qty)
    atr, votes, adx, vwap = _council(df, cfg, rf)
    i = len(df) - 2  # لماذا: آخر شمعة مغلقة (منع إعادة رسم)
    if i < 1: return None
    px = float(df["close"].iat[i])
    is_long, is_short = bool(rf.long_label[i]), bool(rf.short_label[i])
    if not (is_long or is_short): return None
    side = Side.LONG if is_long else Side.SHORT
    if not _otc_ok(df, cfg, atr): return None
    if votes.votes < cfg.min_council_for_entry: return None
    sig = _signal_class(votes)
    sl = px * (1.0 - cfg.hard_sl_pct if side == Side.LONG else 1.0 + cfg.hard_sl_pct)
    if sig is SignalClass.SCALP:
        tps = _build_tp_ladder(side, px, cfg.tps_scalp, (1.0,))
    elif sig is SignalClass.TREND:
        tps = _build_tp_ladder(side, px, cfg.tps_trend, cfg.splits_trend)
    else:
        tps = _build_tp_ladder(side, px, cfg.tps_strong, cfg.splits_strong)
    tps_adj = _ensure_rr(px, sl, [l.price for l in tps], cfg.min_rr)
    tps = [OrderLeg(kind="tp", price=tps_adj[j], size_pct=tps[j].size_pct, reduce_only=True) for j in range(len(tps))]
    legs: List[OrderLeg] = [OrderLeg(kind="entry", price=px, size_pct=1.0, reduce_only=False),
                            OrderLeg(kind="sl", price=sl, size_pct=1.0, reduce_only=True),
                            *tps]
    if sig is SignalClass.STRONG_TREND:
        legs.append(_trail_leg(float(atr[i]), cfg.trail_atr_mult))
    meta = {"sig_class": float({SignalClass.SCALP:0, SignalClass.TREND:1, SignalClass.STRONG_TREND:2}[sig]),
            "votes": float(votes.votes), "strength": float(votes.strength),
            "adx": float(adx[i]), "atr": float(atr[i]), "rf_dir": float(votes.rf_dir),
            "entry_index": float(i)}
    return OrderPlan(side=side, entry_price=px, legs=legs, meta=meta)

# ───────────── CCXT + Sizing (60% * 10x) ─────────────
class ExchangeAdapter:
    def __init__(self, cfg: BotConfig):
        assert ccxt is not None, "ccxt is required"
        if not hasattr(ccxt, cfg.exchange): raise ValueError(f"Unknown exchange id: {cfg.exchange}")
        self.ex = getattr(ccxt, cfg.exchange)({"enableRateLimit": True, "options": {"defaultType": "swap"}})
        self.cfg = cfg
        self.symbol = cfg.symbol
        self.market = None

    def load(self):
        self.ex.load_markets()
        self.market = self.ex.market(self.symbol)
        try:
            if not DRY_RUN:
                self.ex.set_leverage(LEVERAGE, self.symbol)  # لماذا: طلب المستخدم ثابت
        except Exception as e:
            log.warning(f"set_leverage failed: {e}")
        return self

    def fetch_ohlcv(self, limit: int) -> pd.DataFrame:
        o = self.ex.fetch_ohlcv(self.symbol, timeframe=self.cfg.timeframe, limit=limit)
        df = pd.DataFrame(o, columns=["timestamp","open","high","low","close","volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp").sort_index()
        # حذف الشمعة الجارية لمنع إعادة رسم
        now_floor = pd.Timestamp.utcnow().tz_localize("UTC").floor(self.cfg.timeframe)
        if len(df) and df.index[-1] == now_floor:
            df = df.iloc[:-1]
        return df

    def _round_to(self, x: float, step: Optional[float], prec: Optional[int]) -> float:
        if step and step > 0: return float(math.floor(x / step) * step)
        if prec is not None:  return float(round(x, prec))
        return float(x)

    def clamp_amount_price(self, price: float, amount: float) -> Tuple[float, float]:
        m = self.market or self.ex.market(self.symbol)
        prec_p = m.get("precision", {}).get("price", None)
        prec_a = m.get("precision", {}).get("amount", None)
        step_p = (m.get("limits", {}).get("price", {}) or {}).get("min", None)
        step_a = (m.get("limits", {}).get("amount", {}) or {}).get("min", None)
        price  = self._round_to(price, step_p, prec_p)
        amount = self._round_to(amount, step_a, prec_a)
        min_cost = (m.get("limits", {}).get("cost", {}) or {}).get("min", None)
        if min_cost and price * amount < float(min_cost):
            raise ValueError(f"Notional {price*amount:.8f} < min cost {min_cost}")
        return price, amount

    def quote_balance(self) -> float:
        bal = self.ex.fetch_balance()
        q = (self.market or self.ex.market(self.symbol)).get("quote", "USDT")
        if q in bal and isinstance(bal[q], dict):
            return float(bal[q].get("free") or bal[q].get("total") or 0.0)
        return float(bal.get(q, 0.0))

    def create(self, otype: str, side: str, amount: float, price: Optional[float], params: dict):
        if DRY_RUN:
            log.info(f"[DRY] {otype.upper()} {side} {amount} {'@'+str(price) if price else ''} {params}")
            return {"id": "dry"}
        return self.ex.create_order(self.symbol, otype, side, amount, price, params)

    def place(self, plan: OrderPlan) -> List[dict]:
        side = "buy" if plan.side == Side.LONG else "sell"
        # لماذا: الحجم حسب طلبك (60% * 10x)
        balance = self.quote_balance()
        notional = max(0.0, balance * RISK_ALLOC * LEVERAGE)
        amount_raw = notional / max(plan.entry_price, 1e-12)
        _, amount = self.clamp_amount_price(plan.entry_price, amount_raw)

        res = []
        # Entry
        res.append(self.create("market", side, amount, None, {"reduceOnly": False}))
        # SL
        sl = next((l for l in plan.legs if l.kind == "sl"), None)
        if sl:
            sl_price, _ = self.clamp_amount_price(sl.price, amount)
            params = {"reduceOnly": True, "stopPrice": sl_price, "triggerPrice": sl_price}
            res.append(self.create("stop_market", "sell" if side=="buy" else "buy", amount, None, params))
        # TPs
        done = 0.0
        for leg in (l for l in plan.legs if l.kind == "tp"):
            tp_price, tp_amt = self.clamp_amount_price(leg.price, amount * leg.size_pct)
            res.append(self.create("limit", "sell" if side=="buy" else "buy", tp_amt, tp_price, {"reduceOnly": True}))
            done += tp_amt
        # Trail (للترند القوي)
        trail = next((l for l in plan.legs if l.kind == "trail"), None)
        if trail:
            rem = max(0.0, amount - done)
            if rem > 0:
                res.append(self.create("trailing_stop_market", "sell" if side=="buy" else "buy", rem, None,
                                       {"reduceOnly": True, "trailOffsetATR": float(trail.price)}))
        return res

# ───────────── لوج مختصر ─────────────
def log_line(symbol: str, plan: OrderPlan) -> str:
    cls = {0:"SCALP",1:"TREND",2:"STRONG"}[int(plan.meta["sig_class"])]
    side = "BUY" if plan.side == Side.LONG else "SELL"
    return (f"{symbol} | {side} | votes={int(plan.meta['votes'])} | score={plan.meta['strength']:.2f} | "
            f"ADX={plan.meta['adx']:.2f} | ATR={plan.meta['atr']:.4f} | cls={cls} | entry={plan.entry_price:.4f}")

# ───────────── حلقات التشغيل ─────────────
STATE = {"last_plan": None, "last_bar_ts": None}
CFG = BotConfig()
APP = Flask(__name__)

def fetch_df(ex: ExchangeAdapter) -> pd.DataFrame:
    return ex.fetch_ohlcv(CFG.fetch_limit)

def trade_once(ex: ExchangeAdapter):
    df = fetch_df(ex)
    if len(df) < 250: 
        log.warning("insufficient candles"); return
    # تكرار الدخول على نفس الشمعة ممنوع
    last_closed_ts = int(df.index[-1].value // 10**6)  # ms
    if STATE["last_bar_ts"] == last_closed_ts: 
        return
    plan = plan_orders(df, CFG)
    if not plan: 
        return
    log.info(log_line(CFG.symbol, plan))
    try:
        ex.place(plan)
        STATE["last_plan"] = {"ts": last_closed_ts, "plan": plan.meta, "entry": plan.entry_price}
        STATE["last_bar_ts"] = last_closed_ts
    except Exception as e:
        log.error(f"placement failed: {e}")

def trade_loop():
    # لماذا: فصل اللوب عن ويب سيرفر
    if ccxt is None:
        log.error("ccxt not installed"); return
    ex = ExchangeAdapter(CFG).load()
    log.info("🟣 SMART PROFIT AI: Scalp + Trend + Volume Analysis + TP Profile (1+2+3) + Council + Smart RF Patch Activated")
    while True:
        try:
            trade_once(ex)
        except Exception as e:
            log.error(f"trade loop error: {e}")
        # sleep حتى نهاية الشمعة التالية ~15m
        time.sleep(30)  # صغير؛ لأننا نحذف الشمعة الجارية بالفعل

def keepalive_loop():
    while True:
        log.debug("heartbeat ok")
        time.sleep(30)

# ───────────── Flask API ─────────────
@APP.get("/")
def index():
    return jsonify({
        "bot": "SUI SMART RF BOT",
        "version": BOT_VERSION,
        "symbol": CFG.symbol,
        "timeframe": CFG.timeframe,
        "dry_run": DRY_RUN,
        "leverage": LEVERAGE,
        "risk_alloc": RISK_ALLOC,
    })

@APP.get("/health")
def health():
    return jsonify({"ok": True, "dry_run": DRY_RUN})

@APP.get("/last")
def last():
    lp = STATE["last_plan"]
    return jsonify(lp if lp else {"msg": "no trades yet"})

# ───────────── الإقلاع ─────────────
def verify_execution_environment():
    if ccxt is None:
        log.warning("ccxt not installed; DRY mode only.")
    if DRY_RUN:
        log.info("MODE = DRY-RUN (set CONFIRM_LIVE=YES for LIVE)")
    log.info(f"BOT {BOT_VERSION} | SYMBOL: {CFG.symbol} | INTERVAL: {CFG.timeframe} | LEVERAGE: {LEVERAGE}x")

if __name__ == "__main__":
    verify_execution_environment()
    # تشغيل الثريدات
    threading.Thread(target=keepalive_loop, daemon=True).start()
    threading.Thread(target=trade_loop, daemon=True).start()

    log.info("SMART PATCH ACTIVATED: Council Strong Entry + Smart RF Patch (Scalp/Trend/Strong)")
    # 👇 نهاية البوت كما طلبت
    APP.run(host="0.0.0.0", port=PORT, debug=False)
