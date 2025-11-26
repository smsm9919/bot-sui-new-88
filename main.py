# =================== SMART MONEY CONCEPTS (SMC) ENGINE ===================

class SmartMoneyConcepts:
    def __init__(self):
        self.pivot_highs = deque(maxlen=20)
        self.pivot_lows = deque(maxlen=20)
        self.order_blocks = deque(maxlen=15)
        self.fair_value_gaps = deque(maxlen=15)
        self.liquidity_zones = deque(maxlen=10)
        
    def detect_pivot_points(self, df, left_bars=5, right_bars=5):
        """كشف القمم والقيعان المحورية"""
        if len(df) < left_bars + right_bars + 1:
            return
            
        highs = df['high'].astype(float)
        lows = df['low'].astype(float)
        
        for i in range(left_bars, len(df) - right_bars):
            # كشف القمة المحورية
            if highs.iloc[i] == highs.iloc[i-left_bars:i+right_bars+1].max():
                self.pivot_highs.append({
                    'index': i,
                    'price': highs.iloc[i],
                    'timestamp': df['time'].iloc[i],
                    'strength': self._calculate_pivot_strength(highs, i, left_bars, right_bars)
                })
            
            # كشف القاع المحوري
            if lows.iloc[i] == lows.iloc[i-left_bars:i+right_bars+1].min():
                self.pivot_lows.append({
                    'index': i,
                    'price': lows.iloc[i],
                    'timestamp': df['time'].iloc[i],
                    'strength': self._calculate_pivot_strength(lows, i, left_bars, right_bars, False)
                })
    
    def _calculate_pivot_strength(self, series, idx, left, right, is_high=True):
        """حساب قوة القمة/القاع المحوري"""
        if is_high:
            left_min = series.iloc[idx-left:idx].min()
            right_min = series.iloc[idx+1:idx+right+1].min()
            return (series.iloc[idx] - max(left_min, right_min)) / series.iloc[idx] * 100
        else:
            left_max = series.iloc[idx-left:idx].max()
            right_max = series.iloc[idx+1:idx+right+1].max()
            return (min(left_max, right_max) - series.iloc[idx]) / series.iloc[idx] * 100
    
    def detect_order_blocks(self, df, lookback=10):
        """كشف كتل الأوامر (Order Blocks)"""
        if len(df) < lookback + 5:
            return
            
        for i in range(5, len(df) - 1):
            current_candle = {
                'open': float(df['open'].iloc[i]),
                'high': float(df['high'].iloc[i]),
                'low': float(df['low'].iloc[i]),
                'close': float(df['close'].iloc[i])
            }
            next_candle = {
                'open': float(df['open'].iloc[i+1]),
                'high': float(df['high'].iloc[i+1]),
                'low': float(df['low'].iloc[i+1]),
                'close': float(df['close'].iloc[i+1])
            }
            
            # Bullish Order Block: شمعة هابطة يليها شمعة صاعدة
            if (current_candle['close'] < current_candle['open'] and 
                next_candle['close'] > next_candle['open'] and
                next_candle['close'] > current_candle['high']):
                
                self.order_blocks.append({
                    'type': 'bullish',
                    'entry_zone': [current_candle['low'], current_candle['high']],
                    'timestamp': df['time'].iloc[i],
                    'strength': self._calculate_ob_strength(current_candle, next_candle)
                })
            
            # Bearish Order Block: شمعة صاعدة يليها شمعة هابطة
            elif (current_candle['close'] > current_candle['open'] and 
                  next_candle['close'] < next_candle['open'] and
                  next_candle['close'] < current_candle['low']):
                
                self.order_blocks.append({
                    'type': 'bearish',
                    'entry_zone': [current_candle['low'], current_candle['high']],
                    'timestamp': df['time'].iloc[i],
                    'strength': self._calculate_ob_strength(current_candle, next_candle)
                })
    
    def _calculate_ob_strength(self, candle1, candle2):
        """حساب قوة كتلة الأوامر"""
        body1 = abs(candle1['close'] - candle1['open'])
        body2 = abs(candle2['close'] - candle2['open'])
        range1 = candle1['high'] - candle1['low']
        
        if range1 == 0:
            return 0
            
        body_ratio = body1 / range1
        momentum = body2 / body1 if body1 > 0 else 0
        
        return min(5.0, (body_ratio * 3 + momentum * 2))
    
    def detect_fair_value_gaps(self, df):
        """كشف فجوات القيمة العادلة (FVG)"""
        if len(df) < 4:
            return
            
        for i in range(2, len(df) - 1):
            candle_a = {
                'high': float(df['high'].iloc[i-2]),
                'low': float(df['low'].iloc[i-2])
            }
            candle_b = {
                'high': float(df['high'].iloc[i-1]),
                'low': float(df['low'].iloc[i-1])
            }
            candle_c = {
                'high': float(df['high'].iloc[i]),
                'low': float(df['low'].iloc[i])
            }
            
            # Bullish FVG: A-high < C-low
            if candle_a['high'] < candle_c['low']:
                self.fair_value_gaps.append({
                    'type': 'bullish',
                    'zone': [candle_a['high'], candle_c['low']],
                    'size_bps': ((candle_c['low'] - candle_a['high']) / candle_a['high']) * 10000,
                    'timestamp': df['time'].iloc[i]
                })
            
            # Bearish FVG: A-low > C-high
            elif candle_a['low'] > candle_c['high']:
                self.fair_value_gaps.append({
                    'type': 'bearish',
                    'zone': [candle_c['high'], candle_a['low']],
                    'size_bps': ((candle_a['low'] - candle_c['high']) / candle_a['low']) * 10000,
                    'timestamp': df['time'].iloc[i]
                })
    
    def analyze_liquidity_zones(self, current_price):
        """تحليل مناطق السيولة"""
        supply_zones = []
        demand_zones = []
        
        # مناطق العرض (مقاومة)
        for pivot in self.pivot_highs:
            if pivot['price'] > current_price * 1.005:  # فوق السعر ب 0.5%
                supply_zones.append({
                    'price': pivot['price'],
                    'strength': pivot['strength'],
                    'type': 'supply'
                })
        
        # مناطق الطلب (دعم)
        for pivot in self.pivot_lows:
            if pivot['price'] < current_price * 0.995:  # تحت السعر ب 0.5%
                demand_zones.append({
                    'price': pivot['price'],
                    'strength': pivot['strength'],
                    'type': 'demand'
                })
        
        # ترتيب المناطق حسب القوة
        supply_zones.sort(key=lambda x: x['strength'], reverse=True)
        demand_zones.sort(key=lambda x: x['strength'], reverse=True)
        
        return {
            'supply_zones': supply_zones[:3],  # أقوى 3 مناطق عرض
            'demand_zones': demand_zones[:3],  # أقوى 3 مناطق طلب
            'current_strength': self._calculate_current_liquidity_strength(current_price)
        }
    
    def _calculate_current_liquidity_strength(self, current_price):
        """حساب قوة السيولة الحالية"""
        near_supply = any(abs(pivot['price'] - current_price) / current_price < 0.02 
                         for pivot in self.pivot_highs)
        near_demand = any(abs(pivot['price'] - current_price) / current_price < 0.02 
                         for pivot in self.pivot_lows)
        
        return {
            'near_supply': near_supply,
            'near_demand': near_demand,
            'imbalance': len([p for p in self.pivot_highs if p['price'] > current_price]) - 
                        len([p for p in self.pivot_lows if p['price'] < current_price])
        }
    
    def get_smc_analysis(self, df, current_price):
        """تحليل SMC شامل"""
        self.detect_pivot_points(df)
        self.detect_order_blocks(df)
        self.detect_fair_value_gaps(df)
        
        return {
            'pivot_highs': list(self.pivot_highs)[-5:],  # آخر 5 قمم
            'pivot_lows': list(self.pivot_lows)[-5:],    # آخر 5 قيعان
            'order_blocks': list(self.order_blocks)[-3:], # آخر 3 كتل أوامر
            'fair_value_gaps': list(self.fair_value_gaps)[-3:], # آخر 3 فجوات
            'liquidity_pools': self.analyze_liquidity_zones(current_price)
        }

# =================== PRICE STRUCTURE ANALYSIS ===================

def analyze_price_structure(df):
    """تحليل الهيكل السعري المتقدم"""
    if len(df) < 30:
        return {"ok": False, "trend": "neutral", "trend_strength": 0}
    
    highs = df['high'].astype(float)
    lows = df['low'].astype(float)
    closes = df['close'].astype(float)
    
    # تحديد Higher Highs / Lower Lows
    hh = 0
    hl = 0
    lh = 0
    ll = 0
    
    for i in range(2, len(df)):
        # Higher High
        if highs.iloc[i] > highs.iloc[i-1] > highs.iloc[i-2]:
            hh += 1
        # Higher Low
        elif lows.iloc[i] > lows.iloc[i-1] > lows.iloc[i-2]:
            hl += 1
        # Lower High
        elif highs.iloc[i] < highs.iloc[i-1] < highs.iloc[i-2]:
            lh += 1
        # Lower Low
        elif lows.iloc[i] < lows.iloc[i-1] < lows.iloc[i-2]:
            ll += 1
    
    # تحديد الاتجاه
    bullish_strength = hh + hl
    bearish_strength = lh + ll
    total_strength = bullish_strength + bearish_strength
    
    if total_strength == 0:
        return {"ok": False, "trend": "neutral", "trend_strength": 0}
    
    if bullish_strength > bearish_strength * 1.5:
        trend = "bullish"
        strength = bullish_strength / total_strength * 10
    elif bearish_strength > bullish_strength * 1.5:
        trend = "bearish"
        strength = bearish_strength / total_strength * 10
    else:
        trend = "ranging"
        strength = abs(bullish_strength - bearish_strength) / total_strength * 5
    
    # كشف نقاط التحول المحتملة
    potential_reversal = False
    if trend == "bullish" and ll > hh:
        potential_reversal = True
    elif trend == "bearish" and hh > ll:
        potential_reversal = True
    
    return {
        "ok": True,
        "trend": trend,
        "trend_strength": round(strength, 2),
        "bullish_signals": bullish_strength,
        "bearish_signals": bearish_strength,
        "potential_reversal": potential_reversal,
        "structure": f"HH:{hh} HL:{hl} LH:{lh} LL:{ll}"
    }

# =================== VWAP ANALYSIS ===================

def calculate_vwap(df):
    """حساب VWAP (Volume Weighted Average Price)"""
    if len(df) < 20:
        return {"ok": False, "vwap": 0, "deviation": 0}
    
    typical_price = (df['high'].astype(float) + df['low'].astype(float) + df['close'].astype(float)) / 3
    volume = df['volume'].astype(float)
    
    vwap = (typical_price * volume).cumsum() / volume.cumsum()
    current_vwap = vwap.iloc[-1]
    current_price = float(df['close'].iloc[-1])
    
    deviation = ((current_price - current_vwap) / current_vwap) * 100
    
    return {
        "ok": True,
        "vwap": current_vwap,
        "deviation": round(deviation, 4),
        "above_vwap": current_price > current_vwap,
        "aligned": abs(deviation) < 1.0  # ضمن 1% من VWAP
    }

# =================== ENTRY DECISION MASTER ENGINE ===================

def master_entry_engine(council, price):
    """
    أقوى نسخة منطق دخول مرتبة – tier-based + decisive + clean
    """

    tierA = []
    tierB = []
    tierC = []

    det = council["details"]
    gz   = det.get("golden_zone", {})
    smc  = det.get("smc_analysis", {})
    indi = det.get("indicators", {})
    vwap = det.get("vwap_analysis", {})
    cndl = det.get("candle_signals", {})

    # ----------- TIER A (أقوى 3 إشارات لا تفتح صفقة ترند بدونها) -----------

    # GOLDEN ZONE CONFIRMED
    if gz.get("ok") and gz.get("score", 0) >= 6:
        tierA.append(f"GOLDEN({gz['zone']['type']})")

    # STRONG SMC (OB + FVG + Liquidity Sweep)
    smc_strength = 0

    if smc['liquidity_pools']['ok']:
        ph = len(smc['liquidity_pools']['pivot_highs'])
        pl = len(smc['liquidity_pools']['pivot_lows'])
        if ph >= 2 or pl >= 2:
            smc_strength += 2

    if len(smc.get("order_blocks", [])) >= 2:
        smc_strength += 2

    if len(smc.get("fair_value_gaps", [])) >= 2:
        smc_strength += 1

    if smc_strength >= 4:
        tierA.append(f"SMC_STRONG({smc_strength})")

    # LIQUIDITY SWEEP
    lq = smc["liquidity_pools"].get("current_strength", {})
    if lq.get("near_supply") or lq.get("near_demand"):
        tierA.append("LIQ_SWEEP")

    # ----------- TIER B (داعم قوي – يكمل الصفقة) -----------

    # VWAP Alignment
    if vwap.get("ok") and vwap.get("aligned"):
        tierB.append(f"VWAP_ALIGN")

    # Strong Order Block
    obs = smc.get("order_blocks", [])
    if obs and obs[-1]["strength"] >= 3:
        tierB.append(f"OB({obs[-1]['strength']})")

    # Strong FVG
    fvgs = smc.get("fair_value_gaps", [])
    if fvgs and fvgs[-1]["size_bps"] >= 8:
        tierB.append(f"FVG({fvgs[-1]['size_bps']:.1f})")

    # Market Structure
    ps = det.get("price_structure", {})
    if ps.get("ok") and ps.get("trend_strength", 0) >= 2:
        tierB.append(f"STRUCT_{ps['trend']}({ps['trend_strength']})")

    # ----------- TIER C (فلتر وتحسين فقط) -----------

    # RSI context
    rsi = indi.get("rsi", 50)
    if 40 <= rsi <= 60:
        tierC.append(f"RSI_NEUTRAL({rsi:.1f})")
    if rsi < 30:
        tierC.append(f"RSI_OVERSOLD({rsi:.1f})")
    if rsi > 70:
        tierC.append(f"RSI_OVERBOUGHT({rsi:.1f})")

    # ADX
    if indi.get("adx", 0) >= 20:
        tierC.append(f"ADX({indi['adx']:.1f})")

    # Candles
    cs = max(cndl.get("score_buy", 0), cndl.get("score_sell", 0))
    if cs >= 4:
        tierC.append(f"CANDLE({cs})")

    # -----------------------------------
    # --------- DECISION LOGIC ----------
    # -----------------------------------

    score = len(tierA) * 4 + len(tierB) * 2 + len(tierC) * 1

    # Determine Mode
    if len(tierA) >= 1 and score >= 10:
        mode = "trend"
    elif score >= 7:
        mode = "studied_scalp"
    elif score >= 5:
        mode = "cautious_scalp"
    else:
        mode = "reject"

    # Determine Direction
    side = None
    if gz.get("ok"):
        if gz["zone"]["type"] == "golden_bottom": side = "buy"
        if gz["zone"]["type"] == "golden_top":    side = "sell"
    else:
        if cndl.get("score_buy", 0) > cndl.get("score_sell", 0): side = "buy"
        if cndl.get("score_sell", 0) > cndl.get("score_buy", 0): side = "sell"

    if mode == "reject" or side is None:
        return {
            "allow": False,
            "mode": "reject",
            "side": None,
            "score": score,
            "tierA": tierA,
            "tierB": tierB,
            "tierC": tierC,
            "why": "weak_structure"
        }

    return {
        "allow": True,
        "mode": mode,
        "side": side,
        "score": score,
        "tierA": tierA,
        "tierB": tierB,
        "tierC": tierC
    }

# =================== ENHANCED COUNCIL AI WITH SMC ===================

def super_council_ai_enhanced_with_smc(df):
    """نسخة محسنة من المجلس مع دمج SMC والهيكل السعري"""
    try:
        if len(df) < 50:
            return {"b": 0, "s": 0, "score_b": 0.0, "score_s": 0.0, "logs": [], "confidence": 0.0}
        
        current_price = float(df['close'].iloc[-1])
        ind = compute_indicators(df)
        
        # تحليلات متقدمة
        smc_analysis = smc_detector.get_smc_analysis(df, current_price)
        price_structure = analyze_price_structure(df)
        vwap_analysis = calculate_vwap(df)
        gz = golden_zone_check(df, ind)
        candles = compute_candles(df)
        flow = compute_flow_metrics(df)
        
        # تحليل السيولة المتقدم
        liquidity_analysis = smc_detector.analyze_liquidity_zones(current_price)
        
        council_data = {
            "b": 0, "s": 0,
            "score_b": 0.0, "score_s": 0.0,
            "logs": [],
            "confidence": 0.0,
            "details": {
                "smc_analysis": smc_analysis,
                "price_structure": price_structure,
                "vwap_analysis": vwap_analysis,
                "golden_zone": gz,
                "candle_signals": candles,
                "flow_metrics": flow,
                "liquidity_analysis": liquidity_analysis,
                "indicators": ind
            }
        }
        
        # ===== SMC-BASED VOTING =====
        
        # Order Blocks تأثير
        for ob in smc_analysis.get('order_blocks', [])[-2:]:
            if ob['type'] == 'bullish' and ob['strength'] >= 3:
                council_data["score_b"] += ob['strength'] * 0.8
                council_data["b"] += 1
                council_data["logs"].append(f"📦 Bullish OB (str:{ob['strength']:.1f})")
            elif ob['type'] == 'bearish' and ob['strength'] >= 3:
                council_data["score_s"] += ob['strength'] * 0.8
                council_data["s"] += 1
                council_data["logs"].append(f"📦 Bearish OB (str:{ob['strength']:.1f})")
        
        # FVG تأثير
        for fvg in smc_analysis.get('fair_value_gaps', [])[-2:]:
            if fvg['type'] == 'bullish' and fvg['size_bps'] >= 5:
                council_data["score_b"] += min(2.0, fvg['size_bps'] * 0.1)
                council_data["b"] += 1
                council_data["logs"].append(f"📊 Bullish FVG ({fvg['size_bps']:.1f}bps)")
            elif fvg['type'] == 'bearish' and fvg['size_bps'] >= 5:
                council_data["score_s"] += min(2.0, fvg['size_bps'] * 0.1)
                council_data["s"] += 1
                council_data["logs"].append(f"📊 Bearish FVG ({fvg['size_bps']:.1f}bps)")
        
        # الهيكل السعري
        if price_structure["ok"]:
            if price_structure["trend"] == "bullish":
                council_data["score_b"] += price_structure["trend_strength"] * 0.3
                council_data["b"] += int(price_structure["trend_strength"])
                council_data["logs"].append(f"📈 Bullish Structure (str:{price_structure['trend_strength']:.1f})")
            elif price_structure["trend"] == "bearish":
                council_data["score_s"] += price_structure["trend_strength"] * 0.3
                council_data["s"] += int(price_structure["trend_strength"])
                council_data["logs"].append(f"📉 Bearish Structure (str:{price_structure['trend_strength']:.1f})")
        
        # VWAP محاذاة
        if vwap_analysis["ok"] and vwap_analysis["aligned"]:
            if vwap_analysis["above_vwap"]:
                council_data["score_b"] += 1.5
                council_data["b"] += 1
                council_data["logs"].append("🔷 Above VWAP (Bullish)")
            else:
                council_data["score_s"] += 1.5
                council_data["s"] += 1
                council_data["logs"].append("🔷 Below VWAP (Bearish)")
        
        # السيولة
        liq_strength = liquidity_analysis.get('current_strength', {})
        if liq_strength.get('near_demand'):
            council_data["score_b"] += 2.0
            council_data["b"] += 2
            council_data["logs"].append("💧 Near Demand Liquidity")
        if liq_strength.get('near_supply'):
            council_data["score_s"] += 2.0
            council_data["s"] += 2
            council_data["logs"].append("💧 Near Supply Liquidity")
        
        # ===== APPLY MASTER ENTRY ENGINE =====
        master_decision = master_entry_engine(council_data, current_price)
        
        # تحديث البيانات النهائية
        council_data.update({
            "master_decision": master_decision,
            "confidence": min(1.0, (council_data["score_b"] + council_data["score_s"]) / 25.0)
        })
        
        return council_data
        
    except Exception as e:
        log_w(f"Enhanced Council AI with SMC error: {e}")
        return {"b":0,"s":0,"score_b":0.0,"score_s":0.0,"logs":[],"confidence":0.0,"master_decision":{"allow":False}}

# =================== ENHANCED POSITION MANAGEMENT ===================

def manage_position_enhanced_with_smc(df, ind, info):
    """إدارة محسنة للمراكز مع دمج SMC"""
    if not STATE["open"] or STATE["qty"] <= 0:
        return

    px = info["price"]
    entry = STATE["entry"]
    side = STATE["side"]
    mode = STATE.get("mode", "scalp")
    
    pnl_pct = (px - entry) / entry * 100 * (1 if side == "long" else -1)
    STATE["pnl"] = pnl_pct
    
    if pnl_pct > STATE["highest_profit_pct"]:
        STATE["highest_profit_pct"] = pnl_pct

    # ===== SMC-BASED DYNAMIC MANAGEMENT =====
    
    current_price = float(df['close'].iloc[-1])
    smc_analysis = smc_detector.get_smc_analysis(df, current_price)
    liquidity_zones = smc_detector.analyze_liquidity_zones(current_price)
    
    # ترقية السكالب لترند عند إشارات قوية
    if mode == "scalp" and not STATE.get("upgraded_to_trend", False):
        upgrade_conditions = [
            pnl_pct >= 0.8,
            len(smc_analysis.get('order_blocks', [])) >= 2,
            liquidity_zones['current_strength']['imbalance'] > 2
        ]
        
        if sum(upgrade_conditions) >= 2:
            STATE["mode"] = "trend"
            STATE["upgraded_to_trend"] = True
            log_i("🎯 SCALP UPGRADED TO TREND - Strong SMC signals detected")
    
    # وقف خسارة ديناميكي يعتمد على السيولة
    if not STATE.get("dynamic_sl_set", False):
        nearest_support = min([z['price'] for z in liquidity_zones['demand_zones']], default=entry * 0.99)
        nearest_resistance = max([z['price'] for z in liquidity_zones['supply_zones']], default=entry * 1.01)
        
        if side == "long":
            dynamic_sl = nearest_support
        else:
            dynamic_sl = nearest_resistance
            
        STATE["dynamic_stop_loss"] = dynamic_sl
        STATE["dynamic_sl_set"] = True
        log_i(f"🛡️ Dynamic SL set: {dynamic_sl:.6f}")

    # جني أرباح ذكي عند مناطق السيولة
    if side == "long" and liquidity_zones['supply_zones']:
        nearest_supply = liquidity_zones['supply_zones'][0]['price']
        if px >= nearest_supply * 0.998 and pnl_pct > 0.5:  # قرب منطقة العرض
            close_qty = safe_qty(STATE["qty"] * 0.5)
            if close_qty > 0:
                close_side = "sell"
                if MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN:
                    try:
                        params = exchange_specific_params(close_side, is_close=True)
                        ex.create_order(SYMBOL, "market", close_side, close_qty, None, params)
                        log_g(f"🏦 Take Profit at Supply Zone | PnL: {pnl_pct:.2f}%")
                        STATE["qty"] = safe_qty(STATE["qty"] - close_qty)
                    except Exception as e:
                        log_e(f"❌ Supply zone TP failed: {e}")
    
    elif side == "short" and liquidity_zones['demand_zones']:
        nearest_demand = liquidity_zones['demand_zones'][0]['price']
        if px <= nearest_demand * 1.002 and pnl_pct > 0.5:  # قرب منطقة الطلب
            close_qty = safe_qty(STATE["qty"] * 0.5)
            if close_qty > 0:
                close_side = "buy"
                if MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN:
                    try:
                        params = exchange_specific_params(close_side, is_close=True)
                        ex.create_order(SYMBOL, "market", close_side, close_qty, None, params)
                        log_g(f"🏦 Take Profit at Demand Zone | PnL: {pnl_pct:.2f}%")
                        STATE["qty"] = safe_qty(STATE["qty"] - close_qty)
                    except Exception as e:
                        log_e(f"❌ Demand zone TP failed: {e}")
    
    # الإدارة الأساسية
    manage_after_entry_enhanced(df, ind, info)

# =================== ENHANCED TRADE LOOP WITH SMC ===================

def trade_loop_enhanced_with_smc():
    """الدورة الرئيسية المحسنة مع دمج SMC"""
    global wait_for_next_signal_side, compound_pnl
    
    # تهيئة محرك SMC
    global smc_detector
    smc_detector = SmartMoneyConcepts()
    
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
                
            # تحديث محرك SMC
            current_price = float(df['close'].iloc[-1]) if len(df) > 0 else px
            smc_detector.detect_pivot_points(df)
            smc_detector.detect_order_blocks(df)
            smc_detector.detect_fair_value_gaps(df)
            
            info = rf_signal_live(df)
            ind = compute_indicators(df)
            
            # مجلس الإدارة المحسن مع SMC
            council_data = super_council_ai_enhanced_with_smc(df)
            master_decision = council_data.get("master_decision", {})
            
            # لوج قرار المجلس
            if master_decision.get("allow", False):
                log_i(f"🏛 MASTER DECISION: {master_decision['side'].upper()} | {master_decision['mode']} | "
                      f"Score: {master_decision['score']} | "
                      f"TierA: {len(master_decision['tierA'])} | "
                      f"TierB: {len(master_decision['tierB'])} | "
                      f"TierC: {len(master_decision['tierC'])}")
            
            # تنفيذ الدخول بناءً على قرار المجلس
            if (master_decision.get("allow", False) and 
                not STATE["open"] and 
                px is not None):
                
                side = master_decision["side"]
                mode = master_decision["mode"]
                
                # تحديد حجم الصفقة حسب النوع
                if mode == "trend":
                    size_multiplier = 1.0  # 100% حجم
                elif mode == "studied_scalp":
                    size_multiplier = 0.75  # 75% حجم
                else:  # cautious_scalp
                    size_multiplier = 0.5  # 50% حجم
                
                base_qty = compute_size(bal, px)
                final_qty = safe_qty(base_qty * size_multiplier)
                
                if final_qty > 0:
                    ok = open_market_enhanced(side, final_qty, px)
                    if ok:
                        STATE["mode"] = mode
                        STATE["entry_reasons"] = {
                            "tierA": master_decision['tierA'],
                            "tierB": master_decision['tierB'],
                            "tierC": master_decision['tierC'],
                            "total_score": master_decision['score']
                        }
                        log_g(f"🎯 SMART ENTRY: {side.upper()} | {mode} | Size: {size_multiplier*100}%")
            
            # إدارة الصفقة المفتوحة مع SMC
            if STATE["open"]:
                manage_position_enhanced_with_smc(df, ind, {
                    "price": px or info["price"],
                    "smc_analysis": smc_detector.get_smc_analysis(df, current_price),
                    **info
                })
            
            loop_i += 1
            sleep_s = NEAR_CLOSE_S if time_to_candle_close(df) <= 10 else BASE_SLEEP
            time.sleep(sleep_s)
            
        except Exception as e:
            log_e(f"SMC Enhanced loop error: {e}")
            time.sleep(BASE_SLEEP)

# =================== ADVANCED DASHBOARD API ===================

@app.route("/smc_dashboard")
def smc_dashboard():
    """لوحة تحكم متقدمة لعرض قرارات SMC والمجلس"""
    df = fetch_ohlcv(limit=100)
    current_price = price_now()
    
    if df.empty or current_price is None:
        return jsonify({"error": "No data available"})
    
    # جمع كل البيانات
    smc_analysis = smc_detector.get_smc_analysis(df, current_price)
    council_data = super_council_ai_enhanced_with_smc(df)
    master_decision = council_data.get("master_decision", {})
    price_structure = analyze_price_structure(df)
    vwap_analysis = calculate_vwap(df)
    gz = golden_zone_check(df, compute_indicators(df))
    
    return jsonify({
        "timestamp": datetime.utcnow().isoformat(),
        "price": current_price,
        "master_decision": master_decision,
        "council_summary": {
            "buy_votes": council_data.get("b", 0),
            "sell_votes": council_data.get("s", 0),
            "buy_score": council_data.get("score_b", 0),
            "sell_score": council_data.get("score_s", 0),
            "confidence": council_data.get("confidence", 0)
        },
        "smc_analysis": {
            "pivot_highs": len(smc_analysis.get('pivot_highs', [])),
            "pivot_lows": len(smc_analysis.get('pivot_lows', [])),
            "order_blocks": len(smc_analysis.get('order_blocks', [])),
            "fair_value_gaps": len(smc_analysis.get('fair_value_gaps', [])),
            "liquidity_zones": smc_analysis.get('liquidity_pools', {})
        },
        "market_structure": price_structure,
        "vwap_analysis": vwap_analysis,
        "golden_zone": gz,
        "current_position": {
            "open": STATE["open"],
            "side": STATE["side"],
            "mode": STATE.get("mode", "none"),
            "entry_reasons": STATE.get("entry_reasons", {})
        }
    })

# =================== INITIALIZATION ===================

# تهيئة محرك SMC العالمي
smc_detector = SmartMoneyConcepts()

# استبدال الدورة الرئيسية
trade_loop = trade_loop_enhanced_with_smc

# تحديث التحقق من البيئة
def verify_execution_environment():
    print(f"⚙️ EXECUTION ENVIRONMENT", flush=True)
    print(f"🔧 EXCHANGE: {EXCHANGE_NAME.upper()} | SYMBOL: {SYMBOL}", flush=True)
    print(f"🔧 EXECUTE_ORDERS: {EXECUTE_ORDERS} | DRY_RUN: {DRY_RUN}", flush=True)
    print(f"🎯 SMART MONEY CONCEPTS: Full SMC Integration Activated", flush=True)
    print(f"🏛 MASTER ENTRY ENGINE: Tier-Based Decision Making", flush=True)
    print(f"📊 ADVANCED ANALYSIS: Price Structure + VWAP + Liquidity Zones", flush=True)
    print(f"🧠 ENHANCED COUNCIL: SMC + Golden Zones + Volume + Flow", flush=True)

# تشغيل البوت المحدث
if __name__ == "__main__":
    verify_execution_environment()
    
    import threading
    threading.Thread(target=keepalive_loop, daemon=True).start()
    threading.Thread(target=trade_loop, daemon=True).start()
    
    log_i(f"🚀 SUI ULTRA PRO AI BOT STARTED - {BOT_VERSION}")
    log_i(f"🎯 SYMBOL: {SYMBOL} | INTERVAL: {INTERVAL} | LEVERAGE: {LEVERAGE}x")
    log_i(f"💡 SMART MONEY CONCEPTS FULLY INTEGRATED: SMC + Master Entry Engine + Advanced Dashboard")
    
    app.run(host="0.0.0.0", port=PORT, debug=False)
