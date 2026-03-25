"""
A+ Market Phase & Supply-Demand Trading System
Professional TradingView-style implementation with real-time data
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
import time
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

ASSETS = {
    'XAUUSD (Gold)': 'GC=F',
    'WTI (Oil)': 'CL=F',
    'EURUSD': 'EURUSD=X',
    'S&P500': '^GSPC',
    'DAX30': '^GDAXI'
}

TIMEFRAMES = {
    '1 Minute': '1m',
    '5 Minutes': '5m',
    '15 Minutes': '15m',
    '30 Minutes': '30m',
    '1 Hour': '60m',
    '4 Hours': '240m',
    '1 Day': '1d'
}

# TradingView-style color schemes
COLORS = {
    'bullish': '#00ff9d',
    'bearish': '#ff4d4d',
    'neutral': '#ffaa00',
    'supply': '#ff3366',
    'demand': '#33ff66',
    'background': '#131722',
    'grid': '#2a2e39',
    'text': '#d1d4dc'
}

# ============================================================================
# REAL-TIME DATA MANAGER
# ============================================================================

class RealTimeDataManager:
    """Manages real-time price updates"""
    
    def __init__(self):
        self.cache = {}
        self.last_update = {}
        
    def get_realtime_price(self, symbol: str) -> Dict:
        """Get real-time price for symbol"""
        try:
            ticker = yf.Ticker(ASSETS.get(symbol, symbol))
            data = ticker.history(period='1d', interval='1m')
            
            if not data.empty:
                latest = data.iloc[-1]
                current_price = latest['Close']
                
                # Get bid/ask (approximated)
                bid = current_price * 0.9999
                ask = current_price * 1.0001
                
                # Calculate daily change
                daily_open = data.iloc[0]['Open']
                daily_change = ((current_price - daily_open) / daily_open) * 100
                
                return {
                    'price': current_price,
                    'bid': bid,
                    'ask': ask,
                    'change': daily_change,
                    'high': data['High'].max(),
                    'low': data['Low'].min(),
                    'volume': latest['Volume'],
                    'timestamp': datetime.now()
                }
        except Exception as e:
            print(f"Error fetching real-time price: {e}")
            
        return None
        
    def get_historical_data(self, symbol: str, interval: str, period: str = '1mo') -> pd.DataFrame:
        """Get historical data with caching"""
        cache_key = f"{symbol}_{interval}_{period}"
        
        # Check cache (5 seconds TTL for real-time, longer for historical)
        if cache_key in self.cache:
            cache_time = self.cache[cache_key].get('timestamp', datetime.min)
            if (datetime.now() - cache_time).seconds < 5:
                return self.cache[cache_key]['data']
        
        try:
            ticker = ASSETS.get(symbol, symbol)
            df = yf.download(ticker, period=period, interval=interval, progress=False)
            
            if not df.empty:
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                
                # Add additional indicators
                df = self._add_indicators(df)
                
                self.cache[cache_key] = {
                    'data': df,
                    'timestamp': datetime.now()
                }
                
                return df
                
        except Exception as e:
            print(f"Error fetching data: {e}")
            
        return pd.DataFrame()
    
    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators"""
        # Moving Averages
        df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # ATR
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift())
        low_close = abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        
        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        
        # Fibonacci Levels (last swing)
        if len(df) > 50:
            swing_high = df['High'].iloc[-50:].max()
            swing_low = df['Low'].iloc[-50:].min()
            diff = swing_high - swing_low
            df['Fib_0.236'] = swing_high - diff * 0.236
            df['Fib_0.382'] = swing_high - diff * 0.382
            df['Fib_0.5'] = swing_high - diff * 0.5
            df['Fib_0.618'] = swing_high - diff * 0.618
            df['Fib_0.786'] = swing_high - diff * 0.786
        
        return df

# ============================================================================
# MARKET PHASE DETECTOR
# ============================================================================

def detect_market_phase(df: pd.DataFrame) -> Dict:
    """Detect market phase based on your playbook definitions"""
    if len(df) < 50:
        return {'phase': 'Unknown', 'confidence': 0, 'location': 'mid', 'liquidity_swept': False, 'structure_broken': False, 'price_position': 0.5, 'momentum': 0, 'volatility': 0}
    
    # Get recent data
    recent = df.tail(50)
    current_price = df['Close'].iloc[-1]
    
    # Calculate price range and position
    range_high = recent['High'].max()
    range_low = recent['Low'].min()
    price_range = range_high - range_low
    
    if price_range > 0:
        price_position = (current_price - range_low) / price_range
    else:
        price_position = 0.5
    
    # Location classification
    if price_position < 0.3:
        location = 'discount'
    elif price_position > 0.7:
        location = 'premium'
    else:
        location = 'mid'
    
    # Calculate momentum
    momentum = (df['Close'].iloc[-1] - df['Close'].iloc[-20]) / df['Close'].iloc[-20] * 100
    
    # Calculate volatility
    volatility = df['Returns'].std() * 100 if 'Returns' in df.columns else 1
    
    # Phase detection logic
    phase = 'Consolidation'
    confidence = 0.5
    
    # Accumulation detection
    if location == 'discount' and abs(momentum) < 2 and volatility < 1.5:
        phase = 'Accumulation'
        confidence = 0.7 + (1 - price_position) * 0.3
    # Distribution detection
    elif location == 'premium' and abs(momentum) < 2 and volatility < 1.5:
        phase = 'Distribution'
        confidence = 0.7 + (price_position - 0.7) * 0.3
    # Markup detection
    elif momentum > 3 and volatility > 1:
        phase = 'Markup'
        confidence = min(0.9, 0.6 + momentum / 20)
    # Markdown detection
    elif momentum < -3 and volatility > 1:
        phase = 'Markdown'
        confidence = min(0.9, 0.6 + abs(momentum) / 20)
    
    # Liquidity sweep detection
    liquidity_swept = False
    if len(df) > 10:
        recent_highs = df['High'].iloc[-10:-1].max()
        if df['High'].iloc[-1] > recent_highs:
            if df['Close'].iloc[-1] < df['Close'].iloc[-2]:
                liquidity_swept = True
        recent_lows = df['Low'].iloc[-10:-1].min()
        if df['Low'].iloc[-1] < recent_lows:
            if df['Close'].iloc[-1] > df['Close'].iloc[-2]:
                liquidity_swept = True
    
    # Structure break detection
    structure_broken = False
    if len(df) > 20:
        if df['High'].iloc[-1] > df['High'].iloc[-10:-1].max() and df['Close'].iloc[-1] > df['EMA_20'].iloc[-1]:
            structure_broken = True
        elif df['Low'].iloc[-1] < df['Low'].iloc[-10:-1].min() and df['Close'].iloc[-1] < df['EMA_20'].iloc[-1]:
            structure_broken = True
    
    return {
        'phase': phase,
        'confidence': confidence,
        'location': location,
        'liquidity_swept': liquidity_swept,
        'structure_broken': structure_broken,
        'price_position': price_position,
        'momentum': momentum,
        'volatility': volatility
    }

# ============================================================================
# ENHANCED SUPPLY & DEMAND ZONES
# ============================================================================

def find_enhanced_zones(df: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
    """Enhanced supply/demand zone detection with volume confirmation"""
    supply_zones = []
    demand_zones = []
    
    if len(df) < 30:
        return supply_zones, demand_zones
    
    # Detect swing points
    swing_highs = []
    swing_lows = []
    
    for i in range(5, len(df) - 5):
        # Swing high
        if df['High'].iloc[i] == df['High'].iloc[i-5:i+5].max():
            swing_highs.append({
                'price': df['High'].iloc[i],
                'index': i,
                'volume': df['Volume'].iloc[i],
                'time': df.index[i]
            })
        
        # Swing low
        if df['Low'].iloc[i] == df['Low'].iloc[i-5:i+5].min():
            swing_lows.append({
                'price': df['Low'].iloc[i],
                'index': i,
                'volume': df['Volume'].iloc[i],
                'time': df.index[i]
            })
    
    # Identify demand zones
    for low in swing_lows:
        if low['volume'] > df['Volume'].iloc[max(0, low['index']-10):low['index']].mean() * 1.5:
            zone = {
                'price': low['price'],
                'lower': low['price'] * 0.998,
                'upper': low['price'] * 1.002,
                'strength': min(1.0, low['volume'] / df['Volume'].mean()),
                'time': low['time'],
                'touches': 1
            }
            
            # Check if price has returned to this zone
            recent_prices = df['Low'].iloc[-20:]
            if any(abs(p - low['price']) / low['price'] < 0.005 for p in recent_prices):
                zone['strength'] *= 1.2
                
            demand_zones.append(zone)
    
    # Identify supply zones
    for high in swing_highs:
        if high['volume'] > df['Volume'].iloc[max(0, high['index']-10):high['index']].mean() * 1.5:
            zone = {
                'price': high['price'],
                'lower': high['price'] * 0.998,
                'upper': high['price'] * 1.002,
                'strength': min(1.0, high['volume'] / df['Volume'].mean()),
                'time': high['time'],
                'touches': 1
            }
            
            # Check if price has returned to this zone
            recent_prices = df['High'].iloc[-20:]
            if any(abs(p - high['price']) / high['price'] < 0.005 for p in recent_prices):
                zone['strength'] *= 1.2
                
            supply_zones.append(zone)
    
    # Sort by strength
    supply_zones.sort(key=lambda x: x['strength'], reverse=True)
    demand_zones.sort(key=lambda x: x['strength'], reverse=True)
    
    return supply_zones[:5], demand_zones[:5]

# ============================================================================
# MARKET ANALYZER
# ============================================================================

class MarketAnalyzer:
    """Advanced market analysis with institutional concepts"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def detect_order_flow(self) -> Dict:
        """Detect order flow and institutional activity"""
        if len(self.df) < 20:
            return {}
            
        recent = self.df.tail(20)
        
        # Detect absorption
        absorption = False
        if len(recent) > 5:
            price_range = recent['High'].max() - recent['Low'].min()
            avg_volume = recent['Volume'].mean()
            volume_spike = recent['Volume'].iloc[-1] > avg_volume * 1.5
            
            if volume_spike and price_range / recent['Close'].iloc[-1] < 0.005:
                absorption = True
        
        # Detect exhaustion
        exhaustion = False
        if len(recent) > 10:
            price_move = abs(recent['Close'].iloc[-1] - recent['Close'].iloc[-10]) / recent['Close'].iloc[-10]
            volume_trend = recent['Volume'].iloc[-5:].mean() / recent['Volume'].iloc[-10:-5].mean()
            
            if price_move > 0.02 and volume_trend < 0.8:
                exhaustion = True
        
        # Detect accumulation/distribution
        ad_line = (self.df['Close'] - self.df['Low'] - (self.df['High'] - self.df['Close'])) / (self.df['High'] - self.df['Low']) * self.df['Volume']
        ad_line = ad_line.replace([np.inf, -np.inf], 0).fillna(0)
        ad_trend = ad_line.iloc[-20:].mean() - ad_line.iloc[-40:-20].mean() if len(ad_line) > 40 else 0
        
        accumulation = ad_trend > 0 and self.df['Close'].iloc[-1] < self.df['Close'].iloc[-20] if len(self.df) > 20 else False
        distribution = ad_trend < 0 and self.df['Close'].iloc[-1] > self.df['Close'].iloc[-20] if len(self.df) > 20 else False
        
        return {
            'absorption': absorption,
            'exhaustion': exhaustion,
            'accumulation': accumulation,
            'distribution': distribution,
            'ad_line': ad_line.iloc[-1] if len(ad_line) > 0 else 0
        }
    
    def detect_imbalances(self) -> List[Dict]:
        """Detect market imbalances and inefficiencies"""
        imbalances = []
        
        for i in range(1, len(self.df) - 1):
            candle = self.df.iloc[i]
            prev_candle = self.df.iloc[i-1]
            
            # Bullish imbalance
            if candle['Low'] > prev_candle['High'] and candle['Volume'] > self.df['Volume'].iloc[i-5:i].mean() * 1.2:
                imbalances.append({
                    'type': 'bullish',
                    'price': prev_candle['High'],
                    'gap': candle['Low'] - prev_candle['High'],
                    'time': candle.name
                })
            
            # Bearish imbalance
            elif candle['High'] < prev_candle['Low'] and candle['Volume'] > self.df['Volume'].iloc[i-5:i].mean() * 1.2:
                imbalances.append({
                    'type': 'bearish',
                    'price': prev_candle['Low'],
                    'gap': prev_candle['Low'] - candle['High'],
                    'time': candle.name
                })
        
        return imbalances[-10:]

# ============================================================================
# ENHANCED SETUP GENERATOR
# ============================================================================

def generate_enhanced_setups(df: pd.DataFrame, asset: str, phase_data: Dict) -> List[Dict]:
    """Enhanced trade setup generation with institutional concepts"""
    setups = []
    
    # Hard filters from playbook
    if phase_data['confidence'] < 0.6:
        return setups
    
    if not phase_data['liquidity_swept']:
        return setups
    
    if not phase_data['structure_broken']:
        return setups
    
    current_price = df['Close'].iloc[-1]
    atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else current_price * 0.01
    
    # Get supply/demand zones
    supply_zones, demand_zones = find_enhanced_zones(df)
    
    # Get order flow analysis
    analyzer = MarketAnalyzer(df)
    order_flow = analyzer.detect_order_flow()
    
    # LONG SETUP
    if phase_data['location'] == 'discount' and phase_data['phase'] in ['Accumulation', 'Markup']:
        if demand_zones:
            best_demand = demand_zones[0]
            
            # Check if price is near demand zone
            distance_to_zone = abs(current_price - best_demand['price']) / current_price
            
            if distance_to_zone < 0.02:
                entry = current_price
                stop = best_demand['lower'] * 0.995
                
                recent_high = df['High'].iloc[-20:].max()
                target1 = recent_high
                target2 = recent_high + atr
                
                risk = entry - stop
                reward1 = target1 - entry
                
                if risk > 0:
                    rr_ratio = reward1 / risk
                    
                    if rr_ratio >= 2:
                        confidence = phase_data['confidence']
                        
                        if best_demand['strength'] * 0.1:
                            confidence += 0.1
                        if order_flow.get('absorption'):
                            confidence += 0.1
                        if order_flow.get('accumulation'):
                            confidence += 0.1
                        if 'RSI' in df.columns and df['RSI'].iloc[-1] < 40:
                            confidence += 0.05
                        
                        setups.append({
                            'direction': 'LONG',
                            'entry': entry,
                            'stop_loss': stop,
                            'take_profit': target1,
                            'take_profit_2': target2,
                            'risk_reward': rr_ratio,
                            'confidence': min(0.95, confidence),
                            'zone_price': best_demand['price'],
                            'zone_strength': best_demand['strength'],
                            'order_flow': order_flow
                        })
    
    # SHORT SETUP
    elif phase_data['location'] == 'premium' and phase_data['phase'] in ['Distribution', 'Markdown']:
        if supply_zones:
            best_supply = supply_zones[0]
            
            # Check if price is near supply zone
            distance_to_zone = abs(current_price - best_supply['price']) / current_price
            
            if distance_to_zone < 0.02:
                entry = current_price
                stop = best_supply['upper'] * 1.005
                
                recent_low = df['Low'].iloc[-20:].min()
                target1 = recent_low
                target2 = recent_low - atr
                
                risk = stop - entry
                reward1 = entry - target1
                
                if risk > 0:
                    rr_ratio = reward1 / risk
                    
                    if rr_ratio >= 2:
                        confidence = phase_data['confidence']
                        
                        if best_supply['strength'] * 0.1:
                            confidence += 0.1
                        if order_flow.get('exhaustion'):
                            confidence += 0.1
                        if order_flow.get('distribution'):
                            confidence += 0.1
                        if 'RSI' in df.columns and df['RSI'].iloc[-1] > 60:
                            confidence += 0.05
                        
                        setups.append({
                            'direction': 'SHORT',
                            'entry': entry,
                            'stop_loss': stop,
                            'take_profit': target1,
                            'take_profit_2': target2,
                            'risk_reward': rr_ratio,
                            'confidence': min(0.95, confidence),
                            'zone_price': best_supply['price'],
                            'zone_strength': best_supply['strength'],
                            'order_flow': order_flow
                        })
    
    return setups

# ============================================================================
# TELEGRAM ALERTS
# ============================================================================

def send_telegram_alert(bot_token: str, chat_id: str, setup: Dict, asset: str) -> bool:
    """Send trade alert via Telegram"""
    if not bot_token or not chat_id or bot_token == 'YOUR_BOT_TOKEN':
        return False
    
    emoji = "🟢" if setup['direction'] == 'LONG' else "🔴"
    
    message = f"""
{emoji} <b>A+ TRADE SETUP ALERT</b>
━━━━━━━━━━━━━━━━━━━

<b>Asset:</b> {asset}
<b>Direction:</b> {setup['direction']}
<b>Entry:</b> ${setup['entry']:.2f}
<b>Stop Loss:</b> ${setup['stop_loss']:.2f}
<b>Take Profit:</b> ${setup['take_profit']:.2f}

<b>Risk/Reward:</b> 1:{setup['risk_reward']:.1f}
<b>Confidence:</b> {setup['confidence']:.0%}

━━━━━━━━━━━━━━━━━━━
<i>"Discipline over emotion. Wait for the market to invite you."</i>
"""
    
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            'chat_id': chat_id,
            'text': message,
            'parse_mode': 'HTML'
        }
        response = requests.post(url, json=payload, timeout=10)
        return response.status_code == 200
    except:
        return False

# ============================================================================
# TRADINGVIEW-STYLE CHART
# ============================================================================

class TradingViewChart:
    """Create TradingView-style professional charts"""
    
    def __init__(self, df: pd.DataFrame, asset: str):
        self.df = df
        self.asset = asset
        
    def create_candlestick_chart(self, supply_zones=None, demand_zones=None, 
                                  setups=None, show_indicators=True) -> go.Figure:
        """Create main candlestick chart with indicators"""
        
        # Calculate row heights based on indicators
        if show_indicators:
            rows = 4
            row_heights = [0.5, 0.2, 0.15, 0.15]
        else:
            rows = 1
            row_heights = [1]
        
        fig = make_subplots(
            rows=rows, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=row_heights,
            subplot_titles=('Price Action', 'Volume', 'RSI', 'MACD')
        )
        
        # MAIN PRICE CHART
        fig.add_trace(
            go.Candlestick(
                x=self.df.index,
                open=self.df['Open'],
                high=self.df['High'],
                low=self.df['Low'],
                close=self.df['Close'],
                name='Price',
                increasing=dict(line=dict(color='#00ff9d', width=1), fillcolor='#00ff9d'),
                decreasing=dict(line=dict(color='#ff4d4d', width=1), fillcolor='#ff4d4d'),
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Moving Averages
        fig.add_trace(
            go.Scatter(
                x=self.df.index,
                y=self.df['EMA_9'],
                name='EMA 9',
                line=dict(color='#ffaa00', width=1.5),
                opacity=0.8
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=self.df.index,
                y=self.df['EMA_20'],
                name='EMA 20',
                line=dict(color='#00ff9d', width=1.5),
                opacity=0.8
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=self.df.index,
                y=self.df['EMA_50'],
                name='EMA 50',
                line=dict(color='#ffaa66', width=1.5),
                opacity=0.8
            ),
            row=1, col=1
        )
        
        if 'SMA_200' in self.df.columns and not self.df['SMA_200'].isna().all():
            fig.add_trace(
                go.Scatter(
                    x=self.df.index,
                    y=self.df['SMA_200'],
                    name='SMA 200',
                    line=dict(color='#aa66ff', width=2, dash='dash'),
                    opacity=0.7
                ),
                row=1, col=1
            )
        
        # Bollinger Bands
        if 'BB_Upper' in self.df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.df.index,
                    y=self.df['BB_Upper'],
                    name='BB Upper',
                    line=dict(color='#888888', width=1, dash='dot'),
                    opacity=0.5
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=self.df.index,
                    y=self.df['BB_Lower'],
                    name='BB Lower',
                    line=dict(color='#888888', width=1, dash='dot'),
                    opacity=0.5,
                    fill='tonexty',
                    fillcolor='rgba(136, 136, 136, 0.1)'
                ),
                row=1, col=1
            )
        
        # Fibonacci Levels
        if 'Fib_0.236' in self.df.columns:
            for level, color in [('Fib_0.236', '#888888'), ('Fib_0.382', '#888888'), 
                                  ('Fib_0.5', '#ffaa00'), ('Fib_0.618', '#888888'), 
                                  ('Fib_0.786', '#888888')]:
                if level in self.df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=self.df.index,
                            y=self.df[level],
                            name=level.replace('_', ' '),
                            line=dict(color=color, width=1, dash='dash'),
                            opacity=0.5,
                            visible='legendonly'
                        ),
                        row=1, col=1
                    )
        
        # Supply Zones
        if supply_zones:
            for zone in supply_zones[:5]:
                fig.add_hrect(
                    y0=zone['lower'],
                    y1=zone['upper'],
                    fillcolor='#ff3366',
                    opacity=0.15,
                    line_width=1,
                    line_color='#ff3366',
                    line_dash='dash',
                    name=f"Supply Zone (Strength: {zone['strength']:.0%})",
                    row=1, col=1
                )
                
                fig.add_annotation(
                    x=zone['time'],
                    y=zone['upper'],
                    text=f"🔴 SUPPLY",
                    showarrow=True,
                    arrowhead=1,
                    arrowcolor='#ff3366',
                    font=dict(size=10, color='#ff3366'),
                    bgcolor='rgba(0,0,0,0.5)',
                    row=1, col=1
                )
        
        # Demand Zones
        if demand_zones:
            for zone in demand_zones[:5]:
                fig.add_hrect(
                    y0=zone['lower'],
                    y1=zone['upper'],
                    fillcolor='#33ff66',
                    opacity=0.15,
                    line_width=1,
                    line_color='#33ff66',
                    line_dash='dash',
                    name=f"Demand Zone (Strength: {zone['strength']:.0%})",
                    row=1, col=1
                )
                
                fig.add_annotation(
                    x=zone['time'],
                    y=zone['lower'],
                    text=f"🟢 DEMAND",
                    showarrow=True,
                    arrowhead=1,
                    arrowcolor='#33ff66',
                    font=dict(size=10, color='#33ff66'),
                    bgcolor='rgba(0,0,0,0.5)',
                    row=1, col=1
                )
        
        # Trade Setups
        if setups:
            for setup in setups:
                arrow_color = '#33ff66' if setup['direction'] == 'LONG' else '#ff3366'
                
                fig.add_trace(
                    go.Scatter(
                        x=[self.df.index[-1]],
                        y=[setup['entry']],
                        mode='markers+text',
                        marker=dict(
                            symbol='triangle-up' if setup['direction'] == 'LONG' else 'triangle-down',
                            size=15,
                            color=arrow_color,
                            line=dict(color='white', width=1)
                        ),
                        text=[f"{setup['direction']} @ {setup['entry']:.2f}"],
                        textposition='top center',
                        textfont=dict(size=12, color=arrow_color),
                        name=f"{setup['direction']} Setup",
                        showlegend=True
                    ),
                    row=1, col=1
                )
                
                fig.add_hline(
                    y=setup['entry'],
                    line_dash="dash",
                    line_color=arrow_color,
                    opacity=0.7,
                    row=1, col=1
                )
                
                fig.add_hline(
                    y=setup['stop_loss'],
                    line_dash="dash",
                    line_color='#ff4d4d',
                    opacity=0.7,
                    row=1, col=1
                )
                
                fig.add_hline(
                    y=setup['take_profit'],
                    line_dash="dash",
                    line_color='#00ff9d',
                    opacity=0.7,
                    row=1, col=1
                )
        
        # VOLUME CHART
        volume_colors = ['#33ff66' if self.df['Close'].iloc[i] >= self.df['Open'].iloc[i] 
                         else '#ff3366' for i in range(len(self.df))]
        
        fig.add_trace(
            go.Bar(
                x=self.df.index,
                y=self.df['Volume'],
                name='Volume',
                marker_color=volume_colors,
                opacity=0.7,
                showlegend=True
            ),
            row=2, col=1
        )
        
        if 'Volume_MA' in self.df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.df.index,
                    y=self.df['Volume_MA'],
                    name='Volume MA',
                    line=dict(color='#ffaa00', width=1.5, dash='dash'),
                    opacity=0.8
                ),
                row=2, col=1
            )
        
        # RSI CHART
        fig.add_trace(
            go.Scatter(
                x=self.df.index,
                y=self.df['RSI'],
                name='RSI',
                line=dict(color='#aa66ff', width=2),
                fill='tozeroy',
                fillcolor='rgba(170, 102, 255, 0.1)'
            ),
            row=3, col=1
        )
        
        fig.add_hline(y=70, line_dash="dash", line_color="#ff4d4d", opacity=0.5, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#33ff66", opacity=0.5, row=3, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="#888888", opacity=0.5, row=3, col=1)
        
        # MACD CHART
        fig.add_trace(
            go.Scatter(
                x=self.df.index,
                y=self.df['MACD'],
                name='MACD',
                line=dict(color='#00ff9d', width=1.5)
            ),
            row=4, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=self.df.index,
                y=self.df['MACD_Signal'],
                name='Signal',
                line=dict(color='#ffaa00', width=1.5)
            ),
            row=4, col=1
        )
        
        colors_macd = ['#33ff66' if val >= 0 else '#ff3366' for val in self.df['MACD_Histogram']]
        fig.add_trace(
            go.Bar(
                x=self.df.index,
                y=self.df['MACD_Histogram'],
                name='Histogram',
                marker_color=colors_macd,
                opacity=0.7
            ),
            row=4, col=1
        )
        
        # Chart styling
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='#131722',
            plot_bgcolor='#131722',
            font=dict(color='#d1d4dc', size=12),
            title={
                'text': f'{self.asset} - TradingView Style Chart',
                'font': {'size': 20, 'color': '#d1d4dc'},
                'x': 0.5,
                'xanchor': 'center'
            },
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(0,0,0,0.7)',
                font=dict(size=10)
            ),
            xaxis_rangeslider_visible=False,
            dragmode='zoom',
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        # Update axes
        for i in range(1, rows + 1):
            fig.update_xaxes(
                gridcolor='#2a2e39',
                showgrid=True,
                gridwidth=0.5,
                zerolinecolor='#2a2e39',
                row=i, col=1
            )
            fig.update_yaxes(
                gridcolor='#2a2e39',
                showgrid=True,
                gridwidth=0.5,
                zerolinecolor='#2a2e39',
                row=i, col=1
            )
        
        return fig

# ============================================================================
# REAL-TIME PRICE DISPLAY
# ============================================================================

def display_realtime_prices(data_manager: RealTimeDataManager, asset: str):
    """Display real-time price information"""
    
    realtime = data_manager.get_realtime_price(asset)
    
    if realtime:
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            price_color = '#00ff9d' if realtime['change'] >= 0 else '#ff4d4d'
            st.markdown(f"""
            <div style='background: rgba(255,255,255,0.05); padding: 10px; border-radius: 5px;'>
                <small style='color: #888'>Current Price</small><br>
                <span style='font-size: 24px; font-weight: bold; color: {price_color}'>${realtime['price']:.2f}</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style='background: rgba(255,255,255,0.05); padding: 10px; border-radius: 5px;'>
                <small style='color: #888'>24h Change</small><br>
                <span style='font-size: 18px; color: {price_color}'>{realtime['change']:+.2f}%</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style='background: rgba(255,255,255,0.05); padding: 10px; border-radius: 5px;'>
                <small style='color: #888'>Bid/Ask</small><br>
                <span style='font-size: 14px'>${realtime['bid']:.2f} / ${realtime['ask']:.2f}</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div style='background: rgba(255,255,255,0.05); padding: 10px; border-radius: 5px;'>
                <small style='color: #888'>24h Range</small><br>
                <span style='font-size: 14px'>${realtime['low']:.2f} - ${realtime['high']:.2f}</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            st.markdown(f"""
            <div style='background: rgba(255,255,255,0.05); padding: 10px; border-radius: 5px;'>
                <small style='color: #888'>Volume</small><br>
                <span style='font-size: 14px'>{realtime['volume']:,.0f}</span>
            </div>
            """, unsafe_allow_html=True)
        
        return realtime
    return None

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.set_page_config(
        page_title="A+ Trading System - Professional Edition",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .stApp {
        background-color: #131722;
    }
    .main-header {
        background: linear-gradient(135deg, #1e2a3a 0%, #0f172a 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="color: #00ff9d; margin: 0;">🎯 A+ Trading System</h1>
        <p style="color: #d1d4dc; margin: 0;">Professional Market Phase & Supply-Demand Trading Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Trading Configuration")
        
        asset = st.selectbox("Select Asset", list(ASSETS.keys()))
        timeframe = st.selectbox("Timeframe", list(TIMEFRAMES.keys()))
        period = st.selectbox("Data Period", ['1d', '5d', '1mo', '3mo', '6mo', '1y'])
        
        st.markdown("---")
        
        # Chart settings
        st.markdown("### 📊 Chart Settings")
        show_indicators = st.checkbox("Show All Indicators", value=True)
        show_supply_demand = st.checkbox("Show Supply/Demand Zones", value=True)
        
        st.markdown("---")
        
        # Telegram settings
        st.markdown("### 📱 Telegram Alerts")
        use_telegram = st.checkbox("Enable Telegram Alerts")
        bot_token = st.text_input("Bot Token", type="password", value="YOUR_BOT_TOKEN")
        chat_id = st.text_input("Chat ID", value="YOUR_CHAT_ID")
        
        st.markdown("---")
        
        # Real-time settings
        st.markdown("### 🔄 Real-time Settings")
        auto_refresh = st.checkbox("Auto-refresh (5 seconds)", value=True)
        
        st.markdown("---")
        st.markdown("### 📖 Trading Rules")
        with st.expander("A+ Setup Requirements"):
            st.markdown("""
            ✅ **Hard Filters:**
            1. HTF context clear
            2. Correct location (discount/premium)
            3. Liquidity taken
            4. Structure break
            5. Fresh supply/demand zone
            
            🎯 **Risk Management:**
            - Minimum 1:2 risk-reward
            - Maximum 2% risk per trade
            """)
    
    # Initialize data manager
    data_manager = RealTimeDataManager()
    
    # Real-time price display
    st.markdown("### 💹 Real-time Market Data")
    display_realtime_prices(data_manager, asset)
    
    # Fetch historical data
    with st.spinner(f"Loading {asset} data..."):
        df = data_manager.get_historical_data(asset, TIMEFRAMES[timeframe], period)
    
    if df.empty:
        st.error("Failed to fetch data. Please check your internet connection.")
        return
    
    # Add returns column if not present
    if 'Returns' not in df.columns:
        df['Returns'] = df['Close'].pct_change()
    
    # Market analysis
    phase_data = detect_market_phase(df)
    supply_zones, demand_zones = find_enhanced_zones(df)
    setups = generate_enhanced_setups(df, asset, phase_data)
    
    # Order flow analysis
    analyzer = MarketAnalyzer(df)
    order_flow = analyzer.detect_order_flow()
    
    # Market metrics row
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("Market Phase", phase_data['phase'], 
                 delta=f"{phase_data['confidence']:.0%} confidence")
    
    with col2:
        st.metric("Location", phase_data['location'].upper(),
                 delta=f"{phase_data['price_position']:.0%} of range")
    
    with col3:
        st.metric("Liquidity Swept", "✅" if phase_data['liquidity_swept'] else "❌")
    
    with col4:
        st.metric("Structure Break", "✅" if phase_data['structure_broken'] else "❌")
    
    with col5:
        absorption_emoji = "✅" if order_flow.get('absorption') else "❌"
        st.metric("Absorption", absorption_emoji)
    
    with col6:
        exhaustion_emoji = "✅" if order_flow.get('exhaustion') else "❌"
        st.metric("Exhaustion", exhaustion_emoji)
    
    st.markdown("---")
    
    # Professional TradingView-style chart
    chart = TradingViewChart(df, asset)
    fig = chart.create_candlestick_chart(
        supply_zones if show_supply_demand else None,
        demand_zones if show_supply_demand else None,
        setups,
        show_indicators
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Trade setups section
    st.markdown("---")
    st.markdown("### 🎯 A+ Trade Setups")
    
    if setups:
        for i, setup in enumerate(setups):
            with st.container():
                cols = st.columns([1, 2, 2, 2, 1])
                
                direction_color = "#00ff9d" if setup['direction'] == 'LONG' else "#ff4d4d"
                
                with cols[0]:
                    st.markdown(f"""
                    <div style="background: {direction_color}20; padding: 10px; border-radius: 5px; text-align: center;">
                        <span style="font-size: 24px;">{'🟢' if setup['direction'] == 'LONG' else '🔴'}</span><br>
                        <span style="font-weight: bold; color: {direction_color};">{setup['direction']}</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                with cols[1]:
                    st.markdown(f"""
                    **Entry:** ${setup['entry']:.2f}<br>
                    **Stop:** ${setup['stop_loss']:.2f}<br>
                    **Risk:** ${abs(setup['entry'] - setup['stop_loss']):.2f}
                    """, unsafe_allow_html=True)
                
                with cols[2]:
                    st.markdown(f"""
                    **Target 1:** ${setup['take_profit']:.2f}<br>
                    **Target 2:** ${setup['take_profit_2']:.2f}<br>
                    **Reward:** ${abs(setup['take_profit'] - setup['entry']):.2f}
                    """, unsafe_allow_html=True)
                
                with cols[3]:
                    st.markdown(f"""
                    **R:R:** 1:{setup['risk_reward']:.1f}<br>
                    **Confidence:** {setup['confidence']:.0%}<br>
                    **Zone Strength:** {setup['zone_strength']:.0%}
                    """, unsafe_allow_html=True)
                
                with cols[4]:
                    if use_telegram and bot_token != 'YOUR_BOT_TOKEN':
                        if st.button(f"🚀 Execute", key=f"exec_{i}"):
                            if send_telegram_alert(bot_token, chat_id, setup, asset):
                                st.success("Alert sent to Telegram!")
                
                st.markdown("---")
    else:
        st.info("🔍 No A+ setups detected. Waiting for market to invite you...")
        
        # Show what's missing
        missing = []
        if phase_data['confidence'] < 0.6:
            missing.append("HTF context not clear")
        if not phase_data['liquidity_swept']:
            missing.append("No liquidity sweep")
        if not phase_data['structure_broken']:
            missing.append("No structure break")
        if phase_data['location'] not in ['discount', 'premium']:
            missing.append("Wrong location")
        
        if missing:
            st.warning(f"**Missing A+ Criteria:** {', '.join(missing)}")
    
    # Market insights
    st.markdown("---")
    st.markdown("### 📈 Market Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔄 Order Flow Analysis")
        if order_flow.get('absorption'):
            st.success("✅ **Absorption Detected** - Large players absorbing selling pressure")
        if order_flow.get('exhaustion'):
            st.warning("⚠️ **Exhaustion Detected** - Momentum may be running out")
        if order_flow.get('accumulation'):
            st.success("✅ **Accumulation Detected** - Institutional buying present")
        if order_flow.get('distribution'):
            st.warning("⚠️ **Distribution Detected** - Institutional selling present")
        
        if not any(order_flow.values()):
            st.info("No significant order flow signals detected")
    
    with col2:
        st.markdown("#### 📊 Key Levels")
        if supply_zones:
            st.write(f"**Strongest Supply:** ${supply_zones[0]['price']:.2f} (Strength: {supply_zones[0]['strength']:.0%})")
        if demand_zones:
            st.write(f"**Strongest Demand:** ${demand_zones[0]['price']:.2f} (Strength: {demand_zones[0]['strength']:.0%})")
        
        st.write(f"**Current Price:** ${df['Close'].iloc[-1]:.2f}")
        if 'ATR' in df.columns:
            st.write(f"**ATR (14):** ${df['ATR'].iloc[-1]:.2f}")
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(5)
        st.rerun()

if __name__ == "__main__":
    main()
