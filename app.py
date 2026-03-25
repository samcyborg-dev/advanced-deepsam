"""
Professional Trading Platform - TradingView Style
Real-time charts, multi-timeframe analysis, A+ trade setups
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
import threading
from collections import deque
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
    '1m': '1m',
    '5m': '5m',
    '15m': '15m',
    '30m': '30m',
    '1h': '60m',
    '4h': '240m',
    '1d': '1d'
}

# ============================================================================
# REAL-TIME DATA STREAM
# ============================================================================

class RealTimeDataStream:
    """Real-time data streaming with WebSocket-like updates"""
    
    def __init__(self):
        self.cache = {}
        self.price_history = deque(maxlen=1000)
        self.last_update = None
        self.update_interval = 1  # seconds
        
    def stream_data(self, symbol: str) -> pd.DataFrame:
        """Stream real-time data"""
        try:
            ticker = yf.Ticker(ASSETS.get(symbol, symbol))
            
            # Get latest 5 minutes of data for real-time
            data = ticker.history(period='5d', interval='1m')
            
            if not data.empty:
                data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                
                # Add SMA 50
                data['SMA_50'] = data['Close'].rolling(window=50).mean()
                
                # Calculate real-time metrics
                current_price = data['Close'].iloc[-1]
                prev_close = data['Close'].iloc[-2] if len(data) > 1 else current_price
                
                data['Change'] = data['Close'].pct_change() * 100
                data['Volume_Change'] = data['Volume'].pct_change() * 100
                
                return data
                
        except Exception as e:
            print(f"Stream error: {e}")
            
        return pd.DataFrame()
    
    def get_realtime_quote(self, symbol: str) -> Dict:
        """Get current quote"""
        try:
            ticker = yf.Ticker(ASSETS.get(symbol, symbol))
            data = ticker.history(period='1d', interval='1m')
            
            if not data.empty:
                latest = data.iloc[-1]
                daily_open = data.iloc[0]['Open']
                
                return {
                    'price': latest['Close'],
                    'bid': latest['Close'] * 0.9999,
                    'ask': latest['Close'] * 1.0001,
                    'change': ((latest['Close'] - daily_open) / daily_open) * 100,
                    'high': data['High'].max(),
                    'low': data['Low'].min(),
                    'volume': latest['Volume'],
                    'timestamp': datetime.now()
                }
        except:
            pass
            
        return None

# ============================================================================
# PROFESSIONAL CHART ENGINE
# ============================================================================

class ProfessionalChart:
    """TradingView-style professional chart engine"""
    
    def __init__(self, df: pd.DataFrame, asset: str):
        self.df = df
        self.asset = asset
        self.chart_type = 'candlestick'  # candlestick, line, area
        self.chart_theme = 'dark'
        
    def create_chart(self, show_sma: bool = True, 
                     show_volume: bool = True,
                     chart_type: str = 'candlestick',
                     zoom_level: int = 100) -> go.Figure:
        """Create professional chart with interactive controls"""
        
        # Create figure
        if show_volume:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.7, 0.3]
            )
        else:
            fig = make_subplots(rows=1, cols=1)
        
        # Chart styling based on theme
        bg_color = '#131722' if self.chart_theme == 'dark' else '#ffffff'
        grid_color = '#2a2e39' if self.chart_theme == 'dark' else '#e0e0e0'
        text_color = '#d1d4dc' if self.chart_theme == 'dark' else '#000000'
        
        # Price chart based on type
        if chart_type == 'candlestick':
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
                row=1, col=1 if show_volume else 1
            )
        elif chart_type == 'line':
            fig.add_trace(
                go.Scatter(
                    x=self.df.index,
                    y=self.df['Close'],
                    name='Price',
                    line=dict(color='#00ff9d', width=2),
                    fill='tozeroy' if chart_type == 'area' else None,
                    fillcolor='rgba(0, 255, 157, 0.1)'
                ),
                row=1, col=1 if show_volume else 1
            )
        
        # Add SMA 50
        if show_sma and 'SMA_50' in self.df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.df.index,
                    y=self.df['SMA_50'],
                    name='SMA 50',
                    line=dict(color='#ffaa00', width=2),
                    opacity=0.8
                ),
                row=1, col=1 if show_volume else 1
            )
        
        # Volume chart
        if show_volume:
            volume_colors = ['#33ff66' if self.df['Close'].iloc[i] >= self.df['Open'].iloc[i] 
                           else '#ff3366' for i in range(len(self.df))]
            
            fig.add_trace(
                go.Bar(
                    x=self.df.index,
                    y=self.df['Volume'],
                    name='Volume',
                    marker_color=volume_colors,
                    opacity=0.7
                ),
                row=2, col=1
            )
        
        # Update layout with professional styling
        fig.update_layout(
            template='plotly_dark' if self.chart_theme == 'dark' else 'plotly_white',
            paper_bgcolor=bg_color,
            plot_bgcolor=bg_color,
            font=dict(color=text_color, size=12),
            title={
                'text': f'{self.asset} - Interactive Chart',
                'font': {'size': 20, 'color': text_color},
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
            dragmode='zoom',
            margin=dict(l=50, r=50, t=80, b=50),
            xaxis=dict(
                rangeslider=dict(visible=False),
                type='date',
                gridcolor=grid_color,
                showgrid=True,
                gridwidth=0.5
            ),
            yaxis=dict(
                gridcolor=grid_color,
                showgrid=True,
                gridwidth=0.5,
                title_text='Price'
            )
        )
        
        if show_volume:
            fig.update_yaxes(title_text="Volume", row=2, col=1, gridcolor=grid_color)
        
        # Add interactive buttons
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    active=0,
                    x=0.01,
                    y=1.15,
                    buttons=list([
                        dict(label="Reset View",
                             method="relayout",
                             args=[{"xaxis.autorange": True, "yaxis.autorange": True}]),
                        dict(label="Zoom In",
                             method="relayout",
                             args=[{"xaxis.range": [self.df.index[-50], self.df.index[-1]]}]),
                        dict(label="Zoom Out",
                             method="relayout",
                             args=[{"xaxis.range": [self.df.index[-200], self.df.index[-1]]}])
                    ])
                )
            ]
        )
        
        return fig

# ============================================================================
# MARKET PHASE ANALYZER
# ============================================================================

def analyze_market_phase(df: pd.DataFrame) -> Dict:
    """Analyze market phase based on price action"""
    if len(df) < 100:
        return {'phase': 'Analyzing...', 'confidence': 0, 'trend': 'neutral'}
    
    # Calculate key metrics
    sma_50 = df['SMA_50'].iloc[-1]
    current_price = df['Close'].iloc[-1]
    price_above_sma = current_price > sma_50
    
    # Trend analysis
    price_20d_ago = df['Close'].iloc[-20]
    price_50d_ago = df['Close'].iloc[-50] if len(df) > 50 else price_20d_ago
    
    trend_strength = ((current_price - price_20d_ago) / price_20d_ago) * 100
    long_trend = ((current_price - price_50d_ago) / price_50d_ago) * 100
    
    # Volatility analysis
    atr = df['High'].iloc[-20:].max() - df['Low'].iloc[-20:].min()
    volatility = atr / current_price * 100
    
    # Phase detection
    if trend_strength > 2 and long_trend > 5:
        phase = 'Markup'
        confidence = min(0.9, 0.6 + trend_strength / 20)
    elif trend_strength < -2 and long_trend < -5:
        phase = 'Markdown'
        confidence = min(0.9, 0.6 + abs(trend_strength) / 20)
    elif abs(trend_strength) < 1 and volatility < 2:
        # Check location
        price_range = df['High'].iloc[-50:].max() - df['Low'].iloc[-50:].min()
        price_position = (current_price - df['Low'].iloc[-50:].min()) / price_range
        
        if price_position < 0.3:
            phase = 'Accumulation'
            confidence = 0.7 + (1 - price_position) * 0.3
        elif price_position > 0.7:
            phase = 'Distribution'
            confidence = 0.7 + (price_position - 0.7) * 0.3
        else:
            phase = 'Consolidation'
            confidence = 0.5
    else:
        phase = 'Transition'
        confidence = 0.4
    
    # Liquidity sweep detection
    liquidity_swept = False
    if len(df) > 20:
        recent_high = df['High'].iloc[-10:-1].max()
        recent_low = df['Low'].iloc[-10:-1].min()
        
        if df['High'].iloc[-1] > recent_high:
            if df['Close'].iloc[-1] < df['Close'].iloc[-2]:
                liquidity_swept = True
        elif df['Low'].iloc[-1] < recent_low:
            if df['Close'].iloc[-1] > df['Close'].iloc[-2]:
                liquidity_swept = True
    
    # Structure break detection
    structure_broken = False
    if len(df) > 30:
        if df['Close'].iloc[-1] > df['High'].iloc[-20:-1].max():
            structure_broken = True
        elif df['Close'].iloc[-1] < df['Low'].iloc[-20:-1].min():
            structure_broken = True
    
    # Trend direction
    if trend_strength > 1:
        trend = 'bullish'
    elif trend_strength < -1:
        trend = 'bearish'
    else:
        trend = 'neutral'
    
    return {
        'phase': phase,
        'confidence': confidence,
        'trend': trend,
        'trend_strength': trend_strength,
        'volatility': volatility,
        'liquidity_swept': liquidity_swept,
        'structure_broken': structure_broken,
        'price_above_sma': price_above_sma
    }

# ============================================================================
# SUPPLY/DEMAND DETECTOR
# ============================================================================

def find_supply_demand(df: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
    """Find key supply and demand zones"""
    supply = []
    demand = []
    
    if len(df) < 50:
        return supply, demand
    
    # Find swing points
    swings_high = []
    swings_low = []
    
    for i in range(5, len(df) - 5):
        # Swing high
        if df['High'].iloc[i] == df['High'].iloc[i-5:i+5].max():
            swings_high.append({
                'price': df['High'].iloc[i],
                'index': i,
                'volume': df['Volume'].iloc[i],
                'time': df.index[i]
            })
        
        # Swing low
        if df['Low'].iloc[i] == df['Low'].iloc[i-5:i+5].min():
            swings_low.append({
                'price': df['Low'].iloc[i],
                'index': i,
                'volume': df['Volume'].iloc[i],
                'time': df.index[i]
            })
    
    # Identify demand zones (support)
    for low in swings_low[-10:]:  # Last 10 swing lows
        if low['volume'] > df['Volume'].iloc[max(0, low['index']-10):low['index']].mean():
            zone = {
                'price': low['price'],
                'strength': min(1.0, low['volume'] / df['Volume'].mean()),
                'time': low['time'],
                'type': 'demand'
            }
            demand.append(zone)
    
    # Identify supply zones (resistance)
    for high in swings_high[-10:]:  # Last 10 swing highs
        if high['volume'] > df['Volume'].iloc[max(0, high['index']-10):high['index']].mean():
            zone = {
                'price': high['price'],
                'strength': min(1.0, high['volume'] / df['Volume'].mean()),
                'time': high['time'],
                'type': 'supply'
            }
            supply.append(zone)
    
    # Sort by strength
    supply.sort(key=lambda x: x['strength'], reverse=True)
    demand.sort(key=lambda x: x['strength'], reverse=True)
    
    return supply[:3], demand[:3]  # Return top 3 zones

# ============================================================================
# A+ TRADE SETUP GENERATOR
# ============================================================================

def generate_trade_setups(df: pd.DataFrame, asset: str, phase_data: Dict) -> List[Dict]:
    """Generate A+ trade setups based on the playbook"""
    setups = []
    
    # Hard filters - MUST pass all
    if phase_data['confidence'] < 0.6:
        return setups
    
    if not phase_data['liquidity_swept']:
        return setups
    
    if not phase_data['structure_broken']:
        return setups
    
    current_price = df['Close'].iloc[-1]
    supply_zones, demand_zones = find_supply_demand(df)
    
    # Calculate ATR for stop loss
    atr = (df['High'].iloc[-20:].max() - df['Low'].iloc[-20:].min()) / 2
    
    # LONG SETUP
    if phase_data['trend'] in ['bullish', 'neutral'] and phase_data['phase'] in ['Accumulation', 'Markup']:
        if demand_zones:
            best_demand = demand_zones[0]
            
            # Check if price is near demand zone
            distance = abs(current_price - best_demand['price']) / current_price
            
            if distance < 0.02:  # Within 2%
                entry = current_price
                stop = best_demand['price'] - atr * 0.5
                target = best_demand['price'] + atr * 2
                
                risk = entry - stop
                reward = target - entry
                
                if risk > 0 and reward / risk >= 2:
                    setups.append({
                        'direction': 'LONG',
                        'entry': entry,
                        'stop': stop,
                        'target': target,
                        'rr': reward / risk,
                        'confidence': phase_data['confidence'],
                        'zone': best_demand['price']
                    })
    
    # SHORT SETUP
    elif phase_data['trend'] in ['bearish', 'neutral'] and phase_data['phase'] in ['Distribution', 'Markdown']:
        if supply_zones:
            best_supply = supply_zones[0]
            
            # Check if price is near supply zone
            distance = abs(current_price - best_supply['price']) / current_price
            
            if distance < 0.02:  # Within 2%
                entry = current_price
                stop = best_supply['price'] + atr * 0.5
                target = best_supply['price'] - atr * 2
                
                risk = stop - entry
                reward = entry - target
                
                if risk > 0 and reward / risk >= 2:
                    setups.append({
                        'direction': 'SHORT',
                        'entry': entry,
                        'stop': stop,
                        'target': target,
                        'rr': reward / risk,
                        'confidence': phase_data['confidence'],
                        'zone': best_supply['price']
                    })
    
    return setups

# ============================================================================
# TELEGRAM ALERT SYSTEM
# ============================================================================

def send_telegram_alert(bot_token: str, chat_id: str, setup: Dict, asset: str) -> bool:
    """Send trade alert via Telegram"""
    if not bot_token or not chat_id or bot_token == 'YOUR_BOT_TOKEN':
        return False
    
    emoji = "🟢" if setup['direction'] == 'LONG' else "🔴"
    
    message = f"""
{emoji} <b>A+ TRADE SIGNAL</b>
━━━━━━━━━━━━━━━━━━━

<b>Asset:</b> {asset}
<b>Direction:</b> {setup['direction']}
<b>Entry:</b> ${setup['entry']:.2f}
<b>Stop Loss:</b> ${setup['stop']:.2f}
<b>Take Profit:</b> ${setup['target']:.2f}

<b>Risk/Reward:</b> 1:{setup['rr']:.1f}
<b>Confidence:</b> {setup['confidence']:.0%}

━━━━━━━━━━━━━━━━━━━
<i>"Discipline over emotion"</i>
"""
    
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {'chat_id': chat_id, 'text': message, 'parse_mode': 'HTML'}
        response = requests.post(url, json=payload, timeout=10)
        return response.status_code == 200
    except:
        return False

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.set_page_config(
        page_title="A+ Trading Platform",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for professional look
    st.markdown("""
    <style>
    .stApp {
        background-color: #0e0f1a;
    }
    .main-header {
        background: linear-gradient(135deg, #1a1f2e 0%, #0a0e1a 100%);
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-bottom: 2px solid #00ff9d;
    }
    .metric-card {
        background: rgba(20, 25, 40, 0.8);
        border-radius: 8px;
        padding: 12px;
        border-left: 3px solid #00ff9d;
    }
    .price-up {
        color: #00ff9d;
        font-weight: bold;
    }
    .price-down {
        color: #ff4d4d;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="color: #00ff9d; margin: 0;">🎯 A+ Trading Platform</h1>
        <p style="color: #8a8f9a; margin: 0;">Professional Market Analysis & Execution</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize data stream
    data_stream = RealTimeDataStream()
    
    # Sidebar - Trading Controls
    with st.sidebar:
        st.markdown("### 📊 Trading Controls")
        
        # Asset selection
        asset = st.selectbox("Asset", list(ASSETS.keys()))
        
        # Timeframe selection
        timeframe = st.selectbox("Timeframe", list(TIMEFRAMES.keys()))
        
        st.markdown("---")
        
        # Chart Controls
        st.markdown("### 🎨 Chart Settings")
        chart_type = st.selectbox("Chart Type", ['candlestick', 'line', 'area'])
        show_sma = st.checkbox("Show SMA 50", value=True)
        show_volume = st.checkbox("Show Volume", value=True)
        chart_theme = st.selectbox("Theme", ['dark', 'light'])
        
        st.markdown("---")
        
        # Analysis Tools
        st.markdown("### 🔧 Analysis Tools")
        show_supply_demand = st.checkbox("Show Supply/Demand Zones", value=True)
        
        st.markdown("---")
        
        # Telegram Integration
        st.markdown("### 📱 Alerts")
        use_telegram = st.checkbox("Enable Telegram Alerts")
        bot_token = st.text_input("Bot Token", type="password", value="YOUR_BOT_TOKEN")
        chat_id = st.text_input("Chat ID", value="YOUR_CHAT_ID")
        
        st.markdown("---")
        
        # Auto-refresh
        auto_refresh = st.checkbox("Auto-refresh (5s)", value=False)
        
        st.markdown("---")
        
        # Trading Rules
        with st.expander("📖 A+ Trading Rules"):
            st.markdown("""
            **A+ Setup Requirements:**
            1. Clear market phase (confidence > 60%)
            2. Correct location (discount for longs, premium for shorts)
            3. Liquidity swept
            4. Structure break confirmed
            5. Minimum 1:2 risk-reward
            
            **Risk Management:**
            - Never risk more than 2% per trade
            - Use hard stops
            - Let winners run
            """)
    
    # Real-time quote
    st.markdown("### 💹 Live Market Data")
    
    quote = data_stream.get_realtime_quote(asset)
    if quote:
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            price_color = "price-up" if quote['change'] >= 0 else "price-down"
            st.markdown(f"""
            <div class="metric-card">
                <small>Current Price</small><br>
                <span class="{price_color}" style="font-size: 24px;">${quote['price']:.2f}</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            change_color = "price-up" if quote['change'] >= 0 else "price-down"
            st.markdown(f"""
            <div class="metric-card">
                <small>24h Change</small><br>
                <span class="{change_color}" style="font-size: 18px;">{quote['change']:+.2f}%</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <small>Bid/Ask</small><br>
                <span>${quote['bid']:.2f} / ${quote['ask']:.2f}</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <small>24h Range</small><br>
                <span>${quote['low']:.2f} - ${quote['high']:.2f}</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            st.markdown(f"""
            <div class="metric-card">
                <small>Volume</small><br>
                <span>{quote['volume']:,.0f}</span>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Fetch data
    with st.spinner(f"Loading {asset} data..."):
        df = data_stream.stream_data(asset)
        
        if df.empty:
            st.error("Failed to load data. Please check your connection.")
            return
    
    # Analyze market
    phase_data = analyze_market_phase(df)
    supply_zones, demand_zones = find_supply_demand(df)
    setups = generate_trade_setups(df, asset, phase_data)
    
    # Market metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Market Phase", phase_data['phase'], 
                 delta=f"{phase_data['confidence']:.0%} confidence")
    
    with col2:
        st.metric("Trend", phase_data['trend'].upper(),
                 delta=f"{phase_data['trend_strength']:+.1f}%")
    
    with col3:
        st.metric("Liquidity Swept", "✅" if phase_data['liquidity_swept'] else "❌")
    
    with col4:
        st.metric("Structure Break", "✅" if phase_data['structure_broken'] else "❌")
    
    with col5:
        st.metric("Volatility", f"{phase_data['volatility']:.1f}%")
    
    st.markdown("---")
    
    # Professional Chart
    chart_engine = ProfessionalChart(df, asset)
    chart_engine.chart_theme = chart_theme
    
    fig = chart_engine.create_chart(
        show_sma=show_sma,
        show_volume=show_volume,
        chart_type=chart_type
    )
    
    # Add supply/demand zones if enabled
    if show_supply_demand:
        for zone in supply_zones:
            fig.add_hline(
                y=zone['price'],
                line_dash="dash",
                line_color="#ff3366",
                opacity=0.5,
                annotation_text=f"Supply {zone['strength']:.0%}",
                annotation_position="top right"
            )
        
        for zone in demand_zones:
            fig.add_hline(
                y=zone['price'],
                line_dash="dash",
                line_color="#33ff66",
                opacity=0.5,
                annotation_text=f"Demand {zone['strength']:.0%}",
                annotation_position="bottom right"
            )
    
    st.plotly_chart(fig, use_container_width=True, config={
        'scrollZoom': True,
        'displayModeBar': True,
        'modeBarButtonsToAdd': ['drawline', 'drawrect', 'eraseshape']
    })
    
    # Chart controls info
    st.info("💡 **Chart Controls:** Scroll to zoom | Drag to pan | Double-click to reset | Use drawing tools from toolbar")
    
    st.markdown("---")
    
    # Trade Setups
    st.markdown("### 🎯 A+ Trade Setups")
    
    if setups:
        for i, setup in enumerate(setups):
            cols = st.columns([1, 2, 2, 2, 1])
            
            with cols[0]:
                direction_color = "#00ff9d" if setup['direction'] == 'LONG' else "#ff4d4d"
                st.markdown(f"""
                <div style="text-align: center; padding: 10px; background: {direction_color}10; border-radius: 8px;">
                    <span style="font-size: 28px;">{'🟢' if setup['direction'] == 'LONG' else '🔴'}</span><br>
                    <span style="color: {direction_color}; font-weight: bold;">{setup['direction']}</span>
                </div>
                """, unsafe_allow_html=True)
            
            with cols[1]:
                st.markdown(f"""
                **Entry:** ${setup['entry']:.2f}<br>
                **Stop:** ${setup['stop']:.2f}<br>
                **Risk:** ${abs(setup['entry'] - setup['stop']):.2f}
                """)
            
            with cols[2]:
                st.markdown(f"""
                **Target:** ${setup['target']:.2f}<br>
                **Reward:** ${abs(setup['target'] - setup['entry']):.2f}<br>
                **R:R:** 1:{setup['rr']:.1f}
                """)
            
            with cols[3]:
                st.markdown(f"""
                **Confidence:** {setup['confidence']:.0%}<br>
                **Zone:** ${setup['zone']:.2f}<br>
                **Status:** Ready
                """)
            
            with cols[4]:
                if use_telegram and bot_token != 'YOUR_BOT_TOKEN':
                    if st.button(f"🚀 Execute", key=f"exec_{i}"):
                        if send_telegram_alert(bot_token, chat_id, setup, asset):
                            st.success("✅ Alert sent!")
                            st.balloons()
                        else:
                            st.error("❌ Failed to send")
                else:
                    st.button(f"📋 Copy", key=f"copy_{i}")
            
            st.markdown("---")
    else:
        st.info("🔍 No A+ setups detected. Waiting for market conditions...")
        
        # Show what's missing
        missing = []
        if phase_data['confidence'] < 0.6:
            missing.append("Market phase not clear")
        if not phase_data['liquidity_swept']:
            missing.append("No liquidity sweep")
        if not phase_data['structure_broken']:
            missing.append("No structure break")
        
        if missing:
            st.warning(f"**Missing A+ Criteria:** {', '.join(missing)}")
    
    # Key levels
    st.markdown("---")
    st.markdown("### 📊 Key Market Levels")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔴 Supply Zones (Resistance)")
        for zone in supply_zones:
            st.write(f"• ${zone['price']:.2f} - Strength: {zone['strength']:.0%}")
        if not supply_zones:
            st.write("No significant supply zones detected")
    
    with col2:
        st.markdown("#### 🟢 Demand Zones (Support)")
        for zone in demand_zones:
            st.write(f"• ${zone['price']:.2f} - Strength: {zone['strength']:.0%}")
        if not demand_zones:
            st.write("No significant demand zones detected")
    
    # Market insights
    st.markdown("---")
    st.markdown("### 📈 Market Insights")
    
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        st.markdown("**Current Conditions:**")
        st.write(f"• Price is {'above' if phase_data['price_above_sma'] else 'below'} SMA 50")
        st.write(f"• Trend strength: {abs(phase_data['trend_strength']):.1f}%")
        st.write(f"• Market phase: {phase_data['phase']}")
    
    with insight_col2:
        st.markdown("**Trading Recommendations:**")
        if phase_data['phase'] == 'Accumulation':
            st.success("✅ Look for LONG setups at demand zones")
        elif phase_data['phase'] == 'Distribution':
            st.warning("⚠️ Look for SHORT setups at supply zones")
        elif phase_data['phase'] == 'Markup':
            st.success("✅ Trend is your friend - look for pullbacks to enter LONG")
        elif phase_data['phase'] == 'Markdown':
            st.warning("⚠️ Wait for distribution before entering SHORT")
        else:
            st.info("⏸️ Consolidation - wait for clear direction")
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(5)
        st.rerun()

if __name__ == "__main__":
    main()
