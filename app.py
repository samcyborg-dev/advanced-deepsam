"""
Professional Trading Platform - TradingView Style
Fixed data loading with multiple fallback options
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

# Alternative symbols for fallback
FALLBACK_SYMBOLS = {
    'GC=F': 'GOLD',
    'CL=F': 'USOIL',
    'EURUSD=X': 'EURUSD',
    '^GSPC': 'SPY',
    '^GDAXI': 'DAX'
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
# ROBUST DATA FETCHER
# ============================================================================

class DataFetcher:
    """Robust data fetching with multiple fallback options"""
    
    def __init__(self):
        self.cache = {}
        
    def fetch_with_retry(self, symbol: str, interval: str, period: str = '1mo', max_retries: int = 3) -> pd.DataFrame:
        """Fetch data with retry logic"""
        
        for attempt in range(max_retries):
            try:
                # First attempt with original symbol
                df = yf.download(symbol, period=period, interval=interval, progress=False, timeout=30)
                
                if not df.empty:
                    return df
                
                # If empty, try fallback symbol
                if symbol in FALLBACK_SYMBOLS:
                    fallback = FALLBACK_SYMBOLS[symbol]
                    df = yf.download(fallback, period=period, interval=interval, progress=False, timeout=30)
                    
                    if not df.empty:
                        return df
                
                # If still empty, try with different period
                if attempt == 1:
                    df = yf.download(symbol, period='3mo', interval=interval, progress=False, timeout=30)
                    if not df.empty:
                        return df
                        
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(2)  # Wait before retry
                
        return pd.DataFrame()
    
    def generate_sample_data(self, symbol: str, interval: str, period: str = '1mo') -> pd.DataFrame:
        """Generate sample data for testing when live data fails"""
        
        # Create date range
        end_date = datetime.now()
        if period == '1d':
            start_date = end_date - timedelta(days=1)
        elif period == '5d':
            start_date = end_date - timedelta(days=5)
        elif period == '1mo':
            start_date = end_date - timedelta(days=30)
        elif period == '3mo':
            start_date = end_date - timedelta(days=90)
        else:
            start_date = end_date - timedelta(days=30)
        
        # Generate time index
        if interval == '1m':
            dates = pd.date_range(start=start_date, end=end_date, freq='1min')
        elif interval == '5m':
            dates = pd.date_range(start=start_date, end=end_date, freq='5min')
        elif interval == '15m':
            dates = pd.date_range(start=start_date, end=end_date, freq='15min')
        elif interval == '30m':
            dates = pd.date_range(start=start_date, end=end_date, freq='30min')
        elif interval == '60m':
            dates = pd.date_range(start=start_date, end=end_date, freq='1h')
        elif interval == '240m':
            dates = pd.date_range(start=start_date, end=end_date, freq='4h')
        else:
            dates = pd.date_range(start=start_date, end=end_date, freq='1d')
        
        # Generate sample price data
        base_price = {
            'GC=F': 1950,
            'CL=F': 75,
            'EURUSD=X': 1.085,
            '^GSPC': 4500,
            '^GDAXI': 15800
        }.get(symbol, 100)
        
        np.random.seed(42)
        returns = np.random.randn(len(dates)) * 0.01
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Create OHLC data
        df = pd.DataFrame({
            'Open': prices * (1 + np.random.randn(len(dates)) * 0.005),
            'High': prices * (1 + abs(np.random.randn(len(dates)) * 0.01)),
            'Low': prices * (1 - abs(np.random.randn(len(dates)) * 0.01)),
            'Close': prices,
            'Volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
        
        # Ensure OHLC consistency
        df['High'] = df[['High', 'Open', 'Close']].max(axis=1)
        df['Low'] = df[['Low', 'Open', 'Close']].min(axis=1)
        
        return df
    
    def get_data(self, symbol: str, interval: str, period: str = '1mo') -> pd.DataFrame:
        """Get data with fallback to sample data if needed"""
        
        cache_key = f"{symbol}_{interval}_{period}"
        
        # Check cache
        if cache_key in self.cache:
            cache_time = self.cache[cache_key].get('timestamp', datetime.min)
            if (datetime.now() - cache_time).seconds < 10:
                return self.cache[cache_key]['data']
        
        # Try to fetch real data
        df = self.fetch_with_retry(symbol, interval, period)
        
        # If failed, use sample data
        if df.empty:
            st.warning(f"⚠️ Using simulated data for {symbol}. Live data unavailable.")
            df = self.generate_sample_data(symbol, interval, period)
        
        # Standardize columns
        if not df.empty:
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # Calculate SMA 50
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            
            # Calculate additional metrics
            df['Change'] = df['Close'].pct_change() * 100
            df['Volume_Change'] = df['Volume'].pct_change() * 100
            
            # Drop NaN values
            df = df.dropna()
        
        # Cache the data
        self.cache[cache_key] = {
            'data': df,
            'timestamp': datetime.now()
        }
        
        return df

# ============================================================================
# REAL-TIME DATA STREAM
# ============================================================================

class RealTimeDataStream:
    """Real-time data streaming with WebSocket-like updates"""
    
    def __init__(self):
        self.data_fetcher = DataFetcher()
        self.last_update = None
        
    def stream_data(self, symbol: str, interval: str = '5m') -> pd.DataFrame:
        """Stream real-time data"""
        return self.data_fetcher.get_data(symbol, interval, period='5d')
    
    def get_realtime_quote(self, symbol: str) -> Dict:
        """Get current quote"""
        try:
            ticker = yf.Ticker(ASSETS.get(symbol, symbol))
            data = ticker.history(period='1d', interval='1m', timeout=10)
            
            if not data.empty:
                latest = data.iloc[-1]
                daily_open = data.iloc[0]['Open'] if len(data) > 0 else latest['Close']
                
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
        except Exception as e:
            print(f"Quote error: {e}")
            
        # Return simulated quote if live fails
        base_price = {
            'XAUUSD (Gold)': 1950,
            'WTI (Oil)': 75,
            'EURUSD': 1.085,
            'S&P500': 4500,
            'DAX30': 15800
        }.get(symbol, 100)
        
        return {
            'price': base_price,
            'bid': base_price * 0.9999,
            'ask': base_price * 1.0001,
            'change': np.random.randn() * 0.5,
            'high': base_price * 1.01,
            'low': base_price * 0.99,
            'volume': np.random.randint(1000, 5000),
            'timestamp': datetime.now()
        }

# ============================================================================
# PROFESSIONAL CHART ENGINE
# ============================================================================

class ProfessionalChart:
    """TradingView-style professional chart engine"""
    
    def __init__(self, df: pd.DataFrame, asset: str):
        self.df = df
        self.asset = asset
        self.chart_type = 'candlestick'
        self.chart_theme = 'dark'
        
    def create_chart(self, show_sma: bool = True, 
                     show_volume: bool = True,
                     chart_type: str = 'candlestick') -> go.Figure:
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
        
        # Chart styling
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
        
        # Update layout
        fig.update_layout(
            template='plotly_dark' if self.chart_theme == 'dark' else 'plotly_white',
            paper_bgcolor=bg_color,
            plot_bgcolor=bg_color,
            font=dict(color=text_color, size=12),
            title={
                'text': f'{self.asset} - Interactive Chart (Scroll to Zoom | Drag to Pan)',
                'font': {'size': 16, 'color': text_color},
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
                    y=1.12,
                    buttons=list([
                        dict(label="Reset View",
                             method="relayout",
                             args=[{"xaxis.autorange": True, "yaxis.autorange": True}]),
                        dict(label="Zoom In (50 bars)",
                             method="relayout",
                             args=[{"xaxis.range": [self.df.index[-50], self.df.index[-1]]}]),
                        dict(label="Zoom Out (200 bars)",
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
    if len(df) < 50:
        return {'phase': 'Analyzing...', 'confidence': 0, 'trend': 'neutral', 'trend_strength': 0, 'volatility': 0, 'liquidity_swept': False, 'structure_broken': False, 'price_above_sma': False}
    
    # Calculate key metrics
    sma_50 = df['SMA_50'].iloc[-1] if 'SMA_50' in df.columns else df['Close'].iloc[-1]
    current_price = df['Close'].iloc[-1]
    price_above_sma = current_price > sma_50 if not pd.isna(sma_50) else False
    
    # Trend analysis
    price_20d_ago = df['Close'].iloc[-20] if len(df) >= 20 else df['Close'].iloc[0]
    price_50d_ago = df['Close'].iloc[-50] if len(df) >= 50 else df['Close'].iloc[0]
    
    trend_strength = ((current_price - price_20d_ago) / price_20d_ago) * 100 if price_20d_ago > 0 else 0
    long_trend = ((current_price - price_50d_ago) / price_50d_ago) * 100 if price_50d_ago > 0 else 0
    
    # Volatility analysis
    atr = df['High'].iloc[-20:].max() - df['Low'].iloc[-20:].min() if len(df) >= 20 else 0
    volatility = (atr / current_price * 100) if current_price > 0 else 0
    
    # Phase detection
    if trend_strength > 2 and long_trend > 5:
        phase = 'Markup'
        confidence = min(0.9, 0.6 + trend_strength / 20)
    elif trend_strength < -2 and long_trend < -5:
        phase = 'Markdown'
        confidence = min(0.9, 0.6 + abs(trend_strength) / 20)
    elif abs(trend_strength) < 1 and volatility < 2 and len(df) >= 50:
        price_range = df['High'].iloc[-50:].max() - df['Low'].iloc[-50:].min()
        price_position = (current_price - df['Low'].iloc[-50:].min()) / price_range if price_range > 0 else 0.5
        
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
        recent_high = df['High'].iloc[-10:-1].max() if len(df) > 10 else df['High'].iloc[-1]
        recent_low = df['Low'].iloc[-10:-1].min() if len(df) > 10 else df['Low'].iloc[-1]
        
        if df['High'].iloc[-1] > recent_high:
            if df['Close'].iloc[-1] < df['Close'].iloc[-2] if len(df) > 1 else False:
                liquidity_swept = True
        elif df['Low'].iloc[-1] < recent_low:
            if df['Close'].iloc[-1] > df['Close'].iloc[-2] if len(df) > 1 else False:
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
        'confidence': min(0.95, confidence),
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
    
    if len(df) < 30:
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
    
    # Identify demand zones
    for low in swings_low[-10:]:
        avg_volume = df['Volume'].iloc[max(0, low['index']-10):low['index']].mean()
        if low['volume'] > avg_volume:
            zone = {
                'price': low['price'],
                'strength': min(1.0, low['volume'] / df['Volume'].mean()),
                'time': low['time'],
                'type': 'demand'
            }
            demand.append(zone)
    
    # Identify supply zones
    for high in swings_high[-10:]:
        avg_volume = df['Volume'].iloc[max(0, high['index']-10):high['index']].mean()
        if high['volume'] > avg_volume:
            zone = {
                'price': high['price'],
                'strength': min(1.0, high['volume'] / df['Volume'].mean()),
                'time': high['time'],
                'type': 'supply'
            }
            supply.append(zone)
    
    # Remove duplicates and sort
    supply = [dict(t) for t in {tuple(sorted(d.items())) for d in supply}]
    demand = [dict(t) for t in {tuple(sorted(d.items())) for d in demand}]
    
    supply.sort(key=lambda x: x['strength'], reverse=True)
    demand.sort(key=lambda x: x['strength'], reverse=True)
    
    return supply[:3], demand[:3]

# ============================================================================
# A+ TRADE SETUP GENERATOR
# ============================================================================

def generate_trade_setups(df: pd.DataFrame, asset: str, phase_data: Dict) -> List[Dict]:
    """Generate A+ trade setups based on the playbook"""
    setups = []
    
    # Hard filters
    if phase_data['confidence'] < 0.6:
        return setups
    
    if not phase_data['liquidity_swept']:
        return setups
    
    if not phase_data['structure_broken']:
        return setups
    
    current_price = df['Close'].iloc[-1]
    supply_zones, demand_zones = find_supply_demand(df)
    
    # Calculate ATR for stop loss
    atr = (df['High'].iloc[-20:].max() - df['Low'].iloc[-20:].min()) / 2 if len(df) >= 20 else current_price * 0.01
    
    # LONG SETUP
    if phase_data['trend'] in ['bullish', 'neutral'] and phase_data['phase'] in ['Accumulation', 'Markup']:
        if demand_zones:
            best_demand = demand_zones[0]
            
            # Check if price is near demand zone
            distance = abs(current_price - best_demand['price']) / current_price
            
            if distance < 0.02:
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
            
            if distance < 0.02:
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
    
    # Custom CSS
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
    data_fetcher = DataFetcher()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Trading Controls")
        
        asset = st.selectbox("Asset", list(ASSETS.keys()))
        timeframe = st.selectbox("Timeframe", list(TIMEFRAMES.keys()))
        period = st.selectbox("Data Period", ['1d', '5d', '1mo', '3mo'])
        
        st.markdown("---")
        
        st.markdown("### 🎨 Chart Settings")
        chart_type = st.selectbox("Chart Type", ['candlestick', 'line', 'area'])
        show_sma = st.checkbox("Show SMA 50", value=True)
        show_volume = st.checkbox("Show Volume", value=True)
        chart_theme = st.selectbox("Theme", ['dark', 'light'])
        
        st.markdown("---")
        
        st.markdown("### 🔧 Analysis Tools")
        show_supply_demand = st.checkbox("Show Supply/Demand Zones", value=True)
        
        st.markdown("---")
        
        st.markdown("### 📱 Alerts")
        use_telegram = st.checkbox("Enable Telegram Alerts")
        bot_token = st.text_input("Bot Token", type="password", value="YOUR_BOT_TOKEN")
        chat_id = st.text_input("Chat ID", value="YOUR_CHAT_ID")
        
        st.markdown("---")
        
        auto_refresh = st.checkbox("Auto-refresh (5s)", value=False)
        
        st.markdown("---")
        
        with st.expander("📖 A+ Trading Rules"):
            st.markdown("""
            **A+ Setup Requirements:**
            1. Clear market phase (confidence > 60%)
            2. Correct location
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
        symbol = ASSETS.get(asset, asset)
        df = data_fetcher.get_data(symbol, TIMEFRAMES[timeframe], period)
        
        if df.empty:
            st.error("❌ Failed to load data. Please check your internet connection and try again.")
            st.info("💡 Tip: Try selecting a different timeframe or asset, or refresh the page.")
            return
        
        st.success(f"✅ Loaded {len(df)} candles for {asset}")
    
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
        'modeBarButtonsToAdd': ['drawline', 'drawrect', 'eraseshape'],
        'displaylogo': False
    })
    
    # Chart instructions
    st.info("""
    💡 **Chart Controls:**
    - **Scroll** = Zoom in/out
    - **Drag** = Pan across chart
    - **Double-click** = Reset view
    - **Use toolbar** = Drawing tools, zoom, pan, etc.
    """)
    
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
                            st.success("✅ Alert sent to Telegram!")
                            st.balloons()
                        else:
                            st.error("❌ Failed to send alert")
                else:
                    st.info("Enable Telegram to execute")
            
            st.markdown("---")
    else:
        st.info("🔍 No A+ setups detected. Waiting for market conditions...")
        
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
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(5)
        st.rerun()

if __name__ == "__main__":
    main()
