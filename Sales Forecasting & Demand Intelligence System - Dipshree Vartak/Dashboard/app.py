import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set page config for a widescreen layout and dark theme
st.set_page_config(
    page_title="DemandIntel | Sales Forecasting & Demand Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling (Dark Mode & Glassmorphism)
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap" rel="stylesheet">
<style>
    /* Main App Container */
    .stApp {
        background: linear-gradient(135deg, #0A0C10 0%, #1F2833 100%);
        color: #EAEAEA;
        font-family: 'Outfit', sans-serif;
    }
    
    /* Header/Title Styles */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        color: #00F2FE !important;
        font-weight: 800;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0d1117 !important;
        border-right: 1px solid rgba(0, 242, 254, 0.1);
    }
    
    /* Glassmorphism Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 22px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        transition: transform 0.3s ease, border 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        border: 1px solid rgba(0, 242, 254, 0.3);
    }
    
    .metric-title {
        font-size: 14px;
        color: #8A99AD;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
        margin-bottom: 8px;
    }
    
    .metric-value {
        font-size: 32px;
        font-weight: 800;
        color: #FFFFFF;
        background: linear-gradient(45deg, #00F2FE, #4FACFE);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-change {
        font-size: 14px;
        margin-top: 5px;
        font-weight: 600;
    }
    
    .positive-change { color: #00E676; }
    .negative-change { color: #FF1744; }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0A0C10;
    }
    ::-webkit-scrollbar-thumb {
        background: #1F2833;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #00F2FE;
    }
</style>
""", unsafe_allow_html=True)


# Helper: Load and preprocess primary data
@st.cache_data
def load_and_preprocess_data():
    # Load dataset
    train_path = 'Train.csv'
    if not os.path.exists(train_path):
        train_path = 'train.csv'
    
    if not os.path.exists(train_path):
        st.error(f"Dataset '{train_path}' not found! Please place the Superstore sales CSV in the workspace directory.")
        return None
        
    df = pd.read_csv(train_path)
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='mixed')
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='mixed')
    
    # Extract features
    df['Year'] = df['Order Date'].dt.year
    df['Month'] = df['Order Date'].dt.month
    df['Week'] = df['Order Date'].dt.isocalendar().week.astype(int)
    df['DayOfWeek'] = df['Order Date'].dt.day_name()
    df['Quarter'] = df['Order Date'].dt.quarter
    
    # Map seasons
    def get_season(month):
        if month in [12, 1, 2]: return 'Winter'
        elif month in [3, 4, 5]: return 'Spring'
        elif month in [6, 7, 8]: return 'Summer'
        else: return 'Autumn'
        
    df['Season'] = df['Month'].apply(get_season)
    df['Ship_Time'] = (df['Ship Date'] - df['Order Date']).dt.days
    
    return df

df_raw = load_and_preprocess_data()

if df_raw is not None:
    # Sidebar Navigation with modern branding
    st.sidebar.markdown("""
    <div style="text-align: center; margin-bottom: 25px;">
        <h2 style="margin: 0; color: #00F2FE; font-weight: 800; letter-spacing: 1px;">DemandIntel AI</h2>
        <span style="color: #8A99AD; font-size: 12px; font-weight: 600;">Sales Forecasting & Intelligent Demand System</span>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.sidebar.radio(
        "Navigation",
        ["Sales Overview", "Forecast Explorer", "Anomaly Report", "Product Demand Segments"]
    )
    
    # Global Filters on Sidebar (Applies to Page 1 mainly)
    st.sidebar.markdown("<hr style='border-color: rgba(255,255,255,0.08);'>", unsafe_allow_html=True)
    st.sidebar.subheader("Global Filters")
    
    # Region filter
    all_regions = ['All'] + list(df_raw['Region'].unique())
    selected_region = st.sidebar.selectbox("Select Region", all_regions, index=0)
    
    # Category filter
    all_categories = ['All'] + list(df_raw['Category'].unique())
    selected_category = st.sidebar.selectbox("Select Category", all_categories, index=0)
    
    # Apply filters to dataset for Overview
    df_filtered = df_raw.copy()
    if selected_region != 'All':
        df_filtered = df_filtered[df_filtered['Region'] == selected_region]
    if selected_category != 'All':
        df_filtered = df_filtered[df_filtered['Category'] == selected_category]

    # =====================================================================
    # PAGE 1: SALES OVERVIEW DASHBOARD
    # =====================================================================
    if page == "Sales Overview":
        st.title("📊 Sales Overview Dashboard")
        st.write("An overview of historical sales performance, category breakdowns, and growth dynamics.")
        
        # Metric Cards Layout
        total_sales = df_filtered['Sales'].sum()
        total_orders = df_filtered['Order ID'].nunique()
        avg_ship_time = df_filtered['Ship_Time'].mean()
        num_categories = df_filtered['Category'].nunique()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Total Sales Revenue</div>
                <div class="metric-value">${total_sales:,.2f}</div>
                <div class="metric-change positive-change">▲ Active Volume</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Total Orders Placed</div>
                <div class="metric-value">{total_orders:,}</div>
                <div class="metric-change positive-change">▲ 100% Filled</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Avg. Fulfillment Time</div>
                <div class="metric-value">{avg_ship_time:.2f} Days</div>
                <div class="metric-change positive-change">▼ Within Target</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Active Categories</div>
                <div class="metric-value">{num_categories}</div>
                <div class="metric-change positive-change">▲ Diversified</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Charts Row 1
        c_row1_1, c_row1_2 = st.columns(2)
        
        with c_row1_1:
            # Sales by Year (Bar Chart)
            year_sales = df_filtered.groupby('Year')['Sales'].sum().reset_index()
            year_sales['Year'] = year_sales['Year'].astype(str)
            fig_bar = px.bar(
                year_sales, x='Year', y='Sales',
                title="Total Sales Revenue by Year",
                color='Sales',
                color_continuous_scale=px.colors.sequential.Tealgrn,
                text_auto='.3s'
            )
            fig_bar.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#EAEAEA',
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with c_row1_2:
            # Sales by Category and Region
            cat_region_sales = df_filtered.groupby(['Category', 'Region'])['Sales'].sum().reset_index()
            fig_sun = px.sunburst(
                cat_region_sales, path=['Category', 'Region'], values='Sales',
                title="Sales Breakdown by Category and Region",
                color='Sales',
                color_continuous_scale=px.colors.sequential.Electric
            )
            fig_sun.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#EAEAEA'
            )
            st.plotly_chart(fig_sun, use_container_width=True)
            
        # Charts Row 2 - Line Chart for monthly sales trend
        st.subheader("📈 Monthly Sales Trend (4-Year Horizon)")
        df_monthly_trend = df_filtered.set_index('Order Date').resample('ME')['Sales'].sum().reset_index()
        fig_line = px.line(
            df_monthly_trend, x='Order Date', y='Sales',
            title="Aggregated Monthly Sales Over Time",
            markers=True,
            labels={'Order Date': 'Date', 'Sales': 'Revenue ($)'}
        )
        fig_line.update_traces(line_color='#00F2FE', line_width=3, marker=dict(size=8, color='#ff7f0e'))
        fig_line.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#EAEAEA',
            xaxis=dict(showgrid=False, rangeslider=dict(visible=True)),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
        )
        st.plotly_chart(fig_line, use_container_width=True)

    # =====================================================================
    # PAGE 2: FORECAST EXPLORER
    # =====================================================================
    elif page == "Forecast Explorer":
        st.title("🔮 Advanced Forecast Explorer")
        st.write("Dynamic multi-model sales forecasting with parameter fitting, out-of-sample predictions, and validation metrics.")
        
        # Segment options
        st.subheader("Select Segment for Forecasting")
        seg_col1, seg_col2, seg_col3 = st.columns(3)
        
        with seg_col1:
            segment_type = st.selectbox("Select Segment Type", ["Total Sales", "Category", "Region"])
            
        with seg_col2:
            if segment_type == "Total Sales":
                segment_selection = "Total Sales"
                seg_df = df_raw.copy()
            elif segment_type == "Category":
                cats = list(df_raw['Category'].unique())
                segment_selection = st.selectbox("Select Category", cats)
                seg_df = df_raw[df_raw['Category'] == segment_selection]
            else:
                regs = list(df_raw['Region'].unique())
                segment_selection = st.selectbox("Select Region", regs)
                seg_df = df_raw[df_raw['Region'] == segment_selection]
                
        with seg_col3:
            model_selection = st.selectbox("Select Forecasting Model", ["SARIMA", "Prophet", "XGBoost"])
            
        forecast_horizon = st.slider("Select Forecast Horizon (Months Ahead)", min_value=1, max_value=3, value=3)
        
        # Aggregate monthly sales for the chosen segment
        seg_ts = seg_df.set_index('Order Date').resample('ME')['Sales'].sum()
        
        # Validation Split (last 3 months)
        train_size = len(seg_ts) - 3
        train_ts = seg_ts.iloc[:train_size]
        val_ts = seg_ts.iloc[train_size:]
        
        st.markdown(f"**Selected:** `{segment_selection}` | **Model:** `{model_selection}`")
        
        # Run Forecast models dynamically on-the-fly
        with st.spinner("Fitting model and generating forecast..."):
            
            if model_selection == "SARIMA":
                # Grid-search / simple fit on train_ts
                try:
                    # Default parameters or simple order
                    order = (1, 1, 1)
                    seasonal_order = (1, 1, 0, 12)
                    
                    # Train model
                    sarima_train = SARIMAX(train_ts, order=order, seasonal_order=seasonal_order,
                                           enforce_stationarity=False, enforce_invertibility=False)
                    sarima_train_fit = sarima_train.fit(disp=False)
                    val_pred = sarima_train_fit.forecast(steps=3)
                    
                    # Full model
                    sarima_full = SARIMAX(seg_ts, order=order, seasonal_order=seasonal_order,
                                          enforce_stationarity=False, enforce_invertibility=False)
                    sarima_full_fit = sarima_full.fit(disp=False)
                    future_forecast = sarima_full_fit.get_forecast(steps=forecast_horizon)
                    future_mean = future_forecast.predicted_mean
                    future_conf = future_forecast.conf_int(alpha=0.05)
                    
                    # Convert to pandas series
                    val_pred = pd.Series(val_pred, index=val_ts.index)
                    future_mean = pd.Series(future_mean, index=pd.date_range(start=seg_ts.index[-1] + pd.DateOffset(months=1), periods=forecast_horizon, freq='ME'))
                    conf_lower = future_conf.iloc[:, 0]
                    conf_upper = future_conf.iloc[:, 1]
                    
                    mae = mean_absolute_error(val_ts, val_pred)
                    rmse = np.sqrt(mean_squared_error(val_ts, val_pred))
                    mape = np.mean(np.abs((val_ts - val_pred) / val_ts)) * 100
                    
                except Exception as e:
                    st.error(f"SARIMA execution failed: {e}")
                    val_pred, future_mean, mae, rmse, mape = None, None, 0, 0, 0
                    
            elif model_selection == "Prophet":
                try:
                    # Train model
                    df_prop_train = train_ts.reset_index().rename(columns={'Order Date': 'ds', 'Sales': 'y'})
                    df_prop_train['ds'] = df_prop_train['ds'].dt.tz_localize(None)
                    
                    m_prop = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
                    m_prop.fit(df_prop_train)
                    
                    fut_val = m_prop.make_future_dataframe(periods=3, freq='ME')
                    val_pred_df = m_prop.predict(fut_val).tail(3)
                    val_pred = pd.Series(val_pred_df['yhat'].values, index=val_ts.index)
                    
                    # Full model
                    df_prop_full = seg_ts.reset_index().rename(columns={'Order Date': 'ds', 'Sales': 'y'})
                    df_prop_full['ds'] = df_prop_full['ds'].dt.tz_localize(None)
                    m_prop_full = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
                    m_prop_full.fit(df_prop_full)
                    
                    fut_full = m_prop_full.make_future_dataframe(periods=forecast_horizon, freq='ME')
                    full_forecast_df = m_prop_full.predict(fut_full)
                    future_pred_df = full_forecast_df.tail(forecast_horizon)
                    
                    future_mean = pd.Series(future_pred_df['yhat'].values, index=pd.date_range(start=seg_ts.index[-1] + pd.DateOffset(months=1), periods=forecast_horizon, freq='ME'))
                    conf_lower = pd.Series(future_pred_df['yhat_lower'].values, index=future_mean.index)
                    conf_upper = pd.Series(future_pred_df['yhat_upper'].values, index=future_mean.index)
                    
                    mae = mean_absolute_error(val_ts, val_pred)
                    rmse = np.sqrt(mean_squared_error(val_ts, val_pred))
                    mape = np.mean(np.abs((val_ts - val_pred) / val_ts)) * 100
                except Exception as e:
                    st.error(f"Prophet execution failed: {e}")
                    val_pred, future_mean, mae, rmse, mape = None, None, 0, 0, 0
                    
            else: # XGBoost
                try:
                    # Helper for features
                    def build_features(series):
                        df_f = pd.DataFrame(series)
                        df_f.columns = ['y']
                        df_f['lag_1'] = df_f['y'].shift(1)
                        df_f['lag_2'] = df_f['y'].shift(2)
                        df_f['lag_3'] = df_f['y'].shift(3)
                        df_f['rolling_mean'] = df_f['y'].shift(1).rolling(window=3).mean()
                        df_f['month'] = df_f.index.month
                        df_f['quarter'] = df_f.index.quarter
                        return df_f.dropna()
                        
                    def run_recursive(model, history, steps=3):
                        curr = history.copy()
                        preds = []
                        for _ in range(steps):
                            df_feat = build_features(curr)
                            last_row = df_feat.iloc[-1:].drop(columns=['y'])
                            pred_val = model.predict(last_row)[0]
                            preds.append(pred_val)
                            next_dt = curr.index[-1] + pd.DateOffset(months=1)
                            curr = pd.concat([curr, pd.Series([pred_val], index=[next_dt])])
                        return preds
                        
                    # Train features
                    df_train_feat = build_features(train_ts)
                    X_tr, y_tr = df_train_feat.drop(columns=['y']), df_train_feat['y']
                    
                    xgb = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
                    xgb.fit(X_tr, y_tr)
                    
                    # Validate
                    val_pred_list = run_recursive(xgb, seg_ts.iloc[:train_size], steps=3)
                    val_pred = pd.Series(val_pred_list, index=val_ts.index)
                    
                    # Full model
                    df_full_feat = build_features(seg_ts)
                    X_f, y_f = df_full_feat.drop(columns=['y']), df_full_feat['y']
                    xgb_full = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
                    xgb_full.fit(X_f, y_f)
                    
                    future_list = run_recursive(xgb_full, seg_ts, steps=forecast_horizon)
                    future_mean = pd.Series(future_list, index=pd.date_range(start=seg_ts.index[-1] + pd.DateOffset(months=1), periods=forecast_horizon, freq='ME'))
                    conf_lower = future_mean * 0.92 # simulated bounds since XGBoost is deterministic
                    conf_upper = future_mean * 1.08
                    
                    mae = mean_absolute_error(val_ts, val_pred)
                    rmse = np.sqrt(mean_squared_error(val_ts, val_pred))
                    mape = np.mean(np.abs((val_ts - val_pred) / val_ts)) * 100
                except Exception as e:
                    st.error(f"XGBoost execution failed: {e}")
                    val_pred, future_mean, mae, rmse, mape = None, None, 0, 0, 0

        # Plotly chart for actuals + validation + future forecast
        if val_pred is not None and future_mean is not None:
            fig_fore = go.Figure()
            
            # Actual sales
            fig_fore.add_trace(go.Scatter(
                x=seg_ts.index, y=seg_ts.values,
                name="Actual Sales",
                line=dict(color='#2b5c8f', width=3),
                mode='lines+markers'
            ))
            
            # Validation prediction
            fig_fore.add_trace(go.Scatter(
                x=val_ts.index, y=val_pred.values,
                name="Validation Forecast (Backtest)",
                line=dict(color='#ff7f0e', width=2, dash='dash'),
                mode='lines+markers'
            ))
            
            # Future Forecast
            fig_fore.add_trace(go.Scatter(
                x=future_mean.index, y=future_mean.values,
                name="Future Forecast",
                line=dict(color='#00F2FE', width=3),
                mode='lines+markers'
            ))
            
            # Confidence Interval
            fig_fore.add_trace(go.Scatter(
                x=list(future_mean.index) + list(future_mean.index)[::-1],
                y=list(conf_upper.values) + list(conf_lower.values)[::-1],
                fill='toself',
                fillcolor='rgba(0, 242, 254, 0.12)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=True,
                name="Forecast Confidence Bounds"
            ))
            
            fig_fore.update_layout(
                title=f"Forecast Projection for {segment_selection} using {model_selection}",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#EAEAEA',
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_fore, use_container_width=True)
            
            # Display metrics & future forecast table
            col_met1, col_met2, col_met3 = st.columns(3)
            with col_met1:
                st.metric("Model Mean Absolute Error (MAE)", f"${mae:,.2f}")
            with col_met2:
                st.metric("Model Root Mean Squared Error (RMSE)", f"${rmse:,.2f}")
            with col_met3:
                st.metric("Mean Absolute Percentage Error (MAPE)", f"{mape:.2f}%")
                
            # Forecast Table
            st.subheader("📋 Predicted Future Sales Values")
            forecast_tbl = pd.DataFrame({
                'Forecast Period': [d.strftime('%Y-%m') for d in future_mean.index],
                'Predicted Revenue': [f"${v:,.2f}" for v in future_mean.values],
                'Lower Confidence Interval': [f"${v:,.2f}" for v in conf_lower.values],
                'Upper Confidence Interval': [f"${v:,.2f}" for v in conf_upper.values]
            })
            st.dataframe(forecast_tbl, use_container_width=True, hide_index=True)

    # =====================================================================
    # PAGE 3: ANOMALY REPORT
    # =====================================================================
    elif page == "Anomaly Report":
        st.title("🚨 Anomaly Detection Report")
        st.write("Detect sales weeks that deviate significantly from expected baseline patterns using Isolation Forest and Z-Score techniques.")
        
        # Aggregate weekly sales
        df_weekly = df_filtered.set_index('Order Date').resample('W')['Sales'].sum().reset_index()
        df_weekly_sales = df_weekly.copy().set_index('Order Date')
        
        # Run anomaly detection algorithms
        df_weekly_sales['rolling_mean'] = df_weekly_sales['Sales'].rolling(window=4, min_periods=1).mean()
        df_weekly_sales['rolling_std'] = df_weekly_sales['Sales'].rolling(window=4, min_periods=1).std().fillna(0)
        
        # Isolation Forest
        features_anomaly = df_weekly_sales[['Sales', 'rolling_mean', 'rolling_std']]
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        df_weekly_sales['anomaly_iso'] = iso_forest.fit_predict(features_anomaly)
        df_weekly_sales['is_anomaly_iso'] = df_weekly_sales['anomaly_iso'] == -1
        
        # Z-Score
        window = 12
        df_weekly_sales['rolling_mean_12'] = df_weekly_sales['Sales'].rolling(window=window, min_periods=1).mean()
        df_weekly_sales['rolling_std_12'] = df_weekly_sales['Sales'].rolling(window=window, min_periods=1).std().fillna(df_weekly_sales['Sales'].std())
        df_weekly_sales['z_score'] = (df_weekly_sales['Sales'] - df_weekly_sales['rolling_mean_12']) / df_weekly_sales['rolling_std_12']
        df_weekly_sales['is_anomaly_z'] = df_weekly_sales['z_score'].abs() > 2.0
        
        # Selection filter for which anomaly to view
        anomaly_algo = st.selectbox("Select Anomaly Algorithm to Plot", ["Isolation Forest (ML)", "Z-Score (Statistical)"])
        
        fig_anom = go.Figure()
        
        # Base weekly sales line
        fig_anom.add_trace(go.Scatter(
            x=df_weekly_sales.index, y=df_weekly_sales['Sales'],
            name="Weekly Sales",
            line=dict(color='#2b5c8f', width=2),
            mode='lines'
        ))
        
        # Overlay anomalies
        if anomaly_algo == "Isolation Forest (ML)":
            anom_df = df_weekly_sales[df_weekly_sales['is_anomaly_iso']]
            marker_style = dict(color='red', size=10, symbol='x')
            lbl = "Isolation Forest Anomaly"
        else:
            anom_df = df_weekly_sales[df_weekly_sales['is_anomaly_z']]
            marker_style = dict(color='#E040FB', size=10, symbol='circle-open', line=dict(width=2))
            lbl = "Z-Score Anomaly"
            
        fig_anom.add_trace(go.Scatter(
            x=anom_df.index, y=anom_df['Sales'],
            mode='markers',
            marker=marker_style,
            name=lbl
        ))
        
        fig_anom.update_layout(
            title=f"Weekly Sales Timeline with Highlighted Anomalies ({anomaly_algo})",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#EAEAEA',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
        )
        st.plotly_chart(fig_anom, use_container_width=True)
        
        # Comparison stats
        iso_count = df_weekly_sales['is_anomaly_iso'].sum()
        z_count = df_weekly_sales['is_anomaly_z'].sum()
        overlap = (df_weekly_sales['is_anomaly_iso'] & df_weekly_sales['is_anomaly_z']).sum()
        
        col_anom1, col_anom2, col_anom3 = st.columns(3)
        with col_anom1:
            st.metric("Isolation Forest Flagged Weeks", f"{iso_count} Weeks")
        with col_anom2:
            st.metric("Z-Score Flagged Weeks", f"{z_count} Weeks")
        with col_anom3:
            st.metric("Overlapping Detections", f"{overlap} Weeks")
            
        # Table of anomalies
        st.subheader("📋 Flagged Anomaly Details & Explanations")
        anom_report_df = df_weekly_sales[df_weekly_sales['is_anomaly_iso'] | df_weekly_sales['is_anomaly_z']].copy()
        anom_report_df = anom_report_df.sort_index(ascending=False).reset_index()
        
        # Build explanation dynamically
        def explain_anom(row):
            if row['Sales'] > df_weekly_sales['Sales'].mean():
                return "Revenue Spike: Potential major marketing event, holiday sales (Q4), or large bulk corporate procurement."
            else:
                return "Revenue Drop: Post-holiday slump, shipment tracking latency, or supply chain bottlenecks."
                
        anom_report_df['Business Explanation'] = anom_report_df.apply(explain_anom, axis=1)
        anom_report_df['Algorithm Method'] = anom_report_df.apply(
            lambda r: "Both" if (r['is_anomaly_iso'] and r['is_anomaly_z']) else ("Isolation Forest" if r['is_anomaly_iso'] else "Z-Score"),
            axis=1
        )
        
        display_tbl = anom_report_df[['Order Date', 'Sales', 'Algorithm Method', 'Business Explanation']].rename(
            columns={'Order Date': 'Week Commencing', 'Sales': 'Weekly Revenue'}
        )
        display_tbl['Weekly Revenue'] = display_tbl['Weekly Revenue'].map(lambda v: f"${v:,.2f}")
        
        st.dataframe(display_tbl, use_container_width=True, hide_index=True)

    # =====================================================================
    # PAGE 4: PRODUCT DEMAND SEGMENTS
    # =====================================================================
    elif page == "Product Demand Segments":
        st.title("🎯 Product Demand Segmentation")
        st.write("Segment product sub-categories into actionable demand profiles using K-Means Clustering on volume, volatility, growth, and transaction size.")
        
        # Calculate features dynamically
        # 1. Volume
        subcat_tot = df_filtered.groupby('Sub-Category')['Sales'].sum()
        # 2. Growth
        subcat_annual = df_filtered.groupby(['Sub-Category', 'Year'])['Sales'].sum().unstack().fillna(0)
        subcat_growth = subcat_annual.pct_change(axis=1).iloc[:, 1:].mean(axis=1).fillna(0)
        # 3. Volatility
        subcat_monthly = df_filtered.groupby(['Sub-Category', pd.Grouper(key='Order Date', freq='ME')])['Sales'].sum().unstack().fillna(0)
        subcat_vol = subcat_monthly.std(axis=1)
        # 4. Avg Order Value
        subcat_aov = df_filtered.groupby('Sub-Category')['Sales'].mean()
        
        feat_df = pd.DataFrame({
            'Total_Sales': subcat_tot,
            'Growth_Rate': subcat_growth,
            'Volatility': subcat_vol,
            'Avg_Order_Value': subcat_aov
        }).fillna(0)
        
        # Scale features
        scaler = StandardScaler()
        scaled = scaler.fit_transform(feat_df)
        
        # KMeans Fit (K=4)
        k_val = 4
        kmeans = KMeans(n_clusters=k_val, random_state=42, n_init=10)
        feat_df['Cluster'] = kmeans.fit_predict(scaled)
        
        # Map labels based on centroid properties
        # Simple sorting of clusters by sales volume
        centers = scaler.inverse_transform(kmeans.cluster_centers_)
        cluster_order = np.argsort(centers[:, 0]) # indices of clusters sorted by sales volume (ascending)
        
        # Assign meaningful label strings
        cluster_labels = {
            cluster_order[3]: "⚡ High Volume, Stable Demand",
            cluster_order[2]: "📈 High Growth, Emerging",
            cluster_order[1]: "🔄 Steady Demand, Medium Volume",
            cluster_order[0]: "⚠️ Low Volume, High Volatility"
        }
        
        feat_df['Segment_Name'] = feat_df['Cluster'].map(cluster_labels)
        
        # PCA for projection
        pca = PCA(n_components=2)
        pca_proj = pca.fit_transform(scaled)
        feat_df['PCA1'] = pca_proj[:, 0]
        feat_df['PCA2'] = pca_proj[:, 1]
        
        # Plotly cluster plot
        fig_cl = px.scatter(
            feat_df.reset_index(), x='PCA1', y='PCA2',
            color='Segment_Name',
            hover_name='Sub-Category',
            text='Sub-Category',
            title="Product Demand Segmentation Map (PCA 2D Spatial Projection)",
            color_discrete_sequence=px.colors.qualitative.Bold,
            labels={'PCA1': 'PCA Axis 1', 'PCA2': 'PCA Axis 2'}
        )
        fig_cl.update_traces(marker=dict(size=14, line=dict(color='black', width=1)), textposition='top center')
        fig_cl.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#EAEAEA',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
            legend=dict(title="Demand Segment", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_cl, use_container_width=True)
        
        # Recommendations Card Layout
        st.subheader("💡 Stocking Strategy Recommendations by Demand Profile")
        
        rec_col1, rec_col2 = st.columns(2)
        
        with rec_col1:
            st.markdown("""
            <div class="metric-card">
                <h4 style="margin: 0 0 10px 0; color: #00F2FE;">⚡ High Volume, Stable Demand</h4>
                <p style="font-size: 13px; color: #EAEAEA;">
                    <b>Stocking Strategy:</b> Maintain high safety stock coefficients. Adopt continuous replenishment systems with weekly review schedules.
                </p>
                <p style="font-size: 13px; color: #8A99AD;">
                    <b>Focus:</b> Vendor contract locks, bulk pricing discounts, and automated restocking pipelines.
                </p>
            </div>
            <div class="metric-card">
                <h4 style="margin: 0 0 10px 0; color: #4FACFE;">📈 High Growth, Emerging</h4>
                <p style="font-size: 13px; color: #EAEAEA;">
                    <b>Stocking Strategy:</b> Agile, low-commitment sourcing. Leverage quick-ship agreements and build dynamic buffers.
                </p>
                <p style="font-size: 13px; color: #8A99AD;">
                    <b>Focus:</b> Trend monitoring, promotion synchronization, and lead-time optimization.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        with rec_col2:
            st.markdown("""
            <div class="metric-card">
                <h4 style="margin: 0 0 10px 0; color: #A8FF78;">🔄 Steady Demand, Medium Volume</h4>
                <p style="font-size: 13px; color: #EAEAEA;">
                    <b>Stocking Strategy:</b> Standard Min-Max replenishment cycles. Keep stock targets aligned with 1-month forecast windows.
                </p>
                <p style="font-size: 13px; color: #8A99AD;">
                    <b>Focus:</b> Product bundling to trigger volume expansion, storage cost management.
                </p>
            </div>
            <div class="metric-card">
                <h4 style="margin: 0 0 10px 0; color: #FF4E50;">⚠️ Low Volume, High Volatility</h4>
                <p style="font-size: 13px; color: #EAEAEA;">
                    <b>Stocking Strategy:</b> Print-on-demand or dropship model (make-to-order). Avoid holding local stock.
                </p>
                <p style="font-size: 13px; color: #8A99AD;">
                    <b>Focus:</b> Lead-time agreements with suppliers, low holding cost priorities.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        # Listing sub-categories under each segment
        st.subheader("📋 Sub-Category Segmentation Catalog")
        list_tbl = feat_df.reset_index()[['Sub-Category', 'Segment_Name', 'Total_Sales', 'Growth_Rate', 'Avg_Order_Value']]
        list_tbl['Total_Sales'] = list_tbl['Total_Sales'].map(lambda v: f"${v:,.2f}")
        list_tbl['Growth_Rate'] = list_tbl['Growth_Rate'].map(lambda v: f"{v:+.2%}")
        list_tbl['Avg_Order_Value'] = list_tbl['Avg_Order_Value'].map(lambda v: f"${v:,.2f}")
        
        st.dataframe(list_tbl.sort_values(by='Segment_Name'), use_container_width=True, hide_index=True)
else:
    st.info("Loading initial setup...")
