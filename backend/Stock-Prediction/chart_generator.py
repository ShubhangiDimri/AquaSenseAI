import pandas as pd
import matplotlib.pyplot as plt
import io
from prophet import Prophet
import matplotlib.patches as mpatches

# New import for Plotly
import plotly.graph_objects as go

# Suppress informational messages from Prophet
import logging
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

def generate_forecast_chart():
    """Generates an interactive stock forecast chart using Prophet and Plotly."""
    df = pd.read_csv('Fish_CatchNStock.csv')
    
    # Prepare data for Prophet
    df_prophet = df[['Year', 'Stock']].copy()
    df_prophet['Year'] = df_prophet['Year'].apply(lambda x: x.split('-')[0])
    df_prophet.rename(columns={'Year': 'ds', 'Stock': 'y'}, inplace=True)
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

    # Train the model
    model = Prophet()
    model.fit(df_prophet)

    # Make a 5-year future dataframe
    future = model.make_future_dataframe(periods=5, freq='Y')
    forecast = model.predict(future)

    # Create an interactive Plotly figure
    fig = go.Figure()

    # Add the lower and upper forecast bounds (the shaded area)
    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat_upper'],
        fill=None, mode='lines', line_color='rgba(0,100,80,0.2)', name='Upper Bound'
    ))
    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat_lower'],
        fill='tonexty', mode='lines', line_color='rgba(0,100,80,0.2)', name='Lower Bound'
    ))
    
    # Add the main forecast line
    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat'],
        mode='lines', line=dict(color='blue', width=4), name='Forecast'
    ))
    
    # Add the actual historical data points
    fig.add_trace(go.Scatter(
        x=df_prophet['ds'], y=df_prophet['y'],
        mode='markers', marker=dict(color='black', size=8), name='Actual Stock'
    ))

    fig.update_layout(
        title='Interactive Fish Stock 5-Year Forecast 🔮',
        xaxis_title='Year',
        yaxis_title='Stock Quantity',
        legend_title='Legend'
    )
    
    # Return the figure as a JSON object instead of an image
    return fig.to_json()


# This function remains unchanged
def generate_health_check_chart():
    """Generates the overfishing status chart with a detailed legend."""
    df = pd.read_csv('Fish_CatchNStock.csv')
    df['Year'] = df['Year'].apply(lambda x: x.split('-')[0])

    plt.figure(figsize=(12, 7))
    bar_width = 0.4
    index = pd.Index(range(len(df['Year'])))

    colors = ['red' if df['TotalCatch'][i] > 0.8 * df['Stock'][i] else 'green' for i in range(len(df))]

    stock_bar = plt.bar(index, df['Stock'], bar_width, label='Total Stock', color='skyblue')
    plt.bar(index + bar_width, df['TotalCatch'], bar_width, color=colors)
    
    red_patch = mpatches.Patch(color='red', label='Overfishing Alert (>80%)')
    green_patch = mpatches.Patch(color='green', label='Safe (<80%)')
    
    plt.legend(handles=[stock_bar, red_patch, green_patch])

    plt.xlabel('Year')
    plt.ylabel('Quantity')
    plt.title('Overfishing Status: Health Check 🩺')
    plt.xticks(index + bar_width / 2, df['Year'], rotation=45)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return buf