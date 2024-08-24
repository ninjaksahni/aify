import streamlit as st
import os
from dotenv import load_dotenv
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from asset_sentiment_analyzer import SentimentAnalyzer


# Load environment variables from .env file
#load_dotenv()
#openai_key = os.getenv('OPENAI_API_KEY')

# Load the OpenAI API key from Streamlit secrets
openai_key = st.secrets["openai"]["api_key"]

# Check if the API key is loaded
if not openai_key:
    st.error("OpenAI API key is missing.")
else:
    # Add custom CSS
    st.markdown("""
        <style>
        .big-font {
            font-size:40px !important;
            color: #FF6347;
        }
        </style>
        """, unsafe_allow_html=True)

    # Initialize Streamlit app
    st.markdown('<p class="big-font">Stock Sentiment Analysis KSVX</p>', unsafe_allow_html=True)

    # Create columns for layout
    col1, col2 = st.columns([2, 1])

    with col1:
        # Input for stock asset
        asset = st.text_input('Enter the stock asset symbol (e.g., AAPL for Apple or ZOMATO.NS for Indian stocks or ^NSEI for Index):', 'NVDA')

        # Input for date using calendar
        date_input = st.date_input('Select the date:', value=datetime.now())

        # Input for number of days prior
        days_prior = st.slider('Select number of days prior to analyze:', min_value=1, max_value=4, value=3)

        # Explain performance implications
        st.write(f"**Performance Note:** The analysis for 1 day will be faster, while leading to 4 days may take longer.")

        # Define the date range based on days prior
        end_date = date_input
        start_date = end_date - timedelta(days=days_prior)

        # Convert the selected dates to the required format for SentimentAnalyzer
        formatted_end_date = end_date.strftime('%m/%d/%Y')
        formatted_start_date = start_date.strftime('%m/%d/%Y')

        # Format the dates for display
        formatted_display_end_date = end_date.strftime('%A, %d %B %Y')
        formatted_display_start_date = start_date.strftime('%A, %d %B %Y')

        # Show formatted date range
        st.write(f"**Selected Date Range:** {formatted_display_start_date} to {formatted_display_end_date}")

        def fetch_and_display_stock_data():
            try:
                # Fetch stock data from Yahoo Finance
                stock = yf.Ticker(asset)
                stock_info = stock.history(period='1d')  # Fetch the latest data
                
                # Check if stock_info is not empty
                if not stock_info.empty:
                    current_price = stock_info['Close'].iloc[-1]
                else:
                    current_price = 'N/A'
                
                # Get stock info
                stock_info_full = stock.info
                stock_name = stock_info_full.get('shortName', 'N/A')
                last_traded_price = stock_info_full.get('regularMarketPrice', 'N/A')

                # Ensure prices are numbers and format them
                try:
                    current_price = float(current_price)
                except ValueError:
                    current_price = 'N/A'
                try:
                    last_traded_price = float(last_traded_price)
                except ValueError:
                    last_traded_price = 'N/A'

                # Display stock data
                st.write(f"### Stock Information for {asset}")
                st.write(f"**Name:** {stock_name}")
                st.write(f"**Current Price:** ${current_price:.2f}" if current_price != 'N/A' else "**Current Price:** N/A")
                st.write(f"**Last Traded Price:** ${last_traded_price:.2f}" if last_traded_price != 'N/A' else "**Last Traded Price:** N/A")

                # Fetch 1-year historical data
                historical_data = stock.history(period='1y')
                if not historical_data.empty and 'Close' in historical_data.columns:
                    # Calculate Moving Averages
                    historical_data['50_MA'] = historical_data['Close'].rolling(window=50).mean()
                    historical_data['200_MA'] = historical_data['Close'].rolling(window=200).mean()
                    
                    # Calculate RSI
                    delta = historical_data['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    historical_data['RSI'] = 100 - (100 / (1 + rs))

                    # Plot price and moving averages
                    fig_price = go.Figure()
                    fig_price.add_trace(go.Scatter(x=historical_data.index, y=historical_data['Close'], mode='lines+markers', name='Price'))
                    fig_price.add_trace(go.Scatter(x=historical_data.index, y=historical_data['50_MA'], mode='lines', name='50-Day MA', line=dict(color='orange')))
                    fig_price.add_trace(go.Scatter(x=historical_data.index, y=historical_data['200_MA'], mode='lines', name='200-Day MA', line=dict(color='red')))
                    
                    fig_price.update_layout(
                        title=f'{asset} Price Trend with Moving Averages (Last 1 Year)',
                        xaxis_title='Date',
                        yaxis_title='Price',
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='#FFFFFF'
                    )
                    st.plotly_chart(fig_price)

                    # Plot RSI
                    fig_rsi = go.Figure()
                    fig_rsi.add_trace(go.Scatter(x=historical_data.index, y=historical_data['RSI'], mode='lines', name='RSI'))
                    fig_rsi.add_hline(y=70, line=dict(color='red', dash='dash'), annotation_text='Overbought', annotation_position='top left')
                    fig_rsi.add_hline(y=30, line=dict(color='green', dash='dash'), annotation_text='Oversold', annotation_position='bottom left')

                    fig_rsi.update_layout(
                        title=f'{asset} RSI (Last 1 Year)',
                        xaxis_title='Date',
                        yaxis_title='RSI',
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='#FFFFFF'
                    )
                    st.plotly_chart(fig_rsi)

                    # Plot Volume
                    fig_volume = go.Figure()
                    fig_volume.add_trace(go.Bar(x=historical_data.index, y=historical_data['Volume'], name='Volume', marker_color='lightblue'))

                    fig_volume.update_layout(
                        title=f'{asset} Trading Volume (Last 1 Year)',
                        xaxis_title='Date',
                        yaxis_title='Volume',
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='#FFFFFF'
                    )
                    st.plotly_chart(fig_volume)
                else:
                    st.write("No historical data available for the last year.")

            except Exception as e:
                st.error(f"An error occurred while fetching stock data: {e}")

        # Button to analyze sentiment and fetch stock data
        if st.button('Analyze Sentiment'):
            # Fetch and display stock data immediately
            fetch_and_display_stock_data()
            
            try:
                # Initialize SentimentAnalyzer with the asset and OpenAI key
                analyzer = SentimentAnalyzer(asset=asset, openai_key=openai_key)

                # Fetch and analyze sentiment for each date in the range
                sentiment_data = []
                today_date = datetime.now().date()  # Get today's date

                for single_date in (start_date + timedelta(n) for n in range(days_prior + 1)):
                    formatted_date = single_date.strftime('%m/%d/%Y')
                    sentiment = analyzer.get_sentiment(date=formatted_date)
                    sentiment_data.append({
                        'Date': single_date.strftime('%A, %d %B %Y'),  # Formatted date
                        'Sentiment': sentiment.capitalize()
                    })

                # Display sentiment analysis results in a table
                sentiment_df = pd.DataFrame(sentiment_data)

                # Display the sentiment analysis table without custom CSS
                st.write("### Sentiment Analysis Table")
                st.write(sentiment_df.to_html(escape=False, index=False), unsafe_allow_html=True)

                # Fetch news links and display them in a table
                try:
                    news_links = analyzer.fetch_news_links(news_date=formatted_end_date)

                    st.write(f"### News Links for {formatted_display_end_date}")
                    if news_links:
                        news_df = pd.DataFrame({
                            "Link": [f"[News {i+1}]({url})" for i, url in enumerate(news_links)],
                            "View Content": [f'<a href="{url}" target="_blank">Open Content</a>' for url in news_links]
                        })
                        # Display the news links and content in the table
                        st.write(news_df.to_html(escape=False, index=False), unsafe_allow_html=True)
                    else:
                        st.write("No news links available for the selected date.")

                except Exception as e:
                    st.error(f"An error occurred while fetching news links: {e}")

                # Generate and display the report with the selected date
                st.write(f"### News Analysis Report for {formatted_display_end_date}")
                report = analyzer.produce_daily_report(date=formatted_end_date, max_words=150)
                st.write(report)

            except ValueError as e:
                st.error(f"An error occurred: {e}")
