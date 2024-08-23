import streamlit as st
import os
from dotenv import load_dotenv
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from asset_sentiment_analyzer import SentimentAnalyzer
from datetime import datetime
import calendar

# Load environment variables from .env file
load_dotenv()
openai_key = os.getenv('OPENAI_API_KEY')

# Check if the API key is loaded
if not openai_key:
    st.error("OpenAI API key is missing. Please check your .env file.")
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

    # Display a logo
    st.image('https://github.com/ninjaksahni/aify/blob/44404d018aafbea07f654276f9c5a8fd0b5df44e/2.png', width=150)

    # Initialize Streamlit app
    st.markdown('<p class="big-font">Stock Sentiment Analysis KSVX</p>', unsafe_allow_html=True)

    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    with col1:
        # Input for stock asset
        asset = st.text_input('Enter the stock asset symbol (e.g., AAPL for Apple or ZOMATO.NS for Indian stocks):', 'AAPL')

        # Input for date using calendar
        date_input = st.date_input('Select the date:', value=datetime.now())
        
        # Convert the selected date to the required format
        formatted_date = date_input.strftime('%m/%d/%Y')
        date_for_display = date_input.strftime('%d/%m/%Y')

        # Format the date for display
        day_of_week = calendar.day_name[date_input.weekday()]
        day_of_month = date_input.day
        month_name = date_input.strftime('%B')
        year = date_input.year
        formatted_display_date = f"{day_of_week}, {day_of_month} {month_name} {year}"

        # Show formatted date
        st.write(f"**Selected Date:** {formatted_display_date}")

        if st.button('Analyze Sentiment'):
            try:
                # Initialize SentimentAnalyzer with the asset and OpenAI key
                analyzer = SentimentAnalyzer(asset=asset, openai_key=openai_key)
                
                # Fetch sentiment analysis
                sentiment = analyzer.get_sentiment(date=formatted_date)
                
                # Show sentiment prominently
                st.write(f"## Sentiment Analysis")
                sentiment_color = "green" if sentiment == "bullish" else "red" if sentiment == "bearish" else "grey"
                st.markdown(f"<h2 style='color:{sentiment_color};'>{sentiment.capitalize()} Sentiment</h2>", unsafe_allow_html=True)

                # Fetch stock price from Yahoo Finance
                stock = yf.Ticker(asset)  # Use the full ticker symbol as provided
                try:
                    stock_info = stock.history(period="1d", start=date_input, end=date_input)
                    
                    if not stock_info.empty and 'Close' in stock_info.columns:
                        stock_price = stock_info['Close'][0]
                        st.metric(label="Current Stock Price", value=f"${stock_price:.2f}")
                        
                        # Plot stock price chart
                        historical_data = stock.history(period="1mo")  # Adjust the period as needed
                        if not historical_data.empty and 'Close' in historical_data.columns:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=historical_data.index, y=historical_data['Close'], mode='lines+markers', name='Price'))
                            fig.update_layout(
                                title=f'{asset} Price Trend',
                                xaxis_title='Date',
                                yaxis_title='Price',
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font_color='#FFFFFF'
                            )
                            st.plotly_chart(fig)
                        else:
                            st.write("No historical data available for the selected period.")
                    else:
                        st.write("No stock data available for the selected date.")
                except Exception as e:
                    st.error(f"An error occurred while fetching stock data: {e}")

                # Fetch news links
                try:
                    news_links = analyzer.fetch_news_links(news_date=formatted_date)
                    
                    with st.sidebar.expander("News Links", expanded=False):
                        if news_links:
                            for url in news_links:
                                st.sidebar.write(url)
                        else:
                            st.sidebar.write("No news links available for the selected date.")
                    
                    # Display news content in sidebar
                    with st.sidebar.expander("News Content", expanded=False):
                        if news_links:
                            for url in news_links:
                                content = analyzer.show_news_content(url)
                                st.sidebar.write(f"**URL**: {url}")
                                st.sidebar.write(content)
                        else:
                            st.sidebar.write("No news content available for the selected date.")
                except Exception as e:
                    st.error(f"An error occurred while fetching news links: {e}")
                
                # Generate and display the report (reduce by 50%)
                report = analyzer.produce_daily_report(date=formatted_date, max_words=150)
                st.write("### Daily Report")
                st.write(report)

            except ValueError as e:
                st.error(f"An error occurred: {e}")
