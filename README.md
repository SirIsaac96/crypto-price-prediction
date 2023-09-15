# CRYPTO PRICE PREDICTION USING LSTM AND GRU (Group Capstone Project)

## GROUP MEMBERS:
1. ISAAC OLAKA
2. GRACE OGAWO
3. TERESIAH NJOKI

## Problem Description
Cryptocurrency price prediction has emerged as a challenging but exciting endeavor in the ever-changing world of finance and technology. Cryptocurrencies like Bitcoin (BTC), Ethereum (ETH), and others have gained popularity as cutting-edge forms of money that utilize blockchain technology and cryptographic functions to achieve transparency, decentralization, and immutability.
In recent years, these digital currencies have drawn the interest of traders, investors and financial experts. They have significant benefits like faster transactions, lower fees and enhanced security, making them an appealing option for traditional forms of cash. However, their unusual volatility, characterized by extreme price fluctuations, comes with significant drawbacks for traders, investors, and financial experts, thereby presenting a significant hurdle to accurately predicting future price movements. The high volatility of crypto markets is usually caused by: unpredictable demand and supply, social media sentiments, news articles, political dynamics, regulatory policies, among others. This unpredictability highlights the pressing need for robust predictive models capable of assisting investors in making informed decisions to optimize profits and mitigate risks.
The significance of solving this problem cuts across multiple dimensions. For cryptocurrency investors, precise price predictions hold the key to strategically timing investment and trading decisions. In a market known for its extreme price volatility, accurate forecasting can make the difference between substantial gains and considerable losses. Moreover, this challenge extends its relevance to policy decision-makers and financial researchers, who seek insights into the behavior of cryptocurrency markets, aiming to devise well-informed strategies. In the midst of evolution of cryptocurrency market, it becomes strategically significant to predict the price movements.

## Proposed Solution
This project aims to leverage the potential of deep learning techniques, with a focus on the Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) neural network architectures in order to address this multidimensional challenge.
Traditional time-series models like AutoRegressive Integrated Moving Average (ARIMA) exhibit limitations in capturing the complex and non-linear patterns present in cryptocurrency price data. Thus, the application of advanced methodologies becomes essential to navigate the complexities of this evolving market.
Therefore, Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) are advanced models that are excellent in handling sequential data, which makes them suitable for the complex task of time-series forecasting. By training LSTM and GRU models on historical cryptocurrency data, our approach seeks to uncover the complex relationships and non-linear patterns that characterize the cryptocurrency market.
The model created through this project may help investors make wise decisions in the extremely dynamic crypto markets. It may also be useful to regulators who seek ensure a stable financial marketplace.

## Data Acquisition
The datasets for this capstone project were gathered from yahoo finance website (https://finance.yahoo.com/, accessed on August 28, 2023). The historical price data for the most popular cryptocurrencies selected are: Bitcoin (BTC) from January 2, 2015, to August 21, 2023; Ethereum (ETH) from November 10, 2017 to August 21, 2023; Ripple (XRP) from November 10, 2017 to August 21, 2023; Tether (USDT) from November 10, 2017 to August 21, 2023; and Litecoin (LTC) from January 2, 2015, to August 21, 2023.
We created a Python function to download the datasets from the Yahoo Finance API, then saved the resulting datasets as CSV files.
