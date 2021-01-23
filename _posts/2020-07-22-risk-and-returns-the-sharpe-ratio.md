---
title: "Risk and Return - the Sharpe Ratio"
date: 2020-07-22
tags: [data science, finance analysis, data visualization]
header:
  images:
excerpt: "Data Science, Finance Analysis, Data Visualization"
mathjax: "true"
---

## Project: Risk and Returns - the Sharpe Ratio

Let's learn about the Sharpe ratio by calculating it for the stocks of the two tech giants Facebook and Amazon. As benchmark we'll use the S&P 500 that measures the performance of the 500 largest stocks in the US. When we use a stock index instead of the risk-free rate, the result is called the Information Ratio and is used to benchmark the return on active portfolio management because it tells you how much more return for a given unit of risk your portfolio manager earned relative to just putting your money into a low-cost index fund.


```python
# Importing required modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Settings to produce nice plots in a Jupyter notebook
plt.style.use('fivethirtyeight')
%matplotlib inline

# Reading in the data
stock_data = pd.read_csv('datasets/stock_data.csv', parse_dates=['Date'], index_col=['Date']).dropna()
benchmark_data = pd.read_csv('datasets/benchmark_data.csv', parse_dates=['Date'], index_col=['Date']).dropna()
```

### A first glance at the data

Let's take a look the data to find out how many observations and variables we have at our disposal.


```python
# Display summary for stock_data
print('Stocks\n')
stock_data.info()
print(stock_data.head())

# Display summary for benchmark_data
print('\nBenchmarks\n')
benchmark_data.info()
print(benchmark_data.head())
```

    Stocks

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 252 entries, 2016-01-04 to 2016-12-30
    Data columns (total 2 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   Amazon    252 non-null    float64
     1   Facebook  252 non-null    float64
    dtypes: float64(2)
    memory usage: 5.9 KB
                    Amazon    Facebook
    Date                              
    2016-01-04  636.989990  102.220001
    2016-01-05  633.789978  102.730003
    2016-01-06  632.650024  102.970001
    2016-01-07  607.940002   97.919998
    2016-01-08  607.049988   97.330002

    Benchmarks

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 252 entries, 2016-01-04 to 2016-12-30
    Data columns (total 1 columns):
     #   Column   Non-Null Count  Dtype  
    ---  ------   --------------  -----  
     0   S&P 500  252 non-null    float64
    dtypes: float64(1)
    memory usage: 3.9 KB
                S&P 500
    Date               
    2016-01-04  2012.66
    2016-01-05  2016.71
    2016-01-06  1990.26
    2016-01-07  1943.09
    2016-01-08  1922.03


### Plot and summarize daily prices for Amazon and Facebook

Before we compare an investment in either Facebook or Amazon with the index of the 500 largest companies in the US, let's visualize the data, so we better understand what we're dealing with.


```python
# visualize the stock_data
stock_data.plot(title='Stock Data', subplots=True)


# summarize the stock_data
stock_data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Amazon</th>
      <th>Facebook</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>252.000000</td>
      <td>252.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>699.523135</td>
      <td>117.035873</td>
    </tr>
    <tr>
      <th>std</th>
      <td>92.362312</td>
      <td>8.899858</td>
    </tr>
    <tr>
      <th>min</th>
      <td>482.070007</td>
      <td>94.160004</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>606.929993</td>
      <td>112.202499</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>727.875000</td>
      <td>117.765000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>767.882492</td>
      <td>123.902503</td>
    </tr>
    <tr>
      <th>max</th>
      <td>844.359985</td>
      <td>133.279999</td>
    </tr>
  </tbody>
</table>
</div>




![png](output_5_1.png)


### Visualize and summarize daily values for S&P 500

Let's also take a closer look at the value of the S&P 500, our benchmark.


```python
# plot the benchmark_data
benchmark_data.plot(title='Benchmark Data')

# summarize the benchmark_data
benchmark_data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>S&amp;P 500</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>252.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2094.651310</td>
    </tr>
    <tr>
      <th>std</th>
      <td>101.427615</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1829.080000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2047.060000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2104.105000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2169.075000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2271.720000</td>
    </tr>
  </tbody>
</table>
</div>




![png](output_7_1.png)


### The inputs for the Sharpe Ratio: starting with daily stock returns

The Sharpe Ratio uses the difference in returns between the two investment opportunities under consideration.

However, our data show the historical value of each investment, not the return. To calculate the return, we need to calculate the percentage change in value from one day to the next. We'll also take a look at the summary statistics because these will become our inputs as we calculate the Sharpe Ratio.


```python
# calculate daily stock_data returns
stock_returns = stock_data.pct_change()

# plot the daily returns
stock_returns.plot(title='Stock Return')

# summarize the daily returns
stock_returns.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Amazon</th>
      <th>Facebook</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>251.000000</td>
      <td>251.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.000818</td>
      <td>0.000626</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.018383</td>
      <td>0.017840</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-0.076100</td>
      <td>-0.058105</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.007211</td>
      <td>-0.007220</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000857</td>
      <td>0.000879</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.009224</td>
      <td>0.008108</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.095664</td>
      <td>0.155214</td>
    </tr>
  </tbody>
</table>
</div>




![png](output_9_1.png)


### Daily S&P 500 returns

For the S&P 500, calculating daily returns works just the same way, we just need to make sure we select it as a Series using single brackets [] and not as a DataFrame to facilitate the calculations in the next step.


```python
# calculate daily benchmark_data returns
sp_returns = benchmark_data['S&P 500'].pct_change()

# plot the daily returns
sp_returns.plot(title='Daily S&P 500 Returns')

# summarize the daily returns
sp_returns.describe()
```




    count    251.000000
    mean       0.000458
    std        0.008205
    min       -0.035920
    25%       -0.002949
    50%        0.000205
    75%        0.004497
    max        0.024760
    Name: S&P 500, dtype: float64




![png](output_11_1.png)


### Calculating excess returns for Amazon and Facebook vs. S&P 500

Next, we need to calculate the relative performance of stocks vs. the S&P 500 benchmark. This is calculated as the difference in returns between stock_returns and sp_returns for each day.


```python
# calculate the difference in daily returns
excess_returns = stock_returns.sub(sp_returns, axis=0)

# plot the excess_returns
excess_returns.plot(title='Difference in Daily Returns')

# summarize the excess_returns
excess_returns.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Amazon</th>
      <th>Facebook</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>251.000000</td>
      <td>251.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.000360</td>
      <td>0.000168</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.016126</td>
      <td>0.015439</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-0.100860</td>
      <td>-0.051958</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.006229</td>
      <td>-0.005663</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000698</td>
      <td>-0.000454</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.007351</td>
      <td>0.005814</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.100728</td>
      <td>0.149686</td>
    </tr>
  </tbody>
</table>
</div>




![png](output_13_1.png)


### The Sharpe Ratio: the average difference in daily returns

Now we can finally start computing the Sharpe Ratio. First we need to calculate the average of the excess_returns. This tells us how much more or less the investment yields per day compared to the benchmark.


```python
# calculate the mean of excess_returns
avg_excess_return = excess_returns.mean()

# plot avg_excess_returns
avg_excess_return.plot.bar(title='Mean of the Return Difference')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x286dcdbbf88>




![png](output_15_1.png)


### The Sharpe Ration: standard deviation of the return difference

It looks like there was quite a bit of a difference between average daily returns for Amazon and Facebook.

Next, we calculate the standard deviation of the excess_returns. This shows us the amount of risk an investment in the stocks implies as compared to an investment in the S&P 500.


```python
# calculate the standard deviations
sd_excess_return = excess_returns.std()

# plot the standard deviations
sd_excess_return.plot.bar(title='Standard Deviation of the Return Difference')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x286dcdca1c8>




![png](output_17_1.png)


### Putting it all together

Now we just need to compute the ratio of avg_excess_returns and sd_excess_returns. The result is now finally the Sharpe ratio and indicates how much more (or less) return the investment opportunity under consideration yields per unit of risk.

The Sharpe Ratio is often annualized by multiplying it by the square root of the number of periods. We have used daily data as input, so we'll use the square root of the number of trading days (5 days, 52 weeks, minus a few holidays)


```python
# calculate the daily sharpe ratio
daily_sharpe_ratio = avg_excess_return.div(sd_excess_return)

# annualize the sharpe ratio
annual_factor = np.sqrt(252)
annual_sharpe_ratio = daily_sharpe_ratio * annual_factor

# plot the annualized sharpe ratio
annual_sharpe_ratio.plot(title='Annualized Sharpe Ratio: Stock vs S&P 500')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x286dceb3048>




![png](output_19_1.png)


### Conclusion

Given the two Sharpe ratios, which investment should we go for? In 2016, Amazon had a Sharpe ratio twice as high as Facebook. This means that an investment in Amazon returned twice as much compared to the S&P 500 for each unit of risk an investor would have assumed. In other words, in risk-adjusted terms, the investment in Amazon would have been more attractive.

This difference was mostly driven by differences in return rather than risk between Amazon and Facebook. The risk of choosing Amazon over FB (as measured by the standard deviation) was only slightly higher so that the higher Sharpe ratio for Amazon ends up higher mainly due to the higher average daily returns for Amazon.

When faced with investment alternatives that offer both different returns and risks, the Sharpe Ratio helps to make a decision by adjusting the returns by the differences in risk and allows an investor to compare investment opportunities on equal terms, that is, on an 'apples-to-apples' basis.
