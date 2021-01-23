---
title: "Discovery of Handwashing"
date: 2020-09-15
tags: [data manipulation, data visualization]
header:
  images:
excerpt: "Data Manipulation, Data Visualization"
mathjax: "true"
---

## Project: Discovery of Handwashing

In this notebook, we're going to reanalyze the data that made Semmelweis discover the importance of handwashing. Let's start by looking at the data that made Semmelweis realize that something was wrong with the procedures at Vienna General Hospital.


```python
# importing modules
import pandas as pd

# Read datasets/yearly_deaths_by_clinic.csv into yearly
yearly = pd.read_csv('datasets/yearly_deaths_by_clinic.csv')

# Print out yearly
print(yearly)
```

        year  births  deaths    clinic
    0   1841    3036     237  clinic 1
    1   1842    3287     518  clinic 1
    2   1843    3060     274  clinic 1
    3   1844    3157     260  clinic 1
    4   1845    3492     241  clinic 1
    5   1846    4010     459  clinic 1
    6   1841    2442      86  clinic 2
    7   1842    2659     202  clinic 2
    8   1843    2739     164  clinic 2
    9   1844    2956      68  clinic 2
    10  1845    3241      66  clinic 2
    11  1846    3754     105  clinic 2


### The alarming number of deaths

The table above shows the number of women giving birth at the two clinics at the Vienna General Hospital for the years 1841 to 1846. You'll notice that giving birth was very dangerous; an alarming number of women died as the result of childbirth, most of them from childbed fever.

We see this more clearly if we look at the proportion of deaths out of the number of women giving birth. Let's zoom in on the proportion of deaths at Clinic 1.


```python
# Calculate proportion of deaths per no. births
yearly['proportion_deaths'] = yearly['deaths'] / yearly['births']

# Extract clinic 1 data into yearly1 and clinic 2 data into yearly2
yearly1 = yearly[yearly['clinic'] == 'clinic 1']
yearly2 = yearly[yearly['clinic'] == 'clinic 2']

# Print out yearly1
print(yearly1)
```

       year  births  deaths    clinic  proportion_deaths
    0  1841    3036     237  clinic 1           0.078063
    1  1842    3287     518  clinic 1           0.157591
    2  1843    3060     274  clinic 1           0.089542
    3  1844    3157     260  clinic 1           0.082357
    4  1845    3492     241  clinic 1           0.069015
    5  1846    4010     459  clinic 1           0.114464


### Death at the clinics

If we now plot the proportion of deaths at both clinic 1 and clinic 2 we'll see a curious pattern…


```python
# This makes plots appear in the notebook
%matplotlib inline

# Plot yearly proportion of deaths at the two clinics
ax = yearly1.plot(x='year', y='proportion_deaths', label='clinic 1')
yearly2.plot(x='year', y='proportion_deaths', label='clinic 2', ax=ax)
ax.set_ylabel('Proportion deaths')
```




    Text(0, 0.5, 'Proportion deaths')




![png](output_5_1.png)


### The handwashing begins

Let's load in monthly data from Clinic 1 to see if the handwashing had any effect.


```python
# Read datasets/monthly_deaths.csv into monthly
monthly = pd.read_csv('datasets/monthly_deaths.csv', parse_dates=['date'])

# Calculate proportion of deaths per no. births
monthly['proportion_deaths'] = monthly['deaths'] / monthly['births']

# Print out the first rows in monthly
print(monthly.head(1))
```

            date  births  deaths  proportion_deaths
    0 1841-01-01     254      37           0.145669


### The effect of handwashing

With the data loaded we can now look at the proportion of deaths over time. In the plot below we haven't marked where obligatory handwashing started, but it reduced the proportion of deaths to such a degree that you should be able to spot it!


```python
# Plot monthly proportion of deaths
ax = monthly.plot(x='date', y='proportion_deaths')
ax.set_ylabel('Proportion deaths')
```




    Text(0, 0.5, 'Proportion deaths')




![png](output_9_1.png)


### The effect of handwashing highlighted

Starting from the summer of 1847 the proportion of deaths is drastically reduced and, yes, this was when Semmelweis made handwashing obligatory.

The effect of handwashing is made even more clear if we highlight this in the graph.


```python
# Date when handwashing was made mandatory
import pandas as pd
handwashing_start = pd.to_datetime('1847-06-01')

# Split monthly into before and after handwashing_start
before_washing = monthly[monthly['date'] < handwashing_start]
after_washing = monthly[monthly['date'] >= handwashing_start]

# Plot monthly proportion of deaths before and after handwashing
ax = before_washing.plot(x='date', y='proportion_deaths', label='before washing')
after_washing.plot(x='date', y='proportion_deaths', label='after washing', ax=ax)
ax.set_ylabel('Proportion deaths')
```




    Text(0, 0.5, 'Proportion deaths')




![png](output_11_1.png)


### More handwashing, fewer deaths?

Again, the graph shows that handwashing had a huge effect. How much did it reduce the monthly proportion of deaths on average?


```python
# Difference in mean monthly proportion of deaths due to handwashing
before_proportion = before_washing['proportion_deaths']
after_proportion = after_washing['proportion_deaths']
mean_diff = after_proportion.mean() - before_proportion.mean()
mean_diff
```




    -0.0839566075118334



### A Bootstrap analysis of Semmelweis handwashing data

It reduced the proportion of deaths by around 8 percentage points! From 10% on average to just 2% (which is still a high number by modern standards).

To get a feeling for the uncertainty around how much handwashing reduces mortalities we could look at a confidence interval (here calculated using the bootstrap method).


```python
# A bootstrap analysis of the reduction of deaths due to handwashing
boot_mean_diff = []
for i in range(3000):
    boot_before = before_proportion.sample(frac=1, replace=True)
    boot_after = after_proportion.sample(frac=1, replace=True)
    boot_mean_diff.append(boot_after.mean() - boot_before.mean())

# Calculating a 95% confidence interval from boot_mean_diff
confidence_interval = pd.Series(boot_mean_diff).quantile([0.025, 0.975])
confidence_interval
```




    0.025   -0.101067
    0.975   -0.066966
    dtype: float64



#### Conclusion

So handwashing reduced the proportion of deaths by between 6.7 and 10 percentage points, according to a 95% confidence interval. All in all, it would seem that Semmelweis had solid evidence that handwashing was a simple but highly effective procedure that could save many lives.


```python
# The data Semmelweis collected points to that:
doctors_should_wash_their_hands = True
```
