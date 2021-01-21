---
title: "Classify Song Genres from Audio Data"
date: 2020-11-15
tags: [machine learning, data science]
header:
  image: 
excerpt: "Machine Learning, Data Science"
mathjax: "true"
---


## Classify Song Genres from Audio Data

Over the past few years, streaming services with huge catalogs have become the primary means through which most people listen to their favorite music. But at the same time, the sheer amount of music on offer can mean users might be a bit overwhelmed when trying to look for newer music that suits their tastes. For this reason, streaming services have looked into means of categorizing music to allow personalize recommendations. One method involves direct analysis of the raw audio information in a given song, scoring the raw data on a variety of metrics.

Our goal is to look through this dataset and classify songs as being 'Hip-Hop' or 'Rock' all without listening to a single one ourselves. In doing so, we will learn how to clean our data, do some exploratory data visualization, and use feature reduction towards the goal of feeding our data through some simple machine learning algorithm such as decision tree and logistic regression.

Let's start by creating two pandas DataFrames and merge so we have features and label (often referred to as x and y) for the classification later on.


```python
import pandas as pd

# Read in track metadata with genre labels
tracks = pd.read_csv('datasets/fma-rock-vs-hiphop.csv')

# Read in track metrics with the features
echonest_metrics = pd.read_json('datasets/echonest-metrics.json', precise_float=True)

# Merge the relevant columns of tracks and echonest_metrics
echo_tracks = echonest_metrics.merge(tracks[['genre_top', 'track_id']], on='track_id')

# Inspect the resultant dataframe
echo_tracks.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 4802 entries, 0 to 4801
    Data columns (total 10 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   track_id          4802 non-null   int64  
     1   acousticness      4802 non-null   float64
     2   danceability      4802 non-null   float64
     3   energy            4802 non-null   float64
     4   instrumentalness  4802 non-null   float64
     5   liveness          4802 non-null   float64
     6   speechiness       4802 non-null   float64
     7   tempo             4802 non-null   float64
     8   valence           4802 non-null   float64
     9   genre_top         4802 non-null   object
    dtypes: float64(8), int64(1), object(1)
    memory usage: 412.7+ KB



```python
# Print DataFrame
echo_tracks
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
      <th>track_id</th>
      <th>acousticness</th>
      <th>danceability</th>
      <th>energy</th>
      <th>instrumentalness</th>
      <th>liveness</th>
      <th>speechiness</th>
      <th>tempo</th>
      <th>valence</th>
      <th>genre_top</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>0.416675</td>
      <td>0.675894</td>
      <td>0.634476</td>
      <td>1.062807e-02</td>
      <td>0.177647</td>
      <td>0.159310</td>
      <td>165.922</td>
      <td>0.576661</td>
      <td>Hip-Hop</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>0.374408</td>
      <td>0.528643</td>
      <td>0.817461</td>
      <td>1.851103e-03</td>
      <td>0.105880</td>
      <td>0.461818</td>
      <td>126.957</td>
      <td>0.269240</td>
      <td>Hip-Hop</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>0.043567</td>
      <td>0.745566</td>
      <td>0.701470</td>
      <td>6.967990e-04</td>
      <td>0.373143</td>
      <td>0.124595</td>
      <td>100.260</td>
      <td>0.621661</td>
      <td>Hip-Hop</td>
    </tr>
    <tr>
      <th>3</th>
      <td>134</td>
      <td>0.452217</td>
      <td>0.513238</td>
      <td>0.560410</td>
      <td>1.944269e-02</td>
      <td>0.096567</td>
      <td>0.525519</td>
      <td>114.290</td>
      <td>0.894072</td>
      <td>Hip-Hop</td>
    </tr>
    <tr>
      <th>4</th>
      <td>153</td>
      <td>0.988306</td>
      <td>0.255661</td>
      <td>0.979774</td>
      <td>9.730057e-01</td>
      <td>0.121342</td>
      <td>0.051740</td>
      <td>90.241</td>
      <td>0.034018</td>
      <td>Rock</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4797</th>
      <td>124718</td>
      <td>0.412194</td>
      <td>0.686825</td>
      <td>0.849309</td>
      <td>6.000000e-10</td>
      <td>0.867543</td>
      <td>0.367315</td>
      <td>96.104</td>
      <td>0.692414</td>
      <td>Hip-Hop</td>
    </tr>
    <tr>
      <th>4798</th>
      <td>124719</td>
      <td>0.054973</td>
      <td>0.617535</td>
      <td>0.728567</td>
      <td>7.215700e-06</td>
      <td>0.131438</td>
      <td>0.243130</td>
      <td>96.262</td>
      <td>0.399720</td>
      <td>Hip-Hop</td>
    </tr>
    <tr>
      <th>4799</th>
      <td>124720</td>
      <td>0.010478</td>
      <td>0.652483</td>
      <td>0.657498</td>
      <td>7.098000e-07</td>
      <td>0.701523</td>
      <td>0.229174</td>
      <td>94.885</td>
      <td>0.432240</td>
      <td>Hip-Hop</td>
    </tr>
    <tr>
      <th>4800</th>
      <td>124721</td>
      <td>0.067906</td>
      <td>0.432421</td>
      <td>0.764508</td>
      <td>1.625500e-06</td>
      <td>0.104412</td>
      <td>0.310553</td>
      <td>171.329</td>
      <td>0.580087</td>
      <td>Hip-Hop</td>
    </tr>
    <tr>
      <th>4801</th>
      <td>124722</td>
      <td>0.153518</td>
      <td>0.638660</td>
      <td>0.762567</td>
      <td>5.000000e-10</td>
      <td>0.264847</td>
      <td>0.303372</td>
      <td>77.842</td>
      <td>0.656612</td>
      <td>Hip-Hop</td>
    </tr>
  </tbody>
</table>
<p>4802 rows × 10 columns</p>
</div>



### Pairwise relationship between continous variables

We typicallly want to avoid using variables that have strong correlations with each other - hence avoiding feature redundancy for a few reasons:

* To keep the model simple and improve interpretability (with many features, we run risk of overfitting)
* When our datasets are very large, using fewer features can drastically speed up computation time

To get a sense of whether tere are any strong correlated features in our dataframe, we will use build-in function in Pandas package.


```python
# Create a correlation matrix
corr_metrics = echonest_metrics.corr()
corr_metrics.style.background_gradient()
```




<style  type="text/css" >
    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow0_col0 {
            background-color:  #023858;
            color:  #f1f1f1;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow0_col1 {
            background-color:  #eae6f1;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow0_col2 {
            background-color:  #d2d2e7;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow0_col3 {
            background-color:  #9ab8d8;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow0_col4 {
            background-color:  #fff7fb;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow0_col5 {
            background-color:  #ede7f2;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow0_col6 {
            background-color:  #eee8f3;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow0_col7 {
            background-color:  #f0eaf4;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow0_col8 {
            background-color:  #e8e4f0;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow1_col0 {
            background-color:  #fff7fb;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow1_col1 {
            background-color:  #023858;
            color:  #f1f1f1;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow1_col2 {
            background-color:  #fff7fb;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow1_col3 {
            background-color:  #fff7fb;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow1_col4 {
            background-color:  #bdc8e1;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow1_col5 {
            background-color:  #e4e1ef;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow1_col6 {
            background-color:  #d9d8ea;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow1_col7 {
            background-color:  #fff7fb;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow1_col8 {
            background-color:  #f7f0f7;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow2_col0 {
            background-color:  #c0c9e2;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow2_col1 {
            background-color:  #dddbec;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow2_col2 {
            background-color:  #023858;
            color:  #f1f1f1;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow2_col3 {
            background-color:  #adc1dd;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow2_col4 {
            background-color:  #ece7f2;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow2_col5 {
            background-color:  #fff7fb;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow2_col6 {
            background-color:  #b9c6e0;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow2_col7 {
            background-color:  #fdf5fa;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow2_col8 {
            background-color:  #73a9cf;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow3_col0 {
            background-color:  #bbc7e0;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow3_col1 {
            background-color:  #fff7fb;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow3_col2 {
            background-color:  #dcdaeb;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow3_col3 {
            background-color:  #023858;
            color:  #f1f1f1;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow3_col4 {
            background-color:  #d7d6e9;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow3_col5 {
            background-color:  #e3e0ee;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow3_col6 {
            background-color:  #e2dfee;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow3_col7 {
            background-color:  #bfc9e1;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow3_col8 {
            background-color:  #b9c6e0;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow4_col0 {
            background-color:  #fff7fb;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow4_col1 {
            background-color:  #9ebad9;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow4_col2 {
            background-color:  #f6eff7;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow4_col3 {
            background-color:  #b8c6e0;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow4_col4 {
            background-color:  #023858;
            color:  #f1f1f1;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow4_col5 {
            background-color:  #f4eef6;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow4_col6 {
            background-color:  #fff7fb;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow4_col7 {
            background-color:  #ede8f3;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow4_col8 {
            background-color:  #fff7fb;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow5_col0 {
            background-color:  #d8d7e9;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow5_col1 {
            background-color:  #afc1dd;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow5_col2 {
            background-color:  #faf2f8;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow5_col3 {
            background-color:  #adc1dd;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow5_col4 {
            background-color:  #e1dfed;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow5_col5 {
            background-color:  #023858;
            color:  #f1f1f1;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow5_col6 {
            background-color:  #d3d4e7;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow5_col7 {
            background-color:  #f1ebf5;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow5_col8 {
            background-color:  #eee9f3;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow6_col0 {
            background-color:  #e4e1ef;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow6_col1 {
            background-color:  #afc1dd;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow6_col2 {
            background-color:  #bfc9e1;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow6_col3 {
            background-color:  #b9c6e0;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow6_col4 {
            background-color:  #f7f0f7;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow6_col5 {
            background-color:  #dedcec;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow6_col6 {
            background-color:  #023858;
            color:  #f1f1f1;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow6_col7 {
            background-color:  #ece7f2;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow6_col8 {
            background-color:  #d9d8ea;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow7_col0 {
            background-color:  #d6d6e9;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow7_col1 {
            background-color:  #d1d2e6;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow7_col2 {
            background-color:  #f3edf5;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow7_col3 {
            background-color:  #7dacd1;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow7_col4 {
            background-color:  #d2d3e7;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow7_col5 {
            background-color:  #ede8f3;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow7_col6 {
            background-color:  #dad9ea;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow7_col7 {
            background-color:  #023858;
            color:  #f1f1f1;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow7_col8 {
            background-color:  #d2d3e7;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow8_col0 {
            background-color:  #d3d4e7;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow8_col1 {
            background-color:  #cccfe5;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow8_col2 {
            background-color:  #69a5cc;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow8_col3 {
            background-color:  #80aed2;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow8_col4 {
            background-color:  #efe9f3;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow8_col5 {
            background-color:  #eee9f3;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow8_col6 {
            background-color:  #ced0e6;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow8_col7 {
            background-color:  #d8d7e9;
            color:  #000000;
        }    #T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow8_col8 {
            background-color:  #023858;
            color:  #f1f1f1;
        }</style><table id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00c" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >track_id</th>        <th class="col_heading level0 col1" >acousticness</th>        <th class="col_heading level0 col2" >danceability</th>        <th class="col_heading level0 col3" >energy</th>        <th class="col_heading level0 col4" >instrumentalness</th>        <th class="col_heading level0 col5" >liveness</th>        <th class="col_heading level0 col6" >speechiness</th>        <th class="col_heading level0 col7" >tempo</th>        <th class="col_heading level0 col8" >valence</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00clevel0_row0" class="row_heading level0 row0" >track_id</th>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow0_col0" class="data row0 col0" >1.000000</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow0_col1" class="data row0 col1" >-0.279829</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow0_col2" class="data row0 col2" >0.102056</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow0_col3" class="data row0 col3" >0.121991</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow0_col4" class="data row0 col4" >-0.283206</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow0_col5" class="data row0 col5" >-0.004059</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow0_col6" class="data row0 col6" >-0.075077</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow0_col7" class="data row0 col7" >0.004313</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow0_col8" class="data row0 col8" >0.020201</td>
            </tr>
            <tr>
                        <th id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00clevel0_row1" class="row_heading level0 row1" >acousticness</th>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow1_col0" class="data row1 col0" >-0.279829</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow1_col1" class="data row1 col1" >1.000000</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow1_col2" class="data row1 col2" >-0.189599</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow1_col3" class="data row1 col3" >-0.477273</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow1_col4" class="data row1 col4" >0.110033</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow1_col5" class="data row1 col5" >0.041319</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow1_col6" class="data row1 col6" >0.038785</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow1_col7" class="data row1 col7" >-0.110701</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow1_col8" class="data row1 col8" >-0.085436</td>
            </tr>
            <tr>
                        <th id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00clevel0_row2" class="row_heading level0 row2" >danceability</th>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow2_col0" class="data row2 col0" >0.102056</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow2_col1" class="data row2 col1" >-0.189599</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow2_col2" class="data row2 col2" >1.000000</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow2_col3" class="data row2 col3" >0.045345</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow2_col4" class="data row2 col4" >-0.118033</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow2_col5" class="data row2 col5" >-0.143339</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow2_col6" class="data row2 col6" >0.171311</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow2_col7" class="data row2 col7" >-0.094352</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow2_col8" class="data row2 col8" >0.428515</td>
            </tr>
            <tr>
                        <th id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00clevel0_row3" class="row_heading level0 row3" >energy</th>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow3_col0" class="data row3 col0" >0.121991</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow3_col1" class="data row3 col1" >-0.477273</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow3_col2" class="data row3 col2" >0.045345</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow3_col3" class="data row3 col3" >1.000000</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow3_col4" class="data row3 col4" >-0.002412</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow3_col5" class="data row3 col5" >0.045752</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow3_col6" class="data row3 col6" >-0.008645</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow3_col7" class="data row3 col7" >0.227324</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow3_col8" class="data row3 col8" >0.219384</td>
            </tr>
            <tr>
                        <th id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00clevel0_row4" class="row_heading level0 row4" >instrumentalness</th>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow4_col0" class="data row4 col0" >-0.283206</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow4_col1" class="data row4 col1" >0.110033</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow4_col2" class="data row4 col2" >-0.118033</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow4_col3" class="data row4 col3" >-0.002412</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow4_col4" class="data row4 col4" >1.000000</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow4_col5" class="data row4 col5" >-0.058593</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow4_col6" class="data row4 col6" >-0.216689</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow4_col7" class="data row4 col7" >0.023003</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow4_col8" class="data row4 col8" >-0.145200</td>
            </tr>
            <tr>
                        <th id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00clevel0_row5" class="row_heading level0 row5" >liveness</th>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow5_col0" class="data row5 col0" >-0.004059</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow5_col1" class="data row5 col1" >0.041319</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow5_col2" class="data row5 col2" >-0.143339</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow5_col3" class="data row5 col3" >0.045752</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow5_col4" class="data row5 col4" >-0.058593</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow5_col5" class="data row5 col5" >1.000000</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow5_col6" class="data row5 col6" >0.073104</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow5_col7" class="data row5 col7" >-0.007566</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow5_col8" class="data row5 col8" >-0.017886</td>
            </tr>
            <tr>
                        <th id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00clevel0_row6" class="row_heading level0 row6" >speechiness</th>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow6_col0" class="data row6 col0" >-0.075077</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow6_col1" class="data row6 col1" >0.038785</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow6_col2" class="data row6 col2" >0.171311</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow6_col3" class="data row6 col3" >-0.008645</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow6_col4" class="data row6 col4" >-0.216689</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow6_col5" class="data row6 col5" >0.073104</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow6_col6" class="data row6 col6" >1.000000</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow6_col7" class="data row6 col7" >0.032188</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow6_col8" class="data row6 col8" >0.094794</td>
            </tr>
            <tr>
                        <th id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00clevel0_row7" class="row_heading level0 row7" >tempo</th>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow7_col0" class="data row7 col0" >0.004313</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow7_col1" class="data row7 col1" >-0.110701</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow7_col2" class="data row7 col2" >-0.094352</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow7_col3" class="data row7 col3" >0.227324</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow7_col4" class="data row7 col4" >0.023003</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow7_col5" class="data row7 col5" >-0.007566</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow7_col6" class="data row7 col6" >0.032188</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow7_col7" class="data row7 col7" >1.000000</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow7_col8" class="data row7 col8" >0.129911</td>
            </tr>
            <tr>
                        <th id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00clevel0_row8" class="row_heading level0 row8" >valence</th>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow8_col0" class="data row8 col0" >0.020201</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow8_col1" class="data row8 col1" >-0.085436</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow8_col2" class="data row8 col2" >0.428515</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow8_col3" class="data row8 col3" >0.219384</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow8_col4" class="data row8 col4" >-0.145200</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow8_col5" class="data row8 col5" >-0.017886</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow8_col6" class="data row8 col6" >0.094794</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow8_col7" class="data row8 col7" >0.129911</td>
                        <td id="T_74612a2e_5944_11eb_8c8a_b0fc3678f00crow8_col8" class="data row8 col8" >1.000000</td>
            </tr>
    </tbody></table>



### Normalizing the feature data

As mentioned earlier, it can be particularly useful to simplify our models and use as few features as necessary to achieve the best result. Since we didn't find any particular strong correlations between our features, we can instead use a common approach to reduce the number of features called principal component analysis (PCA).

However, since PCA uses the absolute variance of a feature to rotate the data, a feature with a broader range of values will overpower and bias the algorithm relative to the other features. To avoid this, we must first normalize our data. There are a few methods to do this, but a common way is through standardization, such that all features have a mean = 0 and standard deviation = 1 (the resultant is a z-score).


```python
# Define our features
features = echo_tracks.drop(columns=['genre_top', 'track_id'])

# Define our labels
labels = echo_tracks['genre_top']

# Import the StandardScaler
from sklearn.preprocessing import StandardScaler

# Scale the features and set the values to a new variable
scaler = StandardScaler()
scaled_train_features = scaler.fit_transform(features)
```

### Principal Component Analysis on our scaled data

Now that we have preprocessed our data, we are ready to use PCA to determine by how much we can reduce the dimensionality of our data. We can use scree-plots and cumulative explained ratio plots to find the number of components to use in further analyses.

Scree-plots display the number of components against the variance explained by each component, sorted in descending order of variance. Scree-plots help us get a better sense of which components explain a sufficient amount of variance in our data. When using scree plots, an 'elbow' (a steep drop from one data point to the next) in the plot is typically used to decide on an appropriate cutoff.


```python
# This is just to make plots appear in the notebook
%matplotlib inline

# Import our plotting module, and PCA class
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Get our explained variance ratios from PCA using all features
pca = PCA()
pca.fit(scaled_train_features)
exp_variance = pca.explained_variance_ratio_

# plot the explained variance using a barplot
fig, ax = plt.subplots()
ax.bar(range(pca.n_components_), exp_variance)
ax.set_xlabel('Principal Component #')
```




    Text(0.5, 0, 'Principal Component #')




![png](Classify-Song-Genres-From-Audio-Data_files/Classify-Song-Genres-From-Audio-Data_8_1.png)


### Further visualization of PCA

Unfortunately, there does not appear to be a clear elbow in this scree plot, which means it is not straightforward to find the number of intrinsic dimensions using this method.

But all is not lost! Instead, we can also look at the cumulative explained variance plot to determine how many features are required to explain, say, about 85% of the variance (cutoffs are somewhat arbitrary here, and usually decided upon by 'rules of thumb'). Once we determine the appropriate number of components, we can perform PCA with that many components, ideally reducing the dimensionality of our data.


```python
# Import numpy
import numpy as np

# Calculate the cumulative explained variance
cum_exp_variance = np.cumsum(exp_variance)

# Plot the cumulative explained variance and draw a dashed line at 0.85.
fig, ax = plt.subplots()
ax.plot(cum_exp_variance)
ax.axhline(y=0.85, linestyle='--')

# choose the n_components where about 85% of our variance can be explained
n_components = 6

# Perform PCA with the chosen number of components and project data onto components
pca = PCA(n_components, random_state=10)
pca.fit(scaled_train_features)
pca_projection = pca.transform(scaled_train_features)
```


![png](Classify-Song-Genres-From-Audio-Data_files/Classify-Song-Genres-From-Audio-Data_10_0.png)


### Train a decision tree to classify genre

Now we can use the lower dimensional PCA projection of the data to classify songs into genres. To do that, we first need to split our dataset into 'train' and 'test' subsets, where the 'train' subset will be used to train our model while the 'test' dataset allows for model performance validation.


```python
# Import train_test_split function and Decision tree classifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Split our data
train_features, test_features, train_labels, test_labels = train_test_split(pca_projection, labels,
                                                                           random_state=10)

# Train our decision tree
tree = DecisionTreeClassifier(random_state=10)
tree.fit(train_features, train_labels)

# Predict the labels for the test data
pred_labels_tree = tree.predict(test_features)
```

### Compare our decision tree to a logistic regression

Although our tree's performance is decent, it's a bad idea to immediately assume that it's therefore the perfect tool for this job -- there's always the possibility of other models that will perform even better! It's always a worthwhile idea to at least test a few other algorithms and find the one that's best for our data.

Sometimes simplest is best, and so we will start by applying logistic regression. Logistic regression makes use of what's called the logistic function to calculate the odds that a given data point belongs to a given class. Once we have both models, we can compare them on a few performance metrics, such as false positive and false negative rate (or how many points are inaccurately classified).


```python
# Import LogisticRegression
from sklearn.linear_model import LogisticRegression

# Train our logistic regression and predict labels for the test set
logreg = LogisticRegression(random_state=10)
logreg.fit(train_features, train_labels)
pred_labels_logit = logreg.predict(test_features)

# Create the classification report for both models
from sklearn.metrics import classification_report
class_rep_tree = classification_report(test_labels, pred_labels_tree)
class_rep_log = classification_report(test_labels, pred_labels_logit)

print("Decision Tree: \n", class_rep_tree)
print("Logistic Regression: \n", class_rep_log)
```

    Decision Tree:
                   precision    recall  f1-score   support

         Hip-Hop       0.60      0.60      0.60       235
            Rock       0.90      0.90      0.90       966

        accuracy                           0.84      1201
       macro avg       0.75      0.75      0.75      1201
    weighted avg       0.84      0.84      0.84      1201

    Logistic Regression:
                   precision    recall  f1-score   support

         Hip-Hop       0.77      0.54      0.64       235
            Rock       0.90      0.96      0.93       966

        accuracy                           0.88      1201
       macro avg       0.83      0.75      0.78      1201
    weighted avg       0.87      0.88      0.87      1201



### Balance our data for greater performance

Both our models do similarly well, boasting an average precision of 87% each. However, looking at our classification report, we can see that rock songs are fairly well classified, but hip-hop songs are disproportionately misclassified as rock songs.

Why might this be the case? Well, just by looking at the number of data points we have for each class, we see that we have far more data points for the rock classification than for hip-hop, potentially skewing our model's ability to distinguish between classes. This also tells us that most of our model's accuracy is driven by its ability to classify just rock songs, which is less than ideal.

To account for this, we can weight the value of a correct classification in each class inversely to the occurrence of data points for each class. Since a correct classification for "Rock" is not more important than a correct classification for "Hip-Hop" (and vice versa), we only need to account for differences in sample size of our data points when weighting our classes here, and not relative importance of each class.


```python
# Subset only the hip-hop tracks, and then only the rock tracks
hop_only = echo_tracks.loc[echo_tracks['genre_top'] == 'Hip-Hop']
rock_only = echo_tracks.loc[echo_tracks['genre_top'] == 'Rock']

# sample the rocks songs to be the same number as there are hip-hop songs
rock_only = rock_only.sample(hop_only.shape[0], random_state=10)

# concatenate the dataframes rock_only and hop_only
rock_hop_bal = pd.concat([rock_only, hop_only])

# The features, labels, and pca projection are created for the balanced dataframe
features = rock_hop_bal.drop(['genre_top', 'track_id'], axis=1)
labels = rock_hop_bal['genre_top']
pca_projection = pca.fit_transform(scaler.fit_transform(features))

# Redefine the train and test set with the pca_projection from the balanced data
train_features, test_features, train_labels, test_labels = train_test_split(pca_projection, labels,
                                                                            random_state=10)
```

### Does balancing our dataset improve model bias?

We've now balanced our dataset, but in doing so, we've removed a lot of data points that might have been crucial to training our models. Let's test to see if balancing our data improves model bias towards the "Rock" classification while retaining overall classification performance.

Note that we have already reduced the size of our dataset and will go forward without applying any dimensionality reduction. In practice, we would consider dimensionality reduction more rigorously when dealing with vastly large datasets and when computation times become prohibitively large.


```python
# Train our decision tree on the balanced data
tree = DecisionTreeClassifier(random_state=10)
tree.fit(train_features, train_labels)
pred_labels_tree = tree.predict(test_features)

# Train our logistic regression on the balanced data
logreg = LogisticRegression(random_state=10)
logreg.fit(train_features, train_labels)
pred_labels_logit = logreg.predict(test_features)

# Compare the models
print("Decision Tree: \n", classification_report(test_labels, pred_labels_tree))
print("Logistic Regression: \n", classification_report(test_labels, pred_labels_logit))
```

    Decision Tree:
                   precision    recall  f1-score   support

         Hip-Hop       0.74      0.73      0.74       230
            Rock       0.73      0.74      0.73       225

        accuracy                           0.74       455
       macro avg       0.74      0.74      0.74       455
    weighted avg       0.74      0.74      0.74       455

    Logistic Regression:
                   precision    recall  f1-score   support

         Hip-Hop       0.84      0.80      0.82       230
            Rock       0.80      0.85      0.83       225

        accuracy                           0.82       455
       macro avg       0.82      0.82      0.82       455
    weighted avg       0.82      0.82      0.82       455



### Using cross-validation to evaluate our models

Success! Balancing our data has removed bias towards the more prevalent class. To get a good sense of how well our models are actually performing, we can apply what's called cross-validation (CV). This step allows us to compare models in a more rigorous fashion.

Since the way our data is split into train and test sets can impact model performance, CV attempts to split the data multiple ways and test the model on each of the splits. Although there are many different CV methods, all with their own advantages and disadvantages, we will use what's known as K-fold CV here. K-fold first splits the data into K different, equally sized subsets. Then, it iteratively uses each subset as a test set while using the remainder of the data as train sets. Finally, we can then aggregate the results from each fold for a final model performance score.


```python
from sklearn.model_selection import KFold, cross_val_score

# Set up our K-fold cross-validation
kf = KFold(10)

tree = DecisionTreeClassifier(random_state=10)
logreg = LogisticRegression(random_state=10)

# Train our models using KFold cv
tree_score = cross_val_score(tree, pca_projection, labels, cv= kf)
logit_score = cross_val_score(logreg, pca_projection, labels, cv=kf)

# Print the mean of each array of scores
print("Decision Tree:", np.mean(tree_score), "Logistic Regression:", np.mean(logit_score))
```

    Decision Tree: 0.7489010989010989 Logistic Regression: 0.782967032967033



```python

```


```python

```
