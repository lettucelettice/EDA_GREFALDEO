# EXPLORATORY DATA ANALYSIS ON SPOTIFY 2023

This repository contains an exploratory data analysis on the most streamed spotify tracks of 2023. 

The objective of this activity is to analyze and interpret streaming data from Spotify's most popular tracks in 2023 to identify trends and characteristics that contribute to track popularity.

### Preprocessing of Data Set

```python
#Import necessary libraries needed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
```


### OVERVIEW OF DATA SET 
1. How many rows and columns does the data set contain?
2. What are the data types of each column? Are there any missing values?

```python
#load and read the ".csv" file to notebook
df_spoti = pd.read_csv('spotify-2023.csv', encoding = 'latin1')
df_spoti
```
![image](https://github.com/user-attachments/assets/cf94792a-94d8-4adb-be31-4aacf3032992)

```python
#display number of rows and columns of the data set
print ("Number of rows: ", df_spoti.shape[0])
print ("Number of columns: ", df_spoti.shape[1])
```
![image](https://github.com/user-attachments/assets/99d42a9a-b921-4c5f-9aa8-ea87d8da3caf)

```python
#check initial dataset structure and data types
print ("Dataset shape: ", df_spoti.shape)
df_spoti.info()
```
![image](https://github.com/user-attachments/assets/6ff245e1-c8fd-4123-b793-4f04554c48ca)

### Data Cleaning 
Identify and handle missing values and correct data types to ensure consistency and accuracy in the dataset.

```python
#locate non-numeric value for "streams" column
df_spoti['streams'].iloc[574]
```
![image](https://github.com/user-attachments/assets/2885247a-7e61-4ea5-82e3-73e11df93784)
```python
#convert non-numeric value to numeric value
df_spoti['streams'] = pd.to_numeric(df_spoti['streams'], errors = 'coerce')

if pd.isnull(df_spoti['streams'].iloc[574]): #if missing, store a numeric value
    df_spoti.at[574, 'streams'] = df_spoti['streams'].median()
else:
    streams_value_574 = df_spoti['streams'].iloc[574]

print("Index 574 value: ", df_spoti['streams'].iloc[574])
```
![image](https://github.com/user-attachments/assets/6102589a-862c-4ced-8aa5-2b00785df4a1)

```python
#convert non-umeric values to numeric by removing commas and turning it into float
df_spoti['in_deezer_playlists']=df_spoti['in_deezer_playlists'].str.replace(",","").astype(float)
df_spoti['in_deezer_playlists']
```
![image](https://github.com/user-attachments/assets/ad29b2e4-f651-4938-86f0-35df3a0b1951)

```python
#convert non-umeric values to numeric by removing commas and turning it into float
df_spoti['in_shazam_charts']=df_spoti['in_shazam_charts'].str.replace(",","").astype(float)
df_spoti['in_shazam_charts']
```
![image](https://github.com/user-attachments/assets/57d498c0-60e9-4a89-88b9-e1f0d4866617)

```python
#check if data types have been converted from non-numeric to numeric 
print("Dataset shape: ", df_spoti.shape)
df_spoti.info()
```
![image](https://github.com/user-attachments/assets/05547fca-edf9-46f3-bb47-039331145816)

```python
#check which columns have missing values and get their count
print("Missing values per column:\n ",df_spoti.isnull().sum())
```
![image](https://github.com/user-attachments/assets/d760b92a-5d9c-4779-aaf2-7ad16e96b772)

```python
#handling missing values
missing = df_spoti.isnull().sum()

if missing.sum()==0:
    print ("Missing values : 0")
else:
    print ("Missing values: \n")
    print(missing[missing>0])

#check if there are still any missing values in the data set 
df_spoti.isnull().sum()
```
![image](https://github.com/user-attachments/assets/a947543a-a335-48e5-9983-97de22c30988)


### BASIC STATISTICS 
1. What are the mean, median, and standard deviation of the streams column?
2. What is the distribution of released_year and artist_count? Are there any noticeable trends or outliers?

```python
#obtain mean, median, mode for streams
meanstreams=df_spoti['streams'].mean()
medianstreams=df_spoti['streams'].median()
stdstreams=df_spoti['streams'].std()

#output the results for each given
print("Mean of streams column: {:,.2f}".format(meanstreams))
print("Median of streams column: {:,.2f}".format(medianstreams))
print("Standard Deviation of streams column:{:,.2f} ".format(stdstreams))
```
![image](https://github.com/user-attachments/assets/9dceae32-9d13-4768-93f9-a7da08b37a69)

```python
#distribution graph for "released year" column
plt.figure(figsize=(20, 6))
sns.histplot(df_spoti['released_year'], bins=94, kde=True, color='pink')
plt.title('Distribution of Released Year')
plt.xlabel('Released Year')
plt.ylabel('Frequency')
plt.show()
```
![image](https://github.com/user-attachments/assets/e045afef-504b-459a-91ce-4f9caba22920)

```python
#trends
```

```python
#basic outliers using basic statistics for released_year
print(df_spoti['released_year'].describe())
```
![image](https://github.com/user-attachments/assets/1c5fe543-c052-4047-aead-2f30c366691d)

```python
#check outliers for said column
def find_outliers(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    outliers = column[(column < (Q1 - 1.5 * IQR)) | (column > (Q3 + 1.5 * IQR))]
    
    #display outliers in list form
    if not outliers.empty:
        return f"Outliers detected in {column.name}: {', '.join(map(str, outliers.unique()))}"
    else:
        return f"No outliers detected in {column.name}."

#output
released_year_outliers = find_outliers(df_spoti['released_year'])
print(released_year_outliers)
```
![image](https://github.com/user-attachments/assets/7d7c01da-862a-4ee2-bb3f-97cc839660c0)

```python
#graph a box plot for released year outlier
plt.figure(figsize=(20, 6))
plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
sns.boxplot(data=df_spoti, x='released_year', color='#FFDBE0')
plt.title('Box Plot of Released Year')
plt.xlabel('Released Year')

plt.tight_layout() 
plt.show()
```
![image](https://github.com/user-attachments/assets/acf4f297-1160-43fb-acbb-3002439a7af0)

```python
#distribution graph for "artist_count" column
plt.figure(figsize=(15, 6))
sns.histplot(df_spoti['artist_count'], bins=20, kde=True, color='pink')
plt.title('Distribution of Artist Count')
plt.xlabel('Artist Count')
plt.ylabel('Frequency')
plt.show()
```
![image](https://github.com/user-attachments/assets/4e0a6bc2-b74c-4377-a621-48c26d9a734d)

```python
#add trends here for artist_count
```

```python
#basic outliers using basic stats for artist_count
print(df_spoti['artist_count'].describe())
```
![image](https://github.com/user-attachments/assets/28b7d856-f1af-44e1-b537-5b69828ffe9c)

```python
#detect in-depth outliers for artist_count column
def find_outliers(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    outliers = column[(column < (Q1 - 1.5 * IQR)) | (column > (Q3 + 1.5 * IQR))]
    
    #display outliers in list form
    if not outliers.empty:
        return f"Outliers detected in {column.name}: {', '.join(map(str, outliers.unique()))}"
    else:
        return f"No outliers detected in {column.name}."

#output
released_year_outliers = find_outliers(df_spoti['artist_count'])
print(released_year_outliers)
```
![image](https://github.com/user-attachments/assets/8caedb4f-09ae-4b0e-85ac-6c61976dba98)

```python
#box plot for "artist_count" column
plt.figure(figsize=(20, 6))
plt.subplot(1, 2, 2)
sns.boxplot(data=df_spoti, x='artist_count', color='#FFDBE0')
plt.title('Box Plot of Artist Count')
plt.xlabel('Artist Count')

plt.tight_layout()
plt.show()
```
![image](https://github.com/user-attachments/assets/b29bbccb-6774-4d0c-b2f3-40dd1b4b3276)


### TOP PERFORMERS 
1. Which track has the highest number of streams? Display the top 5 most streamed tracks.
2. Who are the top 5 most frequent artists based on the number of tracks in the dataset?

#### Top 5 Tracks with Highest Streams
```python
#top 5 tracks with highest number streams
toptracks = df_spoti[['track_name', 'streams']].sort_values(by='streams', ascending=False).head(5)

#display result
print("Top 5 Most Streamed Tracks:")
print(toptracks)
```
![image](https://github.com/user-attachments/assets/6d562d79-15fa-4c04-a198-3532dd4dd8a4)

#### Top 5 Artists
```python
#top 5 artists
```


### TEMPORAL TRENDS
1. Analyze the trends in the number of tracks released over time. Plot the number of tracks released per year.
2. Does the number of tracks released per month follow any noticeable patterns? Which month sees the most releases?

#### Number of tracks released per year 
```python
releasesyearly = df_spoti['released_year'].value_counts().sort_index()

#graph and plot the trends
plt.figure(figsize=(20, 8))
sns.barplot(x=releasesyearly.index, y=releasesyearly.values, color='pink')
plt.title('Number of Tracks Released Per Year')
plt.xlabel('Release Year')
plt.ylabel('Number of Tracks')
plt.xticks(rotation=45, ha='right')
plt.show()
```
![image](https://github.com/user-attachments/assets/d5a43b42-2c14-43d2-96a0-cd75905397ea)

#### Patterns for number of tracks per month
```python
releases_per_month = df_spoti['released_month'].value_counts().sort_index()

#graph and plot the number of releases per month
plt.figure(figsize=(12, 6))
sns.barplot(x=releases_per_month.index, y=releases_per_month.values, color='pink')
plt.title('Number of Tracks Released Per Month')
plt.xlabel('Month')
plt.ylabel('Number of Tracks')
plt.xticks(ticks=range(12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.show()
```
![image](https://github.com/user-attachments/assets/ea711913-9bd1-4fbf-83e0-4bbf8aa96a45)

```python
#identify which month has the most releases
most_releases_month = releases_per_month.idxmax()
most_releases_count = releases_per_month.max()

print(f"The month with the most releases is {most_releases_month} with {most_releases_count} tracks.")
```
![image](https://github.com/user-attachments/assets/9fdc5f30-73cb-432d-a460-738f4de28191)


### GENRE AND MUSIC CHARACTERISTICS
1. Examine the correlation between streams and musical attributes like bpm, danceability_%, and energy_%. Which attributes seem to influence streams the most?
2. Is there a correlation between danceability_% and energy_%? How about valence_% and acousticness_%?

#### Correlation of "Streams" and Musical Attributes
```python
#obtain correlations between streams and the given musical attributrs
correlationstreams = df_spoti[['streams', 'bpm', 'danceability_%', 'energy_%']].corr()['streams'].drop('streams')

print("Correlation of Streams with Musical Attributes: \n")
print(correlationstreams)
```
![image](https://github.com/user-attachments/assets/07c55a8c-2fea-48f6-950f-a491b44b748e)

##### - Correlation between danceability_% and energy_%.
##### - Correlation between valence_% and acousticness_%.

```python
#get correlation between danceability_% and energy_%
danceabilityenergy=df_spoti['danceability_%'].corr(df_spoti['energy_%'])
print(f"Correlation between Danceability % and Energy %: {danceabilityenergy:.2f}")

#get correlation between valence_% and acousticness_%
valenceacousticness=df_spoti['valence_%'].corr(df_spoti['acousticness_%'])
print(f"Correlation between Valence % and Acousticness %: {valenceacousticness:.2f}")
```
![image](https://github.com/user-attachments/assets/c44bf432-d9df-4480-842b-e976758819c1)

```python
#graph the correlation for the attributes
attributes = df_spoti[['streams', 'bpm', 'danceability_%', 'energy_%', 'valence_%', 'acousticness_%']]
correlation_matrix = attributes.corr()

#create and plot using heatmap to observe the correlations 
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='spring', vmin=-1, vmax=1)
plt.title("Correlation Heatmap of Streams and the different Musical Atrributes")
plt.show()
```
![image](https://github.com/user-attachments/assets/b94d3d1a-aa8e-42c0-8ca7-c7442f7dc251)

### PLATFORM POPULARITY 
1. How do the numbers of tracks in spotify_playlists, spotify_charts, and apple_playlists compare? Which platform seems to favor the most popular tracks?

### ADVANCED ANALYSIS 
1. Based on the streams data, can you identify any patterns among tracks with the same key or mode (Major vs. Minor)?
2. Do certain genres or artists consistently appear in more playlists or charts? Perform an analysis to compare the most frequently appearing artists in playlists or charts.











