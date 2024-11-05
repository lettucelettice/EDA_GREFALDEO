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

if pd.isnull(df_spoti['streams'].iloc[574]):
    # If it's missing, set a default or calculated value (e.g., the median of 'streams' column)
    df_spoti.at[574, 'streams'] = df_spoti['streams'].median()
else:
    # If it's not missing, store it in a variable
    streams_value_574 = df_spoti['streams'].iloc[574]

print("Value at index 574 in 'streams':", df_spoti['streams'].iloc[574])
```
![image](https://github.com/user-attachments/assets/d52c2233-613d-40e2-aa32-be86129f0d5c)

```python
#convert non-umeric values to numeric by removing commas and turning it into float
df_spoti['in_deezer_playlists'] = df_spoti['in_deezer_playlists'].str.replace(",","").astype(float)
df_spoti['in_deezer_playlists']
```
![image](https://github.com/user-attachments/assets/ad29b2e4-f651-4938-86f0-35df3a0b1951)

```python
#convert non-umeric values to numeric by removing commas and turning it into float
df_spoti['in_shazam_charts'] = df_spoti['in_shazam_charts'].str.replace(",","").astype(float)
df_spoti['in_shazam_charts']
```
![image](https://github.com/user-attachments/assets/57d498c0-60e9-4a89-88b9-e1f0d4866617)

```python
#check if data types have been converted from non-numeric to numeric 
print ("Dataset shape: ", df_spoti.shape)
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

if missing.sum() == 0:
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
meanstreams = df_spoti['streams'].mean()
medianstreams = df_spoti['streams'].median()
stdstreams = df_spoti['streams'].std()

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
![image](https://github.com/user-attachments/assets/b7e7342b-f576-43a4-8c07-b8bafa839302)

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
```python
# Find the top 5 tracks with the highest number of streams
toptracks = df_spoti[['track_name', 'streams']].sort_values(by='streams', ascending=False).head(5)

# Display the result
print("Top 5 Most Streamed Tracks:")
print(toptracks)
```









