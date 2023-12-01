# Finger-pain-analysis-using-DL
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans

# In this data is read from .cvs file
data = pd.read_csv('/content/labview data.csv')
data

# In this we find out the maximum value in imported data.
df = pd.read_csv('/content/labview data.csv')
print("Maximum: ",df["pain"].max())

# In this we can see the data signals On Y axis we plot pain intensity and on X axis we plot index of our graph. Pain intensity is between 0 to 100.
fig = px.line(data, y = 'pain', title='Input Data')
fig.show()

# In this declare that how many column and row we have to consider for the result
x = data.iloc[:,1:2] # 1st for rows and second for columns

# In this we declare in how many number of clusters we have to classify the data. We use the K-means algorithm for classifying data in three level of intensity.
kmeans = KMeans(3)
kmeans.fit(x)

# In this data is classified in three categories high, medium, low.0=low, 1=medium, 2=high. It shows the identified cluster by the system.
identified_clusters = kmeans.fit_predict(x)
identified_clusters

# In this we plot our clustered data on graph. It shows data is classified in three categories low, high, medium. Low pain values denoted by red colour, medium pain values denoted by blue colour, high pain values denoted by green colour.We can easily see in above graph Green colour values are more than red and blue. So pain of patient is high.

data_with_clusters = data.copy()
data_with_clusters['Clusters'] = identified_clusters 
plt.scatter(data_with_clusters['pain'],data_with_clusters['y'],c=data_with_clusters['Clusters'],cmap='rainbow')

# Plot same data in another type of graph 
sns.set_color_codes()
sns.distplot(data['pain'].dropna(), bins = 20, kde = True, color = 'red')

# Plot same data in another type of graph. In this we can see the data overlapped on 20 and between 40 and 60 data is overlapped.
plt.figure(figsize=(10, 7))
plt.title("Distribution of pain data")
sns.histplot(x="pain", hue="y", data=data)
plt.show()

# Finally for user understanding we print how many values of high intensity, how many values of medium pain, and how many values of low pain.

mylist=['high','med','low']

for i in range(len(kmeans.cluster_centers_)):
  print("Pain", mylist[i])
  print("Size:", sum(kmeans.labels_ == i))


