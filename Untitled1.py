
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


# In[2]:

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# In[3]:

df  = pd.read_csv('data.csv')
df = df[df['ignore'] == False]
df['time_in_secs'] = df['time'].apply(lambda x: x.split(':')).apply(lambda x:int(x[0])* 360 + int(x[1])* 60 + int(x[2]))


# In[4]:

male_df = df[df['male'] == 1]


# In[5]:

female_df = df[df.male == 0]


# In[6]:

ax.scatter3D(female_df.age , female_df.years_since_run ,female_df.time_in_secs)
plt.show()


# In[ ]:



