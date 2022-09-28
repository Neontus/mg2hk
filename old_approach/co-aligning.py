#!/usr/bin/env python
# coding: utf-8

# ## Initial Project - Coaligning IRIS and AIA data

# ### Inputs

# In[2]:


import os
import numpy
import scipy
import matplotlib.pyplot as plt


# In[3]:


from iris_lmsalpy import extract_irisL2data as ei


# In[4]:


datapath = os.getcwd() + "/data/"


# In[5]:


raster_folder = datapath + "iris_l2_20220607_202829_3620106067_raster/"
aia_folder = datapath + "iris_l2_20220607_202829_3620106067_SDO/"


# ### Accessing Raster Data

# In[6]:


raster_path = raster_folder + "iris_l2_20220607_202829_3620106067_raster_t000_r00000.fits"


# In[7]:


raster_path


# In[8]:


ei.info_fits(raster_path)


# In[9]:


iris_raster = ei.load(raster_path, verbose = True)


# In[10]:


print(type(iris_raster))


# In[11]:


print(iris_raster.windows)


# In[12]:


print(iris_raster.raster.keys())


# In[13]:


print(iris_raster.raster['Mg II k 2796'].keys())


# In[14]:


iris_raster.raster['Mg II k 2796'].data.shape


# In[15]:


iris_raster.quick_look()
