#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import streamlit as st 
import numpy as np
import pickle


# In[2]:


st.title('Employee Promotion Programme')
st.sidebar.header('''Fill in the fields to make the Prediction!''')
st.sidebar.header('Enter Employee Data')


# In[3]:


department_mapping = {
                'Sales & Marketing':1,
                'Operations':2,
                'Technology':3,
                'Analytics':4,
                'R&D':5,
                'Procurement':6,
                'Finance':7,
                'HR':8,
                'Legal':9
}


# In[4]:


region_mapping  = {
      'region_1':1,
      'region_2':2,
      'region_3':3,
      'region_4':4,
      'region_5':5,
      'region_6':6,
      'region_7':7,
      'region_8':8,
      'region_9':9,
      'region_10':10,
      'region_11':11,
      'region_12':12,
      'region_13':13,
      'region_14':14,
      'region_15':15,
      'region_16':16,
      'region_17':17,
      'region_18':18,
      'region_19':19,
      'region_20':20,
      'region_21':21,
      'region_22':22,
      'region_23':23,
      'region_24':24,
      'region_25':25,
      'region_26':26,
      'region_27':27,
      'region_28':28,
      'region_29':29,
      'region_30':30,
      'region_31':31,
      'region_32':32,
      'region_33':33,
      'region_34':34
}


# In[5]:


education_mapping = {
                 'Masters':1,                
                 'Bachelors':2,         
                 'Below Secondary':3,    
}


# In[6]:


gender_mapping ={
              'male':1,
              'female':0
}


# In[7]:


recruitment_channel_mapping = {
                           'sourcing':1,
                           'other':2,
                           'referred':3
}


# In[8]:


awards_won_mapping = {
                'Yes':1,
                'No':0
}


# In[9]:


# Creating the side menu to enter employee data

def user_input_features():
    
    department_feature = st.sidebar.selectbox("Select the Department", ("Sales & Marketing", "Operations",
                                                                           "Technology", "Analytics", "R&D",
                                                                           "Procurement", "Finance", "HR",
                                                                           "Legal"))
    department = department_mapping[department_feature]
    
    region_feature= st.sidebar.selectbox("Select the region",("region_1","region_2","region_3","region_4","region_5","region_6",
                                                              "region_7","region_8" ,"region_9" ,"region_10", "region_11","region_12",
                                                              "region_13","region_14","region_15","region_16" ,"region_17","region_18" ,
                                                              "region_19" ,"region_20","region_21","region_22","region_23","region_24",
                                                              "region_25","region_26" ,"region_27","region_28" ,"region_29" ,"region_30",
                                                              "region_31","region_32","region_33","region_34"))
    region = region_mapping[region_feature]
    
    education_feature = st.sidebar.selectbox("Select the education", ("Masters",               
                                                                      "Bachelors",     
                                                                      "Below Secondary"))
    education = education_mapping[education_feature]
    
    gender_feature = st.sidebar.selectbox("Select the gender", ("male","female"))
    gender = gender_mapping[gender_feature]
    
    recruitment_channel_feature = st.sidebar.selectbox("Select the recruitment channel", ("sourcing","other","referred"))
    recruitment_channel = recruitment_channel_mapping[recruitment_channel_feature] 
    
    no_of_trainings = st.sidebar.slider("How many trainings did you do?", 1, 10, 2) 
    
    age = st.sidebar.slider("Select the age?", 20, 60, 20) 
    
    previous_year_rating = st.sidebar.slider("Previous Year Performance Evaluation", 1, 5, 4)
    
    length_of_service = st.sidebar.slider("Service time", 0, 30, 5)
    
    avg_training_score = st.sidebar.slider("Average grades in training", 39, 99, 70)
    
    awards_won_feature = st.sidebar.selectbox("Awarded this year", ("Yes", "No"))
    
    awards_won = awards_won_mapping[awards_won_feature]
    
    data = {'department':department,
            'region':region,
            'education':education,
            'gender':gender,
            'recruitment_channel':recruitment_channel,
            'no_of_trainings':no_of_trainings,
            'age': age,
            'previous_year_rating': previous_year_rating, 
            'length_of_service': length_of_service,
            'awards_won': awards_won,
            'avg_training_score': avg_training_score }
    
    features = pd.DataFrame(data,index = [0])
    return features


# In[10]:


input_df = user_input_features()


# In[11]:


promotion_test = pd.read_csv('https://github.com/AkashSapariya/Employee-Promotion-Hr-Review-Classification-Model/blob/main/employee_promotion.csv')


# In[16]:


loaded_model=pickle.load(open('D:/Data Science/Project/Project 3/My work Deployment/HRmodel.pkl','rb'))
prediction = loaded_model.predict(input_df)
prediction_probability = loaded_model.predict_proba(input_df)


# In[17]:


st.subheader('Prediction')
result = np.array(['This Employee wont get promotion.','This Employee will be promoted!'])
st.write(result[prediction][0])

st.subheader('Probability of prediction')
st.write('Based on the data selected,\nthis employee have a {0:.2f}% chance of being promoted.'.format(prediction_probability[0][1] * 100))

if prediction == 0:
    st.markdown("![Alt Text](https://i.chzbgr.com/full/8491994880/hDC7CDEBC/good-luck-with-your-tockawocka)")

    
else:
    st.markdown("![Alt Text](https://media0.giphy.com/media/3oz9ZE2Oo9zRC/giphy.gif)")
    st.markdown("![Alt Text](https://i.gifer.com/6ob.gif)")
    


# In[ ]:





# In[ ]:




