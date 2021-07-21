import streamlit as st
import pandas as pd
import numpy as np
import warnings
import plotly.express as px
warnings.filterwarnings('ignore')
pd.set_option('max_column', None)
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
sns.set()
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import scipy.spatial.distance as sdist
import sklearn.cluster as cluster

st.set_page_config('Dashboard', layout="wide")
st.title('User Analytics in the Telecommunication Industry') 
data  = pd.read_csv('data_source.csv', na_values=['?', None])
data.drop('Unnamed: 0', axis=1, inplace=True)
#print(data.head(5))
st.write(data.head(5))

def total_data_description(data, t):
    if t == 'Email':
        st.write('Description of total data volume (in Bytes) for Email ') 
        st.write(data['Email'].describe())
    elif t == 'Social media':
        st.write('Description of total data volume (in Bytes) for Socal media')
        st.write(data['Social Media'].describe())
    elif t == 'Google':
        st.write('Description of total data volume (in Bytes) for Google')
        st.write(data['Google'].describe())
    elif t == 'Youtube':
        st.write('Description of total data volume (in Bytes) for Youtube')
        st.write(data['Youtube'].describe())
    elif t == 'Netflix':
        st.write('Description of total data volume (in Bytes) for Netflix')
        st.write(data['Netflix'].describe())
    elif t == 'Gaming':
        st.write('Description of total data volume (in Bytes) for Gaming')
        st.write(data['Gaming'].describe())
    else:
        st.write(data)

def user_engagement_analysis(data, a):
    if a == 'sessions frequency' :
        # top 10 sessions frequency
        st.write('top 5 sessions frequency')
        sessions_frequency = data.groupby('MSISDN/Number')
        sessions_frequency = sessions_frequency.agg({"Bearer Id": "count"})
        Top10_sessions_frequency = sessions_frequency.sort_values(by='Bearer Id', ascending=False)
        st.write(Top10_sessions_frequency.head(5))
    elif a == 'duration of the session':
        # duration of the session
        st.write('top 5 duration of the session')
        session_duration= data.groupby('MSISDN/Number')
        session_duration = session_duration.agg({"Dur. (ms)": "sum"})
        Top10_session_duration = session_duration.sort_values(by='Dur. (ms)', ascending=False)
        st.write(session_duration.head(5))
    elif a == 'sessions total traffic':
        #the sessions total traffic (download and upload (bytes))
        st.write('top 5 sessions total traffic (download and upload (bytes))')
        total_traffic = data.groupby('MSISDN/Number')
        total_traffic = total_traffic.agg({"Total": "sum"})
        Top10_total_traffic = total_traffic.sort_values(by='Total', ascending=False)
        st.write(Top10_total_traffic.head(5))
def joined(data):
    sessions_frequency = data.groupby('MSISDN/Number')
    sessions_frequency = sessions_frequency.agg({"Bearer Id": "count"}) 
    session_duration= data.groupby('MSISDN/Number')
    session_duration = session_duration.agg({"Dur. (ms)": "sum"})
    total_traffic = data.groupby('MSISDN/Number')
    total_traffic = total_traffic.agg({"Total": "sum"}) 
    return pd.DataFrame(sessions_frequency.join(session_duration, how='left')).join(total_traffic, how='left')

def kmeans(data, k):
    k = 3
    joined_data = joined(data)
    cols_to_standardize = ['Bearer Id',  'Dur. (ms)', 'Total']
    data_to_standardize = joined_data[cols_to_standardize]

    # Create the scaler.
    scaler = StandardScaler().fit(data_to_standardize)

    # Standardize the data
    standardized_data = joined_data.copy()
    standardized_columns = scaler.transform(data_to_standardize)
    standardized_data[cols_to_standardize] = standardized_columns

    st.write('Sample of data to use:')
    st.write(standardized_data.sample(5), '\n')

    model = KMeans(n_clusters = k).fit(standardized_data)

    joined_data['cluster'] = model.predict(standardized_data)

    st.write('Cluster summary:')
    summary = joined_data.groupby(['cluster']).mean()
    summary['count'] = joined_data['cluster'].value_counts()
    summary = summary.sort_values(by='count', ascending=False)
    st.write(summary)

def most_app_used(data):
    all_application = data[[ 'Social Media','Google', 'Email','Youtube','Netflix','Gaming']].sum()
    all_ap = dict(all_application)
    all_sum = dict(sorted(all_ap.items(), key=lambda item: item[1], reverse=True))
    top3, i = {}, 0
    for k, v in all_sum.items():
        if i == 3: break
        i+=1
        top3[k] = v 
    #Get the Keys and store them in a list
    labels = list(top3.keys())
    # Get the Values and store them in a list
    values = list(top3.values())
    #plt.pie(values, labels=labels, autopct='%1.2f%%')
   # plt.show()

    pie_fig = px.pie(values=values, names=labels)
    st.plotly_chart(pie_fig)

st.title("Upload and Download description")
updl = st.selectbox('Choose type of web', ('Email','Social media','Google', 'Youtube', 'Netflix', 'Gaming'))
total_data_description(data, updl)

st.title('User Engagement analysis')
analysis = st.selectbox('Choose customers per engagement metric', 
        ('sessions frequency','duration of the session','sessions total traffic'))
total_data_description(data, analysis)

st.title('K-means (k=3) to classify customers in three groups of engagement.')
kmeans(data, 3)

st.title('Top 3 most used application')
most_app_used(data)