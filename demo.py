import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from io import StringIO 
import pickle
from sklearn import svm 
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from wordcloud import WordCloud

cid = 'c32268dc4ba3438cbb69715eaab346e5'
secret = '0bbe809369164c23b0c88706ffd08522'
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


st.title("Music and Social Dashboard")
st.markdown('Assessing your mental well-being through analysis of your music listening habits.')
#st.markdown('Coronavirus disease (COVID-19) is an infectious disease caused')
st.sidebar.title("Type of Visualization")
st.sidebar.markdown("Select the Charts/Plots accordingly:")

#adding a file uploader

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
  df = pd.read_csv(uploaded_file)
  st.write(df)
  df_filtered = df[df['min_played'] >= 60000] 
  df = df_filtered

def calc_ri(df):
  
  x = (df['track_name'].value_counts())
  no_unique_songs = len(df['track_name'].unique())
  dict_unique = x.to_dict()
  list_freq = list(dict_unique.values())
  total_freq = 0
  total_freq_above_two = 0
  for i in range(no_unique_songs):
      total_freq += list_freq[i]
      if (list_freq[i]>2): 
          total_freq_above_two += list_freq[i]
  ri = (total_freq_above_two/total_freq)
  return ri

def calc_avg_features(df):
  avg_val = sum(df['valence'])/len(df)
  avg_energy = sum(df['energy'])/len(df)
  avg_dance = sum(df['danceability'])/len(df)
  avg_loud = sum(df['loudness'])/len(df)
  avg_acoustic = sum(df['acousticness'])/len(df)
  avg_speechiness = sum(df['speechiness'])/len(df)
  avg_instru = sum(df['instrumentalness'])/len(df)
  avg_liveness = sum(df['liveness'])/len(df)
  values = [avg_val, avg_energy, avg_dance, avg_acoustic, avg_speechiness, avg_instru]
  return values

def calc_monthly_avg_features(df):
  df['month'] = pd.to_datetime(df['date']).dt.month
  monthly_val = []
  for i in range(1,13):
    df_filtered = df[df['month'] == i] 
    print(df_filtered)
    avg_val = sum(df_filtered['valence'])/len(df_filtered)
    avg_energy = sum(df_filtered['energy'])/len(df_filtered)
    avg_dance = sum(df_filtered['danceability'])/len(df_filtered)
    avg_loud = sum(df_filtered['loudness'])/len(df_filtered)
    avg_acoustic = sum(df_filtered['acousticness'])/len(df_filtered)
    avg_speechiness = sum(df_filtered['speechiness'])/len(df_filtered)
    avg_instru = sum(df_filtered['instrumentalness'])/len(df_filtered)
    avg_liveness = sum(df_filtered['liveness'])/len(df_filtered)
    values = [avg_val, avg_energy, avg_dance, avg_acoustic, avg_speechiness, avg_instru]
    monthly_val.append([i, avg_val, avg_energy, avg_dance, avg_acoustic, avg_speechiness, avg_instru])
    
  df_temp = pd.DataFrame(monthly_val, columns =['Month', 'Valence','Energy','Danceability','Acousticness','Speechiness','Instumentalness'])
  print(df_temp.head(5))    
    
  return df_temp
  
  
def plot_radar(df, monthly):

  if(monthly==0):
    values = calc_avg_features(df)
    fig = plt.figure(figsize =(10, 8))
    ax = fig.add_subplot(111, projection="polar")
    label = ['valence', 'energy', 'dance', 'acousticness', 'speechiness', 'instrumentalness']
    
    theta = np.arange(len(values)) / float(len(values)) * 2 * np.pi
    print(theta)
    #theta[len(df1.columns)+1] = theta[0]
    plt.plot(theta, values)
    plt.xticks(theta[:], label, color='grey', size=12)
    ax.tick_params(pad=10) # 
    ax.fill(theta, values, alpha=0.3)

      # plt.legend() # shows the legend, using the label of the line plot (useful when there is more than 1 polygon)
    plt.title("")
    return fig
    
  
  if (monthly==1):
    df1 = calc_monthly_avg_features(df)
    df1.drop(columns=['Month'],inplace=True)
    month = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    # fig = plt.figure(figsize =(8, 6))
    fig, ax = plt.subplots()  
    
    
    # ax = fig.add_subplot(111, projection="polar")
    # theta = np.arange(len(df1.columns)) / float(len(df1.columns)) * 2 * np.pi
    
    # for i in range(len(df1)):
    #   values = df1.iloc[i]
    #   print(values)
     
    #   #l1, = ax.plot(theta, values, color="C2", marker="o", label=c1[i])
    #   plt.plot(theta, values, label=month[i])
    #   plt.xticks(theta[:], df1.columns, color='grey', size=12)
    #   ax.tick_params(pad=10) # 
    #   ax.fill(theta, values, alpha=0.3)
    
    for col in df1.columns:
      vals = df1[col]
      ax.plot(month,vals)
    # plt.legend()
    plt.title("Monthly averaged acoustic features")
    plt.legend()
    return fig
    #plt.show()

def plot_genres(df):
  n = (pd.unique(df['artist_name']))
  x = df['artist_name'].value_counts()
  if (len(x>30)):
    top_artists = x[:30]
    genre_list = []
  else:
    top_artists = x
  for i in range(len(top_artists)):
    result = sp.search(top_artists[i])
    track = result['tracks']['items'][0]
    artist = sp.artist(track['artists'][0]['external_urls']['spotify'])
    artist_genre = artist['genres']
    genre_list.extend(artist_genre)
  unique_gen_dict = {}
  for value in genre_list:
    if value in unique_gen_dict.keys():
      unique_gen_dict[value] += 1
    else:
      unique_gen_dict[value] = 1
  wordcloud = WordCloud(
    background_color='white',
    width=1000,
    height=500
  ).generate_from_frequencies(unique_gen_dict)
  return wordcloud

########################################


choices = ['Never', 'Rarely', 'Sometimes', 'Often', 'Always']
qs = [
  ('When	I	listen	to	music	I	get	stuck	in	bad	memories', choices),
  ('I	hide	in	my	music	because	nobody	understands	me,	and	it blocks	people	out', choices),
  ('Music	helps	me	to	relax', choices),
  ('When	I	try	to	use	music	to	feel better	I	actually	end	up	feeling	worse', choices),
  ('I	feel	happier	after	playing	or	listening	to	music', choices),
  ('Music	gives	me	the	energy	to	get	going', choices),
  ('I	like	to	listen	to	songs	over	and	over	even	though	it	makes	me	feel	worse', choices),
  ('Music	makes	me	feel	bad	 about	who	I	am', choices),
  ('Music	helps	me	to	connect	 with	other	people	who	are	like	me', choices),
  ('Music	gives	me	an	excuse	not	to	face	up	to	the	real	world', choices),
  ('It	can	be	hard	to	stop	listening	to	music	that	connects	me	to	bad	memories', choices),
  ('Music	leads	me	to	do	things	I	shouldn’t	do', choices),
  ('When	I’m	feeling	tense	or	tired in	my	body	music	helps	me	to	relax', choices)
]

# for i in range(len(qs)):
#   placeholder = st.empty()
#   num = st.session_state.num
#   with placeholder.form(key=str(num)):
#     st.radio(qs[num][0], key=num+1, options=qs[num][1])
def submitted():
    st.session_state.submitted = True
def reset():
    st.session_state.submitted = False

form_container = st.container()

mapping = {
  'Never': 1,
  'Rarely': 2,
  'Sometimes': 3,
  'Often': 4,
  'Always': 5
}

with open('svm_weights.pkl', 'rb') as f:
  clf = pickle.load(f)
  
if st.button('Predict distress category', on_click=reset):
  
  with form_container:
    st.write('Please answer the following questions')
    with st.form('hums'):
      for i in range(len(qs)):
        st.radio(qs[i][0], options=qs[i][1], key="value{}".format(i))

      st.form_submit_button("Submit", on_click=submitted)

if 'submitted' in st.session_state:
    if st.session_state.submitted == True:

      responses_list = []
      for i in range(len(qs)):
        responses_list.append(
          eval("st.session_state.value{}".format(i))
        )
      responses = np.array([mapping[i] for i in responses_list]).reshape(1, -1)
      predict_probabs = (clf.predict_proba(responses)[0])*100
      if clf.predict(responses)[0]==0:
        st.markdown('## {:.2f}% likelihood of being at no risk of depression'.format(predict_probabs[0]))
      else:
        st.markdown('## {:.2f}% likelihood of being at risk of depression'.format(predict_probabs[1]))
      reset()

    # if submitted:
    #   st.write('Predicted value: {}'.format(4.3))
    #   st.write('hello world')


if st.button('Get Repetitiveness Index'):
  st.write(calc_ri(df))

if st.sidebar.button('Show average acoustic features'):
  st.pyplot(plot_radar(df,0))

if st.sidebar.button('Show monthly averaged acoustic features'):
  st.pyplot(plot_radar(df,1))
  #st.pyplot(plot_radar(df))

if st.sidebar.button('Show genres of artists I listen to the most'):
  st.image(plot_genres(df).to_array())