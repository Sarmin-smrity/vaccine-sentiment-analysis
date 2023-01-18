import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import numpy as np
from textblob import TextBlob
import nltk
#from nltk.sentiment.vader import SentimentIntensityAnalyzer
#nltk.download('vader_lexicon')
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
import vega 
from streamlit_option_menu import option_menu
import seaborn as sns


# 2. horizontal menu
selected2 = option_menu(None, ["Home", "Overview", "Analyzer", 'Aspects','Ploraity'], 
    icons=['house', 'file-bar-graph', "graph-down", 'stack-overflow'], 
    menu_icon="cast", default_index=0, orientation="horizontal")
selected2

if selected2 == "Home":
    	#title
	st.title('Sentiment analysis of Tweets about COVID-19 vaccination')
	#markdown
	#st.markdown('This application is all about public Sentiments Analysis of tweets on COVID-19 Vaccine in South Asian Countries using this streamlit app.')
	#sidebar
	st.sidebar.title('Tweet Sentiment Analysis')
	# sidebar markdown 
	st.sidebar.markdown("So that, people also can get an overview of vaccination's effects ")
	#loading the data (the csv file is in the same folder)
	#if the file is stored the copy the path and paste in read_csv method.
	data=pd.read_csv('data/tweets_v10.csv')
	#checkbox to show data 
	if st.checkbox("Show Data"):
		st.write(data.head(50))
	#subheader
	st.sidebar.subheader('Tweets Analyser')
	#radio buttons
	tweets=st.sidebar.radio('Sentiment Type',('Positive','Negative','Neutral'))
	st.write(data.query('textblob_sentiment==@tweets')[['text']].sample(1).iat[0,0])
	st.write(data.query('textblob_sentiment==@tweets')[['text']].sample(1).iat[0,0])
	st.write(data.query('textblob_sentiment==@tweets')[['text']].sample(1).iat[0,0])

	# country data
	st.markdown("###  Country data count")

	sa_clean=pd.read_csv('data/sa_clean.csv')
	cntry_cnt=sa_clean['user_country'].value_counts()
	cntry_cnt=pd.DataFrame({'Country':cntry_cnt.index,'Count':cntry_cnt.values})
	fig11 = px.bar(cntry_cnt, x='Country', y='Count', color = 'Country', height= 500)
	st.plotly_chart(fig11)

	#selectbox + visualisation
	# An optional string to use as the unique key for the widget. If this is omitted, a key will be generated for the widget based on its content.
	## Multiple widgets of the same type may not share the same key.
	select=st.sidebar.selectbox('Visualisation Of Tweets',['Histogram','Pie Chart'],key=1)
	sentiment=data['textblob_sentiment'].value_counts()
	sentiment=pd.DataFrame({'textblob_sentiment':sentiment.index,'Tweets':sentiment.values})
	st.markdown("###  Sentiment count")
	if select == "Histogram":
			fig = px.bar(sentiment, x='textblob_sentiment', y='Tweets', color = 'Tweets', height= 500)
			st.plotly_chart(fig)
	else:
			fig = px.pie(sentiment, values='Tweets', names='textblob_sentiment')
			st.plotly_chart(fig)

 # Todo : Add VIZ1




		
	#multiselect
	st.sidebar.subheader("Country tweets by sentiment")
	choice = st.sidebar.multiselect("Countrys", ('Bangladesh', 'Bhutan', 'India', 'Nepal', 'Pakistan', 'Sri Lanka'), key = '0')  
	if len(choice)>0:
		air_data=data[data.user_country.isin(choice)]
		# facet_col = 'sentiment'
		fig1 = px.histogram(air_data, x='user_country', y='textblob_sentiment', histfunc='count', color='textblob_sentiment',labels={'sentiment':'tweets'}, height=600, width=800)
		st.plotly_chart(fig1)

	#multiselect
	### todo: check on the code, why its not showing the graph
	st.sidebar.subheader("Sentiment for Vaccine Aspects")
	choice = st.sidebar.multiselect("Aspects", ('Administration Coverage','Clinical Trials', 'Dose Availability', 'Slot Distribution', 'Usage Approval'), key = '2')  
	if len(choice)>0:
		air_data=data[data.user_country.isin(choice)]
		# facet_col = 'sentiment'
		fig2 = px.histogram(air_data, x='topic_aspects', y='textblob_sentiment', histfunc='count', color='textblob_sentiment',labels={'textblob_sentiment':'topic_aspects'}, height=600, width=800)
		st.plotly_chart(fig2)


if selected2 == "Overview":
	st.title('Overview of the analysis')
	nltk_df=pd.read_csv('data/nltk_sentiment_df.csv')

	select=st.sidebar.selectbox('Visualisation Of Tweets',['Histogram','Pie Chart'],key=1)
	# sentiment=nltk_df['textblob_sentiment'].value_counts()
	# sentiment=pd.DataFrame({'textblob_sentiment':sentiment.index,'Tweets':sentiment.values})
	st.markdown("###  NLTK Vader")
	if select == "Histogram":
			fig = px.bar(nltk_df, x='sentiment', y='percentage', color = 'percentage', height= 500)
			st.plotly_chart(fig)
	else:
			fig = px.pie(nltk_df, values='percentage', names='sentiment')
			st.plotly_chart(fig)

	sentiments_pivot=pd.read_csv('data/df_sentiments_pivot.csv')
	st.markdown("###  Sentiment Pivot")

	pivot_fig_1 = px.bar(sentiments_pivot, x='analyzer', y='percentage', color = 'sentiment',height= 500)
	st.plotly_chart(pivot_fig_1)


elif selected2 == "Aspects":

	st.markdown("###  Aspect Countplot")
	data_df=pd.read_csv('tweets_v10.csv')

	cnp = px.histogram(data_df, y='topic_aspects', color = 'textblob_sentiment',height= 500)
	st.plotly_chart(cnp)

	#word cloud
	st.markdown("###  Word Cloud")
	# wrd_cloud=pd.read_csv('data/word_cloud_data.csv')
	# # categorized tweets in seperate Series
	# positive_tweet =  wrd_cloud[wrd_cloud['sentiment'] == 'Positive']['clean_data']
	# negative_tweet =  wrd_cloud[wrd_cloud['sentiment'] == 'Negative']['clean_data']
	# neutral_tweet =  wrd_cloud[wrd_cloud['sentiment'] == 'Neutral']['clean_data']

	# from wordcloud import WordCloud
	# # Function for creating WordClouds
	# def cloud_of_Words(tweet_cat):
	# 	tweet_cat = tweet_cat.astype(str)
	# 	forcloud = ' '.join([tweet for tweet in tweet_cat])
	# 	wordcloud = WordCloud(width =500,height = 300,random_state =5,max_font_size=110).generate(forcloud)
	# 	return wordcloud

	# # img_1 = cloud_of_Words(positive_tweet, 'Positive')
	# st.markdown("######  Positive")
	# st.write(cloud_of_Words(positive_tweet).to_svg(), unsafe_allow_html=True)
	# st.markdown("######  Negative")
	# st.write(cloud_of_Words(negative_tweet).to_svg(), unsafe_allow_html=True)
	# st.markdown("######  Negative")
	# st.write(cloud_of_Words(neutral_tweet).to_svg(), unsafe_allow_html=True)
	# st.image(cloud_of_Words(positive_tweet, 'Positive'), caption='Word Cloud', use_column_width=True)
	from PIL import Image
	pos = Image.open('img/word_cloud_positive.png')
	neg = Image.open('img/word_cloud_negetive.png')
	neu = Image.open('img/word_cloud_neutral.png')

	st.image(pos, caption='Positive')
	st.image(neg, caption='Negative')
	st.image(neu, caption='Neutral')


elif selected2 == "Ploraity":

	deep =pd.read_csv('data/deep_copy_7.csv')
	total=pd.DataFrame()
	deep['date'] = pd.to_datetime(deep['date'])
	total['date'] = sorted(deep['date'].unique())
	senti=list()
	for date in total['date']:
		senti.append(deep[deep['date']==date].polarity.mean())
	total['Sentiment']=senti
	fig = px.line(total, x="date", y="Sentiment", title='Overall Sentiment around Vaccines')
	st.plotly_chart(fig)


	## showing vaccine polarity

	pfizer = deep[deep['pfizer']==1][['date','polarity']]
	bbiotech = deep[deep['bbiotech']==1][['date','polarity']]
	sputnik = deep[deep['sputnik']==1][['date','polarity']]
	astra = deep[deep['astra']==1][['date','polarity']]
	moderna = deep[deep['moderna']==1][['date','polarity']]

	pfizer = pfizer.sort_values(by='date',ascending=True)
	bbiotech = bbiotech.sort_values(by='date',ascending=True)
	sputnik = sputnik.sort_values(by='date',ascending=True)
	astra = astra.sort_values(by='date',ascending=True)
	moderna = moderna.sort_values(by='date',ascending=True)

	pfizer['Avg Polarity'] = pfizer.polarity.rolling(20, min_periods=3).mean()
	bbiotech['Avg Polarity'] = bbiotech.polarity.rolling(20, min_periods=3).mean()
	sputnik['Avg Polarity'] = sputnik.polarity.rolling(20, min_periods=3).mean()
	astra['Avg Polarity'] = astra.polarity.rolling(5, min_periods=3).mean()
	moderna['Avg Polarity'] = moderna.polarity.rolling(20, min_periods=3).mean()

	a,b,c,d,e = pfizer,bbiotech,sputnik,astra,moderna
	fig = px.line(a, x="date", y="Avg Polarity", title='Pfizer')
	# fig.show()
	st.plotly_chart(fig)
	fig = px.line(b, x="date", y="Avg Polarity", title='Bharat Biotech')
	# fig.show()
	st.plotly_chart(fig)
	fig = px.line(c, x="date", y="Avg Polarity", title='Sputnik')
	# fig.show()
	st.plotly_chart(fig)
	fig = px.line(d, x="date", y="Avg Polarity", title='AstraZence/Covishield')
	# fig.show()
	st.plotly_chart(fig)
	fig = px.line(e, x="date", y="Avg Polarity", title='Moderna')
	# fig.show()
	st.plotly_chart(fig)



elif selected2 == "Analyzer":
    #title
	#st.title('Not Done Yet')
	


	nlp = spacy.load('en_core_web_sm')
	#spacy_text_blob = SpacyTextBlob()
	# nlp.add_pipe("SpacyTextBlob")
	nlp.add_pipe('spacytextblob')
   
	text = 'I am happy to get the covid vaccine!'

	st.title('Sentiment Analyzer')
	user_input = st.text_input("Text", text)
	doc = nlp(user_input)

	st.write('Polarity:', round(doc._.blob.polarity, 2))
	st.write('Subjectivity:', round(doc._.blob.subjectivity, 2))

	score = round(doc._.blob.polarity, 2)

	if score==0:st.write("Neutral 😐")
	elif score<0:st.write("Negative 😫")
	elif score>0:st.write("Positive 😀")
