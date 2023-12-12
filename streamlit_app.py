import pickle

# For building application
import streamlit as st

# Basic data manipulation:
import numpy as np
import pandas as pd

# NLP
import spacy
nlp = spacy.load('en_core_web_sm')

# Page configuration
st.set_page_config(
  page_title='TrustTracker',
  page_icon='👌',
  initial_sidebar_state='expanded')
  
# Function to load the dataset
@st.experimental_memo
def load_data():
    # Define the file path
    file_path = 'https://raw.githubusercontent.com/MikkelONielsen/TrustTracker/main/trust_pilot_reviews_data_2022_06.csv'
    
    # Load the CSV file into a pandas dataframe
    df = pd.read_csv(file_path)
    
    return df


# Load the data using the defined function
df = load_data()

pipe_svm = pickle.load(open('data/model.pkl', 'rb'))

# Defining functions
def text_prepro(texts: pd.Series) -> list:
  # Creating a container for the cleaned texts
  clean_container = []
  # Using spacy's nlp.pipe to preprocess the text
  for doc in nlp.pipe(texts):
    # Extracting lemmatized tokens that are not punctuations, stopwords or non-alphabetic characters
    words = [words.lemma_.lower() for words in doc
            if words.is_alpha and not words.is_stop and not words.is_punct]

    # Adding the cleaned tokens to the container "clean_container"
    clean_container.append(" ".join(words))

  return clean_container

# Defining the ML function
def predict(placetext):
  text_ready = []
  text_ready = text_prepro(pd.Series(placetext))
  result = pipe_svm.predict(text_ready)
  if result == 0:
    return "negative sentiment"
  if result == 1:
    return "positive sentiment"

categories = {
    "Price": [
        "price", "cost", "expensive", "cheap", "value", "pay", "affordable",
        "pricey", "budget", "charge", "fee", "pricing", "rate", "worth", "economical"
    ],
    "Delivery": [
        "deliver", "delivery", "shipping", "dispatch", "courier", "ship", "transit",
        "postage", "mail", "shipment", "logistics", "transport", "send", "carrier", "parcel"
    ],
    "Quality": [
        "quality", "material", "build", "standard", "durability", "craftsmanship",
        "workmanship", "texture", "construction", "condition", "grade", "caliber",
        "integrity", "excellence", "reliability", "sturdiness", "performance"
    ],
    "Service": [
        "service", "support", "assistance", "help", "customer service", "care", "response",
        "satisfaction", "experience", "professionalism", "expertise", "efficiency",
        "friendliness", "availability", "flexibility", "reliability"
    ]
}

def lemmatize_keywords(categories):
    lemmatized_categories = {}
    for category, keywords in categories.items():
        lemmatized_keywords = [nlp(keyword)[0].lemma_ for keyword in keywords]
        lemmatized_categories[category] = lemmatized_keywords
    return lemmatized_categories
    
list_lab = []
def categorize_review(text_review):
    lemmatized_review = " ".join([token.lemma_ for token in nlp(text_review.lower())])
    for category, keywords in lemmatize_keywords(categories).items():
        if any(keyword in lemmatized_review for keyword in keywords):
          list_lab.append(category)
    return list_lab
    if len(list_lab) == 0:
      return "Other"

# Creating the overall company performance

# Creating a sentiment column
df['sentiment']=np.where(df['rating']>=4,1,0) # 1=positive, 0=negative

def overall_and_category_sentiment(df):
    # Calculate the mean sentiment for each company
    overall_sentiment_means = df.groupby('name')['sentiment'].mean()

    # Calculate mean sentiment for each category for each company
    category_sentiments = {}
    for category, keywords in categories.items():
        df_category = df[df['review_text'].str.contains('|'.join(keywords))]
        category_sentiments[category] = df_category.groupby('name')['sentiment'].mean()

    # Determine overall sentiment and category sentiment
    overall = overall_sentiment_means.apply(lambda x: 'Positive' if x >= 0.5 else 'Negative')
    category_sentiment_labels = {category: sentiments.apply(lambda x: 'Positive' if x >= 0.5 else 'Negative') for category, sentiments in category_sentiments.items()}

    return overall, category_sentiment_labels

# Apply the function to your DataFrame
overall_scores, category_scores = overall_and_category_sentiment(df)

def check_company_sentiment(company_name):
    if company_name not in overall_scores:
        return f"{company_name} not found in the data."

    sentiment_info = [f"Overall sentiment for {company_name}: {overall_scores[company_name]}"]
  
    for category, scores in category_scores.items():
        if company_name in scores:
            sentiment_info.append(f"{category} sentiment: {scores[company_name]}")
    return sentiment_info

# The App    
st.title('TrustTracker 👌')
st.markdown('Welcome to TrustTracker! The application where you easily can check the quality, price, service and delivery of your favorite companies.')

tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs(['About', 'Individual Reviews', 'Overall Company Performance', 'Model performance', 'Dataset', 'Visualisations'])


with tab1:
  st.header("About the application")
  st.markdown('This application is divided into 6 tabs:')
  st.markdown('*   **About the application:** The first tab shows the current page, where you can get some information about the application. You can always return to this tab if necessary.')
  st.markdown('*   **Individual Reviews:** In the second tab, you can input a review and get an Aspect-Based Sentiment Analysis of the review that show if the review regards price, quality, service or delivery and if the review is positive or negative.')
  st.markdown('*   **Overall Company Performance:** The third tab contains an overall Aspect-Based Sentiment Analysis for a company. Here you can choose a company from the list and get an output telling you if consumers overall have a positive or negative sentiment regarding the company and a separate sentiment for each of the chosen categories.')
  st.markdown('*   **Model Performance:** The fourth tab explains how the underlying Machine Learning Model performs and how the predictor works.')
  st.markdown('*   **Dataset:** In the fifth tab, you can see where the origin of the dataset that has been used to build the application.')
  st.markdown('*   **Visualisations:** Lastly, for those who are curious about the distribution of the dataset, the application includes an exploratory data analysis of the different variables.')
        
with tab2:

  st.header('Predict Individual Reviews')
  st.write('This tab includes a traditional Sentiment Analysis for Individual Reviews using TF-IDF and SVM.')
  
  review_txt = st.text_input('Enter your review here')

  if st.button('Predict Aspect-Based Sentiment'):
    category = categorize_review(review_txt)
    sentiment = predict(review_txt)
    st.write(f'This review regards: {", ".join(category)}')
    st.write(f'and has a {sentiment}')

with tab3:
  st.header('Predict Overall Company Performance')
  st.write('')
  
  selected_company = st.selectbox('Select company:', df['name'].unique())

  if st.button('Predict Overall Aspect-Based Sentiment of Selected Company'):
    result = check_company_sentiment(selected_company)
    st.write(result[0])

    # Define a list of indices to display (1, 2, 3, 4 in this case)
    indices_to_display = [1, 2, 3, 4]

    # Loop through the indices and display only if they are in the list
    for index in indices_to_display:
        if index < len(result):
            st.write(result[index])

with tab4:
  st.header('Model performance')

  # Confusion Matrix
  st.subheader("Confusion Matrix")
  st.image('images/svmconfusionmatrix.png')

  # Yellowbrick FreqDistVisualizer
  st.subheader("Yellowbrick FreqDistVisualizer")
  st.image('images/svmyellowbrick.png')

  # WordClouds
  st.subheader("Word Clouds")
  st.image('images/wordclouds.png')

with tab5:
  st.header('Dataset')
  # Display dataset overview
  st.subheader("Dataset Overview")
  st.dataframe(df.head())
  
with tab6:
  st.header('Visualisations')

  # Reviews by number of companies
  st.subheader("Reviews by Number of Companies")
  st.image('images/reviewsbycompanies.png')

  # Reviews by year
  st.subheader("Reviews by Year")
  st.image('images/reviewsbyyear.png')

  # Reviews by month in 2022
  st.subheader("Reviews by Month in 2022")
  st.image('images/reviewsbymonth.png')
  
  # Reviews by rating
  st.subheader("Reviews by Rating")
  st.image('images/reviewsbyrating.png')

  # Reviews by user
  st.subheader("Reviews by Consumer Name")
  st.image('images/reviewsbyauthor.png')
