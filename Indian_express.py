# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Streamlit App Title
st.title("Topic Modeling on The Indian Express News Articles")

# Dataset Selection
dataset_options = {
    "Business Data": r"C:\Users\HP\Downloads\archive\business_data.csv",
    "Politics Data": r"C:\Users\HP\Downloads\archive\politics_data.csv",
    "Sports Data": r"C:\Users\HP\Downloads\archive\sports_data.csv",
    "Entertainment Data": r"C:\Users\HP\Downloads\archive\entertainment_data.csv",
    "Technology Data": r"C:\Users\HP\Downloads\archive\technology_data.csv"
}

# User selects dataset
selected_dataset = st.selectbox("Select Dataset", list(dataset_options.keys()))

# Load the selected dataset
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

data = load_data(dataset_options[selected_dataset])

# Display the dataset
if st.checkbox("Show Dataset"):
    st.write(data)
    st.write("Columns in the dataset:", data.columns)  # Display column names for debugging

# Check if 'content' column exists
if 'content' not in data.columns:
    st.error("The dataset does not contain a 'content' column. Please check the dataset.")
else:
    # Check class distribution
    class_counts = data['category'].value_counts()
    st.write("Class Distribution:")
    st.bar_chart(class_counts)

    # Exploratory Data Analysis (EDA)
    if st.checkbox("Show EDA"):
        st.subheader("Distribution of Articles by Category")
        fig, ax = plt.subplots()
        sns.countplot(x='category', data=data, ax=ax)
        st.pyplot(fig)

        # Average word counts per category
        data['word_count'] = data['content'].apply(lambda x: len(str(x).split()))
        avg_word_count = data.groupby('category')['word_count'].mean()
        st.write("Average Word Count per Category:")
        st.write(avg_word_count)

    # Text Preprocessing
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    def preprocess_text(text):
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'\d+', '', text)  # Remove numbers
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        tokens = text.split()  # Tokenize
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]  # Remove stop words and lemmatize
        return ' '.join(tokens)

    # Apply preprocessing
    data['Processed_Content'] = data['content'].apply(preprocess_text)

    # Feature Extraction
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X = tfidf_vectorizer.fit_transform(data['Processed_Content']).toarray()
    y = data['category']

    # Split the dataset into training and testing sets using stratified sampling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Model Building and Training
    models = {
        'Logistic Regression': LogisticRegression(),
        'Multinomial Naive Bayes': MultinomialNB(),
        'Random Forest': RandomForestClassifier()
    }

    # Train and evaluate models
    if st.button("Train Models"):
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            st.subheader(f"Model: {model_name}")
            st.write("Accuracy:", accuracy_score(y_test, y_pred))
            st.write(classification_report(y_test, y_pred))
            st.write(confusion_matrix(y_test, y_pred))

    # Hyperparameter Tuning (Example for Random Forest)
    if st.button("Hyperparameter Tuning"):
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30]
        }

        grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=3)
        grid_search.fit(X_train, y_train)

        st.write("Best parameters:", grid_search.best_params_)

        # Save the model
        joblib.dump(grid_search.best_estimator_, 'best_random_forest_model.pkl')

    # Load the model
    model = joblib.load('best_random_forest_model.pkl')

    # Generate predictions on new data
    new_data = st.text_area("Enter new article text for prediction:")
    if st.button("Predict Category"):
        new_data_processed = [preprocess_text(new_data)]
        new_data_features = tfidf_vectorizer.transform(new_data_processed).toarray()
        predictions = model.predict(new_data_features)
        st.write("Predicted Category:", predictions)

    # Question Input
    question = st.text_input("Ask a question about the articles:")
    if st.button("Get Answer"):
        if question:
            # Simple keyword matching to find relevant articles
            relevant_articles = data[data['Processed_Content'].str.contains(re.sub(r'[^\w\s]', '', question.lower()), na=False)]
            if not relevant_articles.empty:
                st.write("Relevant Articles:")
                st.write(relevant_articles[['content', 'category']])
            else:
                st.write("No relevant articles found.")
        else:
            st.write("Please enter a question.")

    # Generate classification report for the best model
    if st.button("Show Final Model Report"):
        y_pred_final = model.predict(X_test)
        st.write(classification_report(y_test, y_pred_final))
