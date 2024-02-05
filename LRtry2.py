import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Load the dataset
df = pd.read_csv('malayalam_train.tsv', sep='\t', header=0)

# Preprocessing function
def preprocess_text(text):
    # Remove URLs, hashtags, and mentions
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)

    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)

    # Lowercase conversion
    text = text.lower()

    # Add other preprocessing steps as needed

    return text

# Applying the preprocessing function
df['text'] = df['text'].apply(preprocess_text)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['category'], test_size=0.2, random_state=42)

# Building the Pipeline for TF-IDF and Logistic Regression
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', LogisticRegression()),
])

# Training the model
pipeline.fit(X_train, y_train)

# Evaluating the model
predictions = pipeline.predict(X_test)
print(classification_report(y_test, predictions))
