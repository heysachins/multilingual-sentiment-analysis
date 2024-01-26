import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample dataset
documents = ["I love this product", "I hate this product", "This is the best purchase", "This is the worst purchase"]
labels = [1, 0, 1, 0]  # 1 for positive sentiment, 0 for negative sentiment

# Preprocessing
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

# Calculating TF-IDF
tfidf = TfidfVectorizer(stop_words=stop_words, lowercase=True)
tfidf_features = tfidf.fit_transform(documents)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(tfidf_features, labels, test_size=0.3, random_state=42, stratify=labels)

# Training a classifier
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Predicting the sentiment
predictions = classifier.predict(X_test)

# Evaluating the model
print("Accuracy:", accuracy_score(y_test, predictions))
