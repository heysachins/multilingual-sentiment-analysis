import pandas as pd # Imported to enable the use of datastructures like dataframe
from sklearn.feature_extraction.text import TfidfVectorizer # Imported to convert raw documents into a matrix of tf idf features
from sklearn.linear_model import LogisticRegression # Imported to enable the use of logistic regression to classify text
from sklearn.model_selection import train_test_split # Imported to enable the user to split the data into train, test samples
from sklearn.metrics import classification_report, accuracy_score # Imported to calculate the accuracy and also print the classification report

# Importing the dataset
df = pd.read_csv('malayalam_train.tsv', sep='\t')
print(df)

# Preprocess the data (if necessary)


# Exploring the dataset
print(df.shape())
df.head()



# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(
    df.text, 
    df.category,
    test_size=0.2, # 20 % of samples will be present in test dataset
    random_state=42)

# Vectorization using TF-IDF
v = TfidfVectorizer()   
X_train_tfidf = v.fit_transform(X_train)
X_test_tfidf = v.transform(X_test)

# Logistic Regression model
log_reg = LogisticRegression()
log_reg.fit(X_train_tfidf, y_train)

# Predictions and evaluation
predictions = log_reg.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))
