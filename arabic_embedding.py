import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder

# Load the CSV file
training_data = pd.read_csv("New_News_Train.csv")
testing_data = pd.read_csv("New_News_Test.csv")

tfidf = TfidfVectorizer(lowercase=False)
X_train = tfidf.fit_transform(training_data["Removed_stopwords"])
X_test = tfidf.transform(testing_data["Removed_stopwords"])

print(f"Training shape: {X_train.shape}")
print(f"Testing shape: {X_test.shape}")


# Encode the labels
encoder = LabelEncoder()
y_train = encoder.fit_transform(training_data["Type"])
y_test = encoder.transform(testing_data["Type"])


# Train a multi-class classifier
classifier = OneVsRestClassifier(SGDClassifier())
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
