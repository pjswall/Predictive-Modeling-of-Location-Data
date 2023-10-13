import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from joblib import dump, load
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv('preprocessed_data.csv')

df = df.dropna(subset=['location_name'])

X = df['location_name']
y = df[['category', 'country_code']]

# Convert location names to a matrix of TF-IDF features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X).toarray()  # Convert sparse matrix to array for GaussianNB

# Encode the targets
label_encoders = {}
for column in y.columns:
    le = LabelEncoder()
    y[column] = le.fit_transform(y[column])
    label_encoders[column] = le  # Store the label encoder for each column

print(X.shape, y.shape)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create the model
gnb = GaussianNB()
multi_target_gnb = MultiOutputClassifier(gnb, n_jobs=-1)

# Train the model
multi_target_gnb.fit(X_train, y_train)
dump(multi_target_gnb, 'gnb.joblib')

model = load('gnb.joblib')
y_pred = model.predict(X_test)

# Print the classification report for each target feature
for idx, target_name in enumerate(y_test.columns):
    print("Classification report for target feature '{}':".format(target_name))
    print(classification_report(y_test.iloc[:, idx], y_pred[:, idx]))
