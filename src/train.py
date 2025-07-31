import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import dump
import os

# Import all the models we want to test
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb

print("--- Starting Model Training and Selection ---")

# --- Load and Prepare Data ---
df = pd.read_csv("data/raw_posts.csv")
df['text'] = df['title'] + ' ' + df['body'].fillna('')
df['label'] = df['classification'].apply(lambda label: 1 if label == 'NSFW' else 0)

# --- Preprocessing & Feature Engineering ---
print("Vectorizing text data...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2)) 
X = vectorizer.fit_transform(df['text'])
y = df['label']

# --- Split Data into Training and Testing Sets ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Data split into {X_train.shape[0]} training and {X_test.shape[0]} testing samples.")

# --- Define Models to Test ---
scale_pos_weight_value = y_train.value_counts()[0] / y_train.value_counts()[1]
models = {
    "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
    "LinearSVC": LinearSVC(random_state=42, class_weight='balanced', dual="auto"),
    "MultinomialNB": MultinomialNB(),
    "LightGBM": lgb.LGBMClassifier(random_state=42, scale_pos_weight=scale_pos_weight_value),
    "MLPClassifier": MLPClassifier(random_state=42, max_iter=20, hidden_layer_sizes=(100, 50), early_stopping=True)
}

best_model = None
best_f1_score = -1
best_model_name = ""

# --- Loop Through Models to Find the Best One ---
for model_name, model in models.items():
    print(f"\n--- Training {model_name} ---")
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    report = classification_report(y_test, y_pred_test, output_dict=True)
    nsfw_f1 = report['1']['f1-score']
    print(f"NSFW F1-Score for {model_name}: {nsfw_f1:.4f}")

    if nsfw_f1 > best_f1_score:
        best_f1_score = nsfw_f1
        best_model = model
        best_model_name = model_name

print(f"\n🏆 Champion model selected: '{best_model_name}' with F1-Score: {best_f1_score:.4f}")

# --- Save the Final Model and Vectorizer to the Root Directory ---
# These files will be committed to the repository using Git LFS.
print("\nSaving champion model to 'champion_model.pkl'...")
dump(best_model, 'champion_model.pkl')

print("Saving vectorizer to 'vectorizer.joblib'...")
dump(vectorizer, 'vectorizer.joblib')

print("\n✅ Training complete. Model and vectorizer are saved and ready for deployment.")