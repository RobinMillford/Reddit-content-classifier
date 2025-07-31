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
from sklearn.ensemble import VotingClassifier
import lightgbm as lgb

print("--- Starting Model Training, Selection, and Ensembling ---")

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

trained_models = {}
model_scores = {}

# --- Loop Through Individual Models to Find the Best Ones ---
for model_name, model in models.items():
    print(f"\n--- Training {model_name} ---")
    model.fit(X_train, y_train)
    
    # For evaluation, we must convert the test data to a dense array for the MLPClassifier
    X_test_eval = X_test.toarray() if isinstance(model, MLPClassifier) else X_test
    y_pred_test = model.predict(X_test_eval)
    
    report = classification_report(y_test, y_pred_test, output_dict=True)
    nsfw_f1 = report['1']['f1-score']
    
    print(f"--- Evaluation for {model_name} ---")
    print(classification_report(y_test, y_pred_test, target_names=['SFW (0)', 'NSFW (1)']))
    
    trained_models[model_name] = model
    model_scores[model_name] = nsfw_f1

# --- Create and Evaluate Ensemble Model ---
print("\n--- Creating and Evaluating Ensemble Model ---")
top_two_models = sorted(model_scores, key=model_scores.get, reverse=True)[:2]
print(f"Ensembling the top two models: {top_two_models[0]} and {top_two_models[1]}")

ensemble_model = VotingClassifier(
    estimators=[
        (top_two_models[0], trained_models[top_two_models[0]]),
        (top_two_models[1], trained_models[top_two_models[1]])
    ],
    voting='hard'
)

ensemble_model.fit(X_train, y_train)
y_pred_ensemble = ensemble_model.predict(X_test)
report_ensemble = classification_report(y_test, y_pred_ensemble, target_names=['SFW (0)', 'NSFW (1)'], output_dict=True)
ensemble_f1_score = report_ensemble['NSFW (1)']['f1-score']

print(f"--- Evaluation for Ensemble Model ---")
print(classification_report(y_test, y_pred_ensemble, target_names=['SFW (0)', 'NSFW (1)']))

# --- Select the Overall Champion Model ---
best_individual_model_name = max(model_scores, key=model_scores.get)
best_individual_f1_score = model_scores[best_individual_model_name]

if ensemble_f1_score > best_individual_f1_score:
    champion_model = ensemble_model
    champion_model_name = "EnsembleClassifier"
    champion_f1_score = ensemble_f1_score
else:
    champion_model = trained_models[best_individual_model_name]
    champion_model_name = best_individual_model_name
    champion_f1_score = best_individual_f1_score

print(f"\n🏆 Overall Champion model selected: '{champion_model_name}' with F1-Score: {champion_f1_score:.4f}")

# --- Save the Final Model and Vectorizer ---
print("\nSaving champion model to 'champion_model.pkl'...")
dump(champion_model, 'champion_model.pkl')

print("Saving vectorizer to 'vectorizer.joblib'...")
dump(vectorizer, 'vectorizer.joblib')

print("\n✅ Training complete. Champion model and vectorizer are saved.")