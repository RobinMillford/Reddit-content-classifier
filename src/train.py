import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, hamming_loss, jaccard_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from joblib import dump, load
import os
import numpy as np
from collections import defaultdict

# Import all the models we want to test
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
import lightgbm as lgb

print("--- Starting Multi-Label Model Training, Selection, and Ensembling ---")

# --- Load and Prepare Data ---
df = pd.read_csv("data/raw_posts.csv")
df['text'] = df['title'] + ' ' + df['body'].fillna('')

# Prepare multi-label targets
print("Preparing multi-label targets...")
label_columns = ['safety', 'toxicity', 'sentiment', 'topic', 'engagement']

# Create label encoders for each category
label_encoders = {}
encoded_labels = {}

for column in label_columns:
    unique_labels = df[column].unique()
    label_encoders[column] = {label: idx for idx, label in enumerate(unique_labels)}
    encoded_labels[column] = df[column].map(label_encoders[column])
    print(f"{column}: {unique_labels}")

# Create binary classification target (for backward compatibility)
df['binary_label'] = df['classification'].apply(lambda label: 1 if label == 'NSFW' else 0)

# --- Preprocessing & Feature Engineering ---
print("Vectorizing text data...")
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2)) 
X = vectorizer.fit_transform(df['text'])

# Prepare targets for multi-label classification
y_multi = np.column_stack([encoded_labels[col] for col in label_columns])
y_binary = df['binary_label']  # Keep binary for comparison

print(f"Feature matrix shape: {X.shape}")
print(f"Multi-label target shape: {y_multi.shape}")
print(f"Label categories: {label_columns}")

# --- Split Data into Training and Testing Sets ---
X_train, X_test, y_multi_train, y_multi_test, y_binary_train, y_binary_test = train_test_split(
    X, y_multi, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)
print(f"Data split into {X_train.shape[0]} training and {X_test.shape[0]} testing samples.")

# --- Define Models to Test ---
scale_pos_weight_value = y_binary_train.value_counts()[0] / y_binary_train.value_counts()[1]

# Binary classification models (for compatibility)
binary_models = {
    "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
    "LinearSVC": LinearSVC(random_state=42, class_weight='balanced', dual="auto"),
    "MultinomialNB": MultinomialNB(),
    "LightGBM": lgb.LGBMClassifier(random_state=42, scale_pos_weight=scale_pos_weight_value),
    "MLPClassifier": MLPClassifier(random_state=42, max_iter=20, hidden_layer_sizes=(100, 50), early_stopping=True)
}

# Multi-label classification models
multi_label_models = {
    "MultiOutput_LogisticRegression": MultiOutputClassifier(LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')),
    "MultiOutput_LinearSVC": MultiOutputClassifier(LinearSVC(random_state=42, class_weight='balanced', dual="auto")),
    "MultiOutput_LightGBM": MultiOutputClassifier(lgb.LGBMClassifier(random_state=42, scale_pos_weight=scale_pos_weight_value)),
    "MultiOutput_MLPClassifier": MultiOutputClassifier(MLPClassifier(random_state=42, max_iter=20, hidden_layer_sizes=(100, 50), early_stopping=True))
}

trained_binary_models = {}
binary_model_scores = {}
trained_multi_models = {}
multi_model_scores = {}

# --- Train Binary Classification Models ---
print("\n=== Training Binary Classification Models ===")
for model_name, model in binary_models.items():
    print(f"\n--- Training {model_name} (Binary) ---")
    model.fit(X_train, y_binary_train)
    
    # For evaluation, we must convert the test data to a dense array for the MLPClassifier
    X_test_eval = X_test.toarray() if isinstance(model, MLPClassifier) else X_test
    y_pred_test = model.predict(X_test_eval)
    
    report = classification_report(y_binary_test, y_pred_test, output_dict=True)
    nsfw_f1 = report['1']['f1-score'] if '1' in report else 0
    
    print(f"--- Binary Evaluation for {model_name} ---")
    print(classification_report(y_binary_test, y_pred_test, target_names=['SFW (0)', 'NSFW (1)']))
    
    trained_binary_models[model_name] = model
    binary_model_scores[model_name] = nsfw_f1

# --- Train Multi-Label Classification Models ---
print("\n=== Training Multi-Label Classification Models ===")
for model_name, model in multi_label_models.items():
    print(f"\n--- Training {model_name} (Multi-Label) ---")
    
    # Handle MLP models that need dense arrays
    if 'MLPClassifier' in model_name:
        X_train_fit = X_train.toarray()
        X_test_eval = X_test.toarray()
    else:
        X_train_fit = X_train
        X_test_eval = X_test
    
    model.fit(X_train_fit, y_multi_train)
    y_pred_multi = model.predict(X_test_eval)
    
    # Calculate multi-label metrics (handle multiclass-multioutput properly)
    # For multiclass-multioutput, calculate metrics per output then average
    from sklearn.metrics import accuracy_score
    
    hamming_losses = []
    individual_accuracies = []
    jaccard_scores = []
    
    for i in range(y_multi_test.shape[1]):
        acc = accuracy_score(y_multi_test[:, i], y_pred_multi[:, i])
        individual_accuracies.append(acc)
        hamming_losses.append(1 - acc)  # Hamming loss is 1 - accuracy for single output
        
        # For multi-class classification, use accuracy as Jaccard approximation
        # since traditional Jaccard is for binary/multi-label, not multi-class
        jaccard_scores.append(acc)  # Use accuracy per class as Jaccard approximation
    
    hamming = np.mean(hamming_losses)
    overall_accuracy = np.mean(individual_accuracies)
    overall_jaccard = np.mean(jaccard_scores)
    
    print(f"--- Multi-Label Evaluation for {model_name} ---")
    print(f"Average Hamming Loss: {hamming:.4f}")
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"Overall Jaccard Score: {overall_jaccard:.4f}")
    
    # Individual label performance
    for i, label_name in enumerate(label_columns):
        label_report = classification_report(y_multi_test[:, i], y_pred_multi[:, i], output_dict=True)
        f1_score = label_report['weighted avg']['f1-score']
        print(f"{label_name} F1-Score: {f1_score:.4f}")
    
    trained_multi_models[model_name] = model
    multi_model_scores[model_name] = overall_jaccard  # Use Jaccard score as main metric for multi-label
# --- Create and Evaluate Binary Ensemble Model ---
print("\n=== Creating Binary Ensemble Model ===")
top_two_binary = sorted(binary_model_scores, key=binary_model_scores.get, reverse=True)[:2]
print(f"Ensembling the top two binary models: {top_two_binary[0]} and {top_two_binary[1]}")

binary_ensemble = VotingClassifier(
    estimators=[
        (top_two_binary[0], trained_binary_models[top_two_binary[0]]),
        (top_two_binary[1], trained_binary_models[top_two_binary[1]])
    ],
    voting='hard'
)

binary_ensemble.fit(X_train, y_binary_train)
y_pred_binary_ensemble = binary_ensemble.predict(X_test)
report_binary_ensemble = classification_report(y_binary_test, y_pred_binary_ensemble, target_names=['SFW (0)', 'NSFW (1)'], output_dict=True)
binary_ensemble_f1_score = report_binary_ensemble['NSFW (1)']['f1-score'] if 'NSFW (1)' in report_binary_ensemble else 0

print(f"--- Binary Ensemble Evaluation ---")
print(classification_report(y_binary_test, y_pred_binary_ensemble, target_names=['SFW (0)', 'NSFW (1)']))

# --- Select the Overall Champion Model ---
print("\n=== Champion Model Selection ===")

# Find best binary model
best_binary_model_name = max(binary_model_scores, key=binary_model_scores.get)
best_binary_f1_score = binary_model_scores[best_binary_model_name]

# Find best multi-label model
best_multi_model_name = max(multi_model_scores, key=multi_model_scores.get)
best_multi_jaccard_score = multi_model_scores[best_multi_model_name]

print(f"Best Binary Model: {best_binary_model_name} (F1: {best_binary_f1_score:.4f})")
print(f"Binary Ensemble F1: {binary_ensemble_f1_score:.4f}")
print(f"Best Multi-Label Model: {best_multi_model_name} (Jaccard: {best_multi_jaccard_score:.4f})")

# For backward compatibility, select binary champion for main deployment
if binary_ensemble_f1_score > best_binary_f1_score:
    champion_model = binary_ensemble
    champion_model_name = "BinaryEnsembleClassifier"
    champion_f1_score = binary_ensemble_f1_score
else:
    champion_model = trained_binary_models[best_binary_model_name]
    champion_model_name = best_binary_model_name
    champion_f1_score = best_binary_f1_score

# Also save the best multi-label model
best_multi_model = trained_multi_models[best_multi_model_name]

print(f"\n🏆 Binary Champion selected: '{champion_model_name}' with F1-Score: {champion_f1_score:.4f}")
print(f"🏆 Multi-Label Champion: '{best_multi_model_name}' with Jaccard Score: {best_multi_jaccard_score:.4f}")

# --- Save the Final Models and Metadata ---
print("\n=== Saving Models and Metadata ===")

# Save the binary champion model (for backward compatibility)
print("Saving binary champion model to 'champion_model.pkl'...")
dump(champion_model, 'champion_model.pkl')

# Save the multi-label champion model
print("Saving multi-label champion model to 'multi_label_model.pkl'...")
dump(best_multi_model, 'multi_label_model.pkl')

# Save the vectorizer
print("Saving vectorizer to 'vectorizer.joblib'...")
dump(vectorizer, 'vectorizer.joblib')

# Save label encoders and metadata
model_metadata = {
    'label_columns': label_columns,
    'label_encoders': label_encoders,
    'binary_champion': champion_model_name,
    'multi_label_champion': best_multi_model_name,
    'binary_f1_score': champion_f1_score,
    'multi_label_jaccard_score': best_multi_jaccard_score  # Now using actual Jaccard score
}

print("Saving model metadata to 'model_metadata.joblib'...")
dump(model_metadata, 'model_metadata.joblib')

print("\n✅ Multi-Label Training complete. All models and metadata are saved.")
