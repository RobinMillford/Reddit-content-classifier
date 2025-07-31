# End-to-End MLOps: Automated Reddit Content Classifier

[![Scheduled Model Retraining](https://github.com/RobinMillford/Reddit-content-classifier/actions/workflows/main.yml/badge.svg)](https://github.com/RobinMillford/Reddit-content-classifier/actions/workflows/main.yml)

This repository contains the source code for a complete, end-to-end MLOps project that automatically trains, evaluates, and deploys a machine learning model to classify Reddit content as Safe-For-Work (SFW) or Not-Safe-For-Work (NSFW).

---

## 🚀 Live Demo

**You can try the live application here:**

**[https://Reddit-content-classifier/](https://reddit-content-classifier.streamlit.app/)** *(Note: The app may take 30-60 seconds to wake up on the first visit due to the free hosting tier.)*

---

## 📋 Project Overview

The goal of this project was to build a robust, automated pipeline for a real-world text classification task. The system is designed to be self-sustaining, requiring no manual intervention to keep the model updated with the latest trends and language from Reddit.

This project demonstrates a full MLOps lifecycle, including:
- **Automated Data Ingestion:** Sourcing fresh data from a live, dynamic source (Reddit API).
- **Automated Model Training:** Experimenting with multiple classical and deep learning models.
- **Automated Model Selection:** Programmatically identifying the best-performing model based on metrics.
- **Model Versioning:** Using Git LFS to track model artifacts alongside the code that produced them.
- **Continuous Integration & Deployment (CI/CD):** Automatically deploying the updated application whenever a new, better model is trained and pushed to the repository.

## 🛠️ Tech Stack

- **Programming Language:** Python
- **Data Ingestion:** PRAW (Python Reddit API Wrapper)
- **Data Handling:** Pandas
- **ML Models:** Scikit-learn (Logistic Regression, LinearSVC, MultinomialNB, MLPClassifier), LightGBM
- **Model Experimentation:** A custom Python script that emulates experimentation loop.
- **Frontend & Serving:** Streamlit Community Cloud
- **Automation (CI/CD):** GitHub Actions
- **Large File Storage:** Git LFS

---

## ⚙️ How the MLOps Pipeline Works

This project is more than just a model; it's a fully automated system. The entire pipeline is orchestrated by a GitHub Actions workflow.

![MLOps Pipeline Diagram](https://github.com/RobinMillford/Reddit-content-classifier/blob/main/Workflow.png)

**The Automated Workflow:**

1.  **Scheduled Trigger:** A GitHub Action is scheduled to run automatically every three days (and can also be triggered manually). This kicks off the entire process.

2.  **Automated Data Ingestion:**
    - The workflow runs the `src/ingest_data.py` script.
    - This script connects to the Reddit API using credentials stored securely in GitHub Secrets.
    - It fetches thousands of the latest posts from a curated list of SFW, NSFW, and mixed-content subreddits.
    - It processes and labels each post, creating a fresh, up-to-date `data/raw_posts.csv` file inside the temporary runner environment.

3.  **Automated Model Training & Selection:**
    - The workflow then runs the `src/train.py` script.
    - This script trains multiple models (Logistic Regression, LinearSVC, MLPClassifier, etc.) on the new dataset.
    - It evaluates each model's performance on a test set, focusing on the F1-score for the NSFW class.
    - It then creates an ensemble `VotingClassifier` using the top two best-performing models.
    - Finally, it compares the ensemble's score to the best individual model's score and saves the overall winner as `champion_model.pkl` and the corresponding text vectorizer as `vectorizer.joblib`.

4.  **Automated Model Push:**
    - The GitHub Action then stages the newly created `champion_model.pkl` and `vectorizer.joblib` files.
    - It automatically commits these large files (handled by Git LFS) and pushes them back to the `main` branch of this repository. The commit message indicates an automated model update.

5.  **Automated Deployment:**
    - Streamlit Community Cloud is connected to this GitHub repository.
    - It detects the new push to the `main` branch.
    - It automatically pulls the latest version of the code, including the new model files from Git LFS.
    - It installs the dependencies from `requirements.txt` and re-deploys the `app.py` application.

The result is a live application that is automatically kept up-to-date with the best model trained on the freshest data, all without any manual intervention.

---

## 📂 Project Structure
```
├── .github/
│   └── workflows/
│       └── retrain.yml       # GitHub Action for automated training
├── src/
│   ├── ingest_data.py      # Script to fetch and process data from Reddit
│   └── train.py            # Script to train, evaluate, and save the best model
├── .gitattributes          # Configures Git LFS to handle large model files
├── .gitignore              # Specifies which files to exclude from Git
├── app.py                  # The main Streamlit application file (UI and logic)
├── champion_model.pkl      # The current production-ready model (tracked by Git LFS)
├── requirements.txt        # Python dependencies for the project
└── vectorizer.joblib       # The vectorizer for the production model (tracked by Git LFS)
```

## 🚀 How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/RobinMillford/Reddit-content-classifier.git](https://github.com/RobinMillford/Reddit-content-classifier.git)
    cd Reddit-content-classifier
    ```

2.  **Install Git LFS:**
    * Download and install Git LFS from [git-lfs.github.com](https://git-lfs.github.com/).
    * Set it up by running: `git lfs install`

3.  **Pull LFS Files:**
    * Download the model files from Git LFS: `git lfs pull`

4.  **Set up Environment:**
    * Create a Python virtual environment: `python -m venv venv`
    * Activate it: `source venv/bin/activate` (on Linux/macOS) or `venv\Scripts\activate` (on Windows)
    * Install dependencies: `pip install -r requirements.txt`

5.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```
    The application will open in your web browser.

---

## 🔮 Future Improvements

This project provides a solid MLOps foundation. Potential next steps could include:

- **Data Drift Detection:** Implement a step in the pipeline to analyze the statistical properties of new data and only trigger a full retrain if significant drift is detected.
- **Hyperparameter Tuning:** Use a framework like Optuna or Hyperopt to automatically find the best hyperparameters for the models during each training run.
- **More Advanced Models:** Experiment with more complex models like Transformers (BERT) for classification, which would require a more powerful training environment.
- **Dedicated Database:** Replace the CSV file with a proper database (like PostgreSQL or a vector database) for more scalable data storage and retrieval.
