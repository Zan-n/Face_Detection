# Bangla Fake News Detection

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange?logo=jupyter)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red?logo=pytorch)
![HuggingFace](https://img.shields.io/badge/Model-BERT-yellow?logo=huggingface)

A machine learning and deep learning project for detecting fake news in Bangla text. The project uses a labelled Bangla news dataset containing headlines, article content, categories, and labels. It includes exploratory data analysis, Bangla text preprocessing, TF-IDF feature extraction, multiple classical machine learning models, and a BERT-based classifier.

## Project Overview

Fake news spreads quickly through online platforms and can mislead readers, damage trust, and create social confusion. This project aims to classify Bangla news articles as either **authentic** or **fake** using natural language processing and supervised learning.

The notebook explores the dataset, visualises text patterns, cleans Bangla text, trains several machine learning models, evaluates their performance, and tests prediction on custom Bangla news text.

## Dataset

The project uses the dataset `fakenewsdataset - 3045.csv`, which contains **3,044 records** and **4 columns**:

| Column | Description |
|---|---|
| `category` | News category (politics, sports, national, international, entertainment, technology, etc.) |
| `headline` | Bangla news headline |
| `content` | Full Bangla news article body |
| `label` | Target class — `0` for fake news, `1` for authentic news |

### Dataset Distribution

| Label | Count |
|---|---:|
| Authentic news (`1`) | 1,900 |
| Fake news (`0`) | 1,144 |

## Project Structure

```text
.
├── fake_news_detection.ipynb        # Main Jupyter / Google Colab notebook
├── fakenewsdataset - 3045.csv       # Bangla fake news dataset
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
├── LICENSE                          # MIT License
└── .gitignore
```

After running the notebook, the following model files are generated:

```text
tfidf_vectorizer.pkl
logistic_regression_model.pkl
naive_bayes_model.pkl
svm_model.pkl
random_forest_model.pkl
knn_model.pkl
decision_tree_model.pkl
gradient_boosting_model.pkl
adaboost_model.pkl
xgboost_model.pkl
best_bert_model.pt
```

## Technologies Used

| Area | Tools |
|---|---|
| Language | Python 3.8+ |
| Data & Visualisation | Pandas, NumPy, Matplotlib, Seaborn, WordCloud |
| NLP & Preprocessing | BNLP Toolkit, Scikit-learn TF-IDF |
| Classical ML | Logistic Regression, Naive Bayes, SVM, Random Forest, KNN, Decision Tree, Gradient Boosting, AdaBoost, XGBoost |
| Deep Learning | PyTorch, Hugging Face Transformers (`bert-base-multilingual-cased`) |
| Serialisation | Joblib |
| Environment | Jupyter Notebook / Google Colab |

## Workflow

### 1. Exploratory Data Analysis

- Separate authentic and fake news subsets
- Analyse headline and content length distributions
- Generate Bangla word clouds for each class
- Identify most frequent words in authentic vs fake news
- Visualise category-wise and label-wise distributions

### 2. Bangla Text Preprocessing

Using the `BasicTokenizer` from the BNLP toolkit:

```python
def clean_text(text):
    tokens = btokenizer.tokenize(text)
    filtered = [
        token for token in tokens
        if token not in stopwords
        and token not in punctuations
        and token not in digits
    ]
    return " ".join(filtered)
```

Preprocessing steps include tokenisation, stopword removal, punctuation removal, digit removal, and combining cleaned headline and content into a single feature.

### 3. Feature Extraction

TF-IDF vectorisation with bigrams for classical ML models:

```python
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
X_train_tfidf = tfidf.fit_transform(X_train_text)
X_test_tfidf = tfidf.transform(X_test_text)
```

### 4. Model Training & Evaluation

Nine classical ML models and one BERT-based deep learning model are trained and evaluated using accuracy, precision, recall, F1-score, confusion matrix, and ROC curve.

### 5. BERT-Based Classifier

A fine-tuned `bert-base-multilingual-cased` model with a custom classification head (dropout + linear layer), trained for 3 epochs using PyTorch.

## Results

### Classical ML Models (TF-IDF Features)

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| **Logistic Regression** | **0.8367** | 0.7927 | 0.9934 | **0.8818** |
| Random Forest | 0.8286 | 0.7829 | 0.9967 | 0.8770 |
| Naive Bayes | 0.8266 | 0.7795 | 1.0000 | 0.8761 |
| Gradient Boosting | 0.8266 | 0.8114 | 0.9342 | 0.8685 |
| XGBoost | 0.8226 | 0.7983 | 0.9507 | 0.8679 |
| KNN | 0.8226 | 0.8121 | 0.9243 | 0.8646 |
| SVM | 0.8185 | 0.8110 | 0.9178 | 0.8611 |
| AdaBoost | 0.8105 | 0.8125 | 0.8980 | 0.8531 |
| Decision Tree | 0.8145 | 0.8419 | 0.8586 | 0.8502 |

### BERT Model

| Model | Validation Accuracy |
|---|---:|
| bert-base-multilingual-cased | ~82.48% |

**Best classical model:** Logistic Regression (83.67% accuracy, 0.88 F1-score)

### Custom Prediction Example

```text
Headline: ঘোড়া এখন কথা বলতে পারে
Content:  ঘোড়া এখন কথা বলতে পারে — এই খবরটি স্থানীয় সূত্রে জানা গেলেও এর কোনো বৈজ্ঞানিক ভিত্তি নেই।
Result:   Fake News ✗
```

## How to Run

### Option 1 — Google Colab

1. Open `fake_news_detection.ipynb` in Google Colab.
2. Upload `fakenewsdataset - 3045.csv` to the Colab environment.
3. Update the dataset path if needed.
4. Run all cells from top to bottom.

### Option 2 — Local Setup

```bash
# Clone the repository
git clone https://github.com/Zan-n/Fake_News_Detection.git
cd Fake_News_Detection

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook fake_news_detection.ipynb
```

## Future Improvements

- Use a larger and more balanced Bangla fake news dataset
- Compare multilingual BERT with Bangla-specific models such as BanglaBERT
- Apply hyperparameter tuning (GridSearchCV / Optuna)
- Add explainable AI methods (LIME / SHAP) for model interpretability
- Build a web application for real-time detection using Streamlit or FastAPI
- Explore data augmentation techniques for the minority class

## License

This project is licensed under the [MIT License](LICENSE).

## Author

**Zannatul Ferdous**
GitHub: [@Zan-n](https://github.com/Zan-n)
