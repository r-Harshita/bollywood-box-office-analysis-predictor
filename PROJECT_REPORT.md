# Bollywood Hit Prediction System — Detailed Project Report

## Table of Contents
1. [Introduction](#introduction)
2. [Problem Statement](#problem-statement)
3. [Dataset](#dataset)
4. [Data Cleaning & Preprocessing](#data-cleaning--preprocessing)
5. [Feature Engineering](#feature-engineering)
6. [Exploratory Data Analysis](#exploratory-data-analysis)
7. [Machine Learning Models](#machine-learning-models)
8. [Results](#results)
9. [Web Application](#web-application)
10. [Conclusion](#conclusion)
11. [Tools & Technologies](#tools--technologies)
12. [Links](#links)

---

## Introduction
The Bollywood film industry is one of the largest in the world,
producing hundreds of movies every year with budgets ranging from
a few crores to hundreds of crores. Despite massive investments,
there is no guaranteed formula for success.

This project uses data analysis and machine learning to identify
the key factors that determine whether a Bollywood movie will be
a commercial Hit or Flop, and deploys the findings as an
interactive web application.

---

## Problem Statement
**Can we predict whether a Bollywood movie will be a Hit or Flop
before its release, using budget and social media engagement data?**

Key questions explored:
- What percentage of Bollywood movies are profitable?
- Does higher budget guarantee box office success?
- Which genres are most profitable?
- Does YouTube engagement predict box office performance?
- Can a Machine Learning model accurately predict Hit or Flop?

---

## Dataset
- **Source:** Kaggle
- **Size:** 149 Bollywood movies
- **Format:** CSV

### Features:
| Column | Description |
|--------|-------------|
| MovieName | Name of the movie |
| Release Date | Date of release |
| Genre | Movie genre |
| Budget | Production budget (in Crores) |
| BoxOfficeCollection | Total box office earnings (in Crores) |
| YoutubeViews | Total YouTube trailer views |
| YoutubeLikes | Total YouTube trailer likes |
| YoutubeDislikes | Total YouTube trailer dislikes |

---

## Data Cleaning & Preprocessing
The following cleaning steps were applied:

1. **Genre Standardization:**
   - Removed leading/trailing whitespace using `str.strip()`
   - Standardized capitalization using `str.title()`
   - This eliminated duplicate genre entries (e.g. "action" and "Action")

2. **Missing Values:**
   - Dataset had zero missing values across all columns
   - No imputation was required

3. **Data Types:**
   - Release Date converted to datetime format using `pd.to_datetime()`
   - Year extracted from Release Date for temporal analysis

```python
df['Genre'] = df['Genre'].str.strip().str.title()
df['Year'] = pd.to_datetime(df['Release Date'], errors='coerce').dt.year
```

---

## Feature Engineering
Two new features were created:

1. **Profit:**
```python
df['Profit'] = df['BoxOfficeCollection'] - df['Budget']
```

2. **HitOrFlop:**
```python
df['HitOrFlop'] = df['Profit'].apply(lambda x: 'Hit' if x > 0 else 'Flop')
```

Movies with positive profit were classified as Hit, negative as Flop.

---

## Exploratory Data Analysis

### Finding 1 — Hit vs Flop Ratio
- 55.7% of movies in the dataset were profitable (Hit)
- 44.3% were unprofitable (Flop)

### Finding 2 — Budget vs Box Office Collection
- Higher budget does NOT always guarantee success
- Several high budget movies flopped while low budget movies thrived
- PK achieved the highest profit (650 Crores) with a moderate budget

### Finding 3 — Genre Analysis
| Genre | Average Profit (Crores) |
|-------|------------------------|
| Drama | 41.5 |
| Action | 34.8 |
| Romance | 27.5 |
| Comedy | 20.8 |
| Thriller | 6.1 |

Drama and Action are the most profitable genres on average.

### Finding 4 — YouTube Engagement vs Box Office
- YouTube Likes shows strong correlation (0.68) with Box Office Collection
- Movies with high YouTube engagement generally perform better
- Social media buzz quality matters more than quantity

### Finding 5 — Correlation Analysis
| Feature Pair | Correlation |
|--------------|-------------|
| BoxOfficeCollection & Profit | 0.96 |
| YoutubeLikes & YoutubeViews | 0.88 |
| YoutubeLikes & BoxOfficeCollection | 0.68 |
| Budget & Profit | 0.42 |

Budget has the weakest correlation with profit (0.42), confirming
that money alone does not buy box office success.

---

## Machine Learning Models

### Features Used:
- Budget
- YoutubeViews
- YoutubeLikes
- YoutubeDislikes

### Target Variable:
- HitOrFlop (Hit = 1, Flop = 0)

### Train/Test Split:
- 80% training, 20% testing
- StandardScaler applied for feature normalization

### Model 1 — Logistic Regression
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
```
- **Accuracy: 63.33%**

### Model 2 — Random Forest Classifier
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
```
- **Accuracy: 66.67%**
- Selected as final model due to superior performance

---

## Results

### Model Performance:
| Model | Accuracy |
|-------|----------|
| Logistic Regression | 63.33% |
| Random Forest Classifier | 66.67% |

### Feature Importance:
| Feature | Importance |
|---------|------------|
| YouTube Likes | 28.6% |
| YouTube Views | 26.6% |
| YouTube Dislikes | 22.7% |
| Budget | 22.1% |

### Key Insight:
YouTube Likes is the single most important predictor of box office
success — more powerful than budget alone.

### Model Testing:
| Scenario | Budget | YouTube Views | YouTube Likes | Prediction |
|----------|--------|---------------|---------------|------------|
| Low budget, viral | 10 Cr | 15M | 900K | HIT (97%) |
| High budget, low buzz | 150 Cr | 1M | 50K | HIT (51%) |
| High budget, low engagement | 100 Cr | 500K | 10K | FLOP (57%) |

---

## Web Application
The project was deployed as an interactive web application using
Streamlit with the following features:

- **Home Page:** Project overview and key statistics
- **Predict Page:** Interactive sliders for real time Hit/Flop prediction
- **Analysis Page:** Data visualizations with tabs
- **Top Movies Page:** Top 10 most profitable movies with data table

### Deployment:
- Platform: Streamlit Community Cloud
- Language: Python
- Deployment time: Under 5 minutes

---

## Conclusion
1. Social media engagement, particularly YouTube Likes, is a
   stronger predictor of Bollywood success than budget
2. Drama and Action genres consistently outperform others
3. A Random Forest model can predict Hit or Flop with 67% accuracy
4. Producers should invest in social media marketing alongside
   production budgets for better returns

---

## Tools & Technologies
| Tool | Purpose |
|------|---------|
| Python | Core programming language |
| Pandas | Data manipulation |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine learning models |
| Streamlit | Web application deployment |
| Jupyter Notebook | Development environment |
| GitHub | Version control |

---

## Links
- 🌐 **Live App:** https://bollywood-box-office-analysis-predictor-rxhtcibqwnh7vaftpxq4tp.streamlit.app/
- 💻 **GitHub:** https://github.com/r-Harshita/bollywood-box-office-analysis-predictor
