import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Page config
st.set_page_config(
    page_title="Bollywood Predictor",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main { background-color: #0a0a0a; }
.stButton>button {
    background-color: #e74c3c;
    color: white;
    border-radius: 25px;
    padding: 10px 30px;
    font-size: 18px;
    font-weight: bold;
    border: none;
    width: 100%;
}
.stButton>button:hover { background-color: #c0392b; }
.metric-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border-radius: 15px;
    padding: 20px;
    text-align: center;
    border: 1px solid #e74c3c;
}
h1 { color: #e74c3c !important; text-align: center; }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('bollywood.csv')
    df['Genre'] = df['Genre'].str.strip().str.title()
    df['Profit'] = df['BoxOfficeCollection'] - df['Budget']
    df['HitOrFlop'] = df['Profit'].apply(lambda x: 'Hit' if x > 0 else 'Flop')
    return df

df = load_data()

# Train model
@st.cache_resource
def train_model():
    X = df[['Budget', 'YoutubeViews', 'YoutubeLikes', 'YoutubeDislikes']]
    y = df['HitOrFlop'].map({'Hit': 1, 'Flop': 0})
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    return model, scaler

model, scaler = train_model()

# Header
st.markdown("# 🎬 Bollywood Box Office Predictor")
st.markdown("<p style='text-align:center; color:#888; font-size:18px'>Discover what makes a Bollywood movie a HIT</p>", unsafe_allow_html=True)
st.markdown("---")

# Navigation
page = st.sidebar.radio("Navigate", 
    ["🏠 Home", "🎯 Predict", "📊 Analysis", "🏆 Top Movies"])

# HOME PAGE
if page == "🏠 Home":
    st.markdown("## Welcome to Bollywood Box Office Predictor!")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Movies", "149")
    with col2:
        st.metric("Hit Movies", f"{len(df[df['HitOrFlop']=='Hit'])}")
    with col3:
        st.metric("Flop Movies", f"{len(df[df['HitOrFlop']=='Flop'])}")
    with col4:
        st.metric("Model Accuracy", "67%")
    
    st.markdown("---")
    st.markdown("### 🔍 What can you do here?")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("🎯 **Predict** \n\nEnter your movie details and predict Hit or Flop instantly")
    with col2:
        st.info("📊 **Analyze** \n\nExplore insights from 149 Bollywood movies")
    with col3:
        st.info("🏆 **Top Movies** \n\nDiscover the most profitable Bollywood movies")

# PREDICT PAGE
elif page == "🎯 Predict":
    st.markdown("## 🎯 Predict Your Movie's Success")
    st.markdown("Adjust the sliders and click Predict!")
    
    col1, col2 = st.columns(2)
    with col1:
        budget = st.slider("💰 Budget (in Crores)", 1, 200, 50)
        views = st.slider("👀 YouTube Views", 0, 25000000, 5000000)
    with col2:
        likes = st.slider("👍 YouTube Likes", 0, 1000000, 100000)
        dislikes = st.slider("👎 YouTube Dislikes", 0, 100000, 10000)
    
    st.markdown("")
    if st.button("🎬 PREDICT NOW!"):
        input_data = pd.DataFrame({
            'Budget': [budget],
            'YoutubeViews': [views],
            'YoutubeLikes': [likes],
            'YoutubeDislikes': [dislikes]
        })
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)
        
        st.markdown("---")
        if prediction[0] == 1:
            st.success(f"## 🟢 HIT! Probability: {probability[0][1]:.2%}")
            st.balloons()
        else:
            st.error(f"## 🔴 FLOP! Probability of Flop: {probability[0][0]:.2%}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Hit Probability", f"{probability[0][1]:.2%}")
        with col2:
            st.metric("Flop Probability", f"{probability[0][0]:.2%}")

# ANALYSIS PAGE
elif page == "📊 Analysis":
    st.markdown("## 📊 Data Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Hit vs Flop", "Genre Analysis", "Correlation"])
    
    with tab1:
        fig, ax = plt.subplots(figsize=(6,6))
        df['HitOrFlop'].value_counts().plot(kind='pie',
            autopct='%1.1f%%',
            colors=['#2ecc71', '#e74c3c'],
            ax=ax, startangle=90)
        ax.set_title('Hit vs Flop Ratio', fontsize=16, fontweight='bold')
        ax.set_ylabel('')
        st.pyplot(fig)
    
    with tab2:
        fig, ax = plt.subplots(figsize=(10,5))
        genre_analysis = df.groupby('Genre')['Profit'].mean().sort_values(ascending=False)
        genre_analysis.plot(kind='bar', color='#3498db', ax=ax)
        ax.set_title('Average Profit by Genre', fontsize=16, fontweight='bold')
        ax.set_xlabel('Genre')
        ax.set_ylabel('Average Profit (in Crores)')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with tab3:
        fig, ax = plt.subplots(figsize=(8,6))
        correlation = df[['Budget', 'BoxOfficeCollection',
                          'YoutubeViews', 'YoutubeLikes',
                          'YoutubeDislikes', 'Profit']].corr()
        sns.heatmap(correlation, annot=True, cmap='RdYlGn', fmt='.2f', ax=ax)
        ax.set_title('Correlation Heatmap', fontsize=16, fontweight='bold')
        st.pyplot(fig)

# TOP MOVIES PAGE
elif page == "🏆 Top Movies":
    st.markdown("## 🏆 Top 10 Most Profitable Movies")
    
    top10 = df.nlargest(10, 'Profit')[['MovieName', 'Profit', 'Genre', 'HitOrFlop']]
    
    fig, ax = plt.subplots(figsize=(10,6))
    plt.barh(top10['MovieName'], top10['Profit'],
             color='#2ecc71', edgecolor='white')
    plt.title('Top 10 Most Profitable Bollywood Movies',
              fontsize=16, fontweight='bold')
    plt.xlabel('Profit (in Crores)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("### 📋 Detailed Table")
    st.dataframe(top10.reset_index(drop=True), use_container_width=True)