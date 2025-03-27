import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import random
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Political Polarization via LLMs",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #4F8BF9;}
    .sub-header {font-size: 1.5rem; margin-bottom: 1rem;}
    .model-card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        margin-bottom: 1rem;
        border-left: 4px solid #4F8BF9;
    }
    .highlight {background-color: #ffffcc; padding: 0.2rem;}
    .stButton button {background-color: #4F8BF9; color: white;}
    .chart-container {margin-top: 2rem;}
    .comment-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f1f3f5;
        margin-bottom: 1rem;
        border-left: 4px solid #adb5bd;
    }
    .republican {border-left: 4px solid #ff6b6b;}
    .democrat {border-left: 4px solid #339af0;}
    .independent {border-left: 4px solid #20c997;}
    .model-comparison {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
    }
    .model-result {
        flex: 1;
        min-width: 200px;
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        border-left: 4px solid #4F8BF9;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🗳️ Political Analysis")
    st.markdown("---")
    
    st.markdown("### About")
    st.info(
        """
        This tool analyzes political stances and news channel preferences using predictions from multiple LLMs like GPT-4o, Llama, Mistral, and Copilot based on YouTube comment data.
        
        **Features:**
        - Analyze your own text
        - Upload comment datasets
        - Compare multiple LLM predictions
        - Visualize political distributions
        """
    )
    
    st.markdown("### Models Used")
    st.markdown("- GPT-4o-mini")
    st.markdown("- Mistral")
    st.markdown("- Copilot")
    st.markdown("- Llama 3.2 70B")
    
    st.markdown("---")
    st.markdown("Made by Sai Rupa Jhade")

# Main content
st.markdown('<p class="main-header">🗳️ Political Polarization Analysis with LLMs</p>', unsafe_allow_html=True)
st.markdown("""
Analyze political stances and news channel preferences using predictions from multiple Large Language Models 
based on YouTube comment data. Enter your own text, upload a dataset, or explore our sample data.
""")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Text Analysis", "Dataset Analysis", "Model Comparison", "Visualization"])

# Sample data
def generate_sample_data(n=50):
    comments = [
        "The economy is doing great under this administration!",
        "We need to secure our borders and protect American jobs.",
        "Healthcare should be a right for all citizens.",
        "Lower taxes will help stimulate economic growth.",
        "Climate change is the biggest threat we face today.",
        "The second amendment rights should not be infringed.",
        "We need more funding for public education.",
        "Government regulation is hurting small businesses.",
        "Income inequality is a serious problem we must address.",
        "Strong military is essential for our national security."
    ]
    
    channels = ["CNN", "Fox News", "MSNBC", "NewsMax", "ABC News"]
    parties = ["democrat", "republican", "independent"]
    weights = {
        "CNN": {"democrat": 0.7, "republican": 0.2, "independent": 0.1},
        "Fox News": {"democrat": 0.1, "republican": 0.8, "independent": 0.1},
        "MSNBC": {"democrat": 0.8, "republican": 0.1, "independent": 0.1},
        "NewsMax": {"democrat": 0.05, "republican": 0.9, "independent": 0.05},
        "ABC News": {"democrat": 0.4, "republican": 0.3, "independent": 0.3}
    }
    
    data = []
    for _ in range(n):
        comment = random.choice(comments)
        channel = random.choice(channels)
        party_weights = weights[channel]
        party = random.choices(parties, weights=[party_weights[p] for p in parties])[0]
        
        # Generate random date within last 30 days
        days_ago = random.randint(0, 30)
        date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        
        # Generate random likes
        likes = int(np.random.exponential(50))
        
        data.append({
            "comment": comment,
            "channel": channel,
            "actual_party": party,
            "date": date,
            "likes": likes
        })
    
    return pd.DataFrame(data)

# Model predictions function (simulated)
def predict_political_stance(text, model="GPT-4o-mini"):
    # In a real app, this would call the actual LLM APIs
    # For demo purposes, we'll use random predictions with some bias based on keywords
    
    # Simple keyword-based bias
    republican_keywords = ["taxes", "borders", "jobs", "amendment", "military", "regulation"]
    democrat_keywords = ["healthcare", "climate", "inequality", "education", "rights"]
    
    text_lower = text.lower()
    
    # Count keywords
    rep_count = sum(1 for word in republican_keywords if word in text_lower)
    dem_count = sum(1 for word in democrat_keywords if word in text_lower)
    
    # Base probabilities
    rep_prob = 0.3 + (rep_count * 0.1) - (dem_count * 0.05)
    dem_prob = 0.3 + (dem_count * 0.1) - (rep_count * 0.05)
    ind_prob = 1 - rep_prob - dem_prob
    
    # Adjust based on selected model (add some variance)
    if model == "GPT-4o-mini":
        rep_prob += random.uniform(-0.1, 0.1)
        dem_prob += random.uniform(-0.1, 0.1)
    elif model == "Mistral":
        rep_prob += random.uniform(-0.15, 0.15)
        dem_prob += random.uniform(-0.15, 0.15)
    elif model == "Copilot":
        rep_prob += random.uniform(-0.05, 0.05)
        dem_prob += random.uniform(-0.05, 0.05)
    elif model == "Llama 3.2 70B":
        rep_prob += random.uniform(-0.2, 0.2)
        dem_prob += random.uniform(-0.2, 0.2)
    
    # Normalize probabilities
    total = rep_prob + dem_prob + ind_prob
    rep_prob /= total
    dem_prob /= total
    ind_prob /= total
    
    # Determine party
    probs = [rep_prob, dem_prob, ind_prob]
    parties = ["republican", "democrat", "independent"]
    party = parties[probs.index(max(probs))]
    
    # Determine channel preference based on party
    channel_probs = {
        "republican": {"Fox News": 0.6, "NewsMax": 0.3, "ABC News": 0.1},
        "democrat": {"CNN": 0.5, "MSNBC": 0.4, "ABC News": 0.1},
        "independent": {"ABC News": 0.6, "CNN": 0.2, "Fox News": 0.2}
    }
    
    channels = list(channel_probs[party].keys())
    channel_weights = list(channel_probs[party].values())
    channel = random.choices(channels, weights=channel_weights)[0]
    
    # Generate explanation
    explanations = {
        "republican": [
            f"User's language contains {rep_count} conservative-leaning terms, consistent with Republican views.",
            "Comment sentiment aligns with traditional conservative values.",
            "Word choice and phrasing match patterns commonly seen in right-leaning discourse."
        ],
        "democrat": [
            f"Analysis detected {dem_count} progressive-leaning terms, typical of Democratic rhetoric.",
            "Comment shows concern for social issues often championed by Democrats.",
            "Language patterns match those commonly found in left-leaning discourse."
        ],
        "independent": [
            "Comment shows a balanced perspective without strong partisan language.",
            "Analysis found mixed political signals without clear partisan alignment.",
            "Language doesn't strongly correlate with either major party's typical rhetoric."
        ]
    }
    
    explanation = random.choice(explanations[party])
    
    # Add confidence score
    confidence = max(probs) * 100
    
    return {
        "political_party": party,
        "party_probabilities": {
            "republican": round(rep_prob * 100, 1),
            "democrat": round(dem_prob * 100, 1)
        },
        "preferred_channel": channel,
        "explanation": explanation,
        "confidence": round(confidence, 1)
    }

# Tab 1: Text Analysis
with tab1:
    st.markdown('<p class="sub-header">✍️ Analyze Your Text</p>', unsafe_allow_html=True)
    
    user_text = st.text_area(
        "Enter a comment or statement to analyze:",
        height=100,
        placeholder="Example: The economy is doing great under this administration!"
    )
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        model_choice = st.selectbox(
            "🤖 Choose a Language Model",
            options=["GPT-4o-mini", "Mistral", "Copilot", "Llama 3.2 70B"]
        )
        
        analyze_button = st.button("🔍 Analyze Text", use_container_width=True)
    
    with col2:
        compare_models = st.checkbox("Compare all models")
    
    if user_text and analyze_button:
        if compare_models:
            st.markdown("### 📊 Model Comparison")
            
            results_container = st.container()
            with results_container:
                st.markdown('<div class="model-comparison">', unsafe_allow_html=True)
                
                for model in ["GPT-4o-mini", "Mistral", "Copilot", "Llama 3.2 70B"]:
                    result = predict_political_stance(user_text, model)
                    
                    # Determine CSS class based on predicted party
                    party_class = result["political_party"]
                    
                    st.markdown(f'''
                    <div class="model-result {party_class}">
                        <h4>{model}</h4>
                        <p><strong>Political Party:</strong> {result["political_party"].capitalize()}</p>
                        <p><strong>Confidence:</strong> {result["confidence"]}%</p>
                        <p><strong>Channel:</strong> {result["preferred_channel"]}</p>
                    </div>
                    ''', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Show detailed analysis for selected model
            st.markdown("### 🔎 Detailed Analysis")
            selected_result = predict_political_stance(user_text, model_choice)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Party probability chart
                fig, ax = plt.subplots(figsize=(8, 5))
                parties = list(selected_result["party_probabilities"].keys())
                probs = list(selected_result["party_probabilities"].values())
                
                colors = ["#ff6b6b", "#339af0", "#20c997"]
                ax.bar(parties, probs, color=colors)
                ax.set_ylabel("Probability (%)")
                ax.set_title(f"{model_choice} - Party Probability Distribution")
                
                for i, v in enumerate(probs):
                    ax.text(i, v + 1, f"{v}%", ha='center')
                
                st.pyplot(fig)
            
            with col2:
                st.markdown(f"### {model_choice} Analysis")
                st.markdown(f"**Input Text:** {user_text}")
                st.markdown(f"**Political Party:** {selected_result['political_party'].capitalize()}")
                st.markdown(f"**Preferred Channel:** {selected_result['preferred_channel']}")
                st.markdown(f"**Confidence:** {selected_result['confidence']}%")
                st.markdown(f"**Explanation:** {selected_result['explanation']}")
        
        else:
            # Single model analysis
            result = predict_political_stance(user_text, model_choice)
            
            st.markdown("### 🧠 Prediction Result")
            
            # Determine CSS class based on predicted party
            party_class = result["political_party"]
            
            st.markdown(f'''
            <div class="comment-box {party_class}">
                <p><strong>Input Text:</strong> {user_text}</p>
            </div>
            ''', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"### {model_choice} Analysis")
                st.markdown(f"**Political Party:** {result['political_party'].capitalize()}")
                st.markdown(f"**Preferred Channel:** {result['preferred_channel']}")
                st.markdown(f"**Confidence:** {result['confidence']}%")
                st.markdown(f"**Explanation:** {result['explanation']}")
            
            with col2:
                # Party probability chart
                fig, ax = plt.subplots(figsize=(8, 5))
                parties = list(result["party_probabilities"].keys())
                probs = list(result["party_probabilities"].values())
                
                colors = ["#ff6b6b", "#339af0", "#20c997"]
                ax.bar(parties, probs, color=colors)
                ax.set_ylabel("Probability (%)")
                ax.set_title(f"{model_choice} - Party Probability Distribution")
                
                for i, v in enumerate(probs):
                    ax.text(i, v + 1, f"{v}%", ha='center')
                
                st.pyplot(fig)

# Tab 2: Dataset Analysis
with tab2:
    st.markdown('<p class="sub-header">📊 Analyze Comment Dataset</p>', unsafe_allow_html=True)
    
    upload_option = st.radio(
        "Choose data source:",
        ["Upload CSV file", "Use sample dataset"]
    )
    
    if upload_option == "Upload CSV file":
        uploaded_file = st.file_uploader("Upload a CSV file with comments", type=["csv"])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success("File uploaded successfully!")
                
                # Check if required column exists
                if "comment" not in df.columns:
                    st.error("CSV must contain a 'comment' column.")
                    df = None
            except Exception as e:
                st.error(f"Error: {e}")
                df = None
        else:
            df = None
    else:
        # Generate sample data
        sample_size = st.slider("Sample size:", min_value=10, max_value=100, value=50)
        df = generate_sample_data(sample_size)
        st.success(f"Generated sample dataset with {sample_size} comments.")
    
    if df is not None:
        st.markdown("### 📋 Dataset Preview")
        st.dataframe(df.head())
        
        st.markdown("### 🔍 Analyze Dataset")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            dataset_model = st.selectbox(
                "🤖 Choose Model for Dataset Analysis",
                options=["GPT-4o-mini", "Mistral", "Copilot", "Llama 3.2 70B"]
            )
            
            analyze_dataset_button = st.button("🔍 Analyze Dataset", use_container_width=True)
        
        with col2:
            max_comments = st.slider(
                "Maximum comments to analyze:",
                min_value=1,
                max_value=min(50, len(df)),
                value=min(10, len(df))
            )
        
        if analyze_dataset_button:
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Analyze subset of comments
            results = []
            
            for i, row in df.head(max_comments).iterrows():
                # Update progress
                progress = int((i + 1) / max_comments * 100)
                progress_bar.progress(progress)
                status_text.text(f"Analyzing comment {i+1} of {max_comments}...")
                
                # Get prediction
                comment = row["comment"]
                result = predict_political_stance(comment, dataset_model)
                
                # Add to results
                results.append({
                    "comment": comment,
                    "predicted_party": result["political_party"],
                    "confidence": result["confidence"],
                    "preferred_channel": result["preferred_channel"],
                    "explanation": result["explanation"]
                })
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Convert results to DataFrame
            results_df = pd.DataFrame(results)
            
            # Display results
            st.markdown("### 📊 Analysis Results")
            st.dataframe(results_df)
            
            # Summary statistics
            st.markdown("### 📈 Summary Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Party distribution
                party_counts = results_df["predicted_party"].value_counts()
                
                fig, ax = plt.subplots(figsize=(8, 5))
                colors = ["#ff6b6b", "#339af0", "#20c997"]
                party_counts.plot(kind="bar", ax=ax, color=colors[:len(party_counts)])
                ax.set_title("Predicted Political Party Distribution")
                ax.set_ylabel("Count")
                
                for i, v in enumerate(party_counts):
                    ax.text(i, v + 0.1, str(v), ha='center')
                
                st.pyplot(fig)
            
            with col2:
                # Channel distribution
                channel_counts = results_df["preferred_channel"].value_counts()
                
                fig, ax = plt.subplots(figsize=(8, 5))
                channel_counts.plot(kind="bar", ax=ax, color=sns.color_palette("viridis", len(channel_counts)))
                ax.set_title("Preferred News Channel Distribution")
                ax.set_ylabel("Count")
                
                for i, v in enumerate(channel_counts):
                    ax.text(i, v + 0.1, str(v), ha='center')
                
                st.pyplot(fig)
            
            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Analysis Results",
                data=csv,
                file_name=f"political_analysis_{dataset_model}.csv",
                mime="text/csv"
            )

# Tab 3: Model Comparison
with tab3:
    st.markdown('<p class="sub-header">🤖 Compare Model Performance</p>', unsafe_allow_html=True)
    
    st.markdown("""
    This section allows you to compare how different LLMs perform on the same text inputs.
    Enter multiple statements to see how each model classifies them.
    """)
    
    # Input multiple statements
    statements = st.text_area(
        "Enter multiple statements (one per line):",
        height=150,
        placeholder="The economy is doing great under this administration!\nWe need to secure our borders and protect American jobs.\nHealthcare should be a right for all citizens."
    )
    
    compare_button = st.button("🔄 Compare Models", use_container_width=True)
    
    if statements and compare_button:
        # Split statements into list
        statement_list = [s.strip() for s in statements.split('\n') if s.strip()]
        
        if statement_list:
            # Create comparison table
            comparison_data = []
            
            for statement in statement_list:
                row = {"Statement": statement}
                
                for model in ["GPT-4o-mini", "Mistral", "Copilot", "Llama 3.2 70B"]:
                    result = predict_political_stance(statement, model)
                    row[f"{model} Prediction"] = result["political_party"].capitalize()
                    row[f"{model} Confidence"] = f"{result['confidence']}%"
                
                comparison_data.append(row)
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Display comparison table
            st.markdown("### 📊 Model Comparison Results")
            st.dataframe(comparison_df)
            
            # Calculate agreement statistics
            st.markdown("### 🔍 Model Agreement Analysis")
            
            agreement_data = []
            
            for statement in statement_list:
                predictions = {}
                
                for model in ["GPT-4o-mini", "Mistral", "Copilot", "Llama 3.2 70B"]:
                    result = predict_political_stance(statement, model)
                    predictions[model] = result["political_party"]
                
                # Count unique predictions
                unique_predictions = set(predictions.values())
                
                # Calculate agreement percentage
                agreement_pct = (1 - (len(unique_predictions) - 1) / 3) * 100
                
                agreement_data.append({
                    "Statement": statement,
                    "Unique Predictions": len(unique_predictions),
                    "Agreement": f"{agreement_pct:.1f}%",
                    "Predictions": ", ".join([f"{m}: {p.capitalize()}" for m, p in predictions.items()])
                })
            
            agreement_df = pd.DataFrame(agreement_data)
            
            # Display agreement statistics
            st.dataframe(agreement_df)
            
            # Visualize agreement
            st.markdown("### 📈 Model Agreement Visualization")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            agreement_values = [float(a.replace("%", "")) for a in agreement_df["Agreement"]]
            statements_short = [s[:30] + "..." if len(s) > 30 else s for s in agreement_df["Statement"]]
            
            colors = []
            for val in agreement_values:
                if val >= 80:
                    colors.append("#20c997")  # High agreement
                elif val >= 50:
                    colors.append("#ffd43b")  # Medium agreement
                else:
                    colors.append("#ff6b6b")  # Low agreement
            
            ax.barh(statements_short, agreement_values, color=colors)
            ax.set_xlabel("Agreement (%)")
            ax.set_title("Model Agreement by Statement")
            
            for i, v in enumerate(agreement_values):
                ax.text(v + 1, i, f"{v:.1f}%", va='center')
            
            st.pyplot(fig)

# Tab 4: Visualization
with tab4:
    st.markdown('<p class="sub-header">📊 Data Visualization</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Explore visualizations of political polarization patterns across different news channels and time periods.
    This section uses sample data to demonstrate potential insights.
    """)
    
    # Generate larger sample dataset for visualization
    viz_data = generate_sample_data(200)
    
    # Add simulated model predictions
    for model in ["GPT-4o-mini", "Mistral", "Copilot", "Llama 3.2 70B"]:
        predicted_parties = []
        for comment in viz_data["comment"]:
            result = predict_political_stance(comment, model)
            predicted_parties.append(result["political_party"])
        
        viz_data[f"{model}_prediction"] = predicted_parties
    
    # Visualization options
    viz_type = st.selectbox(
        "Select visualization type:",
        ["Channel Distribution", "Time Series Analysis", "Model Accuracy", "Comment Length Analysis"]
    )
    
    if viz_type == "Channel Distribution":
        st.markdown("### 📺 Political Distribution by News Channel")
        
        # Select model for prediction
        viz_model = st.selectbox(
            "Select model for predictions:",
            ["GPT-4o-mini", "Mistral", "Copilot", "Llama 3.2 70B", "Actual"]
        )
        
        # Get data for selected model
        if viz_model == "Actual":
            party_column = "actual_party"
        else:
            party_column = f"{viz_model}_prediction"
        
        # Create crosstab
        channel_party_data = pd.crosstab(
            viz_data["channel"],
            viz_data[party_column],
            normalize="index"
        ) * 100
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        channel_party_data.plot(kind="bar", stacked=True, ax=ax, colormap="viridis")
        ax.set_title(f"Political Party Distribution by News Channel ({viz_model})")
        ax.set_ylabel("Percentage (%)")
        ax.set_xlabel("News Channel")
        ax.legend(title="Political Party")
        
        # Add percentage labels
        for c in ax.containers:
            labels = [f"{v:.1f}%" if v > 5 else "" for v in c.datavalues]
            ax.bar_label(c, labels=labels, label_type="center")
        
        st.pyplot(fig)
        
        # Add explanation
        st.markdown("""
        **Insight:** This chart shows the distribution of political affiliations across different news channels.
        The percentages represent the proportion of comments from each political party on each channel.
        """)
    
    elif viz_type == "Time Series Analysis":
        st.markdown("### 📅 Political Polarization Over Time")
        
        # Convert date to datetime
        viz_data["date"] = pd.to_datetime(viz_data["date"])
        
        # Group by date and count parties
        time_model = st.selectbox(
            "Select model for time series:",
            ["GPT-4o-mini", "Mistral", "Copilot", "Llama 3.2 70B", "Actual"]
        )
        
        if time_model == "Actual":
            party_column = "actual_party"
        else:
            party_column = f"{time_model}_prediction"
        
        # Group by date and get counts
        time_data = viz_data.groupby(["date", party_column]).size().unstack().fillna(0)
        
        # Calculate percentage
        time_data_pct = time_data.div(time_data.sum(axis=1), axis=0) * 100
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        time_data_pct.plot(kind="line", marker="o", ax=ax)
        ax.set_title(f"Political Party Distribution Over Time ({time_model})")
        ax.set_ylabel("Percentage (%)")
        ax.set_xlabel("Date")
        ax.legend(title="Political Party")
        ax.grid(True, linestyle="--", alpha=0.7)
        
        st.pyplot(fig)
        
        # Add explanation
        st.markdown("""
        **Insight:** This chart shows how political polarization has changed over time.
        The lines represent the percentage of comments from each political party on each date.
        """)
    
    elif viz_type == "Model Accuracy":
        st.markdown("### 🎯 Model Prediction Accuracy")
        
        # Calculate accuracy for each model
        accuracy_data = []
        
        for model in ["GPT-4o-mini", "Mistral", "Copilot", "Llama 3.2 70B"]:
            # Compare predictions with actual values
            correct = (viz_data[f"{model}_prediction"] == viz_data["actual_party"]).sum()
            total = len(viz_data)
            accuracy = (correct / total) * 100
            
            # Calculate accuracy by party
            party_accuracy = {}
            for party in ["republican", "democrat", "independent"]:
                party_data = viz_data[viz_data["actual_party"] == party]
                if len(party_data) > 0:
                    party_correct = (party_data[f"{model}_prediction"] == party_data["actual_party"]).sum()
                    party_accuracy[party] = (party_correct / len(party_data)) * 100
                else:
                    party_accuracy[party] = 0
            
            accuracy_data.append({
                "Model": model,
                "Overall Accuracy": accuracy,
                "Republican Accuracy": party_accuracy["republican"],
                "Democrat Accuracy": party_accuracy["democrat"],
                "Independent Accuracy": party_accuracy["independent"]
            })
        
        accuracy_df = pd.DataFrame(accuracy_data)
        
        # Plot overall accuracy
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(
            accuracy_df["Model"],
            accuracy_df["Overall Accuracy"],
            color=sns.color_palette("viridis", len(accuracy_df))
        )
        ax.set_title("Overall Model Accuracy")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        
        # Add percentage labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height + 1,
                f"{height:.1f}%",
                ha='center'
            )
        
        st.pyplot(fig)
        
        # Plot accuracy by party
        st.markdown("### 🎯 Model Accuracy by Political Party")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(accuracy_df))
        width = 0.25
        
        ax.bar(x - width, accuracy_df["Republican Accuracy"], width, label="Republican", color="#ff6b6b")
        ax.bar(x, accuracy_df["Democrat Accuracy"], width, label="Democrat", color="#339af0")
        ax.bar(x + width, accuracy_df["Independent Accuracy"], width, label="Independent", color="#20c997")
        
        ax.set_title("Model Accuracy by Political Party")
        ax.set_ylabel("Accuracy (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(accuracy_df["Model"])
        ax.set_ylim(0, 100)
        ax.legend(title="Political Party")
        ax.grid(True, linestyle="--", alpha=0.3, axis="y")
        
        st.pyplot(fig)
        
        # Add explanation
        st.markdown("""
        **Insight:** These charts show how accurately each model predicts political affiliation compared to the actual values.
        The first chart shows overall accuracy, while the second breaks it down by political party.
        """)
    
    elif viz_type == "Comment Length Analysis":
        st.markdown("### 📏 Comment Length vs. Political Affiliation")
        
        # Add comment length
        viz_data["comment_length"] = viz_data["comment"].apply(len)
        
        # Select model
        length_model = st.selectbox(
            "Select model for analysis:",
            ["GPT-4o-mini", "Mistral", "Copilot", "Llama 3.2 70B", "Actual"]
        )
        
        if length_model == "Actual":
            party_column = "actual_party"
        else:
            party_column = f"{length_model}_prediction"
        
        # Create box plot
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.boxplot(
            x=party_column,
            y="comment_length",
            data=viz_data,
            palette={"republican": "#ff6b6b", "democrat": "#339af0", "independent": "#20c997"},
            ax=ax
        )
        
        ax.set_title(f"Comment Length by Political Affiliation ({length_model})")
        ax.set_xlabel("Political Party")
        ax.set_ylabel("Comment Length (characters)")
        
        st.pyplot(fig)
        
        # Add scatter plot with likes
        st.markdown("### 👍 Comment Engagement vs. Length by Political Affiliation")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for party, color in zip(["republican", "democrat", "independent"], ["#ff6b6b", "#339af0", "#20c997"]):
            party_data = viz_data[viz_data[party_column] == party]
            ax.scatter(
                party_data["comment_length"],
                party_data["likes"],
                alpha=0.7,
                label=party.capitalize(),
                color=color,
                s=50
            )
        
        ax.set_title(f"Comment Engagement vs. Length by Political Affiliation ({length_model})")
        ax.set_xlabel("Comment Length (characters)")
        ax.set_ylabel("Likes")
        ax.legend(title="Political Party")
        ax.grid(True, linestyle="--", alpha=0.3)
        
        st.pyplot(fig)
        
        # Add explanation
        st.markdown("""
        **Insight:** These visualizations show the relationship between comment length and political affiliation.
        The box plot shows the distribution of comment lengths for each political party, while the scatter plot
        shows how comment length relates to engagement (likes) across different political affiliations.
        """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #888;">
        Made with ❤️ by Sai Rupa Jhade | Data Science Portfolio
    </div>
    """, 
    unsafe_allow_html=True
)

