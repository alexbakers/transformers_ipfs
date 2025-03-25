import streamlit as st
import torch
import time
import os

# Set page title and description
st.set_page_config(
    page_title="Transformers-IPFS Demo",
    page_icon="🤗",
    layout="wide"
)

st.title("Transformers-IPFS Demo")
st.markdown("This app demonstrates using Hugging Face Transformers with IPFS-hosted models.")

# Install required packages if not already installed
@st.cache_resource
def install_dependencies():
    import subprocess
    packages = ["transformers", "torch", "transformers-ipfs"]
    for package in packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            st.write(f"Installing {package}...")
            subprocess.check_call(["pip", "install", package, "--quiet"])
    
    # Activate IPFS integration
    import transformers_ipfs
    transformers_ipfs.activate()
    st.success("✅ Dependencies installed and IPFS integration activated!")

# Display a loading spinner while installing dependencies
with st.spinner("Setting up environment..."):
    install_dependencies()

# Load the Transformers model
@st.cache_resource
def load_model(model_uri):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    
    with st.spinner(f"Loading model from {model_uri}..."):
        st.info("This may take a few minutes for the first run as the model is downloaded from IPFS.")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_uri)
            model = AutoModelForSequenceClassification.from_pretrained(model_uri)
            return tokenizer, model
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None, None

# Model selection
model_options = {
    "Sentiment Analysis": {
        "model_uri": "ipfs://bafybeichqdarufyutqc7yd43k77fkxbmeuhhetbihd3g32ghcqvijp6fxi",
        "description": "Sentiment analysis model for tweets (positive, neutral, negative)",
        "labels": ["negative", "neutral", "positive"]
    }
}

# Sidebar for model selection
st.sidebar.title("Model Settings")

selected_model = st.sidebar.selectbox(
    "Select Model",
    list(model_options.keys()),
    index=0
)

# Display model description
st.sidebar.markdown(f"**Description**: {model_options[selected_model]['description']}")
st.sidebar.markdown(f"**Source**: {model_options[selected_model]['model_uri']}")

# Load the selected model
model_info = model_options[selected_model]
tokenizer, model = load_model(model_info["model_uri"])
LABELS = model_info["labels"]

# Main content area
st.header("Sentiment Analysis")

# Analysis function
def analyze_sentiment(text, tokenizer, model, labels):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    sentiment = labels[predicted_class]
    confidence = float(probabilities[0][predicted_class])
    
    # Get all class probabilities for visualization
    all_probs = {label: float(prob) for label, prob in zip(labels, probabilities[0])}
    return sentiment, confidence, all_probs

# Example texts
examples = [
    "I absolutely love this new update! The features are amazing! 😊",
    "This is the worst service I've ever experienced. Very disappointed. 😠",
    "The weather is nice today, perfect for a walk in the park!",
    "This product is not worth the money, don't buy it 👎"
]

# Input section
st.subheader("Enter text to analyze")

# Example selector
selected_example = st.selectbox("Or choose an example:", [""] + examples)
if selected_example:
    input_text = selected_example
else:
    input_text = st.text_area("Text to analyze:", height=150)

# Analysis button
if st.button("Analyze Sentiment") and input_text:
    if tokenizer is not None and model is not None:
        # Show the input text
        st.text_area("Analyzing text:", input_text, height=100, disabled=True)
        
        # Generate the analysis with progress
        with st.spinner("Analyzing sentiment..."):
            start_time = time.time()
            sentiment, confidence, all_probs = analyze_sentiment(input_text, tokenizer, model, LABELS)
            end_time = time.time()
            
            st.success(f"Analysis completed in {end_time - start_time:.2f} seconds")
            
            # Display the results
            st.header("Results")
            
            # Display overall sentiment
            st.metric("Predicted Sentiment", sentiment, f"{confidence:.1%} confidence")
            
            # Display probabilities as a horizontal bar chart
            st.subheader("Sentiment Probabilities")
            probs_df = {"Sentiment": list(all_probs.keys()), "Probability": list(all_probs.values())}
            st.bar_chart(probs_df, x="Sentiment", y="Probability")
    else:
        st.error("Model not loaded. Please check the error message above.")

# Information about the app
st.markdown("---")
st.markdown("""
### How It Works

The `transformers-ipfs` package patches the Hugging Face Transformers library to:

1. Recognize `ipfs://` URIs as valid model identifiers
2. Download model files from IPFS nodes or gateways
3. Cache models locally for faster loading in subsequent runs

This allows you to load models from a decentralized network without changing any of your existing code!
""") 