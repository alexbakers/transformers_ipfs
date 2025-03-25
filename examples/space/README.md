# Transformers-IPFS Demo

This Streamlit app demonstrates how to use Hugging Face Transformers with IPFS models using the transformers-ipfs integration.

## Features

- Load models directly from IPFS using their CID
- Analyze sentiment of text using an IPFS-hosted model
- View confidence scores and probability distribution for each sentiment class
- Try with example texts or input your own

## How It Works

The `transformers-ipfs` package patches the Hugging Face Transformers library to:

1. Recognize `ipfs://` URIs as valid model identifiers
2. Download model files from IPFS nodes or gateways
3. Cache models locally for faster loading in subsequent runs

## Model Used

This demo uses a sentiment analysis model hosted on IPFS with CID:
`bafybeichqdarufyutqc7yd43k77fkxbmeuhhetbihd3g32ghcqvijp6fxi`

The original Hugging Face model is: `riturajpandey739/gpt2-sentiment-analysis-tweets`

## Usage

1. Enter your own text or select from the examples
2. Click "Analyze Sentiment" to get the sentiment prediction
3. View the results: predicted sentiment, confidence, and probability distribution

Note: The first run may take longer as the model is downloaded from IPFS.

## Requirements

- streamlit
- transformers
- torch
- transformers-ipfs
