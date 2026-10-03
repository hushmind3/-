#!/bin/bash

# Create directories if they do not exist
mkdir -p ./dataset
mkdir -p ./checkpoint

# Download the datasets
echo "Downloading datasets..."
curl -L -o ./dataset/trade_data_deepseek_risk_2019_2023.csv https://huggingface.co/datasets/benstaf/nasdaq_2013_2023/resolve/main/trade_data_deepseek_risk_2019_2023.csv
curl -L -o ./dataset/trade_data_deepseek_sentiment_2019_2023.csv https://huggingface.co/datasets/benstaf/nasdaq_2013_2023/resolve/main/trade_data_deepseek_sentiment_2019_2023.csv

# Download the model checkpoint (replace with the actual model checkpoint URL)
echo "Downloading model checkpoint..."
# Example URL for the model checkpoint (replace with the actual URL)
curl -L -o ./checkpoint/model_rl.pth https://huggingface.co/rz2689/finrl-dapo-grpo-sentiment-risk/resolve/main/model_rl.pth

echo "Download completed!"