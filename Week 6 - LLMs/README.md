# 🤖 Exploring LLM Architecture and Performance

This repository contains in-depth practical notebooks designed to explore the architecture, compare the performance, and optimize the inference latency of Large Language Models (LLMs).

## Project Contents

The project is structured around three main practical exercises, each contained in a separate part:

### 1. Proprietary Model Performance Comparison

   To compare the reasoning ability, output quality, and response latency of two leading closed-source LLMs on identical prompts.

   **Models Compared:** GPT (OpenAI) and Gemini (Google).

   This part executes simultaneous API calls, evaluates the quality of the generated output between the two providers.

### 2. Open-Source LLM Deployment via Inference Provider

   To demonstrate how to deploy and utilize open-source LLMs without requiring powerful local GPU resources.

   **Models Used:** Popular models like Llama, Mistral, or other models available on Hugging Face.

   The notebook guides you through using a Hugging Face Inference Provider to perform model inference, including steps for: loading the Tokenizer, loading the model, and setting up the basic inference pipeline.

### 3. Inference Optimization with Context Caching

   To experimentally measure and demonstrate the impact of the Context Caching (KV Caching) technique on LLM performance and latency.

   **Tool:** Uses the Gemini API (or another streaming API provider) for precise measurement.

   - Measure Pre-fill Time: The high latency required to process a long initial prompt.
   - Measure Decoding Speed: The fast, stable speed (tokens/second) of generating subsequent tokens thanks to the cache.
   - Conclusion: Proves the significant difference in computational cost between the initial Attention calculation and efficient Cache retrieval during token generation.

## ⚙️ Setup Guide

To run this notebook, you'll need to set up your Python environment and obtain the necessary API keys.

**1. Prerequisites**
   - Python 3.8+
   - Jupyter Notebook or VS Code (with Jupyter extension)

**2. Library Installation**

   Use pip to install all required libraries:
   ```bash
   pip install google-genai openai transformers requests
   ```

**3. API Keys**

   You will need to set up environment variables for your API Keys. You can create a `.env` file or set them directly within the notebooks:
   `GEMINI_API_KEY`, `OPENAI_API_KEY`, `HF_API_KEY` (Required if using paid Inference Endpoints).

  **Setting up in the Notebook (Recommended for testing):**
  
  ```bash
  import os
  # Note: Replace placeholders with your actual API keys
  os.environ["GEMINI_API_KEY"] = "YOUR_GEMINI_KEY" 
  os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_KEY"
  os.environ["HF_API_KEY"] = "YOUR_HF_API_KEY"
  ```

## 📝 Core Concepts

These practical exercises are built upon a solid understanding of the following core concepts:
- Context Window: The token limit for input and output the model can effectively process.
- Latency Metrics: Understanding the difference between First Token Latency and Time Per Subsequent Token (the key proof of LLM Context Caching efficiency).
