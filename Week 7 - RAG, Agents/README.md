# Hotel RAG Agent Project (LangChain + Gemini)

This project demonstrates how to build an intelligent Retrieval-Augmented Generation (RAG) Agent using the LangChain framework and Google's Gemini 2.5 Flash model. 
The Agent is designed to answer complex user queries that require both private data retrieval (hotel policies) and arithmetic reasoning (calculating refunds or costs).

The core objective is to showcase the Agent's ability to chain together multiple specialized Tools to successfully resolve multi-step inquiries.

## ✨ Key Features

- Retrieval-Augmented Generation (RAG): The Agent is equipped with private knowledge of the hotel's policies, loaded into a Vector Store (ChromaDB), allowing it to answer internal, context-specific questions.
- Multi-Tool Usage: The Agent leverages two specialized tools:
  1. Hotel_Policy_Retriever: Queries cancellation policies, room rates, and payment rules from the `hotel_policies.txt` document.
  2. Calculator: Executes arithmetic operations (using the powerful `LLMMathChain`) needed for cost or refund calculations.
- Zero-Shot ReAct Agent: It utilizes the ReAct (Reasoning and Acting) strategy, enabling the Agent to dynamically reason about the steps required and select the appropriate tool for each step of the query resolution process.

## Technology

LLM: `Gemini 2.5 Flash`, `langchain-google-genai`.

Embedding: `sentence-transformers/all-MiniLM-L6-v2`, `HuggingFaceEmbeddings`.

Framework: Orchestration and Abstraction, `langchain` `langchain-core`.

Vector Store: Store and Retrieve Chunks by `chromadb`.

Logic, Tool Selection and Execution, `langchain-agents` `langchain-experimental`.

## Setup and Installation

To run this Notebook, you must set up your local Python virtual environment and configure the Gemini API Key.

1. Install Dependencies
   Ensure you are in your activated Python virtual environment (`venv`) and run the following command to install all necessary packages:
   ```bash
   pip install -U langchain langchain-google-genai chromadb langchain-experimental langchain-community langchain-text-splitters python-dotenv
   ```
   
2. Configure API Key
   - Create a file named `.env` in the root directory of your project.
   - Add your Gemini API key to the file using the variable name `GEMINI_API_KEY`:
     ```bash
     # .env file content
     GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"
     ```
     *The `load_dotenv()` function in the Notebook will automatically load this key into your environment.*

3. Data Preparation
   Place the source data file, `hotel_policies.txt`, in the project directory.

## Analysis of Experiment Results

The provided Notebook includes two key experiments showcasing the Agent's capabilities:

1. Experiment 1: Simple Policy Query (RAG)
   
   Query: "Giá phòng tiêu chuẩn là bao nhiêu và bao gồm những gì?" (What is the standard room rate and what does it include?)

   Action: The Agent correctly identifies this as a policy question and exclusively uses the `Hotel_Policy_Retriever`.

   Result: Provides a direct answer retrieved from the `hotel_policies.txt` data.

2. Experiment 2: Complex Query (RAG + Math)

   Query: "Tôi đặt phòng 2 đêm và đã trả trước 50%. Nếu tôi hủy trước 8 ngày, tôi sẽ được hoàn lại bao nhiêu tiền VND?" (I booked for 2 nights and paid 50% upfront. If I cancel 8 days in advance, how much VND will I be refunded?)

   ReAct Agent Reasoning:
   - Step 1 (RAG): The Agent uses the `Hotel_Policy_Retriever` to find the refund rule ("cancel 8 days in advance... will be refunded 90% of the amount paid").
   - Step 2 (Inference): The Agent realizes the user paid 50% of the total cost upfront.
   - Step 3 (Math): The Agent uses the Calculator tool with the expression `0.90 * 0.50` to determine the final refund rate is `0.45` (or 45% of the total booking cost).
   
This demonstrates the Agent's ability to chain reasoning by dynamically invoking multiple specialized tools in a sequential manner to solve complex, real-world problems.
