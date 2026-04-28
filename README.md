# ML Advance

## Task 1: News Topic Classifier Using BERT
### Description
This project leverages State-of-the-Art (SOTA) NLP to categorize news headlines into four distinct topics: World, Sports, 
Business, and Sci/Tech. By fine-tuning the bert-base-uncased transformer model on the AG News Dataset, 
we achieve high-precision classification that understands the nuances of journalistic language.

**Key Features**
Transformer Architecture: Fine-tuned bert-base-uncased using the Hugging Face transformers library.
Performance Metrics: Evaluated using Accuracy and F1-score to ensure balanced performance across all classes.
Interactive UI: A live deployment using Gradio where users can input custom headlines and receive instant classifications.

**Tech Stack**
Language: Python
Library: Hugging Face (Transformers, Datasets), PyTorch/TensorFlow
Deployment: Gradio
Dataset: AG News (via Hugging Face)

## Task 2: End-to-End ML Pipeline (Customer Churn)
### Description
A production-ready machine learning solution designed to predict customer churn using the Telco Churn Dataset. 
The core of this project is the Scikit-learn Pipeline API, which ensures a seamless transition from raw data to 
model inference while preventing data leakage.

**Project Highlights**
Automated Preprocessing: Integrated scaling for numerical features and one-hot encoding for categorical variables 
within a single Pipeline object.
Model Benchmarking: Comparative analysis between Logistic Regression and Random Forest classifiers.
Hyperparameter Optimization: Utilized GridSearchCV to find the optimal configuration for maximum predictive power.
Model Persistence: The entire pipeline—including preprocessing steps—is exported via joblib for easy deployment.

**Tech Stack**
Core: Scikit-learn, Pandas, NumPy
Optimization: GridSearchCV
Serialization: Joblib

## Task 4: Context-Aware RAG Chatbot
### Description
This project implements a Retrieval-Augmented Generation (RAG) system using LangChain. Unlike standard LLMs, 
this chatbot retrieves relevant information from a custom vector knowledge base before generating responses, 
ensuring accuracy and reducing hallucinations.

**Features**
Knowledge Retrieval: Uses a vectorized document store to query external PDF or Wikipedia data.
Conversational Memory: Implements ConversationalBufferWindowMemory to allow the chatbot to remember previous 
exchanges within the session.
High-Speed Inference: Integrated with the Groq API for lightning-fast Llama 3 / Mixtral responses.
User Interface: A clean, chat-based interface built with Streamlit.

**How it Works**
Ingestion: Documents are split into chunks and embedded.
Vector Store: Chunks are stored in a vector database (e.g., FAISS).
Retrieval: The system searches for the most relevant context based on user query.
Generation: The Groq LLM synthesizes the context and chat history into a coherent answer.

**Tech Stack**
Framework: LangChain
LLM Provider: Groq Cloud API
Vector DB: FAISS 
Frontend: Streamlit
