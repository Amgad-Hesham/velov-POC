# Bike Company Chatbot - Proof of Concept

This project is a proof-of-concept chatbot for a bike company that answers questions based on local data stored in Markdown and CSV files. It leverages FAISS for vector storage and Mistral AI for answering queries intelligently.

## Features
- Parses Markdown and CSV files to create a vectorized knowledge base.
- Uses Mistral AI to provide intelligent responses.
- Categorizes user queries to select the most relevant data source.
- Retrieves and processes information using FAISS vector storage.

## Prerequisites
Before running the chatbot, ensure you have:
- Python 3.12.9 or later installed.
- A Mistral AI API key.
- Required dependencies installed.

## Installation
1. Clone this repository:
   
2. Create an environment"
   ```sh
   conda create --name my_env python=3.12.9

   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

4. Set up your API key:
   - Create a `.env` file in the project root and add:
     ```env
     MISTRAL_API_KEY=<your-api-key>
     ```

## Usage
### 1. Generate Vector Store
Run the following command to process Markdown and CSV files into a FAISS vector store:
```sh
python create_vector_store.py
```
**KEEP IN MIND THIS WILL TAKE TIME TO DOWNLOAD THE EMBEDDING MODEL THE FIRST TIME IT RUNS**
By default, it reads from `data/` and stores vectors in `vector_store/`.

### 2. Run the Chatbot
Ask a question by running:
```sh
python main.py "Your question here"
```
Example:
```sh
python main.py "what is the 24 hour plan"
```

## Project Structure
```
├── create_vector_store.py  # Script to process Markdown & CSV files into FAISS
├── main.py                 # Chatbot entry point
├── requirements.txt        # List of required dependencies
├── .env                    # Environment variables (add manually)
├── data/                   # Markdown & CSV files (your knowledge base)
├── vector_store/           # FAISS vector storage directory
```

## Notes
- Ensure your dataset is placed inside the `data/` directory before running `create_vector_store.py`.
- The chatbot intelligently routes queries between different data sources (`subscriptions` and `stations`).
- The system uses `sentence-transformers/distiluse-base-multilingual-cased-v1` as the embedding model.


