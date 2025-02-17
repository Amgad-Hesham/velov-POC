import sys
import os
import logging
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_mistralai.chat_models import ChatMistralAI
from mistralai import Mistral

load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

VECTOR_STORE_DIR = "vector_store"
EMBEDDING_MODEL = "sentence-transformers/distiluse-base-multilingual-cased-v1"

PROMPT_TEMPLATE = """
You are an assistant for question-answering tasks.
The provided context is structured into clearly labeled sections and subsections. Each section represents a different plan or pricing detail.

Carefully analyze the context and answer the question by leveraging the headings for better relevance.

CONTEXT:
{context}

QUESTION:
{input}

- If the query is broad (e.g., "Tell me about Vélo’v pricing"), summarize all relevant plans.
- If the query is specific (e.g., "How much does a 24-hour Vélo’v pass cost?"), extract precise details from the most relevant section.
- If no exact answer is found, state that the information is unavailable but don't mention the content.

Provide a well-structured, clear, and concise response.
If the question is about bike plans provide your answer based solely on the above context otherwise respond to the user as you would normally respond.
"""

ROUTING_PROMPT = """
Analyze the user's question and determine which data source it relates to. 
Respond with only one word either "subscriptions" or "stations".

Examples:
- "How much does the annual plan cost?" → subscriptions
- "Bikes available at Part-Dieu" → stations
- "Tell me about youth pricing" → subscriptions
- "Where can I find electric bikes?" → stations

Question: {question}
"""

def determine_data_source(query: str) -> str:
    """
    Determine the appropriate data source for the query using an LLM.
    """
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        logger.error("MISTRAL_API_KEY not set in environment variables.")
        sys.exit(1)

    client = Mistral(api_key=api_key)
    response = client.chat.complete(
        model="open-mistral-nemo",
        messages=[{
            "role": "user",
            "content": ROUTING_PROMPT.format(question=query)
        }]
    )
    return response.choices[0].message.content.strip().lower()

def load_vector_store(source_type: str):
    """
    Load the vector store for the given source type.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    store_path = os.path.join(VECTOR_STORE_DIR, source_type)
    return FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)

def main():
    """
    Main execution function for processing the query.
    """
    if len(sys.argv) < 2:
        logger.error("Usage: python main.py <your_question>")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    print(f"Question: {query}")

    data_source = determine_data_source(query)
    print(f"Routing to: {data_source}")

    try:
        vector_store = load_vector_store(data_source)
    except Exception as e:
        logger.error(f"Error loading vector store for '{data_source}': {e}. Falling back to default 'subscriptions'.")
        vector_store = load_vector_store("subscriptions")

    if data_source == 'subscriptions':
        k = 10
    else:
        k = 30

    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    model = ChatMistralAI(
        model="mistral-large-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY")
    )
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    document_chain = create_stuff_documents_chain(model, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({"input": query})

    # references
    print("\nREFERENCE SOURCES:")
    for i, doc in enumerate(response["context"], 1):
        print(f"Document {i}:")
        print(f"Source: {doc.metadata.get('source', 'Unknown source')}")
        if 'row' in doc.metadata:
            print(f"Row: {doc.metadata['row']}")
        print(f"Content: {doc.page_content}\n{'='*50}\n")

    # Output results
    print("\nANSWER:")
    print(response["answer"])

if __name__ == "__main__":
    main()
