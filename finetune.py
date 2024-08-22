import os
from dotenv import load_dotenv

from flask import Flask, request, jsonify
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
load_dotenv()
# Constants
DATA_PATH = "/Users/yahyanaji/Desktop/PedaBot/Finetune info about Pedagogy.pdf"
API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize Flask app
app = Flask(__name__)

def load_documents():
    document_loader = PyPDFLoader(DATA_PATH)
    return document_loader.load()

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def get_vector_store(text_chunks, api_key):
    texts = [chunk.page_content for chunk in text_chunks]
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vector_store = FAISS.from_texts(texts, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_conversational_chain(api_key):
    prompt_template = """
    You are a friendly and professional customer service assistant for Pedagogy, the portal of Educational Development. Pedagogy is a consulting firm based in Lebanon that offers a wide range of educational services to academic institutions locally and in the MENA region. Respond to users in a way that sounds natural, conversational, and personalized—like a real person would. Keep the tone warm, helpful, and professional, avoiding generic responses. Focus on understanding the user's needs and providing clear, concise, and empathetic answers that reflect the company's mission to foster quality education within learning communities.

    Answer the question as detailed as possible using only the provided context. If the answer is not in the provided context, do not generate any response or provide a wrong answer. Instead, print the contact information from the PDF and say, "Please reach out to our office."

    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3, openai_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    llm_chain = LLMChain(llm=model, prompt=prompt)
    return llm_chain

def user_input(user_question, api_key, vector_store):
    docs = vector_store.similarity_search(user_question)
    chain = get_conversational_chain(api_key)
    context = " ".join([doc.page_content for doc in docs])
    response = chain({"context": context, "question": user_question})
    return response["text"]

# Pre-load the PDF and create the vector store
documents = load_documents()
text_chunks = split_documents(documents)
vector_store = get_vector_store(text_chunks, api_key)

# Define the Flask route for the API
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_question = data.get("question")
    
    if not user_question:
        return jsonify({"error": "No question provided"}), 400
    
    response = user_input(user_question, api_key, vector_store)
    return jsonify({"answer": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
