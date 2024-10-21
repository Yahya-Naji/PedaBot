from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from flask_cors import CORS

# Load environment variables
load_dotenv()

# Flask app setup
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests if needed

# Constants
DATA_PATH = os.getenv("PDF_PATH","/Users/yahyanaji/Desktop/WORK/Pedagogy /PedaBot/static/PEDAGOGY Portfolio for Chatbot(Powered by MaxAI).pdf")  # Adjust path for Render deployment
API_KEY = os.getenv("OPENAI_API_KEY")

# Load documents and vector store during startup
vector_store = None
try:
    documents = PyPDFLoader(DATA_PATH).load()
    text_chunks = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30).split_documents(documents)
    vector_store = FAISS.from_texts([chunk.page_content for chunk in text_chunks], OpenAIEmbeddings(openai_api_key=API_KEY))
except Exception as e:
    print(f"Error loading documents or vector store: {e}")
    documents = None

def get_conversational_chain(API_KEY):
    prompt_template = """
    You are a friendly and professional customer service assistant for Pedagogy, the portal of Educational Development. Pedagogy is a consulting firm based in Lebanon that offers a wide range of educational services to academic institutions locally and in the MENA region. Respond to users in a way that sounds natural, conversational, and personalized—like a real person would. Keep the tone warm, helpful, and professional, avoiding generic responses. Focus on understanding the user's needs and providing clear, concise, and empathetic answers that reflect the company's mission to foster quality education within learning communities.

    Answer the question as detailed as possible using only the provided context. If the answer is not in the provided context, do not generate any response or provide a wrong answer. Instead, print the contact information from the PDF and say, if they asked about projects or price, say Please reach out to our office in a nice way in addition to contact details from the PDF.

    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3, openai_api_key=API_KEY)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    llm_chain = LLMChain(llm=model, prompt=prompt)
    return llm_chain

def user_input(user_question):
    if vector_store is None:
        return "Error: Vector store is not available."
    docs = vector_store.similarity_search(user_question)
    if not docs:
        return "No relevant documents found."
    chain = get_conversational_chain(API_KEY)
    context = " ".join([doc.page_content for doc in docs])
    response = chain({"context": context, "question": user_question})
    return response["text"]

# Define the API endpoint for the chatbot
@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        user_question = request.json.get('question')
        if not user_question:
            return jsonify({"error": "No question provided"}), 400
        response_text = user_input(user_question)
        return jsonify({"response": response_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))  # Use the correct port for Render
    app.run(host='0.0.0.0', port=port)
