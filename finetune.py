import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
load_dotenv()
# Constants
DATA_PATH = "/Users/yahyanaji/Desktop/WORK/Pedagogy /PedaBot/PEDAGOGY Portfolio for Chatbot(Powered by MaxAI).pdf"
API_KEY = os.getenv("OPENAI_API_KEY")

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

def get_vector_store(text_chunks, API_KEY):
    texts = [chunk.page_content for chunk in text_chunks]
    embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)
    vector_store = FAISS.from_texts(texts, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_conversational_chain(API_KEY):
    prompt_template = """
    You are a friendly and professional customer service assistant for Pedagogy, the portal of Educational Development. Pedagogy is a consulting firm based in Lebanon that offers a wide range of educational services to academic institutions locally and in the MENA region. Respond to users in a way that sounds natural, conversational, and personalized—like a real person would. Keep the tone warm, helpful, and professional, avoiding generic responses. Focus on understanding the user's needs and providing clear, concise, and empathetic answers that reflect the company's mission to foster quality education within learning communities.

    Answer the question as detailed as possible using only the provided context. If the answer is not in the provided context, do not generate any response or provide a wrong answer. Instead, print the contact information from the PDF and say,if they asked aboutprojects or price say Please reach out to our office. in nice way in addition to contact details from the pdf only from pdf 

    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3, openai_api_key=API_KEY)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    llm_chain = LLMChain(llm=model, prompt=prompt)
    return llm_chain

def user_input(user_question, API_KEY, vector_store):
    if vector_store is None:
        st.error("Error: Vector store is not loaded correctly.")
        return "Error: Vector store is not loaded correctly."
    
    docs = vector_store.similarity_search(user_question)
    if not docs:
        return "No relevant documents found."
    
    chain = get_conversational_chain(API_KEY)
    
    context = " ".join([doc.page_content for doc in docs])
    response = chain({"context": context, "question": user_question})
    
    return response["text"]

# Pre-load the PDF and create the vector store
documents = load_documents()
text_chunks = split_documents(documents)
vector_store = get_vector_store(text_chunks, API_KEY)

# Streamlit App
def main():
    st.title("Pedagogy Q&A Assistant")
    
    # User Question
    user_question = st.text_input("Ask a question about Pedagogy:")
    
    if user_question:
        response = user_input(user_question, API_KEY, vector_store)
        st.write("**Response:**")
        st.write(response)

if __name__ == "__main__":
    main()
