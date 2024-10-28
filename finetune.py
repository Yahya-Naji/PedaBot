import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Constants
DATA_PATH = "./static/Pedagogy_Portfolio.pdf"
API_KEY = os.getenv("OPENAI_API_KEY")

# Load document and create vector store
@st.cache_data
def load_document():
    document_loader = PyPDFLoader(DATA_PATH)
    document = document_loader.load()
    return document[0].page_content  # Extract the full text if it's only one document

@st.cache_data
def create_vector_store(document_text, API_KEY):
    embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)
    vector_store = FAISS.from_texts([document_text], embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def load_vector_store():
    embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)
    if os.path.exists("faiss_index"):
        return FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
    else:
        document_text = load_document()
        return create_vector_store(document_text, API_KEY)

# Function to set up conversation chain
def get_conversational_chain(API_KEY):
    model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3, openai_api_key=API_KEY)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a friendly and professional customer service assistant for Pedagogy. "
                   "Answer the user's questions based on the provided document information."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    
    llm_chain = LLMChain(llm=model, prompt=prompt)
    return llm_chain

# User input processing with chat history
def user_input(user_question, API_KEY, vector_store, chat_history, document_text):
    if vector_store is None:
        st.error("Error: Vector store is not loaded correctly.")
        return "Error: Vector store is not loaded correctly."
    
    # Directly use the loaded document content as context
    chain = get_conversational_chain(API_KEY)
    response = chain({"context": document_text, "question": user_question, "chat_history": chat_history})
    
    return response["text"]

# Load the vector store and document text
vector_store = load_vector_store()
document_text = load_document()  # Load the full document content once

# Streamlit App
def main():
    st.title("Pedagogy Q&A Assistant")
    
    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # User Question
    user_question = st.text_input("Ask a question about Pedagogy:")
    if user_question:
        # Get the response based on user question and chat history
        response = user_input(user_question, API_KEY, vector_store, st.session_state.chat_history, document_text)
        
        # Update the chat history in session state
        st.session_state.chat_history.append(HumanMessage(content=user_question))
        st.session_state.chat_history.append(AIMessage(content=response))
        
        # Display the response
        st.write("**Response:**")
        st.write(response)

# Run the app
if __name__ == "__main__":
    main()