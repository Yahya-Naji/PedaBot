from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI


app = FastAPI()

class Query(BaseModel):
    question: str
    chat_history: list

# Load PDF and create vector store on startup
loader = PyPDFLoader("/Users/yahyanaji/Desktop/YouTube-Facebook-Messenger-Openai-13112023-main/Finetune info about Pedagogy.pdf")
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=24)
chunks = text_splitter.create_documents([page.page_content for page in pages])
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(chunks, embeddings)

qa_chain = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.1), db.as_retriever())

@app.post("/query")
def answer_question(query: Query):
    try:
        result = qa_chain({"question": query.question, "chat_history": query.chat_history})
        return {"answer": result['answer']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

