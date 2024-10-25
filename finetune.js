import "dotenv/config";
import { ChatOpenAI } from "@langchain/openai";
import { PDFLoader } from "@langchain/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "@langchain/vectorstores/memory";
import { OpenAIEmbeddings } from "@langchain/openai";
import { createRetrievalChain } from "@langchain/chains/retrieval";
import { ChatPromptTemplate } from "@langchain/prompts";

// Load environment variables
const API_KEY = process.env.OPENAI_API_KEY;
const DATA_PATH = "./static/Pedagogy_Portfolio.pdf";

// Step 1: Load the PDF Document
const loadDocuments = async () => {
  const loader = new PDFLoader(DATA_PATH);
  const docs = await loader.load();
  console.log(`Loaded ${docs.length} pages from PDF.`);
  return docs;
};

// Step 2: Split the Documents
const splitDocuments = async (docs) => {
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 300,
    chunkOverlap: 30,
  });
  const textChunks = await textSplitter.splitDocuments(docs);
  console.log(`Split document into ${textChunks.length} chunks.`);
  return textChunks;
};

// Step 3: Create the Vector Store with Embeddings
const createVectorStore = async (textChunks) => {
  const embeddings = new OpenAIEmbeddings({ apiKey: API_KEY });
  const vectorStore = await MemoryVectorStore.fromDocuments(textChunks, embeddings);
  console.log("Vector store created with embeddings.");
  return vectorStore;
};

// Step 4: Set up the RAG Chain and Prompt Template
const getConversationalChain = async (retriever) => {
  const promptTemplate = new ChatPromptTemplate([
    ["system", `
      You are a professional assistant for Pedagogy, an educational consulting firm. Respond to user questions based on the given context.
      Only use information in the provided context to answer. If information is not available, ask the user to reach out to the office.
      Keep responses concise and friendly.
      \n\nContext:\n {context}\n\nQuestion:\n{input}\nAnswer:\n
    `],
    ["user", "{input}"],
  ]);

  const model = new ChatOpenAI({ model: "gpt-3.5-turbo", apiKey: API_KEY });
  const chain = await createRetrievalChain({
    retriever,
    prompt: promptTemplate,
    llm: model,
  });
  return chain;
};

// Step 5: Set up the Question-Answering System
const questionAnswerSystem = async (userQuestion, vectorStore) => {
  const retriever = vectorStore.asRetriever();
  const chain = await getConversationalChain(retriever);

  // Run the chain with user input
  const response = await chain.invoke({ input: userQuestion });
  return response.text || "No relevant answer found in the context.";
};

// Main function
const main = async () => {
  // Load and process documents
  const documents = await loadDocuments();
  const textChunks = await splitDocuments(documents);
  const vectorStore = await createVectorStore(textChunks);

  // Example usage
  const userQuestion = "What services does Pedagogy offer?";
  const answer = await questionAnswerSystem(userQuestion, vectorStore);
  console.log("Answer:", answer);
};

main().catch(console.error);
