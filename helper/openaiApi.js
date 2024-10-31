const fs = require("fs");
const pdfParse = require("pdf-parse");
const { OpenAI } = require("openai");
require("dotenv").config();

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const DATA_PATH = "./static/Pedagogy_Portfolio.pdf";

let embeddingsCache = null;
let chatHistory = [];

// Pedagogy info to introduce the assistant
const pedagogyInfo = `
    I am a customer service assistant for Pedagogy, an educational consultancy based in Tripoli.
    Address: Riad Solh Street, City Complex, Floor 1, Tripoli.
    Business Hours: Open until 5 PM.
    Contact: 06 444 502.
`;

// Load and parse the PDF, generate embeddings only once
const initializeEmbeddings = async () => {
  if (embeddingsCache) return embeddingsCache;

  const dataBuffer = fs.readFileSync(DATA_PATH);
  const pdfData = await pdfParse(dataBuffer);
  const textChunks = splitText(pdfData.text);

  const embeddings = await generateEmbeddings(textChunks);
  embeddingsCache = { textChunks, embeddings };
  return embeddingsCache;
};

// Split text into manageable chunks
const splitText = (text, chunkSize = 300, overlap = 30) => {
  const chunks = [];
  for (let i = 0; i < text.length; i += chunkSize - overlap) {
    chunks.push(text.slice(i, i + chunkSize));
  }
  return chunks;
};

// Generate embeddings for text chunks
const generateEmbeddings = async (chunks) => {
  const embeddings = [];
  for (const chunk of chunks) {
    const embeddingResponse = await openai.embeddings.create({
      model: "text-embedding-ada-002",
      input: chunk,
    });
    embeddings.push({
      content: chunk,
      embedding: embeddingResponse.data[0].embedding,
    });
  }
  return embeddings;
};

// Cosine similarity calculation
const cosineSimilarity = (vecA, vecB) => {
  const dotProduct = vecA.reduce((acc, val, i) => acc + val * vecB[i], 0);
  const magnitudeA = Math.sqrt(vecA.reduce((acc, val) => acc + val ** 2, 0));
  const magnitudeB = Math.sqrt(vecB.reduce((acc, val) => acc + val ** 2, 0));
  return dotProduct / (magnitudeA * magnitudeB);
};

// Find relevant document chunk based on cosine similarity
const findRelevantChunk = (embeddings, questionEmbedding) => {
  let bestMatch = null;
  let highestScore = -Infinity;

  embeddings.forEach(({ content, embedding }) => {
    const score = cosineSimilarity(embedding, questionEmbedding);
    if (score > highestScore) {
      highestScore = score;
      bestMatch = content;
    }
  });

  return bestMatch;
};

// Generate assistant response with conversational history and Pedagogy introduction
const chatCompletion = async (prompt) => {
  try {
    // Ensure embeddings are initialized
    const { textChunks, embeddings } = await initializeEmbeddings();

    // Generate embedding for the user question
    const questionEmbeddingResponse = await openai.embeddings.create({
      model: "text-embedding-ada-002",
      input: prompt,
    });
    const questionEmbedding = questionEmbeddingResponse.data[0].embedding;

    // Find the most relevant context in the document if any
    const relevantChunk = findRelevantChunk(embeddings, questionEmbedding) || "";

    // Set introductory message if it's the first conversation
    if (chatHistory.length === 0) {
      chatHistory.push({ role: "assistant", content: pedagogyInfo });
    }

    // Combine message history
    const messages = [
      { role: "system", content: "You are a friendly assistant for Pedagogy." },
      ...chatHistory,
      { role: "user", content: prompt },
    ];

    // Append relevant document content if found
    if (relevantChunk) {
      messages.push({
        role: "assistant",
        content: `Here is what I found relevant to your question: ${relevantChunk}`,
      });
    }

    // Generate response with OpenAI API
    const response = await openai.chat.completions.create({
      model: "gpt-3.5-turbo",
      messages,
    });

    const content = response.choices[0].message.content;

    // Update chat history with the latest interaction
    chatHistory.push({ role: "user", content: prompt });
    chatHistory.push({ role: "assistant", content });

    return { status: 1, response: content };
  } catch (error) {
    return { status: 0, response: `Error: ${error.message}` };
  }
};

// Optionally clear history for a fresh conversation
const clearChatHistory = () => {
  chatHistory = [];
};

module.exports = { chatCompletion, clearChatHistory };
