const fs = require("fs");
const pdfParse = require("pdf-parse");
const { OpenAI } = require("openai");
require("dotenv").config();

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const DATA_PATH = "./static/Pedagogy_Portfolio.pdf";

let embeddingsCache = null;
let chatHistory = [];

// Initialize embeddings (only run once)
const initializeEmbeddings = async () => {
  if (embeddingsCache) return embeddingsCache;

  const dataBuffer = fs.readFileSync(DATA_PATH);
  const pdfData = await pdfParse(dataBuffer);
  const textChunks = splitText(pdfData.text);

  const embeddings = await generateEmbeddings(textChunks);
  embeddingsCache = { textChunks, embeddings };
  return embeddingsCache;
};

// Split text into chunks
const splitText = (text, chunkSize = 300, overlap = 30) => {
  const chunks = [];
  for (let i = 0; i < text.length; i += chunkSize - overlap) {
    chunks.push(text.slice(i, i + chunkSize));
  }
  return chunks;
};

// Generate embeddings
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

// Find relevant chunk
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

// Generate response with chat history
const chatCompletion = async (prompt) => {
  try {
    const { textChunks, embeddings } = await initializeEmbeddings();

    const questionEmbeddingResponse = await openai.embeddings.create({
      model: "text-embedding-ada-002",
      input: prompt,
    });
    const questionEmbedding = questionEmbeddingResponse.data[0].embedding;

    const relevantChunk = findRelevantChunk(embeddings, questionEmbedding) || "No relevant context found.";

    // Combine message history for contextual responses
    const messages = [
      {
        role: "system",
        content: "You are a friendly assistant. Answer questions based on available document content.",
      },
      ...chatHistory,
      { role: "user", content: `${relevantChunk}\n\nQuestion: ${prompt}` },
    ];

    const response = await openai.chat.completions.create({
      model: "gpt-3.5-turbo",
      messages,
    });

    const content = response.choices[0].message.content;

    // Update chat history
    chatHistory.push({ role: "user", content: prompt });
    chatHistory.push({ role: "assistant", content });

    return { status: 1, response: content };
  } catch (error) {
    return { status: 0, response: `Error: ${error.message}` };
  }
};

module.exports = { chatCompletion };
