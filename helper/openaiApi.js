const fs = require("fs");
const pdfParse = require("pdf-parse");
const { OpenAI } = require("openai");
require("dotenv").config();

// Initialize OpenAI directly with API key
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// PDF file path
const DATA_PATH = "./static/Pedagogy_Portfolio.pdf";

// Load and parse the PDF, generate embedding only once
let embeddingsCache = null;

// Load PDF and generate embedding only once
const initializeEmbedding = async () => {
  if (embeddingsCache) return embeddingsCache;

  // Read and parse the PDF
  const dataBuffer = fs.readFileSync(DATA_PATH);
  const pdfData = await pdfParse(dataBuffer);
  const documentText = pdfData.text;

  // Generate a single embedding for the entire document
  const embeddingResponse = await openai.embeddings.create({
    model: "text-embedding-ada-002",
    input: documentText,
  });
  
  embeddingsCache = { documentContent: documentText, embedding: embeddingResponse.data[0].embedding };
  return embeddingsCache;
};

// Calculate cosine similarity
const cosineSimilarity = (vecA, vecB) => {
  const dotProduct = vecA.reduce((acc, val, i) => acc + val * vecB[i], 0);
  const magnitudeA = Math.sqrt(vecA.reduce((acc, val) => acc + val ** 2, 0));
  const magnitudeB = Math.sqrt(vecB.reduce((acc, val) => acc + val ** 2, 0));
  return dotProduct / (magnitudeA * magnitudeB);
};

// Generate response based on PDF context
const chatCompletion = async (prompt) => {
  try {
    // Ensure embedding is loaded and ready
    const { documentContent, embedding } = await initializeEmbedding();

    // Generate question embedding
    const questionEmbeddingResponse = await openai.embeddings.create({
      model: "text-embedding-ada-002",
      input: prompt,
    });
    const questionEmbedding = questionEmbeddingResponse.data[0].embedding;

    // Calculate similarity score to determine relevance
    const similarityScore = cosineSimilarity(embedding, questionEmbedding);
    const relevantContext = similarityScore > 0.8 ? documentContent : "No highly relevant context found.";

    // Generate the response
    const response = await openai.chat.completions.create({
      model: "gpt-3.5-turbo",
      messages: [
        {
          role: "system",
          content: `You are a friendly and professional customer service assistant for Pedagogy, the portal of Educational Development. Respond to users in a warm, concise, and helpful tone. Avoid repeating greetings like "Thank you for reaching out" or "Hello" if the user has already initiated a conversation. Focus on directly addressing the user's question or request with clear, conversational responses. Only reintroduce yourself and Pedagogy's mission if the user is new or has specific questions about the company.`,
        },
        { role: "user", content: `${relevantContext}\n\nQuestion: ${prompt}` },
      ],
    });

    const content = response.choices[0].message.content;

    return {
      status: 1,
      response: content,
    };
  } catch (error) {
    return {
      status: 0,
      response: `Error: ${error.message}`,
    };
  }
};

module.exports = {
  chatCompletion,
};
