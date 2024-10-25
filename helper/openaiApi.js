const fs = require("fs");
const pdfParse = require("pdf-parse");
const { OpenAI } = require("openai");
require("dotenv").config();

// Initialize OpenAI directly with API key
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// PDF file path
const DATA_PATH ="./static/Pedagogy_Portfolio.pdf";


// Load and parse the PDF, generate embeddings
let embeddingsCache = null;

// Load PDF and generate embeddings only once
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

// Generate embeddings for each chunk
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

// Calculate cosine similarity
const cosineSimilarity = (vecA, vecB) => {
  const dotProduct = vecA.reduce((acc, val, i) => acc + val * vecB[i], 0);
  const magnitudeA = Math.sqrt(vecA.reduce((acc, val) => acc + val ** 2, 0));
  const magnitudeB = Math.sqrt(vecB.reduce((acc, val) => acc + val ** 2, 0));
  return dotProduct / (magnitudeA * magnitudeB);
};

// Find the most relevant chunk for a question
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

// Generate response based on PDF context
const chatCompletion = async (prompt) => {
  try {
    // Ensure embeddings are loaded and ready
    const { textChunks, embeddings } = await initializeEmbeddings();

    // Generate question embedding
    const questionEmbeddingResponse = await openai.embeddings.create({
      model: "text-embedding-ada-002",
      input: prompt,
    });
    const questionEmbedding = questionEmbeddingResponse.data[0].embedding;

    // Find the most relevant context chunk
    const relevantChunk = findRelevantChunk(embeddings, questionEmbedding) || "No relevant context found.";

    // Generate the response
    const response = await openai.chat.completions.create({
      model: "gpt-3.5-turbo",
      messages: [
        {
          role: "system",
          content: `You are a friendly and professional customer service assistant for Pedagogy, the portal of Educational Development. Pedagogy is a consulting firm based in Lebanon that offers a wide range of educational services to academic institutions locally and in the MENA region. Respond to users in a way that sounds natural, conversational, and personalized—like a real person would. Keep the tone warm, helpful, and professional, avoiding generic responses. Focus on understanding the user's needs and providing clear, concise, and empathetic answers that reflect the company's mission to foster quality education within learning communities.`,
        },
        { role: "user", content: `${relevantChunk}\n\nQuestion: ${prompt}` },
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
