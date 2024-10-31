const fs = require("fs");
const pdfParse = require("pdf-parse");
const { OpenAI } = require("openai");
require("dotenv").config();

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const DATA_PATH = "./static/Pedagogy_Portfolio.pdf";

let embeddingsCache = null;
let chatHistory = [];

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
const findRelevantChunk = async (query) => {
  const { textChunks, embeddings } = await initializeEmbeddings();
  const questionEmbeddingResponse = await openai.embeddings.create({
    model: "text-embedding-ada-002",
    input: query,
  });
  const questionEmbedding = questionEmbeddingResponse.data[0].embedding;

  let bestMatch = null;
  let highestScore = -Infinity;

  embeddings.forEach(({ content, embedding }) => {
    const score = cosineSimilarity(embedding, questionEmbedding);
    if (score > highestScore) {
      highestScore = score;
      bestMatch = content;
    }
  });

  return bestMatch || "I'm here to assist with Pedagogy’s offerings or answer any questions you have about our services. Feel free to ask!";
};

// Generate assistant response with conversational history
const chatCompletion = async (prompt) => {
  try {
    const relevantInfo = await findRelevantChunk(prompt);

    // Prompt for the assistant's response structure
    const messages = [
      {
        role: "system",
        content: `
          You are Pedagogy’s customer service assistant. Pedagogy is an educational consulting firm based in Lebanon with branches across the MENA region. Provide information from the portfolio document if relevant, focusing on services like accreditation consulting, educational technology integration, and curriculum design. For contact info, use:
          - Address: Riad Solh Street, City Complex, Floor 1, Tripoli, Lebanon
          - Phone: +961 6 444 502
          - Email: info@pedagogycenter.com
          - Hours: Mon-Fri: 8 AM - 5 PM, Sat: 8 AM - 1 PM
          
          When the document lacks specific information, respond professionally and direct users to Pedagogy’s main offerings or contact details.
        `,
      },
      ...chatHistory,
      { role: "user", content: prompt },
      { role: "assistant", content: relevantInfo },
    ];

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

module.exports = { chatCompletion };
