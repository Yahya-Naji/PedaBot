const fs = require("fs");
const pdfParse = require("pdf-parse");
const { OpenAI } = require("openai");
const { LangChain } = require("langchain");  // Ensure LangChain is installed
require("dotenv").config();

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const DATA_PATH = "./static/Pedagogy_Portfolio.pdf";
let embeddingsCache = null;
let chatHistory = [];

// Pedagogy info for assistant introduction
const pedagogyInfo = `
    I am a customer service assistant for Pedagogy, an educational consultancy based in Tripoli, Lebanon, with locations across the MENA region.
    We specialize in educational services, including accreditation consulting, curriculum design, educational technology integration, and capacity-building workshops.
    Contact Information:
    - Address: Riad Solh Street, City Complex, Floor 1, Tripoli, Lebanon
    - Phone: +961 6 444 502
    - Email: info@pedagogycenter.com
    - Business Hours: Mon-Fri: 8 AM - 5 PM, Sat: 8 AM - 1 PM
`;

// Initialize PDF content with Langchain
const initializeEmbeddings = async () => {
  if (embeddingsCache) return embeddingsCache;

  const dataBuffer = fs.readFileSync(DATA_PATH);
  const pdfData = await pdfParse(dataBuffer);
  const textChunks = splitText(pdfData.text);

  const embeddings = await generateEmbeddings(textChunks);
  embeddingsCache = { textChunks, embeddings };
  return embeddingsCache;
};

// Langchain setup for better retrieval
const langchain = new LangChain({
  texts: [pdfData.text],  // Complete text parsed from PDF
  embeddings: openai.embeddings.create({
    model: "text-embedding-ada-002",
  }),
});

// Enhanced search function using LangChain retrieval
const findRelevantInfo = async (query) => {
  const result = await langchain.query(query);
  return result || "I couldn't find specific details in the document, but here’s more general information about Pedagogy's services.";
};

// Conversational function with Langchain PDF search
const chatCompletion = async (prompt) => {
  try {
    // Initialize embeddings if needed
    const { textChunks } = await initializeEmbeddings();

    // Set introductory message if first conversation
    if (chatHistory.length === 0) {
      chatHistory.push({ role: "assistant", content: pedagogyInfo });
    }

    // Search the document if relevant
    const relevantInfo = await findRelevantInfo(prompt);

    // Combine message history and build response with the improved prompt
    const messages = [
      {
        role: "system",
        content: `
          You are Pedagogy’s dedicated customer service assistant. Pedagogy is an educational consulting firm based in Lebanon with branches across the MENA region, offering expertise in accreditation consulting, curriculum design, educational technology, and school management systems. You are here to answer questions from users based on the information you have about Pedagogy’s services, mission, values, and offerings.
          
          If a user’s question can be answered with details from the Pedagogy portfolio document, retrieve that specific information and share it clearly. For example:
          - For accreditation inquiries, explain Pedagogy’s APIS framework.
          - For educational technology, discuss how Pedagogy integrates SMART technologies.
          - For contact information, provide Pedagogy’s phone number, location, or email.

          Contact Information:
          - Address: Riad Solh Street, City Complex, Floor 1, Tripoli, Lebanon
          - Phone: +961 6 444 502
          - Email: info@pedagogycenter.com
          - Business Hours: Mon-Fri: 8 AM - 5 PM, Sat: 8 AM - 1 PM

          If you cannot find an answer in the document, respond warmly and professionally, directing them to Pedagogy’s main services or contact information. Always ensure responses are friendly, concise, and aligned with Pedagogy’s mission to enhance education quality and build competencies.

          When asked general questions, start by introducing Pedagogy’s core mission and offerings before answering the specific query. Maintain a professional and helpful tone in all responses.
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

    // Update chat history
    chatHistory.push({ role: "user", content: prompt });
    chatHistory.push({ role: "assistant", content });

    return { status: 1, response: content };
  } catch (error) {
    return { status: 0, response: `Error: ${error.message}` };
  }
};

module.exports = { chatCompletion };
