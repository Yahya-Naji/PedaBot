const { OpenAI } = require("openai");
require('dotenv').config();
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const chatCompletion = async (prompt) => {
  try {
    const response = await openai.chat.completions.create({
      messages: [
        { 
          role: "system", 
          content: `You are a friendly and professional customer service assistant for Pedagogy, the portal of Educational Development. Pedagogy is a consulting firm based in Lebanon that offers a wide range of educational services to academic institutions locally and in the MENA region. Respond to users in a way that sounds natural, conversational, and personalized—like a real person would. Keep the tone warm, helpful, and professional, avoiding generic responses. Focus on understanding the user's needs and providing clear, concise, and empathetic answers that reflect the company's mission to foster quality education within learning communities.` 
        },
        { role: "user", content: prompt }
      ],
      model: "gpt-3.5-turbo",
    });

    let content = response.choices[0].message.content;

    return {
      status: 1,
      response: content
    };
  } catch (error) {
    return {
      status: 0,
      response: `Error: ${error.message}`
    };
  }
};

module.exports = {
  chatCompletion
};
