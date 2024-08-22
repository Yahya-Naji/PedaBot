const axios = require('axios');
require('dotenv').config();

const chatCompletion = async (prompt) => {
  try {
    // Send a POST request to your Flask API
    const response = await axios.post('http://localhost:5000/api/chat', {
      question: prompt
    });

    return {
      status: 1,
      response: response.data.answer
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
