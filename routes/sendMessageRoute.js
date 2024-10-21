const express = require('express');
const axios = require('axios'); // Ensure axios is imported for making HTTP requests
const router = express.Router();
require('dotenv').config();

const { sendMessage, setTypingOff, setTypingOn } = require('../helper/messengerApi');

router.post('/', async (req, res) => {
  try {
    let body = req.body;
    let senderId = body.senderId;
    let query = body.query;

    // Start typing indication
    await setTypingOn(senderId);

    // Send query to the Flask API hosted on Render
    const flaskApiUrl = process.env.FLASK_API_URL || 'https://pedabot.onrender.com/chatbot';  // Use environment variable for Flask API URL
    const pythonApiResponse = await axios.post(flaskApiUrl, {
      question: query
    });

    // Extract the response text from the Flask API
    if (pythonApiResponse.status === 200 && pythonApiResponse.data.response) {
      let result = pythonApiResponse.data.response;

      // Send the response back to the user via the Messenger API
      await sendMessage(senderId, result);

      // Stop typing indication
      await setTypingOff(senderId);

      console.log(`Sender ID: ${senderId}`);
      console.log(`Response: ${result}`);
    } else {
      throw new Error('No valid response from the Flask API.');
    }
  } catch (error) {
    console.error('Error in sendmessageroute:', error.message);
    await sendMessage(senderId, 'Sorry, something went wrong while processing your request.');
    await setTypingOff(senderId);
  }
  
  res.status(200).send('OK');
});

module.exports = {
  router
};
