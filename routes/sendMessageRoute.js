const express = require('express');
const axios = require('axios'); // Add axios for making HTTP requests
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

    // Send query to the Python API
    const pythonApiResponse = await axios.post('http://localhost:5001/chatbot', {
      question: query
    });

    // Extract the response text from the Python API
    let result = pythonApiResponse.data.response;

    // Send the response back to the user via the Messenger API
    await sendMessage(senderId, result);

    // Stop typing indication
    await setTypingOff(senderId);

    console.log(`Sender ID: ${senderId}`);
    console.log(`Response: ${result}`);
  } catch (error) {
    console.error('Error in sendmessageroute:', error);
  }
  
  res.status(200).send('OK');
});

module.exports = {
  router
};
