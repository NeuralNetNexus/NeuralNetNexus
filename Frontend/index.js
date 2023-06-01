const express = require('express');
const path = require('path');

const app = express();
const indexPath = path.join(__dirname, 'index.html');

app.get('/', (req, res) => {
  res.sendFile(indexPath);
});

app.listen(3001, () => {
  console.log('Server is running on port 3001');
});
