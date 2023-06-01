const express = require('express');
const fs = require('fs');
const path = require('path');
const multer = require('multer');
const unzipper = require('unzipper');
const archiver = require('archiver');
const cors = require('cors');
const mongoose = require('mongoose');
require('dotenv').config();
const { check, validationResult } = require('express-validator');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const app = express();
const bodyParser = require('body-parser');
// Models
const User = require('./models/user');


mongoose.connect(process.env.MONGODB_CONNECTION, { useNewUrlParser: true, useUnifiedTopology: true });
mongoose.connect(process.env.MONGODB_CONNECTION, { useNewUrlParser: true, useUnifiedTopology: true })
  .then(() => console.log('Connected to MongoDB'))
  .catch(err => console.log('Failed to connect to MongoDB', err));

app.use(cors());
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
const upload = multer({ dest: 'uploads/' });

// GET endpoint that returns "Hello World!"
app.get('/', (req, res) => {
  res.send('Hello World!');
});

// POST - Register a user

app.post('/register', [
  check('email').isEmail().withMessage('Email is not valid'),
  check('password').isLength({ min: 6 }).withMessage('Password must be at least 6 characters long'),
  check('role').isIn(['client', 'admin']).withMessage('Role must be either "client" or "admin"'),
  check('name').optional().isLength({ min: 3 }).withMessage('Name must be at least 3 characters long'),
  ], async (req, res, next) => {

  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({ errors: errors.array() });
  }
  
  const { email, password, role, name } = req.body;
  const hashedPassword = await bcrypt.hash(password, 10);

  const user = new User({ email, password: hashedPassword, role, name });

  try {
      await user.save();
      res.status(201).send("User registered successfully");
  } catch (error) {
      if (error.code === 11000) { // MongoDB duplicate key error code
      return res.status(409).send("Username is already taken");
      }
      res.status(500).send('Error when registering your account');
  }
});

// user login route
app.post('/login', [
  check('email').notEmpty().withMessage('Email is required'),
  check('password').notEmpty().withMessage('Password is required'),
], async (req, res, next) => {

  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({ errors: errors.array() });
  }
  const { email, password } = req.body;
  try {
    const user = await User.findOne({ email: email });
    if (!user) {
      return res.status(401).send('Invalid credentials');
    }
    
    const isMatch = await bcrypt.compare(password, user.password);
    if (!isMatch) {
      return res.status(401).send('Invalid credentials');
    }
    const token = jwt.sign({ userId: user._id, role: user.role }, process.env.JWT_SECRET, { expiresIn: '90d' });
    res.json({ token: token, userId: user._id });
  } catch (err) {
    res.status(500).send('Error with login');
  }
});


// POST endpoint to handle the ZIP file upload
app.post('/upload', upload.single('zipFile'), (req, res) => {
  const zipPath = req.file.path;

  // Extract the uploaded ZIP file
  unzipper.Open.file(zipPath)
    .then((archive) => {
      const outputDir = 'extracted_files/';

      archive.extract({
        path: outputDir,
        concurrency: 5, // Number of concurrent extractions
      })
      .then(() => {
        console.log('Extraction completed');

        // Divide files into batches
        const divisionFactor = 2; // Specify the division factor here
        const divisionResults = divideFilesIntoBatches(outputDir, divisionFactor);
        console.log('File division results:', divisionResults);

        res.status(200).send('ZIP file received');
      })
      .catch((err) => {
        console.error('Error during extraction:', err);
        res.status(500).send('Error during extraction');
      });
    })
    .catch((err) => {
      console.error('Error opening ZIP file:', err);
      res.status(500).send('Error opening ZIP file');
    });
});

// Function to divide files into batches and create ZIP files
function divideFilesIntoBatches(directoryPath, divisionFactor) {
  const divisionResults = [];

  fs.readdir(directoryPath, (err, files) => {
    if (err) {
      console.error('Error reading directory:', err);
      return;
    }

    files.forEach((file) => {
      const folderPath = path.join(directoryPath, file);

      fs.stat(folderPath, (err, stats) => {
        if (err) {
          console.error(`Error getting stats for ${file}:`, err);
          return;
        }

        if (stats.isDirectory()) {
          fs.readdir(folderPath, (err, subFiles) => {
            if (err) {
              console.error(`Error reading files from ${file}:`, err);
              return;
            }

            const dividedFiles = subFiles.filter((subFile, index) => (index % divisionFactor) === 0);
            const zipOutputPath = path.join(directoryPath, `${file}.zip`);

            const zipStream = fs.createWriteStream(zipOutputPath);
            const zipArchive = archiver('zip');

            zipStream.on('close', () => {
              console.log(`Zip file created for ${file} with divided files`);
              divisionResults.push({ folder: file, filesCount: dividedFiles.length });
            });

            zipArchive.on('error', (err) => {
              console.error(`Error creating zip for ${file}:`, err);
            });

            zipArchive.pipe(zipStream);

            dividedFiles.forEach((subFile) => {
              const filePath = path.join(folderPath, subFile);
              zipArchive.file(filePath, { name: subFile });
            });

            zipArchive.finalize();
          });
        }
      });
    });
  });

  return divisionResults;
}

// Start the server
app.listen(3000, () => {
  console.log('Server is running on port 3000');
});
