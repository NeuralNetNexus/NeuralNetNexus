const express = require('express');
const fs = require('fs');
const path = require('path');
const multer = require('multer');
const unzipper = require('unzipper');
const archiver = require('archiver');
const cors = require('cors');
const mongoose = require('mongoose');
require('dotenv').config();

const app = express();

mongoose.connect(process.env.MONGODB_CONNECTION, { useNewUrlParser: true, useUnifiedTopology: true });
mongoose.connect(process.env.MONGODB_CONNECTION, { useNewUrlParser: true, useUnifiedTopology: true })
  .then(() => console.log('Connected to MongoDB'))
  .catch(err => console.log('Failed to connect to MongoDB', err));
  
app.use(cors());
const upload = multer({ dest: 'uploads/' });

// GET endpoint that returns "Hello World!"
app.get('/', (req, res) => {
  res.send('Hello World!');
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
