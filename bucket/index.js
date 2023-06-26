const express = require('express');
const bodyParser = require('body-parser');
const multer = require('multer');
const fs = require('fs');
const path = require('path');

const app = express();

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

const datasetStorage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, '/usr/app/datasets');
    },
    filename: (req, file, cb) => {
        cb(null, file.originalname);
    }
});

const modelStorage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, '/usr/app/models');
    },
    filename: (req, file, cb) => {
        cb(null, file.originalname);
    }
});

const uploadDataset = multer({ storage: datasetStorage });
const uploadModel = multer({ storage: modelStorage });

app.post('/datasets', uploadDataset.single('dataset'), (req, res) => {
    res.status(200).send({'message': 'Dataset uploaded.'});
});

app.post('/models', uploadModel.single('model'), (req, res) => {
    res.status(200).send({'message:': 'Model uploaded.'});
});

app.get('/datasets/:fileName', (req, res) => {
    const fileName = req.params.fileName;
    const directoryPath = path.join(__dirname, '/datasets/');
    res.download(directoryPath + fileName, fileName, (err) => {
        if (err) {
            res.status(500).send({
                message: "Could not download the file. " + err,
            });
        }
    });
});

app.get('/models/:fileName', (req, res) => {
    const fileName = req.params.fileName;
    const directoryPath = path.join(__dirname, '/models/');
    res.download(directoryPath + fileName, fileName, (err) => {
        if (err) {
            res.status(500).send({
                message: "Could not download the file. " + err,
            });
        }
    });
});

const PORT = process.env.PORT || 3003;
app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}.`);
});

module.exports = app;
