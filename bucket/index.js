const express = require('express');
const bodyParser = require('body-parser');
const multer = require('multer');

const app = express();

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

const datasetStorage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, '/datasets');
    },
    filename: (req, file, cb) => {
        cb(null, file.originalname);
    }
});

const modelStorage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, '/models');
    },
    filename: (req, file, cb) => {
        cb(null, file.originalname);
    }
});

const uploadDataset = multer({ storage: datasetStorage });
const uploadModel = multer({ storage: modelStorage });

app.post('/datasets', uploadDataset.single('dataset'), (req, res) => {
    res.status(200).send('Dataset uploaded.');
});

app.post('/models', uploadModel.single('model'), (req, res) => {
    res.status(200).send('Model uploaded.');
});

app.get('/datasets', (req, res) => {
    res.status(200).send('Get request received for datasets.');
});

app.get('/models', (req, res) => {
    res.status(200).send('Get request received for models.');
});

const PORT = 3000;
app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}.`);
});

module.exports = app;
