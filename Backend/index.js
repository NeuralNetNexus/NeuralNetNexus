const express = require("express");
const fs = require("fs");
const path = require("path");
const multer = require("multer");
const cors = require("cors");
const mongoose = require("mongoose");
require("dotenv").config();
const { check, validationResult } = require("express-validator");
const axios = require("axios");
const app = express();
const bodyParser = require("body-parser");
const FormData = require('form-data');
const k8s = require('@kubernetes/client-node');
const Project = require("./models.project");
const k8sObjects = require('./kubernetes-objects');
const kc = new k8s.KubeConfig();
kc.loadFromDefault();

const k8sApi = kc.makeApiClient(k8s.BatchV1Api);

const PORT = process.env.PORT || 3001;

var database_uri = process.env.MONGODB_CONNECTION || process.env.DATABASE_URI || "localhost";
const MONGODB_URI = database_uri.startsWith("mongodb://")
  ? database_uri
  : `mongodb://${database_uri}:27017/neuralnetnexus`;
mongoose
  .connect(MONGODB_URI, {
    useNewUrlParser: true,
    useUnifiedTopology: true,
  })
  .then(() => console.log("Connected to MongoDB"))
  .catch((err) => {
    console.log("Failed to connect to MongoDB", err);
    process.exit(5);
  });

app.use(
  cors({
    origin: "*",
  })
);

const upload = multer({
  dest: "/usr/app/",
  limits: { fileSize: 200 * 1024 * 1024 }, // 200MB
  fileFilter: function (req, file, cb) {
    if (path.extname(file.originalname) == ".zip") {
      // Allow the upload
      cb(null, true);
    } else {
      // Reject the upload
      cb(new multer.MulterError('LIMIT_UNEXPECTED_FILE', 'Only .zip files are allowed!'));
    }
  }
});

app.use(bodyParser.json({ limit: '200mb' }));
app.use(bodyParser.urlencoded({ limit: '200mb', extended: true, parameterLimit: 100000 }));
app.use(express.json());

// GET endpoint that returns "Hello World!"
app.get("/", (req, res) => {
  res.send("Hello World!");
});

// GET - retrieve project
app.get("/projects", 
async (req, res) => {
  try{
    const projects = await Project.find().sort({createdAt: -1});
    res.json({ projects: projects});
  } catch (err) {
    res.status(500).json({ 
      error: 'GET_PROJECTS_FAILED',
      message: 'Error when retrieving projects'
    });
  }
});

// GET - retrieve project
app.get("/projects/:projectId",
async (req, res) => {
  const { projectId } = req.params;
  try{
    const project = await Project.findOne({_id: projectId});
    res.json({ project: project});
  } catch (err) {
    res.status(500).json({ 
      error: 'GET_PROJECT_FAILED',
      message: 'Error when retrieving project'
    });
  }
});

// GET - retrieve project's files
app.get("/projects/:projectId/files",
async (req, res) => {
  const { projectId } = req.params;
  try{

    const folderPath = `/usr/app/models/${projectId}`;

    fs.readdir(folderPath, (err, files) => {
      if (err) {
        console.error('Error reading folder:', err);
        return;
      }
      res.json({files: files });
    });
    
  } catch (err) {
    res.status(500).json({ 
      error: 'GET_PROJECT_FAILED',
      message: 'Error when retrieving project'
    });
  }
});


app.get("/models/:filename", async (req, res) => {
  const { filename } = req.params;
  try {

    const response = await axios.get(`http://bucket-service/models/${filename}`, { responseType: 'stream' });

    // Set the headers for the response
    const headers = {
      'Content-Type': response.headers['content-type'],
      'Content-Length': response.headers['content-length'],
    };
    res.writeHead(200, headers);

    response.data.pipe(res);
  } catch (err) {
    res.status(500).json({
      error: 'GET_PROJECT_FAILED',
      message: 'Error when retrieving project'
    });
  }
});

// GET - retrieve project's file by name
app.get("/projects/:projectId/files/:filename",
async (req, res) => {
  const { projectId, filename } = req.params;
  try{
    let file = `/usr/app/models/${projectId}/${filename}.`
    if(filename.includes("confusion_matrix")){
      file = file + "png";
    } else{
      file = file + "pth";
    }
    res.sendFile(file);
    
  } catch (err) {
    console.log(err)
    res.status(500).json({ 
      error: 'GET_PROJECT_FAILED',
      message: 'Error when retrieving project'
    });
  }
});

// POST endpoint to handle the ZIP file upload
app.post("/upload", upload.single("dataset"), async (req, res, next) => {
  const { model, projectName } = req.body;

  // Check if a model was provided in the request
  if (!model) {
    return res.status(400).json({
      error: 'UPLOAD_FAILED',
      message: 'No model provided'
    });
  }

  // Check if a file was sent in the request
  if (!req.file) {
    return res.status(400).json({
      error: 'UPLOAD_FAILED',
      message: 'No file uploaded'
    });
  }

  const fileExtension = path.extname(req.file.originalname);
  if (fileExtension !== '.zip') {
    return res.status(400).json({
      error: 'UPLOAD_FAILED',
      message: 'Only ZIP file is allowed'
    });
  }

  // Check if a project name was given
  if (!projectName) {
    return res.status(400).json({
      error: 'UPLOAD_FAILED',
      message: 'No project name was given'
    });
  }

  const zipFilePath = req.file.path;

  try {
    // Define the properties for the new Project instance
    const projectInfo = {
      dataset: req.file.originalname,
      size: req.file.size / (1024 * 1024), // Convert bytes to MB
      model,
      state: 'Pending',
      name: projectName,
      logs: [],
      aggregator: {
        accuracy: 0,
        loss: 0,
        f1_score: 0,
        precision: 0,
        recall: 0
      }
    };

    // console.log(projectInfo)

    const project = new Project(projectInfo);
    const savedProject = await project.save();
    const projectId = savedProject._id;
    
    // Store Dataset in Bucket
    const fileData = fs.readFileSync(zipFilePath);
    const formData = new FormData();
    const filename = `${projectId}${fileExtension}`;
    formData.append('dataset', fileData, { filename: filename });
    axios.post("http://bucket-service/datasets", formData, {
      headers: formData.getHeaders()
    })
    .then(response => {
      console.log('File uploaded successfully');
      // Delete the file after upload
      // fs.unlinkSync(modifiedFilePath);
    })
    .catch(error => {
      console.error('Error uploading file:', error);
    });

    // Trigger the train-suppervisor job
    const jobManifest = k8sObjects.getTrainSupervisorObject(projectId, model);
    // console.log(jobManifest)
    k8sApi.createNamespacedJob('default', jobManifest)
      .then((response) => {
        // console.log('Job created with response:', response.body);
        return res.status(200).json({
          message: 'ZIP file received',
          projectId: projectId
        });
      })
      .catch((err) => {
        console.error('Error creating job:', err);
        return res.status(500).json({
          error: 'UPLOAD_FAILED',
          message: 'Error creating job'
        });
      });

  } catch (error) { 
    console.error('Error saving file information to MongoDB:', error);
    res.status(500).json({
      error: 'UPLOAD_FAILED',
      message: 'Error saving file information'
    });
  }
});

// PATCH - update project's number of splits
app.patch("/projects/:projectId/n-splits", [
  check("splits")
  .notEmpty().withMessage("Number of splits is required")
  .isFloat({ gt: 0 }).withMessage("Number of splits must be greater than 0")],
async (req, res) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({ error: 'PATCH_PROJECT_FAILED', message: errors.array() });
  }

  const { projectId } = req.params;
  const { splits } = req.body;
  try{
    const project = await Project.findOne({_id: projectId});
    project.n_splits = splits
    project.splits = []
    project.logs = []
    for (let i = 0; i < splits; i++) {
      project.splits.push({
        id: i + 1,
        train_accuracies: [],
        val_accuracies: [],
        train_losses: [],
        val_losses: [],
        cpu_usage: 0,
        ram_usage: 0,
        logs: [],
      })
    }
    await project.save()
    res.json({ project: project});
  } catch (err) {
    res.status(500).json({ 
      error: 'PATCH_PROJECT_FAILED',
      message: 'Error when updating project'
    });
  }
});

// PATCH - update project's state
app.patch("/projects/:projectId/state", [
  check("state")
  .notEmpty().withMessage("State is required")],
async (req, res) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({ error: 'PATCH_PROJECT_FAILED', message: errors.array() });
  }

  const { projectId } = req.params;
  const { state } = req.body;
  try{
    const project = await Project.findOne({_id: projectId});
    project.state = state
    await project.save()
    res.json({ project: project});
  } catch (err) {
    res.status(500).json({ 
      error: 'PATCH_PROJECT_FAILED',
      message: 'Error when updating project'
    });
  }
});

// PATCH - Add logs to project split
app.patch("/projects/:projectId/splits/:splitId/logs",
async (req, res) => {
    const { projectId, splitId } = req.params;
    const { logs } = req.body;

    try {
      const project = await Project.findOne({ _id: projectId });
      project.splits[splitId].logs.push(logs);

      await project.save();
      res.json({ project: project });
    } catch (err) {
      res.status(500).json({ 
        error: 'PUT_PROJECT_FAILED',
        message: 'Error when updating project'
      });
    }
});

// PATCH - Add logs to project
app.patch("/projects/:projectId/logs",
async (req, res) => {
    const { projectId } = req.params;
    const { logs } = req.body;

    try {
      const project = await Project.findOne({ _id: projectId });
      project.logs.push(logs);

      await project.save();
      res.json({ project: project });
    } catch (err) {
      res.status(500).json({ 
        error: 'PUT_PROJECT_FAILED',
        message: 'Error when updating project'
      });
    }
});

// PATCH - Add filepath to project
// app.patch("/projects/:projectId/filepath",
// async (req, res) => {
//     const { projectId, splitId } = req.params;
//     const { logs } = req.body;

//     try {
//       const project = await Project.findOne({ _id: projectId });
//       project.logs.push(logs);

//       await project.save();
//       res.json({ project: project });
//     } catch (err) {
//       res.status(500).json({ 
//         error: 'PUT_PROJECT_FAILED',
//         message: 'Error when updating project'
//       });
//     }
// });

// PATCH - Add accuracy to split
app.patch("/projects/:projectId/splits/:splitId/metrics",
async (req, res) => {
    const { projectId, splitId } = req.params;
    const { train_accuracy, val_accuracy, train_loss, val_loss, cpu_usage, ram_usage } = req.body;

    try {
      const project = await Project.findOne({ _id: projectId });

      project.splits[splitId].train_accuracies.push(train_accuracy);
      project.splits[splitId].val_accuracies.push(val_accuracy);
      project.splits[splitId].train_losses.push(train_loss);
      project.splits[splitId].val_losses.push(val_loss);

      project.splits[splitId].cpu_usage = cpu_usage;
      project.splits[splitId].ram_usage = ram_usage;

      await project.save();
      res.json({ project: project });
    } catch (err) {
      res.status(500).json({ 
        error: 'PUT_PROJECT_FAILED',
        message: 'Error when updating project'
      });
    }
});

// PATCH - Add accuracy to split
app.patch("/projects/:projectId/aggregatormetrics",
async (req, res) => {
    const { projectId, splitId } = req.params;
    const { loss, accuracy, precision, recall, f1_score } = req.body;

    try {
      const project = await Project.findOne({ _id: projectId });
      project.aggregator.loss = loss;
      project.aggregator.accuracy = accuracy;
      project.aggregator.precision = precision;
      project.aggregator.recall = recall;
      project.aggregator.f1_score = f1_score;

      await project.save();
      res.json({ project: project });
    } catch (err) {
      res.status(500).json({ 
        error: 'PUT_PROJECT_FAILED',
        message: 'Error when updating project'
      });
    }
});

app.listen(PORT, () => {
  console.log("Server is running on port " + PORT);
});