const express = require("express");
const fs = require("fs");
const path = require("path");
const multer = require("multer");
const cors = require("cors");
const mongoose = require("mongoose");
require("dotenv").config();
const { check, validationResult } = require("express-validator");
const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");
const app = express();
const bodyParser = require("body-parser");
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
  .catch((err) => console.log("Failed to connect to MongoDB", err));

app.use(
  cors({
    origin: "*",
  })
);

const upload = multer({
  dest: "/usr/app/datasets/",
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
      name: projectName
    };
    console.log(projectInfo)

    const project = new Project(projectInfo);
    const savedProject = await project.save();
    const projectId = savedProject._id;

    // Rename the ZIP file with modified name format
    const modifiedFileName = `${projectId}${fileExtension}`;
    const modifiedFilePath = path.join(req.file.destination, modifiedFileName);
    fs.rename(zipFilePath, modifiedFilePath, (error) => {
      if (error) {
        console.error('Error renaming dataset file:', error);
      } else {
        console.log('New dataset saved as:', modifiedFileName);
      }
    });

    // Trigger the train-suppervisor job
    const jobManifest = k8sObjects.getTrainSupervisorObject(projectId, model);
    console.log(jobManifest)
    k8sApi.createNamespacedJob('default', jobManifest)
      .then((response) => {
        console.log('Job created with response:', response.body);
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

// PATCH - update project's aggregated accuracy
app.patch("/projects/:projectId/aggregated_accuracy", [
  check("aggregated_accuracy")
  .notEmpty().withMessage("Aggregated accuracy is required")
  .isFloat({ min: 0, max: 100 }).withMessage("Aggregated accuracy must be between 0 and 100")],
async (req, res) => {
  const { projectId } = req.params;
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({ error: 'PUT_PROJECT_FAILED', message: errors.array() });
  }
  const {aggregated_accuracy} = req.body;
  try{
    const project = await Project.findOne({_id: projectId});
    project.aggregated_accuracy = aggregated_accuracy
    await project.save()
    res.json({ project: project});
  } catch (err) {
    res.status(500).json({ 
      error: 'PUT_PROJECT_FAILED',
      message: 'Error when updating project'
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
    for (let i = 0; i < splits; i++) {
      project.splits.push({
        id: i + 1,
        train_accuracies: [],
        val_accuracies: [],
        train_losses: [],
        val_losses: []
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

// PATCH - Add accuracy to split
app.patch("/projects/:projectId/splits/:splitId/metrics",
async (req, res) => {
    const { projectId, splitId } = req.params;
    const { train_accuracy, val_accuracy, train_loss, val_loss } = req.body;

    try {
      const project = await Project.findOne({ _id: projectId });
      project.splits[splitId].train_accuracies.push(train_accuracy);
      project.splits[splitId].val_accuracies.push(val_accuracy);
      project.splits[splitId].train_losses.push(train_loss);
      project.splits[splitId].val_losses.push(val_loss);

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