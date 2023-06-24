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
// Models
const User = require("./models.user");
const Project = require("./models.project");
// Kubernetes
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

app.use(bodyParser.json({ limit: '100mb' }));
app.use(bodyParser.urlencoded({ limit: '100mb', extended: true }));

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
app.get("/projects/:id", 
async (req, res) => {
  const id = req.params.id;
  try{
    const project = await Project.findOne({_id: id});
    res.json({ project: project});
  } catch (err) {
    res.status(500).json({ 
      error: 'GET_PROJECT_FAILED',
      message: 'Error when retrieving project'
    });
  }
});
// PUT - update project's number of splits
app.put("/projects/:id/splits", [
  check("splits")
  .notEmpty().withMessage("Number of splits is required")
  .isFloat({ gt: 0 }).withMessage("Number of splits must be greater than 0")],
async (req, res) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({ error: 'PUT_PROJECT_FAILED', message: errors.array() });
  }
  const id = req.params.id;
  const {splits} = req.body;
  try{
    const project = await Project.findOne({_id: id});
    project.n_splits = splits
    await project.save()
    res.json({ project: project});
  } catch (err) {
    res.status(500).json({ 
      error: 'PUT_PROJECT_FAILED',
      message: 'Error when updating project'
    });
  }
});

// PUT - update project's accuracies 
app.put("/projects/:id/accuracies", [
  check("accuracies")
      .notEmpty().withMessage("Accuracies are required")
      .isArray().withMessage("Accuracies must be a float array")
      .custom((value) => {
        for (const num of value) {
          if (typeof num !== "number" || num < 0 || num > 100) {
            throw new Error("Accuracy must be between 0 and 100");
          }
        }
        return true;
      })
], async (req, res) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({ error: 'PUT_PROJECT_FAILED', message: errors.array() });
  }
  const id = req.params.id;
  const { accuracies } = req.body;
  try {
    const project = await Project.findOne({ _id: id });
    project.accuracies = project.accuracies.concat(accuracies);
    await project.save();
    res.json({ project: project });
  } catch (err) {
    res.status(500).json({ 
      error: 'PUT_PROJECT_FAILED',
      message: 'Error when updating project'
    });
  }
});

// PUT - update project's aggregated accuracy
app.put("/projects/:id/aggregated_accuracy", [
  check("aggregated_accuracy")
  .notEmpty().withMessage("Aggregated accuracy is required")
  .isFloat({ min: 0, max: 100 }).withMessage("Aggregated accuracy must be between 0 and 100")],
  async (req, res) => {
  const id = req.params.id;
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({ error: 'PUT_PROJECT_FAILED', message: errors.array() });
  }
  const {aggregated_accuracy} = req.body;
  try{
    const project = await Project.findOne({_id: id});
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
// POST - Register a user
app.post(
  "/register",
  [
    check("email").isEmail().withMessage("Email is not valid"),
    check("password")
      .isLength({ min: 6 })
      .withMessage("Password must be at least 6 characters long"),
    check("role")
      .isIn(["client", "admin"])
      .withMessage('Role must be either "client" or "admin"'),
    check("name")
      .optional()
      .isLength({ min: 3 })
      .withMessage("Name must be at least 3 characters long"),
  ],
  async (req, res, next) => {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ error: 'REGISTER_FAILED', message: errors.array() });
    }

    const { email, password, role, name } = req.body;
    const hashedPassword = await bcrypt.hash(password, 10);

    const user = new User({ email, password: hashedPassword, role, name });

    try {
      await user.save();
      res.status(201).json({ message : "User registered successfully" });

    } catch (error) {
      if (error.code === 11000) {
        // MongoDB duplicate key error code
        return res.status(409).json({message: "Username is already taken"});
      }
      return res.status(500).json({ error: 'REGISTER_FAILED', message: "Error when registering your account" });
    }
  }
);

// user login route
app.post(
  "/login",
  [
    check("email").notEmpty().withMessage("Email is required"),
    check("password").notEmpty().withMessage("Password is required"),
  ],
  async (req, res, next) => {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ 
        error: 'LOGIN_FAILED',
        message:  errors.array()
      });
    }
    const { email, password } = req.body;
    try {
      const user = await User.findOne({ email: email });
      if (!user) {
        return res.status(401).json({ 
          error: 'LOGIN_FAILED',
          message: 'Invalid credentials'
        });
      }

      const isMatch = await bcrypt.compare(password, user.password);
      if (!isMatch) {
        return res.status(401).json({ 
          error: 'LOGIN_FAILED',
          message: 'Invalid credentials'
        });
      }
      const token = jwt.sign(
        { userId: user._id, role: user.role },
        process.env.JWT_SECRET,
        { expiresIn: "90d" }
      );
      res.json({ token: token, userId: user._id });
    } catch (err) {
      res.status(500).json({ 
        error: 'LOGIN_FAILED',
        message: 'Error with login'
      });
    }
  }
);

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
      state: 'pending',
      name: projectName
    };
    console.log(projectInfo)

    const project = new Project(projectInfo);
    const savedProject = await project.save();
    const projectId = savedProject._id;

    // Rename the ZIP file with modified name format
    const modifiedFileName = `pvc-dataset-${projectId}${fileExtension}`;
    const modifiedFilePath = path.join(req.file.destination, modifiedFileName);
    fs.rename(zipFilePath, modifiedFilePath, (error) => {
      if (error) {
        console.error('Error renaming dataset file:', error);
      } else {
        console.log('New dataset saved as:', modifiedFileName);
      }
    });

    res.status(200).json({
      message: 'ZIP file received for project: ' + projectId 
    });

    // TODO - Trigger the train-suppervisor job here
    const jobManifest = k8sObjects.getTrainSupervisorObject(projectId);
    console.log(jobManifest)
    k8sApi.createNamespacedJob('default', jobManifest)
      .then((response) => {
        console.log('Job created with response:', response.body);
        res.status(200).json({
          message: 'ZIP file received for project: ' + projectId 
        });
      })
      .catch((err) => {
        console.error('Error creating job:', err);
        res.status(500).json({
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


// Start the server
app.listen(PORT, () => {
  console.log("Server is running on port " + PORT);
});
