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
const k8sObjects = require('./kubernetes-objects');

const kc = new k8s.KubeConfig();
kc.loadFromDefault();

const k8sApi = kc.makeApiClient(k8s.CoreV1Api);

// Models
const User = require("./models/user");
const Project = require("./models/project");

const PORT = process.env.PORT || 3001;

var database_uri = process.env.MONGODB_CONNECTION || "localhost";
const MONGODB_URI = database_uri.startsWith("mongodb+srv://")
  ? database_uri
  : `mongodb://${database_uri}:27017/fileexchangehub`;

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

app.use(bodyParser.json({ limit: '200mb' }));
app.use(bodyParser.urlencoded({ limit: '200mb', extended: true, parameterLimit: 100000 }));
app.use(express.json());

const upload = multer({
  dest: "datasets/",
  limits: { fileSize: 200 * 1024 * 1024 }, //200MB
  fileFilter: function (req, file, cb) {
   if (file.originalname.endsWith('.zip')) {
      // Allow the upload
      cb(null, true);
    } 
    else {
      // Reject the upload
      cb(new multer.MulterError('LIMIT_UNEXPECTED_FILE', 'Only one file allowed!'));
    }
  }
});


// GET endpoint that returns "Hello World!"
app.get("/", (req, res) => {
  res.send("Hello World!");
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
app.post("/upload", (req, res, next) => {
  upload.single("file")(req, res, function (err) {
    if (err instanceof multer.MulterError) {
      if (err.code === "LIMIT_UNEXPECTED_FILE") {
        // Handle the unexpected file error
        return res.status(400).json({ error: 'UPLOAD_FAILED', message: "Only one file allowed" });
      }
      // Handle other multer errors if needed
    } else if (err) {
      // Handle other errors if needed
    }

    // Proceed to the next middleware or route handler
    next();
  });
}, async function (req, res, next) {
  
  const { model, projectName } = req.body;

  // Check if a model was provided in the request
  if (!model) {
    res.status(400).json({ 
      error: 'UPLOAD_FAILED',
      message: 'No model provided'
    });
    return;
  }
  // Check if a file was sent in the request
  if (!req.file) {
    res.status(400).json({ 
      error: 'UPLOAD_FAILED',
      message: 'No file uploaded'
    });
    return;
  }
  const fileExtension = path.extname(req.file.originalname);
  if (fileExtension != '.zip') {
    res.status(400).json({ 
      error: 'UPLOAD_FAILED',
      message: 'Only ZIP file is allowed'
    });
    return; // Add return statement here
  }
  // Check if a file was sent in the request
  if (!projectName) {
    res.status(400).json({ 
      error: 'UPLOAD_FAILED',
      message: 'No project name was given'
    });
    return;
  }
  // Save the uploaded ZIP file internally
  const zipFilePath = req.file.path;

  try {
    // Define the properties for the new Project instance
    const projectInfo = {
      dataset: req.file.originalname,
      size: req.file.size / (1024 * 1024 * 1024), // Convert bytes to GB
      model,
      state: 'pending',
      name: projectName
    };

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
      message: 'ZIP file received'
    });

    // TODO - Trigger the train-suppervisor job here
    k8sApi.createNamespacedJob('default', k8sObjects.train_suppervisorObject).then((response) => {
      console.log('Job created with response:', response.body);
    }).catch((err) => {
        console.error('Error creating job:', err);
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
