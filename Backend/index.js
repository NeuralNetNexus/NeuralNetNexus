const express = require("express");
const fs = require("fs");
const path = require("path");
const multer = require("multer");
const unzipper = require("unzipper");
const archiver = require("archiver");
const cors = require("cors");
const mongoose = require("mongoose");
require("dotenv").config();
const { check, validationResult } = require("express-validator");
const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");
const app = express();
const bodyParser = require("body-parser");
// Models
const User = require("./models/user");
const File = require("./models/file");

const PORT = process.env.PORT || 3001;

mongoose
  .connect(process.env.MONGODB_CONNECTION, {
    useNewUrlParser: true,
    useUnifiedTopology: true,
  })
  .then(() => console.log("Connected to MongoDB"))
  .catch((err) => console.log("Failed to connect to MongoDB", err));

app.use(cors());
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
const upload = multer({ dest: "datasets/" }, {
  fileFilter: function (req, file, cb) {
    var error_msg = error instanceof multer.MulterError
    if (file.originalname.endsWith('.zip')) {
      // Allow the upload
      cb(null, true);
    } 
    if(error_msg) {
      // Reject the upload
      cb(new multer.MulterError('LIMIT_UNEXPECTED_FILE', 'Only one file allowed!'));
    }
  },
  
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
      message: 'Only 1 ZIP file is allowed'
    });
    return; // Add return statement here
  }

  // Save the uploaded ZIP file internally
  const zipFilePath = req.file.path;

  try {
    // Save the information from req.file in MongoDB
    const fileInfo = {
      originalName: req.file.originalname,
      sizeGB: req.file.size / (1024 * 1024 * 1024), // Convert bytes to GB
    };

    // Create a new File instance
    const file = new File(fileInfo);

    const savedFile = await file.save();
    const fileId = savedFile._id;

    // Rename the ZIP file with modified name format
    const fileExtension = path.extname(req.file.originalname);
    const modifiedFileName = `pvc-dataset-${fileId}${fileExtension}`;
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
  console.log("Server is running on port "+PORT);
});
