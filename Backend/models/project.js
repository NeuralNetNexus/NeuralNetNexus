const mongoose = require('mongoose');
const { Schema } = mongoose;

const myModelSchema = new Schema({
  dataset: {
    type: String,
    required: true,
  },
  size: {
    type: Number,
    required: true,
  },
  model: {
    type: String,
    required: true,
  },
  state: {
    type: String,
    required: true,
  },
}, {
  timestamps: true,
});

module.exports = mongoose.model('Project', myModelSchema);