const mongoose = require('mongoose');
const { Schema } = mongoose;

const myModelSchema = new Schema({
  name: {
    type: String,
    required: true,
  },
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
  n_splits: {
    type: Number,
    required: false
  },
  aggregated_accuracy: {
    type: Number,
    required: false
  },
  accuracies: {
    type: [Number],
    required: false
  }
}, {
  timestamps: true,
});

module.exports = mongoose.model('Project', myModelSchema);