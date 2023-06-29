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
  logs: {
    type: [String],
    required: true,
  },
  splits: [
    {
      id: {
        type: Number,
        required: true,
      },
      cpu_usage: {
        type: Number,
        required: true,
      },
      ram_usage: {
        type: Number,
        required: true,
      },
      train_accuracies: {
        type: [Number],
        required: true,
      },
      val_accuracies: {
        type: [Number],
        required: true,
      },
      train_losses: {
        type: [Number],
        required: true,
      },
      val_losses: {
        type: [Number],
        required: true,
      },
      logs: {
        type: [String],
        required: true,
      }
    },
  ],
  file_paths: {
    type: [String],
    required: true,
  },
}, {
  timestamps: true,
});

module.exports = mongoose.model('Project', myModelSchema);