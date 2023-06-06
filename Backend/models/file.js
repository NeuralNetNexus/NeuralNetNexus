const mongoose = require('mongoose');

const fileSchema = new mongoose.Schema({
    originalName: { type: String, required: true, unique: true },
    sizeGB: { type: Number, required: true },
    }, { timestamps: { createdAt: true, updatedAt: false } });

module.exports = mongoose.model('File', fileSchema);