const mongoose = require('mongoose');

const userSchema = new mongoose.Schema({
    email: { type: String, required: true, unique: true },
    password: { type: String, required: true },
    role: { type: String, enum: ['client', 'admin'], required: true },
    name: { type: String, required: false, unique: false },    
}, { timestamps: { createdAt: true, updatedAt: false } });
 
module.exports = mongoose.model('User', userSchema);