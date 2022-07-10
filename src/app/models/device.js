const mongoose = require('mongoose');
const Schema = mongoose.Schema;

const Device = new Schema({
    Name: { type: String, require: true, unique: true },
    State: { type: Boolean },
    Show: { type: String },
    type: { type: String },
})

module.exports = mongoose.model('Device', Device)