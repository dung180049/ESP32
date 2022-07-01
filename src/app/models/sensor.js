const mongoose = require('mongoose');
const Schema = mongoose.Schema;
var timeSave = new Date()

var timeInMilliseconds = timeSave.getTime()
var timeInUTC7 = timeInMilliseconds + (7 * 60 * 60 * 1000)
var timeSaveInUTC7 = new Date(timeInUTC7)

const Sensor = new Schema({
    Month: { type: Number, require: true, default: timeSaveInUTC7.getMonth() + 1 },
    Date: { type: Number, require: true, default: timeSaveInUTC7.getDate() },
    Hour: { type: Number, require: true, default: timeSaveInUTC7.getHours() },
    temperature: { type: Number, require: true },
    humidity: { type: Number, require: true },
}, );

module.exports = mongoose.model('Sensor', Sensor);