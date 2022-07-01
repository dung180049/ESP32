const json2csv = require('json2csv').parse
const fs = require('fs')
const fields = ['Month', 'Date', 'Hour', 'temperature', 'humidity'];
const csvReader = require('xlsx')
const Sensor = require('../models/sensor')
const { multipleMongooseToObject, mongooseToObject } = require('../../util/mongoose')

class Controller {
    show(req, res, next) {
        var regex = new RegExp(req.query.day)
        var dayShow = new Date(regex)
        var dateShow = dayShow.getDate()
        var monthShow = dayShow.getMonth() + 1

        Sensor.find({ Month: monthShow })
            .find({ Date: dateShow })
            .then(sensors => {
                res.render('show', {
                    sensors: multipleMongooseToObject(sensors),
                    dateShow,
                    monthShow
                })
            })
            .catch(next);
    }

    history(req, res, next) {
        var regex = new RegExp(req.query.day)
        var dayShow = new Date(regex)
        var dateShow = dayShow.getDate()
        var monthShow = dayShow.getMonth() + 1
        res.render('history', {
            dateShow,
            monthShow
        })
    }

    searchParams(req, res) {
        res.render('param')
    }

    control(req, res) {
        res.render('control')
    }

    


    update(req, res) {
        Sensor.find({}, function(err, sensors) {
            if (err) {
                return res.render('predict')
            } else {
                let csv
                try {
                    csv = json2csv(sensors, { fields })
                } catch (err) {
                    return res.status(500).json({ err })
                }
                const filePath = ('forecast/data.csv')
                fs.writeFile(filePath, csv, function(err) {
                    if (err) {
                        return res.render('control')
                    } else {
                        /* const { spawn } = require('child_process');
                        const pyProg = spawn('python', ['forecast/forecast.py']);

                        pyProg.stdout.on('data', function(data) {
                            console.log(data.toString()); */
                        return res.render('home')
                            // })
                    }
                })
            }
        })
    }

    home(req, res) {
        res.render('home')
    }

    predict(req, res) {
        var time = new Date()
        var timeInMilliseconds = time.getTime()
        var timeTomorrow = timeInMilliseconds + ((7 + 24) * 60 * 60 * 1000)
        var timeShow = new Date(timeTomorrow)
        var dateShow = timeShow.getDate()
        var monthShow = timeShow.getMonth() + 1

        
        const filePathCSV = `forecast/csv/${dateShow}${monthShow}.csv`
        const tomorrowData = csvReader.readFile(filePathCSV)
        const sheets = tomorrowData.SheetNames
        const prediction = csvReader.utils.sheet_to_json(tomorrowData.Sheets[sheets])
        res.render('predict', {
            prediction,
            dateShow,
            monthShow,
        })
    }

    store(req, res, next) {
        const sensor = new Sensor(req.query)
        sensor.save()
        res.render('home')
        res.render('control')
    }
}

module.exports = new Controller