const express = require('express')
const router = express.Router()

const Controller = require('../app/controllers/Controller')

router.get('/', Controller.history)

module.exports = router