const express = require('express')
const router = express.Router()

const Controller = require('../app/controllers/Controller')

router.get('/:id', Controller.switch)
router.get('/', Controller.control)

module.exports = router