const express = require('express')
const router = express.Router()

const Controller = require('../app/controllers/Controller')

router.get('/show', Controller.show)
router.get('/', Controller.searchParams)

module.exports = router