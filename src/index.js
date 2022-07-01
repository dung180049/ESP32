const path = require('path')
const express = require('express')
const morgan = require('morgan')
const { engine } = require('express-handlebars')
const app = express()
const port = process.env.PORT || 3000

const route = require('./routes')
const db = require('./config/db')

// Connect DB
db.connect()

app.use(express.static(path.join(__dirname, 'public')))

app.use(express.urlencoded({
    extended: true,
}))
app.use(express.json())

// HTTP logger
app.use(morgan('combined'))

// Template engine

app.engine(
    'hbs',
    engine({
        extname: '.hbs',
        helpers: {
            sum: (a, b) => a + b,
        }
    }),
)

app.set('view engine', 'hbs');
app.set('views', path.join(__dirname, 'resources', 'views'))

route(app)

app.listen(port, () => {
    console.log(`App listening on port ${port}`)
})