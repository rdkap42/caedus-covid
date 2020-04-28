from flask import Flask
from flask import render_template
import os

# App flask config
server = Flask(__name__)
server.debug = True

@server.route("/")
def index():
    return render_template('index.html')

@server.route("/about")
def about():
    return render_template('about.html')

@server.route("/data")
def data():
    return render_template('data_models.html')

@server.route("/products")
def products():
    return render_template('products.html')

@server.route("/contact")
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    server.run( host='0.0.0.0', port=os.environ.get('PORT',8050))