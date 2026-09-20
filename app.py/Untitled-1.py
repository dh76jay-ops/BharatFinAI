"""
Flask app for Stella's romantic page
Run with: python app.py
"""

from flask import Flask, render_template, send_from_directory
import os

app = Flask(__name__)

@app.route('/')
def index():
    """Render the main romantic page for Stella"""
    return render_template('index.html')

@app.route('/music')
def music():
    """Serve background music"""
    return send_from_directory('static', 'music.mp3')

if __name__ == '__main__':
    print("💖 Starting server for Stella...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000)