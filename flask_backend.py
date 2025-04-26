from flask import Flask, request, jsonify
import json

app = Flask(__name__)

# Sample user database (in-memory)
users_db = {"john": {"password": "password123"}}

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    if username in users_db and users_db[username]["password"] == password:
        return jsonify({"message": "Login successful", "status": "success"})
    else:
        return jsonify({"message": "Invalid credentials", "status": "error"}), 401

@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    if username in users_db:
        return jsonify({"message": "User already exists", "status": "error"}), 400

    # In a real app, you'd hash the password and store it in a database
    users_db[username] = {"password": password}
    return jsonify({"message": "Signup successful", "status": "success"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
