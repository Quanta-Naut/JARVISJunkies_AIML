from flask import Flask, jsonify
from flask_cors import CORS

from src.convoAI.convoAI import ConvoAI
from src.emotionDetect.emotionDetect import EmotionDetect

# Create a Flask app instance
app = Flask(__name__)

# Enable CORS for all routes
CORS(app)
convoAgent = ConvoAI()
emotionAgent = EmotionDetect()

# Sample route
@app.route('/api/hi', methods=['GET'])
def get_data():
    
    data = {
        convoAgent.get_response("Hello")
    }
    return jsonify(str(data))

@app.route('/api/data2', methods=['GET'])
def get_data2():
    data = {
        emotionAgent.get_emotion("C:/Users/tarun/Desktop/Folders/ConversationalAI/happy.png")
    }
    return jsonify(str(data))

if __name__ == '__main__':
    app.run(debug=True)
