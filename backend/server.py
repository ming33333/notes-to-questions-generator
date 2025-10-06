from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all routes and origins

# Load the Hugging Face model for question generation
question_generator = pipeline("text2text-generation", model="t5-base", device=-1)

@app.route("/generate", methods=["OPTIONS", "POST"])
def generate():
    if request.method == "OPTIONS":
        # Handle preflight request
        response = jsonify({"message": "CORS preflight"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        return response

    # Handle POST request
    data = request.json
    notes = data.get("notes", "")

    # Generate questions using the Hugging Face model
    prompt = f"Generate 1 sentence questions from the following notes, make sure to give some backgroun on the question: {notes}"
    results = question_generator(prompt, max_length=100, num_return_sequences=3)

    # Format the results
    questions = [{"text": notes, "question": result["generated_text"]} for result in results]
    return jsonify(questions)

if __name__ == "__main__":
    app.run(debug=True)