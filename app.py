from flask import Flask, render_template_string, request

from src.pipeline.prediction_pipeline import PredictionPipeline

app = Flask(__name__)
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Movie Sentiment Analyzer</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f1f5f9;
            margin: 0;
            padding: 40px 15px;
        }
        .container {
            max-width: 650px;
            margin: auto;
            background: white;
            padding: 30px;
            border-radius: 14px;
            box-shadow: 0 4px 18px #00000012;
        }
        h1 { color: #1e293b; }
        textarea {
            width: 100%;
            height: 150px;
            box-sizing: border-box;
            padding: 12px;
            font-size: 16px;
            border: 1px solid #cbd5e1;
            border-radius: 8px;
        }
        button {
            margin-top: 15px;
            padding: 12px 22px;
            background: #2563eb;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
        }
        .result {
            margin-top: 20px;
            padding: 15px;
            background: #eff6ff;
            border-radius: 8px;
            font-weight: bold;
        }
        .error { color: #b91c1c; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Movie Sentiment Analyzer</h1>
        <p>Enter a movie review to predict its sentiment.</p>

        <form method="POST">
            <textarea name="review" required
                placeholder="Write your movie review here...">{{ review }}</textarea>
            <button type="submit">Predict Sentiment</button>
        </form>

        {% if result %}
        <div class="result">Predicted Sentiment: {{ result }}</div>
        {% endif %}

        {% if error %}
        <p class="error">{{ error }}</p>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/" , methods = ["GET" , "POST"])

def home():
    review = ""
    result = None
    error = None

    if request.method == "POST":
        review = request.form.get("review" , "").strip()
        if not review:
            error = "Please enter a movie review."
        else:
            try:
                pipeline = PredictionPipeline()
                result  = pipeline.predict(review)
            except Exception:
                app.logger.exception("prediction request failed")
                error = "request failed/check the application.logs"
    return render_template_string(
        HTML , review = review , error = error , result = result
    )

if __name__ == "__main__":
    app.run(debug=False)