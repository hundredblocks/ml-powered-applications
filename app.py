from flask import Flask, render_template, request

from ml_editor.ml_editor import get_recommendations_from_input
import ml_editor.model_v2 as v2_model
import ml_editor.model_v3 as v3_model

app = Flask(__name__)


@app.route("/")
def landing_page():
    return render_template("landing.html")


@app.route("/v1", methods=["POST", "GET"])
def v1():
    return handle_text_request(request, "v1.html")


@app.route("/v2", methods=["POST", "GET"])
def v2():
    return handle_text_request(request, "v2.html")


@app.route("/v3", methods=["POST", "GET"])
def v3():
    return handle_text_request(request, "v3.html")


def get_model_from_template(template_name):
    return template_name.split(".")[0]


def retrieve_recommendations_for_model(question, model):
    if model == "v1":
        return get_recommendations_from_input(question)
    if model == "v2":
        return v2_model.get_pos_score_from_text(question)
    if model == "v3":
        return v3_model.get_recommendation_and_prediction_from_text(question)
    raise ValueError("Incorrect Model passed")


def handle_text_request(request, template_name):
    if request.method == "POST":
        question = request.form.get("question")
        model_name = get_model_from_template(template_name)
        suggestions = retrieve_recommendations_for_model(question, model_name)
        payload = {
            "input": question,
            "suggestions": suggestions,
            "model_name": model_name,
        }
        return render_template("results.html", ml_result=payload)
    else:
        return render_template(template_name)
