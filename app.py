from functools import lru_cache

from flask import Flask, render_template, request

from ml_editor.ml_editor import get_recommendations_from_input
import ml_editor.model_v2 as v2_model
import ml_editor.model_v3 as v3_model

app = Flask(__name__)


@app.route("/")
def landing_page():
    """
    Renders landing page
    """
    return render_template("landing.html")


@app.route("/v1", methods=["POST", "GET"])
def v1():
    """
    Renders v1 model input form and results
    """
    return handle_text_request(request, "v1.html")


@app.route("/v2", methods=["POST", "GET"])
def v2():
    """
        Renders v2 model input form and results
        """
    return handle_text_request(request, "v2.html")


@app.route("/v3", methods=["POST", "GET"])
def v3():
    """
        Renders v3 model input form and results
    """
    return handle_text_request(request, "v3.html")


def get_model_from_template(template_name):
    """
    Get the name of the relevant model from the name of the template
    :param template_name: name of html template
    :return: name of the model
    """
    return template_name.split(".")[0]


@lru_cache(maxsize=128)
def retrieve_recommendations_for_model(question, model):
    """
    This function computes or retrieves recommendations
    We use an LRU cache to store results we process. If we see the same question
    twice, we can retrieve cached results to serve them faster
    :param question: the input text to the model
    :param model: which model to use
    :return: a model's recommendations
    """
    if model == "v1":
        return get_recommendations_from_input(question)
    if model == "v2":
        return v2_model.get_pos_score_from_text(question)
    if model == "v3":
        return v3_model.get_recommendation_and_prediction_from_text(question)
    raise ValueError("Incorrect Model passed")


def handle_text_request(request, template_name):
    """
    Renders an input form for GET requests and displays results for the given
    posted question for a POST request
    :param request: http request
    :param template_name: name of the requested template (e.g. v2.html)
    :return: Render an input form or results depending on request type
    """
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
