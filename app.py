from flask import Flask, render_template, request

from ml_editor.ml_editor import get_recommendations_from_input

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


def handle_text_request(request, template_name):
    if request.method == "POST":
        question = request.form.get("question")
        suggestions = get_recommendations_from_input(question)
        payload = {"input": question, "suggestions": suggestions}
        return render_template("v1_results.html", ml_result=payload)
    else:
        return render_template(template_name)
