import logging
import requests
from threading import Thread
from flask import Flask, jsonify, render_template, request
from scripts import database as db
import pickle

app1 = Flask("app1")
logged_in = False


@app1.route("/")
def home():
    global logged_in
    if logged_in:
        return render_template("img_form.html")
    else:
        return render_template("login.html")


@app1.route("/login", methods=["POST", "GET"])
def login_user():
    if request.method == "GET":
        return f"The URL /login is accessed directly. Try going to '/' to submit form"
    if request.method == "POST":
        r = db.find_user(request.form.to_dict())
        if r:
            return render_template("img_form.html")
        else:
            return render_template("login.html", bad_login="Not right buddy try again")


@app1.route("/signup")
def signup_page():
    return render_template("signup.html")


@app1.route("/reg", methods=["POST", "GET"])
def reg_user():
    if request.method == "GET":
        return (
            f"The URL /reg is accessed directly. Try going to '/signup' to submit form"
        )
    if request.method == "POST":
        if db.add_data_to_db(request.form.to_dict()):
            return render_template("login.html")
    return render_template("login.html", bad_login="Email already registered")


@app1.route("/submit", methods=["POST", "GET"])
def post_image():
    if request.method == "GET":
        return render_template("img_form.html")
    if request.method == "POST":
        file = request.files["file"]
        img_bytes = file.read()
        r = requests.post("http://127.0.0.1:5000/predict", files={"file": img_bytes})

        r.raise_for_status()

        return render_template("response.html", resp=r)


if __name__ == "__main__":
    app1.run(port=5001)
