from flask import Flask ,render_template,url_for

app=Flask(__name__)


@app.route("/")
def home():
    return "welcome to flask file "


if __name__== "__main__" :
    app.run(debug=True)