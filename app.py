from flask import Flask, request, render_template, jsonify, send_file
from create_anime_image import create_anime_image

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/result", methods=["GET", "POST"])
def result():
    if request.method == "GET":
        return render_template("index.html")
    elif request.method == "POST":
        file = request.files["image"]
        file.save(f"./static/images/{file.filename}")
        return render_template("result.html", file_name=file.filename)


@app.route("/generate_image", methods=["POST"])
def generate_image():
    file_name = request.form["file_name"]
    result_bool = create_anime_image(file_name=file_name)
    if result_bool:
        return jsonify({"file_name": file_name})
    else:
        return jsonify({"error": "Error"})


@app.route("/download_image")
def download_image():
    return send_file("./static/images/result.png", as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
