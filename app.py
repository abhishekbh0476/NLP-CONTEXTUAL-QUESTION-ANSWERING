from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from qa_system import QuestionAnsweringSystem
from werkzeug.utils import secure_filename
import os
import PyPDF2
import docx

UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
ALLOWED_EXTENSIONS = {"pdf", "txt", "docx"}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

qa_system = QuestionAnsweringSystem(use_translation=True)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(path):
    text = []
    try:
        reader = PyPDF2.PdfReader(path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    except Exception:
        pass
    return "\n".join(text)

def extract_text_from_docx(path):
    try:
        doc = docx.Document(path)
        paragraphs = [p.text for p in doc.paragraphs if p.text]
        return "\n".join(paragraphs)
    except Exception:
        return ""

def extract_text(path):
    ext = path.rsplit(".", 1)[1].lower()
    if ext == "pdf":
        return extract_text_from_pdf(path)
    if ext == "docx":
        return extract_text_from_docx(path)
    if ext == "txt":
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception:
            return ""
    return ""

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask_question():
    try:
        data = request.get_json(force=True)
        context = data.get("context", "").strip()
        question = data.get("question", "").strip()
        lang = data.get("lang", "en").strip().lower()
        if not context or not question:
            return jsonify({"error": "Both context and question are required"}), 400
        target = lang if lang in {"en", "hi", "kn"} else "en"
        answer = qa_system.get_answer(context, question, target_lang=target)
        return jsonify({"answer": answer, "status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/ask-file", methods=["POST", "OPTIONS"])
def ask_file():
    try:
        question = request.form.get("question", "").strip()
        context_text = request.form.get("context", "").strip()
        lang = request.form.get("lang", "en").strip().lower()
        target = lang if lang in {"en", "hi", "kn"} else "en"
        file = request.files.get("file")
        extracted_text = ""
        if file and file.filename != "":
            filename = secure_filename(file.filename)
            if not allowed_file(filename):
                return jsonify({"error": f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            try:
                file.save(save_path)
                extracted_text = extract_text(save_path)
            finally:
                try:
                    os.remove(save_path)
                except Exception:
                    pass
            if extracted_text:
                context_text = (context_text + "\n\n" + extracted_text).strip() if context_text else extracted_text
        if not context_text:
            return jsonify({"error": "No context provided and no text extracted from file"}), 400
        if question == "preview_only":
            return jsonify({"status": "success", "extracted_text": extracted_text or "", "answer": ""})
        answer = qa_system.get_answer(context_text, question, target_lang=target)
        return jsonify({"answer": answer, "status": "success", "extracted_text": extracted_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
