"""
MeetMind — Automated Meeting Summarizer
Uses IBM Watson REST APIs directly (no SDK) — works on all platforms.
"""

import os
import json
import base64
import logging
import tempfile
import smtplib
import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder=BASE_DIR)
CORS(app)


def ibm_auth(api_key):
    token = base64.b64encode(f"apikey:{api_key}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def get_content_type(filename):
    filename = filename.lower()
    if filename.endswith(".wav"):  return "audio/wav"
    if filename.endswith(".flac"): return "audio/flac"
    if filename.endswith(".mp3"):  return "audio/mp3"
    if filename.endswith(".ogg"):  return "audio/ogg"
    return None


def transcribe_audio(file_path, content_type, api_key, service_url):
    url = service_url.rstrip("/") + "/v1/recognize"
    params = {"model": "en-US_BroadbandModel", "smart_formatting": "true"}
    headers = {**ibm_auth(api_key), "Content-Type": content_type}

    with open(file_path, "rb") as f:
        resp = requests.post(url, headers=headers, params=params, data=f, timeout=120)

    if resp.status_code != 200:
        raise ValueError(f"STT Error {resp.status_code}: {resp.text}")

    results = resp.json().get("results", [])
    parts = [r["alternatives"][0]["transcript"].strip() for r in results if r.get("alternatives")]

    if not parts:
        raise ValueError("No speech detected in the uploaded audio.")

    return " ".join(parts)


def analyze_transcript(transcript, api_key, service_url):
    if len(transcript.strip()) < 50:
        raise ValueError("Transcript is too short for meaningful analysis.")

    url = service_url.rstrip("/") + "/v1/analyze"
    headers = {**ibm_auth(api_key), "Content-Type": "application/json"}
    payload = {
        "text": transcript,
        "features": {
            "keywords":   {"limit": 8, "sentiment": True},
            "entities":   {"limit": 8, "sentiment": True},
            "sentiment":  {"document": True},
            "concepts":   {"limit": 5},
            "categories": {"limit": 3},
        }
    }

    resp = requests.post(url, headers=headers, params={"version": "2022-04-07"}, json=payload, timeout=60)

    if resp.status_code != 200:
        raise ValueError(f"NLU Error {resp.status_code}: {resp.text}")

    return resp.json()


def find_action_items(transcript):
    action_words = ["will", "should", "need to", "have to", "must", "going to",
                    "plan to", "deadline", "follow up", "todo", "task", "assign", "responsible"]
    sentences = transcript.replace("?", ".").replace("!", ".").split(".")
    items = []
    for s in sentences:
        s = s.strip()
        if s and any(w in s.lower() for w in action_words):
            items.append(s)
    return items[:10]


def format_summary(transcript, insights, action_items):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    sep = "=" * 60
    lines = [sep, "MEETMIND - MEETING SUMMARY", sep, f"Generated: {now}", ""]

    sentiment = insights.get("sentiment", {}).get("document", {})
    label = sentiment.get("label", "neutral")
    score = round(sentiment.get("score", 0), 2)
    lines += [f"Overall Tone: {label.upper()} (score: {score})", ""]

    keywords = insights.get("keywords", [])
    if keywords:
        lines.append("Key Topics:")
        for kw in keywords:
            rel = round(kw.get("relevance", 0) * 100)
            lines.append(f"  - {kw.get('text','')} ({rel}% relevant)")
        lines.append("")

    entities = insights.get("entities", [])
    if entities:
        lines.append("Important Entities:")
        for ent in entities:
            lines.append(f"  - [{ent.get('type','?')}] {ent.get('text','')}")
        lines.append("")

    concepts = insights.get("concepts", [])
    if concepts:
        lines.append("Concepts:")
        for c in concepts:
            lines.append(f"  - {c.get('text','')}")
        lines.append("")

    categories = insights.get("categories", [])
    if categories:
        lines.append("Categories:")
        for cat in categories:
            lines.append(f"  - {cat.get('label','')}")
        lines.append("")

    if action_items:
        lines.append("Action Items:")
        for i, item in enumerate(action_items, 1):
            lines.append(f"  {i}. {item}")
        lines.append("")

    lines += ["-" * 60, "Transcript:", "-" * 60, transcript, "", sep]
    return "\n".join(lines)


def send_email_smtp(recipients, summary_text):
    sender   = os.getenv("SENDER_EMAIL")
    password = os.getenv("SENDER_PASSWORD")
    if not sender or not password:
        raise RuntimeError("SENDER_EMAIL / SENDER_PASSWORD not set.")

    msg = MIMEMultipart()
    msg["Subject"] = f"MeetMind Summary — {datetime.datetime.now().strftime('%b %d, %Y')}"
    msg["From"]    = sender
    msg["To"]      = ", ".join(recipients)
    msg.attach(MIMEText(summary_text, "plain"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, password)
        server.sendmail(sender, recipients, msg.as_string())


@app.route("/", methods=["GET"])
def home():
    index_path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(index_path):
        return send_from_directory(BASE_DIR, "index.html")
    return jsonify({"message": "MeetMind backend is running"})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/summarize", methods=["POST"])
def summarize():
    temp_file_path = None
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No audio file uploaded"}), 400

        audio_file = request.files["audio"]
        if audio_file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        stt_key = request.form.get("stt_api_key") or os.getenv("STT_API_KEY")
        stt_url = request.form.get("stt_url")     or os.getenv("STT_URL")
        nlu_key = request.form.get("nlu_api_key") or os.getenv("NLU_API_KEY")
        nlu_url = request.form.get("nlu_url")     or os.getenv("NLU_URL")

        if not all([stt_key, stt_url, nlu_key, nlu_url]):
            return jsonify({"error": "Missing IBM Watson credentials in environment."}), 400

        content_type = get_content_type(audio_file.filename)
        if not content_type:
            return jsonify({"error": "Unsupported file. Use WAV, FLAC, MP3, or OGG."}), 400

        suffix = "." + audio_file.filename.rsplit(".", 1)[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            audio_file.save(tmp.name)
            temp_file_path = tmp.name

        transcript   = transcribe_audio(temp_file_path, content_type, stt_key, stt_url)
        insights     = analyze_transcript(transcript, nlu_key, nlu_url)
        action_items = find_action_items(transcript)
        summary_text = format_summary(transcript, insights, action_items)

        email_status = "not requested"
        emails_raw   = request.form.get("emails", "").strip()
        if emails_raw:
            recipients = [e.strip() for e in emails_raw.split(",") if e.strip()]
            try:
                send_email_smtp(recipients, summary_text)
                email_status = f"sent to {len(recipients)} recipient(s)"
            except Exception as e:
                email_status = f"failed: {e}"

        return jsonify({
            "success":      True,
            "transcript":   transcript,
            "summary_text": summary_text,
            "email_status": email_status,
            "insights": {
                "keywords":   insights.get("keywords",   []),
                "entities":   insights.get("entities",   []),
                "sentiment":  insights.get("sentiment",  {}),
                "concepts":   insights.get("concepts",   []),
                "categories": insights.get("categories", []),
            },
            "action_items": action_items,
        })

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 422
    except Exception as e:
        logger.exception("Unexpected error")
        return jsonify({"error": f"Something went wrong: {e}"}), 500
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.route("/send-email", methods=["POST"])
def api_send_email():
    data       = request.get_json(force=True)
    summary    = data.get("summary", "")
    recipients = data.get("recipients", [])
    if not summary:    return jsonify({"error": "No summary provided."}), 400
    if not recipients: return jsonify({"error": "No recipients provided."}), 400
    try:
        send_email_smtp(recipients, summary)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
    print(f"Starting MeetMind on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
