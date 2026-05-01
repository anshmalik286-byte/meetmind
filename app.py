import os
import logging
import tempfile
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

from ibm_watson import SpeechToTextV1, NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import (
    Features,
    KeywordsOptions,
    EntitiesOptions,
    SentimentOptions,
    ConceptsOptions,
    CategoriesOptions,
)
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator


# ---------------------------------------------------
# Basic setup
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")

# force loading .env from project folder
load_dotenv(dotenv_path=ENV_PATH)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder=BASE_DIR)
CORS(app)


# ---------------------------------------------------
# Helper: clean input values
# ---------------------------------------------------
def clean_value(value):
    if value is None:
        return None

    value = value.strip()

    # reject empty values
    if value == "":
        return None

    # reject placeholder values
    placeholders = [
        "paste_your_stt_api_key_here",
        "paste_your_stt_service_url_here",
        "paste_your_nlu_api_key_here",
        "paste_your_nlu_service_url_here",
        "your_key_here",
        "your_url_here"
    ]

    if value in placeholders:
        return None

    return value


# ---------------------------------------------------
# IBM Watson client helpers
# ---------------------------------------------------
def get_stt_client(api_key, url):
    if not api_key:
        raise ValueError("STT API key is missing.")
    if not url:
        raise ValueError("STT service URL is missing.")

    authenticator = IAMAuthenticator(api_key)
    stt = SpeechToTextV1(authenticator=authenticator)
    stt.set_service_url(url)
    return stt


def get_nlu_client(api_key, url):
    if not api_key:
        raise ValueError("NLU API key is missing.")
    if not url:
        raise ValueError("NLU service URL is missing.")

    authenticator = IAMAuthenticator(api_key)
    nlu = NaturalLanguageUnderstandingV1(
        version="2022-04-07",
        authenticator=authenticator
    )
    nlu.set_service_url(url)
    return nlu


# ---------------------------------------------------
# File/content helpers
# ---------------------------------------------------
def get_content_type(filename):
    filename = filename.lower()

    if filename.endswith(".wav"):
        return "audio/wav", ".wav"
    if filename.endswith(".flac"):
        return "audio/flac", ".flac"
    if filename.endswith(".mp3"):
        return "audio/mp3", ".mp3"
    if filename.endswith(".ogg"):
        return "audio/ogg", ".ogg"

    return None, None


# ---------------------------------------------------
# Speech to text
# ---------------------------------------------------
def transcribe_audio(stt_client, file_path, content_type):
    transcript_parts = []

    with open(file_path, "rb") as audio_file:
        result = stt_client.recognize(
            audio=audio_file,
            content_type=content_type,
            model="en-US_BroadbandModel",
            smart_formatting=True
        ).get_result()

    for item in result.get("results", []):
        alternatives = item.get("alternatives", [])
        if alternatives:
            text = alternatives[0].get("transcript", "").strip()
            if text:
                transcript_parts.append(text)

    transcript = " ".join(transcript_parts).strip()

    if not transcript:
        raise ValueError("No speech detected in the uploaded audio.")

    return transcript


# ---------------------------------------------------
# Natural language analysis
# ---------------------------------------------------
def analyze_transcript(nlu_client, transcript):
    if len(transcript.strip()) < 50:
        raise ValueError("Transcript is too short for meaningful analysis.")

    response = nlu_client.analyze(
        text=transcript,
        features=Features(
            keywords=KeywordsOptions(limit=8, sentiment=True),
            entities=EntitiesOptions(limit=8, sentiment=True),
            sentiment=SentimentOptions(document=True),
            concepts=ConceptsOptions(limit=5),
            categories=CategoriesOptions(limit=3)
        )
    ).get_result()

    return response


# ---------------------------------------------------
# Action item detection
# ---------------------------------------------------
def find_action_items(transcript):
    action_words = [
        "will",
        "should",
        "need to",
        "have to",
        "must",
        "going to",
        "plan to",
        "deadline",
        "follow up",
        "todo",
        "task",
        "assign",
        "responsible"
    ]

    cleaned = transcript.replace("?", ".").replace("!", ".")
    sentences = cleaned.split(".")
    action_items = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        lower_sentence = sentence.lower()
        if any(word in lower_sentence for word in action_words):
            action_items.append(sentence)

    return action_items[:10]


# ---------------------------------------------------
# Summary formatter
# ---------------------------------------------------
def format_summary(transcript, insights, action_items):
    lines = []
    lines.append("=" * 60)
    lines.append("MEETMIND - MEETING SUMMARY")
    lines.append("=" * 60)
    lines.append("")

    sentiment = insights.get("sentiment", {}).get("document", {})
    label = sentiment.get("label", "neutral")
    score = round(sentiment.get("score", 0), 2)

    lines.append(f"Overall Tone: {label.upper()} (score: {score})")
    lines.append("")

    keywords = insights.get("keywords", [])
    if keywords:
        lines.append("Key Topics:")
        for kw in keywords:
            relevance = round(kw.get("relevance", 0) * 100)
            lines.append(f"  - {kw.get('text', '')} ({relevance}% relevant)")
        lines.append("")

    entities = insights.get("entities", [])
    if entities:
        lines.append("Important Entities:")
        for ent in entities:
            ent_type = ent.get("type", "Unknown")
            ent_text = ent.get("text", "")
            lines.append(f"  - [{ent_type}] {ent_text}")
        lines.append("")

    concepts = insights.get("concepts", [])
    if concepts:
        lines.append("Concepts:")
        for concept in concepts:
            lines.append(f"  - {concept.get('text', '')}")
        lines.append("")

    categories = insights.get("categories", [])
    if categories:
        lines.append("Categories:")
        for category in categories:
            lines.append(f"  - {category.get('label', '')}")
        lines.append("")

    if action_items:
        lines.append("Action Items:")
        for i, item in enumerate(action_items, start=1):
            lines.append(f"  {i}. {item}")
        lines.append("")

    lines.append("-" * 60)
    lines.append("Transcript:")
    lines.append("-" * 60)
    lines.append(transcript)
    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


# ---------------------------------------------------
# Email helper
# ---------------------------------------------------
def send_email(recipients, summary_text, sender_email, sender_password):
    msg = MIMEMultipart()
    msg["Subject"] = "MeetMind Meeting Summary"
    msg["From"] = sender_email
    msg["To"] = ", ".join(recipients)

    msg.attach(MIMEText(summary_text, "plain"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipients, msg.as_string())


# ---------------------------------------------------
# Routes
# ---------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    index_path = os.path.join(BASE_DIR, "index.html")

    if os.path.exists(index_path):
        return send_from_directory(BASE_DIR, "index.html")

    return jsonify({
        "message": "MeetMind backend is running",
        "available_routes": {
            "home": "/",
            "health": "/health",
            "summarize": "/summarize"
        }
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "message": "MeetMind backend is running",
        "env_loaded": os.path.exists(ENV_PATH),
        "stt_key_found": bool(clean_value(os.getenv("STT_API_KEY"))),
        "stt_url_found": bool(clean_value(os.getenv("STT_URL"))),
        "nlu_key_found": bool(clean_value(os.getenv("NLU_API_KEY"))),
        "nlu_url_found": bool(clean_value(os.getenv("NLU_URL")))
    })


@app.route("/favicon.ico", methods=["GET"])
def favicon():
    favicon_path = os.path.join(BASE_DIR, "favicon.ico")
    if os.path.exists(favicon_path):
        return send_from_directory(BASE_DIR, "favicon.ico")
    return "", 204


@app.route("/meta.json", methods=["GET"])
def meta():
    meta_path = os.path.join(BASE_DIR, "meta.json")
    if os.path.exists(meta_path):
        return send_from_directory(BASE_DIR, "meta.json")

    return jsonify({
        "name": "MeetMind",
        "status": "ok"
    })


@app.route("/summarize", methods=["POST"])
def summarize():
    temp_file_path = None

    try:
        if "audio" not in request.files:
            return jsonify({"error": "No audio file uploaded"}), 400

        audio_file = request.files["audio"]

        if audio_file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        # get values from form first, else from .env
        stt_api_key = clean_value(request.form.get("stt_api_key")) or clean_value(os.getenv("STT_API_KEY"))
        stt_url = clean_value(request.form.get("stt_url")) or clean_value(os.getenv("STT_URL"))
        nlu_api_key = clean_value(request.form.get("nlu_api_key")) or clean_value(os.getenv("NLU_API_KEY"))
        nlu_url = clean_value(request.form.get("nlu_url")) or clean_value(os.getenv("NLU_URL"))

        logger.info("Using STT URL: %s", stt_url)
        logger.info("Using NLU URL: %s", nlu_url)
        logger.info("STT key found: %s", bool(stt_api_key))
        logger.info("NLU key found: %s", bool(nlu_api_key))

        if not all([stt_api_key, stt_url, nlu_api_key, nlu_url]):
            return jsonify({
                "error": "Missing IBM Watson credentials. Check your .env file or frontend form fields."
            }), 400

        content_type, suffix = get_content_type(audio_file.filename)
        if not content_type:
            return jsonify({
                "error": "Unsupported file type. Use WAV, FLAC, MP3, or OGG."
            }), 400

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            audio_file.save(temp_file.name)
            temp_file_path = temp_file.name

        logger.info("Audio file saved temporarily: %s", temp_file_path)

        stt_client = get_stt_client(stt_api_key, stt_url)
        transcript = transcribe_audio(stt_client, temp_file_path, content_type)

        nlu_client = get_nlu_client(nlu_api_key, nlu_url)
        insights = analyze_transcript(nlu_client, transcript)

        action_items = find_action_items(transcript)
        summary_text = format_summary(transcript, insights, action_items)

        email_status = "not requested"

        emails_raw = request.form.get("emails", "").strip()
        sender_email = clean_value(request.form.get("sender_email")) or clean_value(os.getenv("SENDER_EMAIL"))
        sender_password = clean_value(request.form.get("sender_password")) or clean_value(os.getenv("SENDER_PASSWORD"))

        if emails_raw:
            recipients = [email.strip() for email in emails_raw.split(",") if email.strip()]

            if sender_email and sender_password:
                try:
                    send_email(recipients, summary_text, sender_email, sender_password)
                    email_status = f"sent to {len(recipients)} recipient(s)"
                except Exception as email_error:
                    email_status = f"failed: {str(email_error)}"
            else:
                email_status = "failed: sender email or app password missing"

        return jsonify({
            "success": True,
            "transcript": transcript,
            "summary_text": summary_text,
            "email_status": email_status,
            "insights": {
                "keywords": insights.get("keywords", []),
                "entities": insights.get("entities", []),
                "sentiment": insights.get("sentiment", {}),
                "concepts": insights.get("concepts", []),
                "categories": insights.get("categories", [])
            },
            "action_items": action_items
        })

    except ValueError as ve:
        logger.warning("Validation error: %s", ve)
        return jsonify({"error": str(ve)}), 422

    except Exception as e:
        logger.exception("Unexpected server error")
        return jsonify({"error": f"Something went wrong: {str(e)}"}), 500

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.info("Temporary file deleted: %s", temp_file_path)


@app.route("/debug-env", methods=["GET"])
def debug_env():
    return jsonify({
        "stt_key_set": bool(os.getenv("STT_API_KEY")),
        "stt_key_preview": os.getenv("STT_API_KEY", "")[:6] + "...",
        "stt_url_set": bool(os.getenv("STT_URL")),
        "nlu_key_set": bool(os.getenv("NLU_API_KEY")),
        "nlu_url_set": bool(os.getenv("NLU_URL")),
    })


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)