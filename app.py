"""
app.py — Flask Application with MongoDB Authentication
Fake News Detection System | Ayush Deval
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import time, logging, os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)
app.secret_key = os.environ.get("SECRET_KEY")

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Load ML Models ────────────────────────────
try:
    from utils.model_loader import load_all_models
    models_cache = load_all_models()
    logger.info("All ML models loaded into memory.")
except KeyboardInterrupt:
    logger.info("Startup interrupted by user.")
    raise SystemExit(0)


# ══════════════════════════════════════════════
# FRONTEND ROUTES
# ══════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/login")
def login_page():
    return render_template("login.html")


# ══════════════════════════════════════════════
# AUTH ROUTES
# ══════════════════════════════════════════════

@app.route("/auth/register", methods=["POST"])
def register():
    data     = request.get_json(silent=True) or {}
    username = data.get("username", "").strip()
    email    = data.get("email", "").strip()
    password = data.get("password", "")
    from utils.auth import register_user
    result = register_user(username, email, password)
    status = 201 if result["success"] else 400
    return jsonify(result), status


@app.route("/auth/login", methods=["POST"])
def login():
    data     = request.get_json(silent=True) or {}
    email    = data.get("email", "").strip()
    password = data.get("password", "")
    from utils.auth import login_user
    result = login_user(email, password)
    status = 200 if result["success"] else 401
    return jsonify(result), status


@app.route("/auth/logout", methods=["POST"])
def logout():
    return jsonify({"success": True, "message": "Logged out successfully."})


@app.route("/auth/me", methods=["GET"])
def get_me():
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    if not token:
        token = request.cookies.get("auth_token", "")
    from utils.auth import verify_token, get_user_stats
    payload = verify_token(token)
    if not payload:
        return jsonify({"error": "Not authenticated"}), 401
    stats = get_user_stats(payload["user_id"])
    return jsonify({"success": True,
                    "user": {"id": payload["user_id"],
                             "username": payload["username"],
                             "email": payload["email"]},
                    "stats": stats})


# ══════════════════════════════════════════════
# HISTORY ROUTES
# ══════════════════════════════════════════════

@app.route("/history", methods=["GET"])
def get_history():
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    if not token:
        return jsonify({"error": "Authentication required."}), 401
    from utils.auth import verify_token, get_search_history, get_user_stats
    payload = verify_token(token)
    if not payload:
        return jsonify({"error": "Session expired. Please login again."}), 401
    history = get_search_history(payload["user_id"], limit=50)
    stats   = get_user_stats(payload["user_id"])
    return jsonify({"success": True, "username": payload["username"],
                    "history": history, "stats": stats})


@app.route("/history/delete", methods=["DELETE"])
def delete_search_item():
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    if not token:
        return jsonify({"error": "Authentication required."}), 401
    from utils.auth import verify_token, delete_search
    payload = verify_token(token)
    if not payload:
        return jsonify({"error": "Session expired."}), 401
    data = request.get_json(silent=True) or {}
    search_id = data.get("search_id", "")
    if not search_id:
        return jsonify({"error": "search_id is required."}), 400
    success = delete_search(payload["user_id"], search_id)
    if success:
        return jsonify({"success": True, "message": "Search deleted."}), 200
    return jsonify({"error": "Search not found."}), 404


@app.route("/history/delete-all", methods=["DELETE"])
def delete_all_history_route():
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    if not token:
        return jsonify({"error": "Authentication required."}), 401
    from utils.auth import verify_token, delete_all_history
    payload = verify_token(token)
    if not payload:
        return jsonify({"error": "Session expired."}), 401
    count = delete_all_history(payload["user_id"])
    return jsonify({"success": True, "message": f"Deleted {count} records.",
                    "deleted_count": count}), 200


# ══════════════════════════════════════════════
# HEALTH CHECK
# ══════════════════════════════════════════════

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok",
                    "models_loaded": list(models_cache.keys()),
                    "groq_enabled":  bool(os.environ.get("GROQ_API_KEY")),
                    "mongo_enabled": bool(os.environ.get("MONGO_URI"))}), 200


# ══════════════════════════════════════════════
# PREDICT ROUTE — FIXED: MongoDB save now executes
# ══════════════════════════════════════════════

@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Request JSON:
        { "text": "news content here", "model": "lr" | "lstm" | "bert" }

    Response JSON:
        {
            "final_verdict":    "Fake" | "Real",
            "final_confidence": 94,
            "ml_verdict":       "Fake",
            "ml_confidence":    91,
            "groq_verdict":     "Fake",
            "groq_explanation": "This is false because...",
            "groq_confidence":  "High",
            "agreement":        "Both models agree",
            "model_used":       "BERT",
            "time_ms":          1420.5
        }
    Body: { text, model }
    Headers: Authorization: Bearer <token>  (optional — saves history if provided)

    FIXES APPLIED:
    1. Removed early return that was blocking MongoDB save
    2. Fixed run_prediction call — now correctly unpacks 2 values (not 3)
    3. MongoDB save now always executes after prediction
    """
    start_time = time.time()

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    text         = data.get("text", "").strip()
    model_choice = data.get("model", "lr").lower()

    if not text or len(text) < 10:
        return jsonify({"error": "Text must be at least 10 characters."}), 400
    if model_choice not in ("lr", "lstm", "bert"):
        return jsonify({"error": "Invalid model. Choose: lr, lstm, or bert"}), 400

    # ── Step 1: ML Model ──────────────────────
    # FIX: run_prediction returns 2 values (prediction, confidence) — not 3
    try:
        from utils.predict import run_prediction
        ml_prediction, ml_confidence = run_prediction(text, model_choice, models_cache)
    except Exception as exc:
        logger.error("ML inference error: %s", exc, exc_info=True)
        return jsonify({"error": "ML model inference failed.", "detail": str(exc)}), 500

    # ── Step 2: Groq LLaMA 3 70B ──────────────
    try:
        from utils.groq_client import analyze_with_groq
        groq_result = analyze_with_groq(text)
    except Exception as exc:
        logger.error("Groq error: %s", exc)
        groq_result = {"verdict": "Unavailable", "explanation": str(exc),
                       "confidence": "N/A", "groq_used": False}

    # ── Step 3: Combine verdicts ───────────────
    from utils.groq_client import combined_verdict
    result             = combined_verdict(ml_prediction, ml_confidence, groq_result)
    result["model_used"] = model_choice.upper()
    result["time_ms"]    = round((time.time() - start_time) * 1000, 2)

    # ── Step 4: Save to MongoDB (if logged in) ─
    # FIX: This now executes — the early return bug is removed
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    if token:
        try:
            from utils.auth import verify_token, save_search
            payload = verify_token(token)
            if payload:
                save_search(
                    user_id  = payload["user_id"],
                    username = payload["username"],
                    search_data = {
                        "text":             text,
                        "model_used":       result["model_used"],
                        "final_verdict":    result["final_verdict"],
                        "final_confidence": result["final_confidence"],
                        "ml_verdict":       result.get("ml_verdict", ""),
                        "ml_confidence":    result.get("ml_confidence", 0),
                        "groq_verdict":     result.get("groq_verdict", ""),
                        "groq_explanation": result.get("groq_explanation", ""),
                        "agreement":        result.get("agreement", ""),
                        "time_ms":          result["time_ms"],
                    }
                )
                logger.info("Search saved to DB | User: %s | Verdict: %s",
                            payload["username"], result["final_verdict"])
        except Exception as exc:
            logger.warning("DB save failed (non-critical): %s", exc)

    # ── Step 5: Return response ────────────────
    return jsonify(result)


# ══════════════════════════════════════════════
# ERROR HANDLERS
# ══════════════════════════════════════════════

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found."}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed."}), 405

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error."}), 500


# ══════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════

if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV", "production") == "development"
    try:
        app.run(host="0.0.0.0", port=port, debug=debug, use_reloader=False)
    except KeyboardInterrupt:
        logger.info("Server stopped by user.")
