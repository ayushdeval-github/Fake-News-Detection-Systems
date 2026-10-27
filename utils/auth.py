"""
utils/auth.py — Authentication & MongoDB Integration
Fake News Detection System | Ayush Deval
"""

import os
import logging
from datetime import datetime, timedelta
from functools import wraps

import jwt
import bcrypt
from pymongo import MongoClient, errors
from flask import request, jsonify

logger = logging.getLogger(__name__)

_client = None
_db     = None

def get_db():
    global _client, _db
    if _db is None:
        mongo_uri = os.environ.get("MONGO_URI")
        if not mongo_uri:
            raise ValueError("MONGO_URI not set in .env file")
        try:
            _client = MongoClient(
                mongo_uri,
                serverSelectionTimeoutMS=5000,
                tls=True,
                tlsAllowInvalidCertificates=True,
                tlsAllowInvalidHostnames=True
            )
            _db = _client["fake_news_detector"]
            # Force a quick ping to validate the connection early.
            _client.admin.command("ping")
            logger.info("MongoDB Atlas connected successfully")
        except errors.PyMongoError as exc:
            logger.error("MongoDB connection failed: %s", exc)
            raise
    return _db

def get_users_collection():
    return get_db()["users"]

def get_history_collection():
    return get_db()["search_history"]

SECRET_KEY   = os.environ.get("SECRET_KEY", "fallback_secret_key_change_this")
TOKEN_EXPIRY = 24

def generate_token(user_id, username, email):
    payload = {
        "user_id":  str(user_id),
        "username": username,
        "email":    email,
        "exp":      datetime.utcnow() + timedelta(hours=TOKEN_EXPIRY),
        "iat":      datetime.utcnow(),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

def verify_token(token):
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    except Exception:
        return None

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        if not token:
            token = request.cookies.get("auth_token", "")
        if not token:
            return jsonify({"error": "Authentication required. Please login."}), 401
        payload = verify_token(token)
        if not payload:
            return jsonify({"error": "Session expired. Please login again."}), 401
        request.current_user = payload
        return f(*args, **kwargs)
    return decorated

def register_user(username, email, password):
    if not username or len(username.strip()) < 3:
        return {"success": False, "error": "Username must be at least 3 characters."}
    if not email or "@" not in email:
        return {"success": False, "error": "Invalid email address."}
    if not password or len(password) < 6:
        return {"success": False, "error": "Password must be at least 6 characters."}

    username = username.strip().lower()
    email = email.strip().lower()

    try:
        users = get_users_collection()
        if users.find_one({"email": email}):
            return {"success": False, "error": "Email already registered. Please login."}
        if users.find_one({"username": username}):
            return {"success": False, "error": "Username already taken. Choose another."}

        hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
        user_doc = {
            "username":       username,
            "email":          email,
            "password":       hashed,
            "created_at":     datetime.utcnow(),
            "last_login":     datetime.utcnow(),
            "total_searches": 0,
            "is_active":      True,
        }
        result = users.insert_one(user_doc)
    except errors.PyMongoError as exc:
        logger.error("MongoDB query failed during registration: %s", exc)
        return {"success": False, "error": "Database unavailable. Please try again later."}

    user_id = str(result.inserted_id)
    token = generate_token(user_id, username, email)
    logger.info("New user registered: %s (%s)", username, email)
    return {"success": True, "token": token,
            "user": {"id": user_id, "username": username, "email": email}}

def login_user(email, password):
    if not email or not password:
        return {"success": False, "error": "Email and password are required."}
    email = email.strip().lower()
    try:
        users = get_users_collection()
        user = users.find_one({"email": email})
    except errors.PyMongoError as exc:
        logger.error("MongoDB query failed during login: %s", exc)
        return {"success": False, "error": "Database unavailable. Please try again later."}

    if not user:
        return {"success": False, "error": "Email not found. Please register first."}
    if not bcrypt.checkpw(password.encode("utf-8"), user["password"]):
        return {"success": False, "error": "Incorrect password. Please try again."}

    try:
        users.update_one({"_id": user["_id"]}, {"$set": {"last_login": datetime.utcnow()}})
    except errors.PyMongoError as exc:
        logger.warning("Could not update last_login for %s: %s", email, exc)

    user_id = str(user["_id"])
    username = user["username"]
    token = generate_token(user_id, username, email)
    logger.info("User logged in: %s (%s)", username, email)
    return {
        "success": True, "token": token,
        "user": {
            "id":             user_id,
            "username":       username,
            "email":          email,
            "total_searches": user.get("total_searches", 0),
            "created_at":     user["created_at"].strftime("%d %b %Y"),
        }
    }

def save_search(user_id, username, search_data):
    try:
        from bson import ObjectId
        history = get_history_collection()
        users   = get_users_collection()
        doc = {
            "user_id":          user_id,
            "username":         username,
            "news_text":        search_data.get("text", "")[:500],
            "model_used":       search_data.get("model_used", ""),
            "final_verdict":    search_data.get("final_verdict", ""),
            "final_confidence": search_data.get("final_confidence", 0),
            "ml_verdict":       search_data.get("ml_verdict", ""),
            "ml_confidence":    search_data.get("ml_confidence", 0),
            "groq_verdict":     search_data.get("groq_verdict", ""),
            "groq_explanation": search_data.get("groq_explanation", ""),
            "agreement":        search_data.get("agreement", ""),
            "time_ms":          search_data.get("time_ms", 0),
            "timestamp":        datetime.utcnow(),
        }
        history.insert_one(doc)
        users.update_one({"_id": ObjectId(user_id)}, {"$inc": {"total_searches": 1}})
        logger.info("Search saved for user: %s | verdict: %s", username, doc["final_verdict"])
    except Exception as exc:
        logger.error("Failed to save search: %s", exc)

def get_search_history(user_id, limit=20):
    try:
        history = get_history_collection()
        results = history.find({"user_id": user_id}).sort("timestamp", -1).limit(limit)
        records = []
        for r in results:
            records.append({
                "search_id":        str(r["_id"]),  # Include ID for deletion
                "news_text":        r.get("news_text", ""),
                "model_used":       r.get("model_used", ""),
                "final_verdict":    r.get("final_verdict", ""),
                "final_confidence": r.get("final_confidence", 0),
                "groq_explanation": r.get("groq_explanation", ""),
                "agreement":        r.get("agreement", ""),
                "timestamp":        r["timestamp"].strftime("%d %b %Y, %I:%M %p") if r.get("timestamp") else "",
            })
        return records
    except errors.PyMongoError as exc:
        logger.error("Failed to get history: %s", exc)
        return []
    except Exception as exc:
        logger.error("Failed to get history: %s", exc)
        return []

def get_user_stats(user_id):
    try:
        history = get_history_collection()
        total = history.count_documents({"user_id": user_id})
        fake  = history.count_documents({"user_id": user_id, "final_verdict": "Fake"})
        real  = history.count_documents({"user_id": user_id, "final_verdict": "Real"})
        return {"total_searches": total, "fake_detected": fake, "real_detected": real}
    except Exception as exc:
        logger.error("Failed to get stats: %s", exc)
        return {"total_searches": 0, "fake_detected": 0, "real_detected": 0}

def delete_search(user_id, search_id):
    """Delete a single search history item."""
    try:
        from bson import ObjectId
        history = get_history_collection()
        result = history.delete_one({"_id": ObjectId(search_id), "user_id": user_id})
        if result.deleted_count > 0:
            logger.info("Search deleted for user: %s | search_id: %s", user_id, search_id)
            return True
        return False
    except Exception as exc:
        logger.error("Failed to delete search: %s", exc)
        return False

def delete_all_history(user_id):
    """Delete all search history for a user."""
    try:
        history = get_history_collection()
        result = history.delete_many({"user_id": user_id})
        logger.info("All searches deleted for user: %s | count: %d", user_id, result.deleted_count)
        return result.deleted_count
    except Exception as exc:
        logger.error("Failed to delete all history: %s", exc)
        return 0