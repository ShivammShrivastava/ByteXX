"""
Firebase Configuration for SatQuery AI Backend
================================================
Uses Firebase Admin SDK to access:
  - Firebase Storage  (read uploaded images)
  - Realtime Database (listen for queries, write answers)

Project:  satellite-efa0a
RTDB URL: https://satellite-efa0a-default-rtdb.firebaseio.com
"""

import os
import firebase_admin
from firebase_admin import credentials, db, storage

# ── Firebase client-side config (matches web app in Firebase Console) ─────────
FIREBASE_CONFIG = {
    "apiKey":            "AIzaSyDJtjpV4DpD-Ev0BeRJDsfZV4k5U63dpW4",
    "authDomain":        "satellite-efa0a.firebaseapp.com",
    "projectId":         "satellite-efa0a",
    "storageBucket":     "satellite-efa0a.firebasestorage.app",
    "messagingSenderId": "504899672780",
    "appId":             "1:504899672780:web:4f61a1b212930752cdc069",
    "measurementId":     "G-TY956H2RJF",
    "databaseURL":       "https://satellite-efa0a-default-rtdb.firebaseio.com",
}

# ── Service-account key — try new name first, fall back to legacy name ─────────
_BACKEND_DIR = os.path.dirname(__file__)

_SA_CANDIDATES = [
    os.path.join(_BACKEND_DIR, "satellite-efa0a-firebase-adminsdk-fbsvc-075fca07b4.json"),
    os.path.join(_BACKEND_DIR, "serviceAccountKey.json"),
]

SERVICE_ACCOUNT_PATH: str = next(
    (p for p in _SA_CANDIDATES if os.path.exists(p)),
    _SA_CANDIDATES[0],   # default (will fail gracefully if missing)
)

# ── Realtime DB paths ──────────────────────────────────────────────────────────
DB_QUERIES_PATH    = "queries"        # /queries/{id}/
DB_RESULTS_PATH    = "results"        # /results/{id}/
DB_AGENT_PATH      = "agent_results"  # /agent_results/{id}/
DB_CNN_PATH        = "cnn_results"    # /cnn_results/{id}/
DB_USERS_PATH      = "users"          # /users/{uid}/

# ── Storage ───────────────────────────────────────────────────────────────────
STORAGE_IMAGES_FOLDER = "uploads"

_initialized = False


def init_firebase() -> bool:
    """
    Initialize Firebase Admin SDK (call once at startup).

    Returns True if successfully initialized, False otherwise.
    """
    global _initialized
    if _initialized:
        return True

    if not os.path.exists(SERVICE_ACCOUNT_PATH):
        print(f"  ⚠  Firebase service account key NOT found at:")
        for p in _SA_CANDIDATES:
            print(f"       {p}")
        print(f"  The backend will start but Firebase RTDB features won't work.")
        return False

    try:
        cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
        firebase_admin.initialize_app(cred, {
            "databaseURL":   FIREBASE_CONFIG["databaseURL"],
            "storageBucket": FIREBASE_CONFIG["storageBucket"],
        })
        _initialized = True
        sa_name = os.path.basename(SERVICE_ACCOUNT_PATH)
        print(f"  ✅ Firebase initialized — key: {sa_name}")
        print(f"     RTDB: {FIREBASE_CONFIG['databaseURL']}")
        return True
    except Exception as e:
        print(f"  ❌ Firebase init failed: {e}")
        return False


def is_initialized() -> bool:
    """Return True if Firebase Admin SDK is ready."""
    return _initialized


def get_db() -> db.Reference:
    """Get Realtime Database root reference."""
    return db.reference("/")


def get_bucket():
    """Get Firebase Storage bucket."""
    return storage.bucket()
