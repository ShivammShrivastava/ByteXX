"""
Firebase Configuration for SatQuery AI Backend
================================================
Uses Firebase Admin SDK to access:
  - Firebase Storage  (read uploaded images)
  - Realtime Database (listen for queries, write answers)
"""

import firebase_admin
from firebase_admin import credentials, db, storage

# ── Firebase config (matches your web app) ──
FIREBASE_CONFIG = {
    "apiKey": "AIzaSyDJtjpV4DpD-Ev0BeRJDsfZV4k5U63dpW4",
    "authDomain": "satellite-efa0a.firebaseapp.com",
    "projectId": "satellite-efa0a",
    "storageBucket": "satellite-efa0a.firebasestorage.app",
    "messagingSenderId": "504899672780",
    "appId": "1:504899672780:web:4f61a1b212930752cdc069",
    "measurementId": "G-TY956H2RJF",
    "databaseURL": "https://satellite-efa0a-default-rtdb.firebaseio.com",
}

# Path to your service account JSON key file
# Download from: Firebase Console → Project Settings → Service Accounts → Generate new private key
SERVICE_ACCOUNT_PATH = "backend/serviceAccountKey.json"

# Realtime DB paths
DB_QUERIES_PATH = "queries"       # /queries/{id}/
DB_RESULTS_PATH = "results"       # /results/{id}/

# Storage folder for uploaded images
STORAGE_IMAGES_FOLDER = "uploads"

_initialized = False


def init_firebase():
    """Initialize Firebase Admin SDK (call once at startup)."""
    global _initialized
    if _initialized:
        return

    import os
    if os.path.exists(SERVICE_ACCOUNT_PATH):
        # Use service account key (full access)
        cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
        firebase_admin.initialize_app(cred, {
            "databaseURL": FIREBASE_CONFIG["databaseURL"],
            "storageBucket": FIREBASE_CONFIG["storageBucket"],
        })
        print(f"  Firebase initialized with service account")
    else:
        print(f"  ⚠ Service account key not found at: {SERVICE_ACCOUNT_PATH}")
        print(f"    Download from Firebase Console → Project Settings → Service Accounts")
        print(f"    The backend will still start but Firebase features won't work.")
        return

    _initialized = True


def get_db():
    """Get Realtime Database reference."""
    return db.reference("/")


def get_bucket():
    """Get Firebase Storage bucket."""
    return storage.bucket()
