"""
Firebase Setup & Connection Test
==================================
Run this script ONCE to verify your Firebase connection
and initialize the database structure.

Usage:
    python backend/firebase_setup.py

Prerequisites:
    1. Download your service account key from Firebase Console:
       Firebase Console → Project Settings → Service Accounts → Generate new private key
    2. Save the downloaded JSON as: backend/serviceAccountKey.json
"""

import sys
import os
import json
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Check service account key exists ──
KEY_PATH = os.path.join(os.path.dirname(__file__), "serviceAccountKey.json")
if not os.path.exists(KEY_PATH):
    print("=" * 60)
    print("  SETUP REQUIRED: Service Account Key Missing")
    print("=" * 60)
    print()
    print("  1. Go to: https://console.firebase.google.com")
    print("  2. Select project: satellite-efa0a")
    print("  3. Click the gear icon → Project Settings")
    print("  4. Go to 'Service Accounts' tab")
    print("  5. Click 'Generate new private key'")
    print("  6. Save the downloaded JSON as:")
    print(f"     {KEY_PATH}")
    print()
    print("  Then re-run this script.")
    sys.exit(1)

# ── Initialize Firebase ──
print("=" * 60)
print("  SatQuery AI — Firebase Setup & Test")
print("=" * 60)

import firebase_admin
from firebase_admin import credentials, db, storage

print("\n[1/4] Initializing Firebase Admin SDK...")
cred = credentials.Certificate(KEY_PATH)
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://satellite-efa0a-default-rtdb.firebaseio.com",
    "storageBucket": "satellite-efa0a.firebasestorage.app",
})
print("  OK - Firebase Admin initialized")

# ── Test Realtime DB ──
print("\n[2/4] Testing Realtime Database connection...")
ref = db.reference("/")
ref.child("_health_check").set({
    "status": "ok",
    "timestamp": int(time.time() * 1000),
    "message": "SatQuery AI backend connected"
})
val = ref.child("_health_check").get()
print(f"  OK - DB write/read successful: {val}")

# ── Initialize DB schema ──
print("\n[3/4] Initializing database schema...")
ref.child("_schema").set({
    "version": "1.0",
    "description": "SatQuery AI - VQA Firebase Schema",
    "queries_path": "/queries/{queryId}",
    "results_path": "/results/{queryId}",
    "last_updated": int(time.time() * 1000),
})
print("  OK - Schema initialized")

# ── Test Storage ──
print("\n[4/4] Testing Firebase Storage connection...")
try:
    bucket = storage.bucket()
    print(f"  OK - Storage bucket: {bucket.name}")
except Exception as e:
    print(f"  Warning: Storage test failed: {e}")

# ── Done ──
print()
print("=" * 60)
print("  SETUP COMPLETE")
print("=" * 60)
print()
print("  Realtime DB URL:  https://satellite-efa0a-default-rtdb.firebaseio.com")
print("  Storage Bucket:   satellite-efa0a.firebasestorage.app")
print()
print("  Database structure:")
print("  /queries/{queryId}")
print("    image_url  : string  (Storage download URL)")
print("    question   : string  (user's question)")
print("    task       : string  (vqa | caption | refer)")
print("    status     : string  (pending | processing | done | error)")
print("    timestamp  : number  (Unix ms)")
print()
print("  /results/{queryId}")
print("    answer     : string  (model's answer)")
print("    confidence : number  (0.0 - 1.0)")
print("    task       : string")
print("    processed_at: number (Unix ms)")
print()
print("  Start backend with:")
print("    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload")
