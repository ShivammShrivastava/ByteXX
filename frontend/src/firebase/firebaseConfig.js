// Firebase configuration and services for ByteX SatQuery AI
import { initializeApp, getApps, getApp } from 'firebase/app';
import { 
  getAuth, 
  signInWithEmailAndPassword, 
  createUserWithEmailAndPassword, 
  signInWithPopup, 
  GoogleAuthProvider, 
  signOut as fbSignOut, 
  onAuthStateChanged,
  updateProfile
} from 'firebase/auth';
import { 
  getDatabase, 
  ref, 
  set, 
  onValue, 
  remove
} from 'firebase/database';

// ✅ Real Firebase configuration — satellite-efa0a (ByteX project)
export const firebaseConfig = {
  apiKey: "AIzaSyDJtjpV4DpD-Ev0BeRJDsfZV4k5U63dpW4",
  authDomain: "satellite-efa0a.firebaseapp.com",
  databaseURL: "https://satellite-efa0a-default-rtdb.firebaseio.com",
  projectId: "satellite-efa0a",
  storageBucket: "satellite-efa0a.firebasestorage.app",
  messagingSenderId: "504899672780",
  appId: "1:504899672780:web:4f61a1b212930752cdc069",
  measurementId: "G-TY956H2RJF"
};

let app, auth, db;

try {
  // Reuse existing app instance if already initialized (handles Vite HMR)
  app = getApps().length > 0 ? getApp() : initializeApp(firebaseConfig);
  auth = getAuth(app);
  db = getDatabase(app);
  console.log("✅ Firebase initialized for project: satellite-efa0a");
} catch (error) {
  console.warn("Firebase init error:", error.message);
}

const googleProvider = new GoogleAuthProvider();
googleProvider.setCustomParameters({ prompt: 'select_account' });

// ─── User Data ─────────────────────────────────────────────────────────────

export async function saveUserData(user, extraData = {}) {
  if (!user?.uid) return null;

  const userData = {
    uid: user.uid,
    email: user.email || '',
    displayName: user.displayName || extraData.displayName || user.email?.split('@')[0] || 'Analyst',
    photoURL: user.photoURL || `https://api.dicebear.com/7.x/bottts/svg?seed=${user.uid}`,
    lastLogin: new Date().toISOString(),
    createdAt: extraData.createdAt || new Date().toISOString(),
    role: 'Satellite Analyst',
    ...extraData
  };

  // localStorage (instant, offline)
  try {
    localStorage.setItem('bytex_active_user', JSON.stringify(userData));
    localStorage.setItem(`bytex_user_${user.uid}`, JSON.stringify(userData));
  } catch (e) {}

  // Firebase RTDB at /users/{uid}/profile
  if (db) {
    try {
      await set(ref(db, `users/${user.uid}/profile`), userData);
      console.log(`✅ User saved to RTDB: /users/${user.uid}/profile`);
    } catch (e) {
      console.warn("RTDB write error:", e.message);
    }
  }

  return userData;
}

// ─── Recents ───────────────────────────────────────────────────────────────

export async function saveRecentQuery(uid, queryItem) {
  if (!uid) return null;

  const item = {
    ...queryItem,
    id: queryItem.id || `q_${Date.now()}`,
    timestamp: queryItem.timestamp || Date.now()
  };

  // localStorage cache
  try {
    const key = `bytex_recents_${uid}`;
    const existing = JSON.parse(localStorage.getItem(key) || '[]');
    const updated = [item, ...existing.filter(i => i.id !== item.id)].slice(0, 50);
    localStorage.setItem(key, JSON.stringify(updated));
  } catch (e) {}

  // RTDB: /users/{uid}/recents/{id}
  if (db) {
    try {
      await set(ref(db, `users/${uid}/recents/${item.id}`), item);
      console.log(`✅ Recent saved to RTDB: /users/${uid}/recents/${item.id}`);
    } catch (e) {
      console.warn("RTDB recents write:", e.message);
    }
  }

  return item;
}

export function subscribeToUserRecents(uid, onUpdate) {
  if (!uid) return () => {};

  // Seed from localStorage immediately
  try {
    const cached = JSON.parse(localStorage.getItem(`bytex_recents_${uid}`) || '[]');
    if (cached.length > 0) onUpdate(cached);
  } catch (e) {}

  if (!db) return () => {};

  try {
    const recentsRef = ref(db, `users/${uid}/recents`);
    const unsub = onValue(recentsRef, (snap) => {
      if (snap.exists()) {
        const items = Object.values(snap.val())
          .sort((a, b) => (b.timestamp || 0) - (a.timestamp || 0));
        localStorage.setItem(`bytex_recents_${uid}`, JSON.stringify(items));
        onUpdate(items);
      }
    }, (err) => console.warn("RTDB listener:", err.message));
    return unsub;
  } catch (e) {
    return () => {};
  }
}

export async function deleteRecentQuery(uid, queryId) {
  if (!uid || !queryId) return;
  try {
    const key = `bytex_recents_${uid}`;
    const existing = JSON.parse(localStorage.getItem(key) || '[]');
    localStorage.setItem(key, JSON.stringify(existing.filter(i => i.id !== queryId)));
  } catch (e) {}
  if (db) {
    try {
      await remove(ref(db, `users/${uid}/recents/${queryId}`));
    } catch (e) {}
  }
}

// ─── Auth Actions ──────────────────────────────────────────────────────────

export async function loginWithEmail(email, password) {
  if (!auth) throw new Error("Firebase Auth not initialized");
  const cred = await signInWithEmailAndPassword(auth, email, password);
  await saveUserData(cred.user);
  return cred.user;
}

export async function registerWithEmail(email, password, displayName) {
  if (!auth) throw new Error("Firebase Auth not initialized");
  const cred = await createUserWithEmailAndPassword(auth, email, password);
  if (displayName) {
    try { await updateProfile(cred.user, { displayName }); } catch (_) {}
  }
  // Re-read user after profile update
  await saveUserData({ ...cred.user, displayName }, { displayName, createdAt: new Date().toISOString() });
  return cred.user;
}

export async function loginWithGoogle() {
  if (!auth) throw new Error("Firebase Auth not initialized");
  const cred = await signInWithPopup(auth, googleProvider);
  await saveUserData(cred.user);
  return cred.user;
}

export async function logoutUser() {
  try {
    localStorage.removeItem('bytex_active_user');
    if (auth) await fbSignOut(auth);
  } catch (e) {}
}

export function onAuthChange(callback) {
  if (auth) {
    return onAuthStateChanged(auth, async (firebaseUser) => {
      if (firebaseUser) {
        // Enrich with any extra data from RTDB
        callback(firebaseUser);
      } else {
        // Check localStorage for a cached local/demo session
        try {
          const cached = localStorage.getItem('bytex_active_user');
          callback(cached ? JSON.parse(cached) : null);
        } catch (_) {
          callback(null);
        }
      }
    });
  }
  try {
    const cached = localStorage.getItem('bytex_active_user');
    callback(cached ? JSON.parse(cached) : null);
  } catch (_) {
    callback(null);
  }
  return () => {};
}

export { auth, db };
