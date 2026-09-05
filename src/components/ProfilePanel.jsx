import React from 'react';
import { X, Mail, User, Calendar, BarChart2, Satellite, LogOut, Shield } from 'lucide-react';
import './ProfilePanel.css';

export default function ProfilePanel({ user, recentsCount, pinnedCount, onLogout, onClose }) {
  const displayName = user?.displayName || user?.email?.split('@')[0] || 'Analyst';
  const email = user?.email || 'analyst@bytex.ai';
  const isDemo = user?.uid?.startsWith('user_') || user?.uid?.startsWith('demo');
  const avatarUrl = user?.photoURL || `https://api.dicebear.com/7.x/bottts/svg?seed=${user?.uid || 'analyst'}`;

  const joinDate = (() => {
    if (user?.metadata?.creationTime) {
      return new Date(user.metadata.creationTime).toLocaleDateString('en-US', { month: 'long', year: 'numeric' });
    }
    return 'Today';
  })();

  return (
    <div className="profile-overlay" onClick={onClose}>
      <div className="profile-panel" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="profile-panel-header">
          <span className="profile-panel-title">My Profile</span>
          <button className="profile-close-btn" onClick={onClose}><X size={18} /></button>
        </div>

        {/* Avatar + Name */}
        <div className="profile-hero">
          <div className="profile-avatar-ring">
            <img src={avatarUrl} alt={displayName} className="profile-avatar-lg" />
            {!isDemo && <div className="profile-online-dot"></div>}
          </div>
          <div className="profile-hero-info">
            <h2 className="profile-display-name">{displayName}</h2>
            <span className={`profile-role-badge ${isDemo ? 'demo' : 'real'}`}>
              <Shield size={12} />
              {isDemo ? 'Demo Account' : 'Verified Analyst'}
            </span>
          </div>
        </div>

        {/* Info Cards */}
        <div className="profile-info-grid">
          <div className="profile-info-card">
            <Mail size={16} className="info-card-icon" />
            <div>
              <span className="info-card-label">Email</span>
              <span className="info-card-value">{email}</span>
            </div>
          </div>

          <div className="profile-info-card">
            <User size={16} className="info-card-icon" />
            <div>
              <span className="info-card-label">User ID</span>
              <span className="info-card-value uid-val">{user?.uid?.substring(0, 20)}...</span>
            </div>
          </div>

          <div className="profile-info-card">
            <Calendar size={16} className="info-card-icon" />
            <div>
              <span className="info-card-label">Member Since</span>
              <span className="info-card-value">{joinDate}</span>
            </div>
          </div>

          <div className="profile-info-card">
            <Satellite size={16} className="info-card-icon" />
            <div>
              <span className="info-card-label">Backend Model</span>
              <span className="info-card-value">Qwen2.5-VL-3B</span>
            </div>
          </div>
        </div>

        {/* Stats */}
        <div className="profile-stats-row">
          <div className="profile-stat">
            <BarChart2 size={20} className="stat-icon" />
            <span className="stat-number">{recentsCount}</span>
            <span className="stat-label">Analyses Run</span>
          </div>
          <div className="profile-stat-divider"></div>
          <div className="profile-stat">
            <span className="stat-number">{pinnedCount}</span>
            <span className="stat-label">Pinned Queries</span>
          </div>
          <div className="profile-stat-divider"></div>
          <div className="profile-stat">
            <span className="stat-number">3</span>
            <span className="stat-label">Modalities</span>
          </div>
        </div>

        {/* Firebase status */}
        <div className="profile-firebase-status">
          <div className={`firebase-dot ${isDemo ? 'offline' : 'online'}`}></div>
          <span>{isDemo ? 'Local session — register for cloud sync' : 'Synced with Firebase RTDB: satellite-efa0a'}</span>
        </div>

        {/* Logout */}
        <button className="profile-logout-btn" onClick={() => { onLogout(); onClose(); }}>
          <LogOut size={16} />
          <span>Sign Out</span>
        </button>
      </div>
    </div>
  );
}
