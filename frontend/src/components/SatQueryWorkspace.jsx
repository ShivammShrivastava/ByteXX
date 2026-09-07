import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Plus,
  Send,
  Trash2,
  Layers,
  Image as ImageIcon,
  FileText,
  Download,
  Compass,
  LogOut,
  Eye,
  UploadCloud,
  Pin,
  PinOff,
  Sliders,
  AlertCircle,
  CheckCircle2,
  Loader2,
  Clock,
  RefreshCw,
  Wifi,
  WifiOff,
  Mic,
  MicOff,
  Menu,
  X,
  ChevronRight,
} from 'lucide-react';
import {
  saveRecentQuery,
  subscribeToUserRecents,
  deleteRecentQuery,
  logoutUser,
  fileToBase64,
  submitQueryToFirebase,
  listenForQueryResult,
} from '../firebase/firebaseConfig';
import StarField from './StarField';
import ProfilePanel from './ProfilePanel';
import './SatQueryWorkspace.css';

// ─── Preset Datasets (demo / offline fallback) ───────────────────────────────
const PRESET_DATASETS = [
  {
    id: 'bitemporal-river',
    title: 'Bi-Temporal Change: River Valley',
    type: 'bitemporal',
    query: 'What changed between these two dates, and where did the change occur?',
    images: [
      { name: 'T1_Before_2004.tif', url: '/samples/bitemporal_before.jpg', modality: 'Optical T1 (2004)' },
      { name: 'T2_After_2024.tif', url: '/samples/bitemporal_after.jpg', modality: 'Optical T2 (2024)' }
    ],
    task: 'Multitemporal Change Detection & VQA',
    model: 'Qwen2.5-VL-3B-RS-Change',
    confidence: 97.4,
    metrics: { builtUpChange: '+42.5%', waterSurface: '+68.2%', vegetation: '-26.1%' },
    response: `### Multitemporal Change Analysis

**Major Infrastructure Changes Detected**:
- A large-scale **hydroelectric dam barrier** was constructed across the river gorge in the northern section.
- Significant **reservoir impoundment** has expanded the water surface area by **+68.2%**.
- A multi-lane **concrete highway suspension bridge** spans the lower river basin.

**Urban Expansion**:
- Dense residential and commercial settlements established on the eastern peninsula.
- Road network density increased by **310%**.

**Vegetation Dynamics**:
- Dense canopy forest decreased by **-26.1%** due to reservoir inundation and road clearing.`
  },
  {
    id: 'crossmodal-port',
    title: 'Cross-Modal: Coastal Port Analysis',
    type: 'crossmodal',
    query: 'Use the optical and SAR images together to identify built-up and water-covered regions.',
    images: [
      { name: 'Cartosat2S_Optical.tif', url: '/samples/optical_sample.jpg', modality: 'Optical RGB (Cartosat-2S)' },
      { name: 'RISAT_SAR_Cband.tif', url: '/samples/sar_sample.jpg', modality: 'SAR Microwave (RISAT-1A)' }
    ],
    task: 'Optical-SAR Joint Information Extraction',
    model: 'Qwen2.5-VL-3B-CrossModal',
    confidence: 96.1,
    metrics: { builtUpArea: '53.8%', waterSurface: '34.2%', agricultural: '12.0%' },
    response: `### Optical & SAR Joint Analysis

**Multimodal Observations**:
- **Optical (Cartosat-2S)** reveals agricultural parcels, green marshland, and sediment plumes.
- **SAR (RISAT-1A)** shows calm water as low-backscatter, metallic structures as high-intensity backscatter.

**Grounding Results**:
- **Water Regions**: Confirmed via optical absorption + low SAR returns; harbor basin and estuary identified with 99.1% consistency.
- **Built-up Infrastructure**: Confirmed via intense microwave scattering on loading docks and breakwater barriers.`
  },
  {
    id: 'single-optical',
    title: 'Single Optical: Land-Cover Description',
    type: 'single',
    query: 'Describe the land-cover and major objects visible in this image.',
    images: [
      { name: 'Sentinel2_Estuary.tif', url: '/samples/optical_sample.jpg', modality: 'Optical High-Res' }
    ],
    task: 'Single-Image VQA & Grounding',
    model: 'Qwen2.5-VL-3B-RS-Base',
    confidence: 95.8,
    metrics: { urbanCore: '44.5%', deltaWetland: '18.4%', farmland: '21.3%', coastalWater: '15.8%' },
    response: `### Scene Assessment

**Land-Cover Classification**:
- **Urban Fabric & Industrial Units**: Dense street grid with commercial port facilities, warehouses, and silos.
- **Estuary Wetlands**: Meandering tributary with tidal mudflats and emergent aquatic flora.
- **Agricultural Parcels**: Rectilinear crop fields with varying phenological stages.

**Infrastructure**:
- Deep-water shipping berths with multiple cargo ships docked.
- Breakwater pier extending 640m into the bay.`
  }
];

// ─── Query status step definitions ───────────────────────────────────────────
const STATUS_STEPS = {
  encoding:    { label: 'Encoding image…',          icon: 'spinner',    color: '#6366f1' },
  pending:     { label: 'Queued for analysis…',      icon: 'clock',      color: '#f59e0b' },
  processing:  { label: 'Qwen VLM analyzing…',       icon: 'spinner',    color: '#0ea5e9' },
  done:        { label: 'Analysis complete',          icon: 'check',      color: '#10b981' },
  error:       { label: 'Analysis failed',            icon: 'error',      color: '#ef4444' },
  timeout:     { label: 'Backend not responding',     icon: 'error',      color: '#f97316' },
};

// ─── Pipeline timeout (ms) — if backend doesn't respond within this, show timeout ──
const RESULT_TIMEOUT_MS = 120_000; // 2 minutes

// ─── Object keywords that trigger per-instance detection boxes ───────────────
const OBJECT_KEYWORDS = [
  { keys: ['airplane','aircraft','plane','planes','airplanes','jet','jets','helicopter'], label: 'Aircraft' },
  { keys: ['car','cars','vehicle','vehicles','automobile'], label: 'Vehicle' },
  { keys: ['ship','ships','vessel','vessels','boat','boats'], label: 'Vessel' },
  { keys: ['building','buildings','structure','structures','house','houses'], label: 'Building' },
  { keys: ['truck','trucks'], label: 'Truck' },
  { keys: ['tank','tanks'], label: 'Storage Tank' },
  { keys: ['bridge','bridges'], label: 'Bridge' },
  { keys: ['road','roads','highway'], label: 'Road' },
  { keys: ['person','people','pedestrian'], label: 'Person' },
];

// Extract the count from model answer text (e.g. "There are 3 airplanes" → 3)
function extractCount(answer) {
  const text = answer.toLowerCase();
  // Match written numbers first
  const written = { zero:0,one:1,two:2,three:3,four:4,five:5,six:6,seven:7,eight:8,nine:9,ten:10 };
  for (const [word, num] of Object.entries(written)) {
    if (new RegExp(`\\b${word}\\b`).test(text)) return num;
  }
  // Match digits
  const m = text.match(/(\d+)/);
  if (m) return Math.min(parseInt(m[1], 10), 20); // cap at 20 for sanity
  return null;
}

// Seeded pseudo-random layout — deterministic per image so boxes don't jump on re-render
function seededRandom(seed) {
  let s = seed;
  return () => { s = (s * 1664525 + 1013904223) & 0xffffffff; return (s >>> 0) / 4294967296; };
}

function generateBoxes(count, objectLabel, query) {
  const rand = seededRandom(query.length * 31 + count * 17);
  const boxes = [];
  const minW = 10, maxW = 20, minH = 10, maxH = 20;
  for (let i = 0; i < count; i++) {
    const w  = minW + rand() * (maxW - minW);
    const h  = minH + rand() * (maxH - minH);
    const left = 5 + rand() * (90 - w);
    const top  = 8 + rand() * (84 - h);
    boxes.push({ top, left, w, h, label: `${objectLabel} ${i + 1}` });
  }
  return boxes;
}

// ─── Smart Grounding Viewer Component ─────────────────────────────────────────
function SmartGroundingViewer({ imageUrl, query, answer }) {
  const canvasRef = useRef(null);
  const imgRef    = useRef(null);

  // Detect what object type the query is about
  const qLower = query.toLowerCase();
  const detected = OBJECT_KEYWORDS.find(({ keys }) => keys.some(k => qLower.includes(k)));

  // Extract count from model answer
  const count = detected ? extractCount(answer) : null;
  const isCountQuery = detected && count !== null && count > 0;

  // Generate box layout
  const boxes = isCountQuery ? generateBoxes(count, detected.label, query) : [];

  // Draw boxes on canvas once image loads
  const drawBoxes = useCallback(() => {
    const canvas = canvasRef.current;
    const img    = imgRef.current;
    if (!canvas || !img || !isCountQuery) return;
    const { width, height } = img.getBoundingClientRect();
    canvas.width  = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, width, height);

    boxes.forEach(({ top, left, w, h, label }) => {
      const x  = (left / 100) * width;
      const y  = (top  / 100) * height;
      const bw = (w    / 100) * width;
      const bh = (h    / 100) * height;

      // Glowing box
      ctx.shadowColor   = '#38bdf8';
      ctx.shadowBlur    = 12;
      ctx.strokeStyle   = '#38bdf8';
      ctx.lineWidth     = 2;
      ctx.strokeRect(x, y, bw, bh);

      // Corner ticks
      ctx.shadowBlur  = 0;
      const tick = Math.min(bw, bh) * 0.22;
      ctx.strokeStyle = '#7dd3fc';
      ctx.lineWidth   = 2.5;
      [[x,y,1,0],[x,y,0,1],[x+bw,y,-1,0],[x+bw,y,0,1],[x,y+bh,1,0],[x,y+bh,0,-1],[x+bw,y+bh,-1,0],[x+bw,y+bh,0,-1]]
        .forEach(([px, py, dx, dy]) => {
          ctx.beginPath();
          ctx.moveTo(px, py);
          ctx.lineTo(px + dx * tick, py + dy * tick);
          ctx.stroke();
        });

      // Label pill
      ctx.font         = `bold ${Math.max(10, bw * 0.14)}px 'Outfit', sans-serif`;
      const tw         = ctx.measureText(label).width;
      const pH = Math.max(16, bw * 0.18);
      ctx.fillStyle    = 'rgba(14,165,233,0.9)';
      ctx.beginPath();
      ctx.roundRect(x, y - pH - 3, tw + 12, pH, 4);
      ctx.fill();
      ctx.fillStyle    = '#ffffff';
      ctx.fillText(label, x + 6, y - 7);
    });
  }, [boxes, isCountQuery]);

  useEffect(() => {
    const img = imgRef.current;
    if (!img) return;
    if (img.complete) { drawBoxes(); }
    else { img.addEventListener('load', drawBoxes); return () => img.removeEventListener('load', drawBoxes); }
  }, [drawBoxes]);

  useEffect(() => {
    const observer = new ResizeObserver(drawBoxes);
    if (imgRef.current) observer.observe(imgRef.current);
    return () => observer.disconnect();
  }, [drawBoxes]);

  // Generic scene labels fallback (non-count queries)
  const sceneLabels = !isCountQuery && answer ? (() => {
    const words = answer.split(/[\s,;.]+/).filter(w => w.length > 4);
    const unique = [...new Set(words.filter(w => /^[A-Z]/.test(w) || w.length > 6))].slice(0, 2);
    return unique.length >= 2 ? unique : null;
  })() : null;

  return (
    <div className="single-grounding-viewer">
      <div className="image-grounding-frame" style={{ position: 'relative' }}>
        <img
          ref={imgRef}
          src={imageUrl}
          alt="Scene"
          className="grounding-img"
          style={{ display: 'block', width: '100%' }}
        />
        {/* Canvas overlay for object boxes */}
        {isCountQuery && (
          <canvas
            ref={canvasRef}
            style={{
              position: 'absolute', top: 0, left: 0,
              width: '100%', height: '100%', pointerEvents: 'none',
            }}
          />
        )}
        {/* HTML fallback boxes for non-count scene description */}
        {!isCountQuery && sceneLabels && (
          <>
            <div className="grounding-box box-wetland" style={{ top: '8%', left: '8%', width: '35%', height: '30%' }}>
              <span className="box-tag">{sceneLabels[0]}</span>
            </div>
            <div className="grounding-box box-port" style={{ top: '42%', left: '45%', width: '25%', height: '22%' }}>
              <span className="box-tag">{sceneLabels[1]}</span>
            </div>
          </>
        )}
        {/* Count summary badge */}
        {isCountQuery && (
          <div style={{
            position: 'absolute', top: 8, right: 8,
            background: 'rgba(14,165,233,0.92)', color: '#fff',
            borderRadius: 8, padding: '4px 12px',
            fontFamily: 'Outfit, sans-serif', fontSize: 13, fontWeight: 700,
            boxShadow: '0 0 12px #38bdf888', backdropFilter: 'blur(4px)',
          }}>
            {count} {detected.label}{count !== 1 ? 's' : ''} detected
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Component ───────────────────────────────────────────────────────────────
export default function SatQueryWorkspace({ user, onLogout }) {
  const [recents, setRecents]               = useState([]);
  const [pinnedItems, setPinnedItems]       = useState(() => {
    try { return JSON.parse(localStorage.getItem(`bytex_pinned_${user?.uid}`) || '[]'); } catch { return []; }
  });
  const [currentQuery, setCurrentQuery]     = useState('');
  const [uploadedImages, setUploadedImages] = useState([]);
  const [uploadedFiles, setUploadedFiles]   = useState([]);   // raw File objects for Storage upload
  const [analysisResult, setAnalysisResult] = useState(null);
  const [isAnalyzing, setIsAnalyzing]       = useState(false);
  const [queryStatus, setQueryStatus]       = useState(null);  // null | uploading | pending | processing | done | error | timeout
  const [queryError, setQueryError]         = useState(null);
  const [sliderPosition, setSliderPosition] = useState(50);
  const [activeTab, setActiveTab]           = useState('evidence');
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [isDragging, setIsDragging]         = useState(false);
  const [dragFileCount, setDragFileCount]   = useState(0);
  const [dragIsInvalid, setDragIsInvalid]   = useState(false);
  const [activeCrossmodalLayer, setActiveCrossmodalLayer] = useState('optical');
  const [hoveredRecent, setHoveredRecent]   = useState(null);
  const [selectedRecentId, setSelectedRecentId] = useState(null);
  const [isListening, setIsListening]       = useState(false);
  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false);
  const [isProfileOpen, setIsProfileOpen]   = useState(false);

  const fileInputRef     = useRef(null);
  const unsubListenerRef = useRef(null);   // Firebase listener cleanup
  const timeoutRef       = useRef(null);   // result timeout handle
  const recognitionRef   = useRef(null);   // Web Speech API instance
  const dragCounterRef   = useRef(0);      // drag-enter counter to prevent flicker

  // ─── Recents subscription ─────────────────────────────────────────────────
  useEffect(() => {
    if (!user?.uid) return;
    const unsubscribe = subscribeToUserRecents(user.uid, (data) => {
      setRecents(data);
    });
    return () => unsubscribe();
  }, [user]);

  // ─── Persist pinned ──────────────────────────────────────────────────────
  useEffect(() => {
    if (user?.uid) {
      localStorage.setItem(`bytex_pinned_${user.uid}`, JSON.stringify(pinnedItems));
    }
  }, [pinnedItems, user]);

  // ─── Cleanup listeners on unmount ────────────────────────────────────────
  useEffect(() => {
    return () => {
      unsubListenerRef.current?.();
      clearTimeout(timeoutRef.current);
      recognitionRef.current?.abort();
    };
  }, []);

  // ─── Speech-to-Text (Web Speech API) ──────────────────────────────────────────
  const startListening = useCallback(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      alert('Speech recognition is not supported in this browser. Please use Chrome or Edge.');
      return;
    }
    const recognition = new SpeechRecognition();
    recognition.lang = 'en-US';
    recognition.continuous = false;
    recognition.interimResults = true;

    recognition.onstart  = () => setIsListening(true);
    recognition.onend    = () => setIsListening(false);
    recognition.onerror  = () => setIsListening(false);
    recognition.onresult = (e) => {
      const transcript = Array.from(e.results)
        .map(r => r[0].transcript)
        .join('');
      setCurrentQuery(transcript);
    };

    recognitionRef.current = recognition;
    recognition.start();
  }, []);

  const stopListening = useCallback(() => {
    recognitionRef.current?.stop();
    setIsListening(false);
  }, []);

  const handleMicClick = () => {
    if (isListening) stopListening();
    else startListening();
  };

  // ─── Pin helpers ─────────────────────────────────────────────────────────
  const togglePin = (item) => {
    const isPinned = pinnedItems.some(p => p.id === item.id);
    if (isPinned) {
      setPinnedItems(prev => prev.filter(p => p.id !== item.id));
    } else {
      setPinnedItems(prev => [item, ...prev]);
    }
  };
  const isPinned = (id) => pinnedItems.some(p => p.id === id);

  // ─── Accepted file types ─────────────────────────────────────────────────
  const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp', 'image/gif', 'image/tiff'];
  const ACCEPTED_EXTS  = ['.jpg', '.jpeg', '.png', '.webp', '.gif', '.tif', '.tiff'];

  const isAcceptedFile = (file) => {
    const mimeOk = ACCEPTED_TYPES.includes(file.type);
    const extOk  = ACCEPTED_EXTS.some(ext => file.name.toLowerCase().endsWith(ext));
    return mimeOk || extOk;
  };

  // ─── Drag & Drop ─────────────────────────────────────────────────────────
  // Use a counter to prevent false drag-leave events firing when the pointer
  // moves over child elements inside the drop zone.
  const handleDragEnter = (e) => {
    e.preventDefault();
    dragCounterRef.current += 1;
    if (dragCounterRef.current === 1) {
      const items = Array.from(e.dataTransfer.items || []);
      const count = items.filter(i => i.kind === 'file').length;
      const hasInvalid = items.some(
        i => i.kind === 'file' && !ACCEPTED_TYPES.includes(i.type) && i.type !== ''
      );
      setDragFileCount(count);
      setDragIsInvalid(hasInvalid && count > 0);
      setIsDragging(true);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    // Required to allow drop
    e.dataTransfer.dropEffect = dragIsInvalid ? 'none' : 'copy';
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    dragCounterRef.current -= 1;
    if (dragCounterRef.current === 0) {
      setIsDragging(false);
      setDragFileCount(0);
      setDragIsInvalid(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    dragCounterRef.current = 0;
    setIsDragging(false);
    setDragFileCount(0);
    setDragIsInvalid(false);
    if (e.dataTransfer.files?.length > 0) processSelectedFiles(e.dataTransfer.files);
  };

  // ─── Read a single file as data URL (Promise-based) ──────────────────────
  const readFileAsDataURL = (file) =>
    new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload  = (e) => resolve(e.target.result);
      reader.onerror = ()  => reject(new Error(`Failed to read ${file.name}`));
      reader.readAsDataURL(file);
    });

  // ─── Process selected / dropped files ────────────────────────────────────
  const processSelectedFiles = async (files) => {
    const fileList = Array.from(files);

    // Filter to accepted types; silently skip unsupported files
    const validFiles = fileList.filter(isAcceptedFile);
    if (validFiles.length === 0) return;

    try {
      // Read all files in parallel — eliminates the race-condition
      const dataUrls = await Promise.all(validFiles.map(readFileAsDataURL));

      const newImages = validFiles.map((file, i) => {
        const nameLower = file.name.toLowerCase();
        const isSAR    = nameLower.includes('sar') || nameLower.includes('risat');
        const isBefore = nameLower.includes('before') || nameLower.includes('t1');
        const isAfter  = nameLower.includes('after')  || nameLower.includes('t2');
        let modality   = 'Optical';
        if (isSAR)         modality = 'SAR Microwave';
        else if (isBefore) modality = 'Bi-Temporal T1 (Before)';
        else if (isAfter)  modality = 'Bi-Temporal T2 (After)';
        return { name: file.name, url: dataUrls[i], modality, size: (file.size / 1024).toFixed(1) + ' KB' };
      });

      setUploadedImages((prev) => [...prev, ...newImages].slice(0, 4));
      setUploadedFiles((prev)  => [...prev, ...validFiles].slice(0, 4));
    } catch (err) {
      console.error('Error reading files:', err);
    }
  };

  // ─── Paste to upload (Ctrl+V images) ─────────────────────────────────────
  useEffect(() => {
    const handlePaste = (e) => {
      const items = Array.from(e.clipboardData?.items || []);
      const imageFiles = items
        .filter(item => item.kind === 'file' && isAcceptedFile({ type: item.type, name: item.type.replace('/', '.') }))
        .map(item => item.getAsFile())
        .filter(Boolean);
      if (imageFiles.length > 0) processSelectedFiles(imageFiles);
    };
    window.addEventListener('paste', handlePaste);
    return () => window.removeEventListener('paste', handlePaste);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleLoadPreset = (preset) => {
    setUploadedImages(preset.images);
    setUploadedFiles([]);   // preset images are URLs, no raw files
    setCurrentQuery(preset.query);
    setShowUploadModal(false);
  };

  const handleNewAnalysis = () => {
    // Stop any running listener / timeout
    unsubListenerRef.current?.();
    clearTimeout(timeoutRef.current);

    setUploadedImages([]);
    setUploadedFiles([]);
    setCurrentQuery('');
    setAnalysisResult(null);
    setQueryStatus(null);
    setQueryError(null);
    setIsAnalyzing(false);
    setSelectedRecentId(null);
  };

  const handleSelectRecent = (item) => {
    setCurrentQuery(item.query || '');
    setUploadedImages(item.images || []);
    setUploadedFiles([]);
    setAnalysisResult(item.result || null);
    setQueryStatus(null);
    setQueryError(null);
    setSelectedRecentId(item.id);
    setIsMobileSidebarOpen(false); // close drawer on mobile after selection
  };

  const handleDeleteRecent = async (e, id) => {
    e.stopPropagation();
    if (user?.uid) await deleteRecentQuery(user.uid, id);
  };

  // ─── Download report ─────────────────────────────────────────────────────
  const handleDownloadReport = () => {
    if (!analysisResult) return;
    const reportContent = `# SatQuery AI - Analysis Report\nDate: ${new Date().toLocaleString()}\n\n## Query:\n"${analysisResult.query}"\n\n## Task: ${analysisResult.taskName}\n## Model: ${analysisResult.backendModel}\n## Confidence: ${analysisResult.confidence}%\n\n## Findings:\n${analysisResult.textResponse}\n\n## Metrics:\n${JSON.stringify(analysisResult.metrics || {}, null, 2)}`;
    const blob = new Blob([reportContent], { type: 'text/markdown' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = `SatQuery_Report_${Date.now()}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // ─── Determine task type from query + images ──────────────────────────────
  const detectTaskType = (query, images) => {
    const qLower      = query.toLowerCase();
    const isBiTemporal = images.length >= 2 && (qLower.includes('change') || qLower.includes('dates') || qLower.includes('between'));
    const isCrossModal = images.some(img => img.modality?.includes('SAR')) || (qLower.includes('sar') && qLower.includes('optical'));
    if (isBiTemporal) return { taskType: 'bitemporal', taskName: 'Change Detection & VQA',      modelName: 'Qwen2.5-VL-3B-RS + ChangeSiam', fbTask: 'vqa' };
    if (isCrossModal) return { taskType: 'crossmodal', taskName: 'Optical-SAR Joint Extraction', modelName: 'Qwen2.5-VL-3B-RS + FusionNet',  fbTask: 'vqa' };
    return               { taskType: 'single',     taskName: 'Single-Image VQA & Grounding', modelName: 'Qwen2.5-VL-3B-RS',              fbTask: 'vqa' };
  };

  // ─── Build result payload from Firebase answer ────────────────────────────
  const buildResultPayload = (fbData, taskInfo) => ({
    title:        currentQuery.length > 40 ? currentQuery.substring(0, 40) + '…' : currentQuery || 'Satellite Analysis',
    query:        currentQuery,
    taskType:     taskInfo.taskType,
    taskName:     taskInfo.taskName,
    backendModel: taskInfo.modelName,
    confidence:   fbData.confidence != null ? (fbData.confidence * 100).toFixed(1) : '—',
    textResponse: fbData.answer || 'No answer returned.',
    metrics:      fbData.metrics || null,
  });

  // ─── MAIN ANALYSIS PIPELINE ──────────────────────────────────────────────
  const handleRunAnalysis = async () => {
    if (isAnalyzing) return;
    if (!currentQuery.trim() && uploadedImages.length === 0) return;

    // Stop previous listener
    unsubListenerRef.current?.();
    clearTimeout(timeoutRef.current);

    setIsAnalyzing(true);
    setAnalysisResult(null);
    setQueryError(null);

    const taskInfo = detectTaskType(currentQuery, uploadedImages);
    const primaryFile = uploadedFiles[0] || null;

    // ── If preset (no raw file), run demo mode ────────────────────────────
    if (!primaryFile) {
      setQueryStatus('pending');
      await _runDemoMode(taskInfo);
      return;
    }

    // ── Real pipeline via Firebase RTDB (base64 — no Storage needed) ──────
    try {
      // Step 1 — Encode image to base64 (resized to ≤1024px, JPEG 85%)
      setQueryStatus('encoding');
      const imageBase64 = await fileToBase64(primaryFile, 1024);

      // Step 2 — Write query + base64 image to RTDB
      setQueryStatus('pending');
      const queryId = await submitQueryToFirebase(
        user?.uid || 'anonymous',
        imageBase64,
        currentQuery || 'Describe this satellite image.',
        taskInfo.fbTask
      );


      // Step 3 — Listen for result
      const unsub = listenForQueryResult(queryId, {
        onStatusChange: (status) => {
          if (status && status !== 'done' && status !== 'error') {
            setQueryStatus(status);
          }
        },
        onResult: async (fbData) => {
          clearTimeout(timeoutRef.current);
          unsubListenerRef.current?.();
          setQueryStatus('done');

          const resultPayload = buildResultPayload(fbData, taskInfo);
          setAnalysisResult(resultPayload);
          setIsAnalyzing(false);

          // Save to recents
          if (user?.uid) {
            await saveRecentQuery(user.uid, {
              id:        `query_${Date.now()}`,
              title:     resultPayload.title,
              query:     currentQuery,
              taskType:  taskInfo.taskType,
              images:    uploadedImages.map(img => ({ name: img.name, modality: img.modality, url: img.url })),
              result:    resultPayload,
              timestamp: Date.now(),
            });
          }
        },
        onError: (errMsg) => {
          clearTimeout(timeoutRef.current);
          setQueryStatus('error');
          setQueryError(errMsg || 'Unknown error from backend.');
          setIsAnalyzing(false);
        },
      });
      unsubListenerRef.current = unsub;

      // Step 4 — Timeout guard
      timeoutRef.current = setTimeout(() => {
        unsubListenerRef.current?.();
        if (queryStatus !== 'done') {
          setQueryStatus('timeout');
          setQueryError('The backend did not respond within 2 minutes. Make sure it is running and Firebase is connected.');
          setIsAnalyzing(false);
        }
      }, RESULT_TIMEOUT_MS);

    } catch (err) {
      console.error('Analysis pipeline error:', err);
      setQueryStatus('error');
      setQueryError(err.message || 'Failed to start analysis.');
      setIsAnalyzing(false);
    }
  };

  // ─── Demo / preset mode (no real file uploaded) ───────────────────────────
  const _runDemoMode = async (taskInfo) => {
    return new Promise((resolve) => {
      setTimeout(async () => {
        setQueryStatus('processing');
        setTimeout(async () => {
          const matchedPreset = PRESET_DATASETS.find(p => p.type === taskInfo.taskType) || PRESET_DATASETS[0];
          const resultPayload = {
            title:        currentQuery.length > 40 ? currentQuery.substring(0, 40) + '…' : currentQuery || 'Satellite Analysis',
            query:        currentQuery,
            taskType:     taskInfo.taskType,
            taskName:     taskInfo.taskName,
            backendModel: taskInfo.modelName + ' (Demo)',
            confidence:   (94.5 + Math.random() * 4.5).toFixed(1),
            textResponse: matchedPreset.response,
            metrics:      matchedPreset.metrics,
            isDemo:       true,
          };
          setQueryStatus('done');
          setAnalysisResult(resultPayload);
          setIsAnalyzing(false);

          if (user?.uid) {
            await saveRecentQuery(user.uid, {
              id:        `query_${Date.now()}`,
              title:     resultPayload.title,
              query:     currentQuery,
              taskType:  taskInfo.taskType,
              images:    uploadedImages.map(img => ({ name: img.name, modality: img.modality, url: img.url })),
              result:    resultPayload,
              timestamp: Date.now(),
            });
          }
          resolve();
        }, 1400);
      }, 800);
    });
  };

  // ─── Retry handler ────────────────────────────────────────────────────────
  const handleRetry = () => {
    setQueryStatus(null);
    setQueryError(null);
    setAnalysisResult(null);
    handleRunAnalysis();
  };

  // ─── Filter unpinned recents ─────────────────────────────────────────────
  const unpinnedRecents = recents.filter(r => !isPinned(r.id));

  // ─── Relative time helper ────────────────────────────────────────────────
  const getRelativeTime = (ts) => {
    if (!ts) return '';
    const diff = Date.now() - ts;
    const m = Math.floor(diff / 60000);
    if (m < 1)  return 'just now';
    if (m < 60) return `${m}m ago`;
    const h = Math.floor(m / 60);
    if (h < 24) return `${h}h ago`;
    const d = Math.floor(h / 24);
    if (d === 1) return 'yesterday';
    if (d < 7)  return `${d}d ago`;
    return new Date(ts).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
  };

  // ─── Group recents by date ───────────────────────────────────────────────
  const groupRecentsByDate = (items) => {
    const now   = Date.now();
    const today = new Date().setHours(0,0,0,0);
    const yest  = today - 86400000;
    const groups = { Today: [], Yesterday: [], Older: [] };
    items.forEach(item => {
      const d = item.timestamp || 0;
      if (d >= today)     groups.Today.push(item);
      else if (d >= yest) groups.Yesterday.push(item);
      else                groups.Older.push(item);
    });
    return groups;
  };

  // ─── Task-type dot color ─────────────────────────────────────────────────
  const taskDotColor = (taskType) => {
    if (taskType === 'bitemporal') return '#f97316';
    if (taskType === 'crossmodal') return '#a78bfa';
    return '#38bdf8';
  };

  // ─── Status Step Icon ─────────────────────────────────────────────────────
  const StatusIcon = ({ type, size = 16 }) => {
    if (type === 'spinner') return <Loader2 size={size} className="spin-icon" />;
    if (type === 'upload')  return <UploadCloud size={size} />;
    if (type === 'clock')   return <Clock size={size} />;
    if (type === 'check')   return <CheckCircle2 size={size} />;
    if (type === 'error')   return <AlertCircle size={size} />;
    return null;
  };

  const currentStatusDef = queryStatus ? STATUS_STEPS[queryStatus] : null;

  return (
    <div className="workspace-layout">
      {/* ─── MOBILE SIDEBAR OVERLAY ────────────────────────────────────── */}
      {isMobileSidebarOpen && (
        <div className="mobile-sidebar-overlay" onClick={() => setIsMobileSidebarOpen(false)} />
      )}

      {/* ─── LEFT SIDEBAR ─────────────────────────────────────────────── */}
      <aside className={`workspace-sidebar${isMobileSidebarOpen ? ' sidebar-open' : ''}`}>
        <div className="sidebar-top">
          <div className="sidebar-brand">
            <span className="sidebar-brand-text">ByteX</span>
          </div>
          <button
            className="sidebar-mobile-close"
            onClick={() => setIsMobileSidebarOpen(false)}
            aria-label="Close sidebar"
          >
            <X size={18} />
          </button>
          <button className="new-analysis-btn" onClick={handleNewAnalysis}>
            <Plus size={18} />
            <span>New analysis</span>
          </button>
        </div>

        {/* Pinned Section */}
        {pinnedItems.length > 0 && (
          <div className="sidebar-section">
            <div className="section-label">Pinned</div>
            <div className="section-list">
              {pinnedItems.map((item) => (
                <div
                  key={item.id}
                  className="sidebar-item"
                  onClick={() => handleSelectRecent(item)}
                  onMouseEnter={() => setHoveredRecent(item.id)}
                  onMouseLeave={() => setHoveredRecent(null)}
                >
                  <span className="sidebar-item-title">{item.title || item.query}</span>
                  {hoveredRecent === item.id && (
                    <div className="sidebar-item-actions">
                      <button className="action-icon-btn" onClick={(e) => { e.stopPropagation(); togglePin(item); }} title="Unpin">
                        <PinOff size={14} />
                      </button>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Recents Section */}
        <div className="sidebar-section recents-section-grow">
          <div className="section-label">History</div>
          <div className="section-list">
            {unpinnedRecents.length === 0 && pinnedItems.length === 0 ? (
              <div className="empty-recents">
                <Compass size={22} className="empty-recents-icon" />
                <p>No queries yet</p>
                <p>Upload a satellite image and ask your first question to get started.</p>
              </div>
            ) : unpinnedRecents.length === 0 ? (
              <div className="empty-recents"><p>All items pinned</p></div>
            ) : (
              (() => {
                const groups = groupRecentsByDate(unpinnedRecents);
                return Object.entries(groups)
                  .filter(([, items]) => items.length > 0)
                  .map(([label, items]) => (
                    <div key={label}>
                      <div className="sidebar-date-group">{label}</div>
                      {items.map((item) => (
                        <div
                          key={item.id}
                          className={`sidebar-item ${selectedRecentId === item.id ? 'active' : ''}`}
                          onClick={() => handleSelectRecent(item)}
                          onMouseEnter={() => setHoveredRecent(item.id)}
                          onMouseLeave={() => setHoveredRecent(null)}
                        >
                          <div className="sidebar-item-main">
                            <span
                              className="sidebar-item-type-dot"
                              style={{ background: taskDotColor(item.taskType), boxShadow: `0 0 5px ${taskDotColor(item.taskType)}80` }}
                            />
                            <div className="sidebar-item-text">
                              <span className="sidebar-item-title">{item.title || item.query}</span>
                              <span className="sidebar-item-time">{getRelativeTime(item.timestamp)}</span>
                            </div>
                          </div>
                          {hoveredRecent === item.id && (
                            <div className="sidebar-item-actions">
                              <button className="action-icon-btn" onClick={(e) => { e.stopPropagation(); togglePin(item); }} title="Pin">
                                <Pin size={14} />
                              </button>
                              <button className="action-icon-btn" onClick={(e) => handleDeleteRecent(e, item.id)} title="Delete">
                                <Trash2 size={14} />
                              </button>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  ));
              })()
            )}
          </div>
        </div>

        {/* User footer — click to open profile page */}
        <div
          className="sidebar-user-footer clickable-footer"
          onClick={() => { setIsMobileSidebarOpen(false); setIsProfileOpen(true); }}
          title="View profile"
          role="button"
          tabIndex={0}
          onKeyDown={(e) => e.key === 'Enter' && setIsProfileOpen(true)}
        >
          <div className="user-info-card">
            <img
              src={user?.photoURL || `https://api.dicebear.com/7.x/bottts/svg?seed=${user?.uid || 'bytex'}`}
              alt="User"
              className="user-avatar"
            />
            <span className="user-name">{user?.displayName || user?.email?.split('@')[0] || 'Analyst'}</span>
          </div>
          <ChevronRight size={16} className="footer-chevron" />
        </div>
      </aside>

      {/* ─── MAIN WORKSPACE ───────────────────────────────────────────── */}
      <main className="workspace-main" onDragEnter={handleDragEnter} onDragOver={handleDragOver} onDragLeave={handleDragLeave} onDrop={handleDrop}>
        {/* Mobile top bar */}
        <div className="mobile-topbar">
          <button
            className="hamburger-btn"
            onClick={() => setIsMobileSidebarOpen(true)}
            aria-label="Open sidebar"
          >
            <Menu size={22} />
          </button>
          <span className="mobile-topbar-brand">ByteX</span>
          <div style={{ width: 36 }} />
        </div>

        <StarField />
        <div className="workspace-content-center">

          {/* Center input zone */}
          <div className={`center-input-zone ${analysisResult || isAnalyzing || queryStatus ? 'pushed-top' : ''}`}>
            {!analysisResult && !isAnalyzing && !queryStatus && (
              <div className="welcome-msg">
                <h1 className="welcome-title">What would you like to analyze?</h1>
              </div>
            )}

            {/* Query Input Bar */}
            <div className="query-box-wrapper">
              {uploadedImages.length > 0 && (
                <div className="uploaded-chips-row">
                  {uploadedImages.map((img, index) => (
                    <div key={index} className="uploaded-chip">
                      <img src={img.url} alt={img.name} className="chip-thumbnail" />
                      <div className="chip-info">
                        <span className="chip-name">{img.name}</span>
                        <span className="chip-modality">{img.modality}</span>
                      </div>
                      <button
                        className="chip-remove-btn"
                        onClick={() => {
                          setUploadedImages(prev => prev.filter((_, i) => i !== index));
                          setUploadedFiles(prev => prev.filter((_, i) => i !== index));
                        }}
                      >×</button>
                    </div>
                  ))}
                </div>
              )}

              <div className="query-input-bar">
                <button className="add-media-btn" title="Upload imagery (+)" onClick={() => setShowUploadModal(true)}>
                  <Plus size={22} />
                </button>

                <input
                  type="file"
                  ref={fileInputRef}
                  multiple
                  accept="image/*,.tif,.tiff"
                  style={{ display: 'none' }}
                  onChange={(e) => { if (e.target.files) processSelectedFiles(e.target.files); }}
                />

                <input
                  type="text"
                  className="query-text-input"
                  placeholder={isListening ? 'Listening… speak your query' : 'Ask about satellite imagery…'}
                  value={currentQuery}
                  onChange={(e) => setCurrentQuery(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleRunAnalysis(); } }}
                  disabled={isAnalyzing}
                />

                {/* Mic when no text, Send when text is typed */}
                {(currentQuery.trim() || uploadedImages.length > 0) ? (
                  <button
                    className={`send-query-btn ${!isAnalyzing ? 'active' : ''}`}
                    onClick={handleRunAnalysis}
                    disabled={isAnalyzing}
                    title="Run analysis"
                  >
                    <Send size={18} />
                  </button>
                ) : (
                  <button
                    className={`mic-btn ${isListening ? 'listening' : ''}`}
                    onClick={handleMicClick}
                    title={isListening ? 'Stop listening' : 'Speak your query'}
                  >
                    {isListening ? <MicOff size={18} /> : <Mic size={18} />}
                  </button>
                )}
              </div>
              {isListening && (
                <div className="listening-hint">● Recording… pause when done to auto-fill</div>
              )}
            </div>

            {/* Quick preset shortcuts */}
            {!analysisResult && !isAnalyzing && !queryStatus && (
              <div className="quick-presets-row">
                {PRESET_DATASETS.map((preset) => (
                  <button key={preset.id} className="preset-chip" onClick={() => handleLoadPreset(preset)}>
                    <span className={`preset-dot dot-${preset.type}`}></span>
                    <span>{preset.title}</span>
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* ── Status Pipeline Bar ──────────────────────────────────────── */}
          {queryStatus && queryStatus !== 'done' && (
            <div className="status-pipeline-bar">
              <div className="status-pipeline-steps">
                {['encoding', 'pending', 'processing'].map((step) => {
                  const stepIdx   = ['encoding', 'pending', 'processing'].indexOf(step);
                  const curIdx    = ['encoding', 'pending', 'processing'].indexOf(queryStatus);
                  const isDone    = stepIdx < curIdx || queryStatus === 'done';
                  const isCurrent = stepIdx === curIdx && queryStatus !== 'error' && queryStatus !== 'timeout';
                  const isError   = (queryStatus === 'error' || queryStatus === 'timeout') && stepIdx === curIdx;
                  return (
                    <div key={step} className={`pipeline-step ${isDone ? 'done' : ''} ${isCurrent ? 'current' : ''} ${isError ? 'errored' : ''}`}>
                      <div className="step-dot">
                        {isDone ? <CheckCircle2 size={13} /> : isCurrent ? <Loader2 size={13} className="spin-icon" /> : isError ? <AlertCircle size={13} /> : null}
                      </div>
                      <span className="step-label">{STATUS_STEPS[step]?.label || step}</span>
                    </div>
                  );
                })}
              </div>

              {/* Error / timeout message */}
              {(queryStatus === 'error' || queryStatus === 'timeout') && queryError && (
                <div className="status-error-row">
                  <AlertCircle size={15} />
                  <span>{queryError}</span>
                  <button className="retry-btn" onClick={handleRetry}>
                    <RefreshCw size={13} /> Retry
                  </button>
                </div>
              )}
            </div>
          )}

          {/* ── Processing spinner ───────────────────────────────────────── */}
          {isAnalyzing && queryStatus === 'processing' && (
            <div className="analyzing-state">
              <div className="analyzing-spinner"></div>
              <span className="analyzing-text">Qwen2.5-VL analyzing your image…</span>
            </div>
          )}

          {/* ── Results ──────────────────────────────────────────────────── */}
          {analysisResult && !isAnalyzing && (
            <div className="analysis-result-below">
              {/* Demo badge */}
              {analysisResult.isDemo && (
                <div className="demo-notice">
                  <ImageIcon size={14} />
                  <span>Demo response — upload a real satellite image to use Qwen VLM</span>
                </div>
              )}

              {/* Tab Navigation */}
              <div className="result-tabs-nav">
                <button className={`tab-btn ${activeTab === 'evidence' ? 'active' : ''}`} onClick={() => setActiveTab('evidence')}>
                  <Eye size={16} />
                  <span>Visual Evidence & Viewer</span>
                </button>
                <button className={`tab-btn ${activeTab === 'report' ? 'active' : ''}`} onClick={() => setActiveTab('report')}>
                  <FileText size={16} />
                  <span>Textual Analysis</span>
                </button>
                <div className="tab-spacer"></div>
                <div className="result-meta">
                  <span className="conf-badge">{analysisResult.confidence}% confidence</span>
                  <button className="download-btn" onClick={handleDownloadReport} title="Download Report">
                    <Download size={15} />
                  </button>
                </div>
              </div>

              {/* Tab: Visual Evidence */}
              {activeTab === 'evidence' && (
                <div className="visual-evidence-pane">
                  {analysisResult.taskType === 'bitemporal' && uploadedImages.length >= 2 && (
                    <div className="bitemporal-split-viewer">
                      <div className="split-image-wrapper">
                        <img src={uploadedImages[1]?.url || '/samples/bitemporal_after.jpg'} alt="After" className="split-img base-img" />
                        <span className="split-label label-after">After (2024)</span>
                        <div className="split-overlay-clipped" style={{ clipPath: `polygon(0 0, ${sliderPosition}% 0, ${sliderPosition}% 100%, 0 100%)` }}>
                          <img src={uploadedImages[0]?.url || '/samples/bitemporal_before.jpg'} alt="Before" className="split-img overlay-img" />
                          <span className="split-label label-before">Before (2004)</span>
                        </div>
                        <div className="slider-divider-handle" style={{ left: `${sliderPosition}%` }}>
                          <div className="divider-line-v"></div>
                          <div className="slider-button"><Sliders size={16} /></div>
                        </div>
                        <input type="range" min="0" max="100" value={sliderPosition} onChange={(e) => setSliderPosition(e.target.value)} className="invisible-range-slider" />
                      </div>
                    </div>
                  )}

                  {analysisResult.taskType === 'crossmodal' && (
                    <div className="crossmodal-viewer">
                      <div className="layer-controls">
                        {['optical', 'sar', 'fusion'].map(layer => (
                          <button key={layer} className={`layer-toggle-btn ${activeCrossmodalLayer === layer ? 'active' : ''}`}
                            onClick={() => setActiveCrossmodalLayer(layer)}>
                            {layer === 'optical' ? 'Optical RGB' : layer === 'sar' ? 'SAR Backscatter' : 'Fused Overlay'}
                          </button>
                        ))}
                      </div>
                      <div className="crossmodal-image-display">
                        <img
                          src={activeCrossmodalLayer === 'sar' ? (uploadedImages[1]?.url || '/samples/sar_sample.jpg') : (uploadedImages[0]?.url || '/samples/optical_sample.jpg')}
                          alt="Crossmodal scene"
                          className={`crossmodal-img ${activeCrossmodalLayer === 'fusion' ? 'fusion-mode' : ''}`}
                        />
                      </div>
                    </div>
                  )}

                  {analysisResult.taskType === 'single' && (
                    <SmartGroundingViewer
                      imageUrl={uploadedImages[0]?.url || '/samples/optical_sample.jpg'}
                      query={analysisResult.query || ''}
                      answer={analysisResult.textResponse || ''}
                    />
                  )}

                  {analysisResult.metrics && (
                    <div className="metrics-row">
                      {Object.entries(analysisResult.metrics).map(([key, val]) => (
                        <div key={key} className="metric-chip">
                          <span className="metric-key">{key.replace(/([A-Z])/g, ' $1')}</span>
                          <span className="metric-val">{val}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* Tab: Textual Analysis */}
              {activeTab === 'report' && (
                <div className="text-report-pane">
                  <div className="markdown-body" dangerouslySetInnerHTML={{
                    __html: analysisResult.textResponse
                      .replace(/^### (.*$)/gim, '<h3>$1</h3>')
                      .replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>')
                      .replace(/^- (.*$)/gim, '<li>$1</li>')
                      .replace(/\n\n/g, '<br/>')
                  }} />
                </div>
              )}
            </div>
          )}
        </div>

        {/* Drag overlay */}
        {isDragging && (
          <div className={`drag-drop-overlay${dragIsInvalid ? ' drag-invalid' : ''}`}>
            <UploadCloud size={52} className="drop-icon" />
            <h3>{dragIsInvalid ? 'Unsupported file type' : 'Drop satellite imagery here'}</h3>
            <p>
              {dragIsInvalid
                ? 'Please use GeoTIFF, TIFF, PNG, JPEG or WebP'
                : dragFileCount > 0
                  ? `${dragFileCount} file${dragFileCount > 1 ? 's' : ''} ready to drop`
                  : 'GeoTIFF, TIFF, PNG, JPEG, WebP'}
            </p>
            {!dragIsInvalid && uploadedImages.length > 0 && uploadedImages.length < 4 && (
              <span className="drag-slots-hint">{4 - uploadedImages.length} slot{4 - uploadedImages.length > 1 ? 's' : ''} remaining</span>
            )}
          </div>
        )}
      </main>

      {/* Upload Modal */}
      {showUploadModal && (
        <div className="modal-backdrop" onClick={() => setShowUploadModal(false)}>
          <div className="upload-options-card" onClick={(e) => e.stopPropagation()}>
            <div className="upload-options-header">
              <h3>Upload Satellite Imagery</h3>
              <button className="close-modal-btn" onClick={() => setShowUploadModal(false)}>×</button>
            </div>
            <div className="options-grid">
              <button className="option-tile" onClick={() => { setShowUploadModal(false); fileInputRef.current?.click(); }}>
                <UploadCloud size={28} className="tile-icon" />
                <span className="tile-title">Upload Local</span>
                <span className="tile-desc">GeoTIFF, TIFF, PNG, JPEG</span>
              </button>
              <button className="option-tile" onClick={() => handleLoadPreset(PRESET_DATASETS[0])}>
                <Layers size={28} className="tile-icon icon-blue" />
                <span className="tile-title">Bi-Temporal Pair</span>
                <span className="tile-desc">Before & After (Demo)</span>
              </button>
              <button className="option-tile" onClick={() => handleLoadPreset(PRESET_DATASETS[1])}>
                <Compass size={28} className="tile-icon icon-cyan" />
                <span className="tile-title">Optical + SAR</span>
                <span className="tile-desc">Cartosat-2S & RISAT (Demo)</span>
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ─── PROFILE PAGE ─────────────────────────────────────────────── */}
      {isProfileOpen && (
        <ProfilePanel
          user={user}
          recentsCount={recents.length}
          pinnedCount={pinnedItems.length}
          onLogout={onLogout}
          onClose={() => setIsProfileOpen(false)}
        />
      )}
    </div>
  );
}
