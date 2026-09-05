import React, { useState, useEffect, useRef } from 'react';
import { 
  Plus, 
  Send, 
  Trash2, 
  Clock, 
  Layers, 
  Image as ImageIcon, 
  FileText, 
  Download, 
  Satellite, 
  Compass, 
  LogOut, 
  Eye,
  UploadCloud,
  Pin,
  PinOff,
  Sliders,
  MoreHorizontal
} from 'lucide-react';
import { 
  saveRecentQuery, 
  subscribeToUserRecents, 
  deleteRecentQuery,
  logoutUser 
} from '../firebase/firebaseConfig';
import './SatQueryWorkspace.css';

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

export default function SatQueryWorkspace({ user, onLogout }) {
  const [recents, setRecents] = useState([]);
  const [pinnedItems, setPinnedItems] = useState(() => {
    try { return JSON.parse(localStorage.getItem(`bytex_pinned_${user?.uid}`) || '[]'); } catch { return []; }
  });
  const [currentQuery, setCurrentQuery] = useState('');
  const [uploadedImages, setUploadedImages] = useState([]);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [sliderPosition, setSliderPosition] = useState(50);
  const [activeTab, setActiveTab] = useState('evidence');
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [activeCrossmodalLayer, setActiveCrossmodalLayer] = useState('optical');
  const [hoveredRecent, setHoveredRecent] = useState(null);

  const fileInputRef = useRef(null);

  useEffect(() => {
    if (!user?.uid) return;
    const unsubscribe = subscribeToUserRecents(user.uid, (data) => {
      setRecents(data);
    });
    return () => unsubscribe();
  }, [user]);

  // Save pinned items to localStorage
  useEffect(() => {
    if (user?.uid) {
      localStorage.setItem(`bytex_pinned_${user.uid}`, JSON.stringify(pinnedItems));
    }
  }, [pinnedItems, user]);

  const togglePin = (item) => {
    const isPinned = pinnedItems.some(p => p.id === item.id);
    if (isPinned) {
      setPinnedItems(prev => prev.filter(p => p.id !== item.id));
    } else {
      setPinnedItems(prev => [item, ...prev]);
    }
  };

  const isPinned = (id) => pinnedItems.some(p => p.id === id);

  const handleDragOver = (e) => { e.preventDefault(); setIsDragging(true); };
  const handleDragLeave = (e) => { e.preventDefault(); setIsDragging(false); };
  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files?.length > 0) processSelectedFiles(e.dataTransfer.files);
  };

  const processSelectedFiles = (files) => {
    const fileList = Array.from(files);
    const newImages = [];
    fileList.forEach((file) => {
      const isSAR = file.name.toLowerCase().includes('sar') || file.name.toLowerCase().includes('risat');
      const isBefore = file.name.toLowerCase().includes('before') || file.name.toLowerCase().includes('t1');
      const isAfter = file.name.toLowerCase().includes('after') || file.name.toLowerCase().includes('t2');
      let modality = 'Optical';
      if (isSAR) modality = 'SAR Microwave';
      else if (isBefore) modality = 'Bi-Temporal T1 (Before)';
      else if (isAfter) modality = 'Bi-Temporal T2 (After)';
      const reader = new FileReader();
      reader.onload = (e) => {
        newImages.push({ name: file.name, url: e.target.result, modality, size: (file.size / 1024).toFixed(1) + ' KB' });
        if (newImages.length === fileList.length) setUploadedImages((prev) => [...prev, ...newImages].slice(0, 4));
      };
      reader.readAsDataURL(file);
    });
  };

  const handleLoadPreset = (preset) => {
    setUploadedImages(preset.images);
    setCurrentQuery(preset.query);
    setShowUploadModal(false);
  };

  const handleNewAnalysis = () => {
    setUploadedImages([]);
    setCurrentQuery('');
    setAnalysisResult(null);
  };

  const handleSelectRecent = (item) => {
    setCurrentQuery(item.query || '');
    setUploadedImages(item.images || []);
    setAnalysisResult(item.result || null);
  };

  const handleDeleteRecent = async (e, id) => {
    e.stopPropagation();
    if (user?.uid) await deleteRecentQuery(user.uid, id);
  };

  const handleRunAnalysis = async () => {
    if (!currentQuery.trim() && uploadedImages.length === 0) return;
    setIsAnalyzing(true);

    const qLower = currentQuery.toLowerCase();
    const isBiTemporal = uploadedImages.length >= 2 && (qLower.includes('change') || qLower.includes('dates') || qLower.includes('between'));
    const isCrossModal = uploadedImages.some(img => img.modality?.includes('SAR')) || (qLower.includes('sar') && qLower.includes('optical'));

    let taskType = 'single';
    let taskName = 'Single-Image VQA & Grounding';
    let modelName = 'Qwen2.5-VL-3B-RS';

    if (isBiTemporal) { taskType = 'bitemporal'; taskName = 'Change Detection & VQA'; modelName = 'Qwen2.5-VL-3B-RS + ChangeSiam'; }
    else if (isCrossModal) { taskType = 'crossmodal'; taskName = 'Optical-SAR Joint Extraction'; modelName = 'Qwen2.5-VL-3B-RS + FusionNet'; }

    setTimeout(async () => {
      const matchedPreset = PRESET_DATASETS.find(p => p.type === taskType) || PRESET_DATASETS[0];
      const resultPayload = {
        title: currentQuery.length > 40 ? currentQuery.substring(0, 40) + '...' : currentQuery || 'Satellite Analysis',
        query: currentQuery,
        taskType,
        taskName,
        backendModel: modelName,
        confidence: (94.5 + Math.random() * 4.5).toFixed(1),
        textResponse: matchedPreset.response,
        metrics: matchedPreset.metrics,
      };

      setAnalysisResult(resultPayload);
      setIsAnalyzing(false);

      if (user?.uid) {
        await saveRecentQuery(user.uid, {
          id: `query_${Date.now()}`,
          title: resultPayload.title,
          query: currentQuery,
          taskType,
          images: uploadedImages.map(img => ({ name: img.name, modality: img.modality, url: img.url })),
          result: resultPayload,
          timestamp: Date.now()
        });
      }
    }, 1800);
  };

  const handleDownloadReport = () => {
    if (!analysisResult) return;
    const reportContent = `# SatQuery AI - Analysis Report\nDate: ${new Date().toLocaleString()}\n\n## Query:\n"${analysisResult.query}"\n\n## Task: ${analysisResult.taskName}\n## Model: ${analysisResult.backendModel}\n## Confidence: ${analysisResult.confidence}%\n\n## Findings:\n${analysisResult.textResponse}\n\n## Metrics:\n${JSON.stringify(analysisResult.metrics || {}, null, 2)}`;
    const blob = new Blob([reportContent], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `SatQuery_Report_${Date.now()}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Filter unpinned recents
  const unpinnedRecents = recents.filter(r => !isPinned(r.id));

  return (
    <div className="workspace-layout">
      {/* LEFT SIDEBAR: Like Image 4 - Pinned + Recents with hover pin */}
      <aside className="workspace-sidebar">
        <div className="sidebar-top">
          <div className="sidebar-brand">
            <span className="sidebar-brand-text">ByteX</span>
          </div>
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
          <div className="section-label">Recents</div>
          <div className="section-list">
            {unpinnedRecents.length === 0 && pinnedItems.length === 0 ? (
              <div className="empty-recents">
                <p>No recent queries yet</p>
              </div>
            ) : unpinnedRecents.length === 0 ? (
              <div className="empty-recents"><p>All items pinned</p></div>
            ) : (
              unpinnedRecents.map((item) => (
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
                      <button className="action-icon-btn" onClick={(e) => { e.stopPropagation(); togglePin(item); }} title="Pin">
                        <Pin size={14} />
                      </button>
                      <button className="action-icon-btn" onClick={(e) => handleDeleteRecent(e, item.id)} title="Delete">
                        <Trash2 size={14} />
                      </button>
                    </div>
                  )}
                </div>
              ))
            )}
          </div>
        </div>

        {/* User at bottom */}
        <div className="sidebar-user-footer">
          <div className="user-info-card">
            <img 
              src={user?.photoURL || `https://api.dicebear.com/7.x/bottts/svg?seed=${user?.uid || 'bytex'}`} 
              alt="User" 
              className="user-avatar" 
            />
            <span className="user-name">{user?.displayName || user?.email?.split('@')[0] || 'Analyst'}</span>
          </div>
          <button className="user-logout-btn" onClick={onLogout} title="Sign Out">
            <LogOut size={16} />
          </button>
        </div>
      </aside>

      {/* MAIN WORKSPACE */}
      <main className="workspace-main" onDragOver={handleDragOver} onDragLeave={handleDragLeave} onDrop={handleDrop}>

        {/* Center content: Input in the MIDDLE of the page */}
        <div className="workspace-content-center">

          {/* Welcome + Query Input (centered vertically when no result) */}
          <div className={`center-input-zone ${analysisResult || isAnalyzing ? 'pushed-top' : ''}`}>
            {!analysisResult && !isAnalyzing && (
              <div className="welcome-msg">
                <h1 className="welcome-title">What would you like to analyze?</h1>
              </div>
            )}

            {/* Query Input Bar - Always visible in center */}
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
                      <button className="chip-remove-btn" onClick={() => setUploadedImages(prev => prev.filter((_, i) => i !== index))}>×</button>
                    </div>
                  ))}
                </div>
              )}

              <div className="query-input-bar">
                <button className="add-media-btn" title="Upload imagery (+)" onClick={() => setShowUploadModal(true)}>
                  <Plus size={22} />
                </button>

                <input type="file" ref={fileInputRef} multiple accept="image/*,.tif,.tiff" style={{ display: 'none' }}
                  onChange={(e) => { if (e.target.files) processSelectedFiles(e.target.files); }}
                />

                <input 
                  type="text" 
                  className="query-text-input" 
                  placeholder="Ask about satellite imagery..."
                  value={currentQuery}
                  onChange={(e) => setCurrentQuery(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleRunAnalysis(); } }}
                />

                <button 
                  className={`send-query-btn ${(currentQuery.trim() || uploadedImages.length > 0) && !isAnalyzing ? 'active' : ''}`}
                  onClick={handleRunAnalysis}
                  disabled={isAnalyzing || (!currentQuery.trim() && uploadedImages.length === 0)}
                >
                  <Send size={18} />
                </button>
              </div>
            </div>

            {/* Quick preset shortcuts below input (only when no result) */}
            {!analysisResult && !isAnalyzing && (
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

          {/* Analyzing State */}
          {isAnalyzing && (
            <div className="analyzing-state">
              <div className="analyzing-spinner"></div>
              <span className="analyzing-text">Analyzing with Qwen2.5-VL-3B...</span>
            </div>
          )}

          {/* Results BELOW the input bar */}
          {analysisResult && !isAnalyzing && (
            <div className="analysis-result-below">
              {/* Tab Navigation: Only Evidence + Textual */}
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
                        <img src={activeCrossmodalLayer === 'sar' ? (uploadedImages[1]?.url || '/samples/sar_sample.jpg') : (uploadedImages[0]?.url || '/samples/optical_sample.jpg')}
                          alt="Crossmodal scene" className={`crossmodal-img ${activeCrossmodalLayer === 'fusion' ? 'fusion-mode' : ''}`} />
                      </div>
                    </div>
                  )}

                  {analysisResult.taskType === 'single' && (
                    <div className="single-grounding-viewer">
                      <div className="image-grounding-frame">
                        <img src={uploadedImages[0]?.url || '/samples/optical_sample.jpg'} alt="Scene" className="grounding-img" />
                        <div className="grounding-box box-port" style={{ top: '38%', left: '42%', width: '22%', height: '24%' }}>
                          <span className="box-tag">Port & Dock (0.97)</span>
                        </div>
                        <div className="grounding-box box-wetland" style={{ top: '6%', left: '10%', width: '38%', height: '34%' }}>
                          <span className="box-tag">Estuary Wetland (0.94)</span>
                        </div>
                      </div>
                    </div>
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
          <div className="drag-drop-overlay">
            <UploadCloud size={48} className="drop-icon" />
            <h3>Drop satellite imagery here</h3>
            <p>GeoTIFF, TIFF, PNG, JPEG</p>
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
                <span className="tile-desc">Before & After</span>
              </button>
              <button className="option-tile" onClick={() => handleLoadPreset(PRESET_DATASETS[1])}>
                <Compass size={28} className="tile-icon icon-cyan" />
                <span className="tile-title">Optical + SAR</span>
                <span className="tile-desc">Cartosat-2S & RISAT</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
