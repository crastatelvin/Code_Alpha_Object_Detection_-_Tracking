import React, { useState, useEffect, useRef } from 'react';
import { 
  Eye, 
  Tv, 
  Activity, 
  Settings, 
  TrendingUp, 
  TrendingDown, 
  Zap, 
  Sliders,
  CheckCircle,
  AlertTriangle
} from 'lucide-react';
import './App.css';

const AVAILABLE_CLASSES = [
  "person", "car", "bicycle", "motorcycle", "bus", 
  "truck", "dog", "cat", "traffic light", "stop sign"
];

function App() {
  const isLocalDev = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
  const apiBase = isLocalDev ? '' : 'http://localhost:8000';

  // Connection and telemetry states
  const [wsConnected, setWsConnected] = useState(false);
  const [telemetry, setTelemetry] = useState({
    fps: 0,
    in_count: 0,
    out_count: 0,
    active_tracks: 0
  });

  // Track configurations
  const [trackerType, setTrackerType] = useState('sort');
  const [confThreshold, setConfThreshold] = useState(0.35);
  const [selectedClasses, setSelectedClasses] = useState(['person', 'car']);
  const [videoSource, setVideoSource] = useState('videos/traffic_highway.mp4');

  // WebSocket reference
  const wsRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);

  // Connect to telemetry WebSocket
  useEffect(() => {
    function connect() {
      // Determine WebSocket URL based on current host (proxied by Vite or fallback to localhost:8000)
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsHost = isLocalDev ? window.location.host : 'localhost:8000';
      const wsUrl = `${protocol}//${wsHost}/ws`;
      
      console.log(`Connecting WebSocket to: ${wsUrl}`);
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log("WebSocket connection established.");
        setWsConnected(true);
        // Push initial config on connection
        sendConfig(ws, {
          tracker_type: trackerType,
          conf_threshold: confThreshold,
          filter_classes: selectedClasses,
          source: videoSource
        });
      };

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          if (message.type === 'telemetry') {
            setTelemetry(message.data);
          } else if (message.type === 'config_updated') {
            const cfg = message.config;
            setTrackerType(cfg.tracker_type);
            setConfThreshold(cfg.conf_threshold);
            setSelectedClasses(cfg.filter_classes);
            if (cfg.source) {
              setVideoSource(cfg.source);
            }
          }
        } catch (err) {
          console.error("Error parsing WebSocket message:", err);
        }
      };

      ws.onclose = () => {
        console.log("WebSocket connection closed. Attempting reconnect...");
        setWsConnected(false);
        reconnectTimeoutRef.current = setTimeout(connect, 3000);
      };

      ws.onerror = (err) => {
        console.error("WebSocket error:", err);
        ws.close();
      };
    }

    connect();

    // Clean up
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, []);

  // Helper to send configuration updates
  const sendConfig = (wsInstance, cfgPatch) => {
    const ws = wsInstance || wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({
        type: 'update_config',
        config: cfgPatch
      }));
    }
  };

  // Handle configuration changes
  const handleTrackerChange = (type) => {
    setTrackerType(type);
    sendConfig(null, {
      tracker_type: type,
      conf_threshold: confThreshold,
      filter_classes: selectedClasses,
      source: videoSource
    });
  };

  const handleConfChange = (e) => {
    const val = parseFloat(e.target.value);
    setConfThreshold(val);
    sendConfig(null, {
      tracker_type: trackerType,
      conf_threshold: val,
      filter_classes: selectedClasses,
      source: videoSource
    });
  };

  const toggleClass = (cls) => {
    let updated;
    if (selectedClasses.includes(cls)) {
      // Keep at least one class
      if (selectedClasses.length === 1) return;
      updated = selectedClasses.filter(c => c !== cls);
    } else {
      updated = [...selectedClasses, cls];
    }
    setSelectedClasses(updated);
    sendConfig(null, {
      tracker_type: trackerType,
      conf_threshold: confThreshold,
      filter_classes: updated,
      source: videoSource
    });
  };

  const handleSourceChange = (src) => {
    setVideoSource(src);
    sendConfig(null, {
      tracker_type: trackerType,
      conf_threshold: confThreshold,
      filter_classes: selectedClasses,
      source: src
    });
  };

  return (
    <div className="app-container">
      {/* Header section */}
      <header className="dashboard-header">
        <div className="header-title-section">
          <Eye className="header-icon" size={32} />
          <div>
            <h1>DETECT & TRACK</h1>
            <p>Real-Time Object Detection & Multi-Object Tracking Console</p>
          </div>
        </div>
        <div className="status-badge">
          <span className={`status-dot ${wsConnected ? 'connected' : 'disconnected'}`}></span>
          <span>{wsConnected ? 'LIVE FEED ACTIVE' : 'CONNECTION OFFLINE'}</span>
        </div>
      </header>

      {/* Main Grid */}
      <div className="dashboard-grid">
        {/* Left Side: Viewport Monitor */}
        <div className="viewport-panel">
          <div className="viewport-header">
            <div className="viewport-title">
              <Tv size={18} className="header-icon" />
              <span>PRIMARY SYSTEM VIEWPORT</span>
            </div>
            <div className="rec-indicator">
              <span className="rec-dot"></span>
              <span>STREAMING</span>
            </div>
          </div>
          
          <div className="viewport-monitor">
            {/* HUD Overlays */}
            <div className="viewport-hud">
              <div className="hud-corner hud-corner-tl"></div>
              <div className="hud-corner hud-corner-tr"></div>
              <div className="hud-corner hud-corner-bl"></div>
              <div className="hud-corner hud-corner-br"></div>
              <div className="hud-crosshair"></div>
            </div>

            {wsConnected ? (
              <img 
                src={`${apiBase}/api/video_feed`} 
                className="viewport-stream" 
                alt="Object Tracking Stream"
                onError={(e) => {
                  console.log("Stream load failed, trying reconnection");
                }}
              />
            ) : (
              <div className="viewport-placeholder">
                <AlertTriangle size={48} color="#FF0055" />
                <p>System offline. Waiting for FastAPI host...</p>
              </div>
            )}
          </div>
        </div>

        {/* Right Side: System Telemetry and Controls */}
        <div className="glass-panel">
          <div className="panel-title">
            <Activity size={20} className="header-icon" />
            <span>REAL-TIME TELEMETRY</span>
          </div>

          {/* Counts metrics */}
          <div className="telemetry-row">
            <div className="metric-box">
              <div className="metric-title">
                <TrendingUp size={14} style={{ marginRight: '4px', verticalAlign: 'middle', display: 'inline' }} />
                Crossed In
              </div>
              <div className="metric-number in">{telemetry.in_count}</div>
            </div>

            <div className="metric-box">
              <div className="metric-title">
                <TrendingDown size={14} style={{ marginRight: '4px', verticalAlign: 'middle', display: 'inline' }} />
                Crossed Out
              </div>
              <div className="metric-number out">{telemetry.out_count}</div>
            </div>
          </div>

          <div className="telemetry-row">
            <div className="metric-box">
              <div className="metric-title">
                <Zap size={14} style={{ marginRight: '4px', verticalAlign: 'middle', display: 'inline' }} />
                System FPS
              </div>
              <div className="metric-number speed">{telemetry.fps}</div>
            </div>

            <div className="metric-box">
              <div className="metric-title">
                <Eye size={14} style={{ marginRight: '4px', verticalAlign: 'middle', display: 'inline' }} />
                Active Tracks
              </div>
              <div className="metric-number active">{telemetry.active_tracks}</div>
            </div>
          </div>

          {/* System Control Settings */}
          <div className="panel-title" style={{ marginTop: '0.5rem' }}>
            <Settings size={20} className="header-icon" />
            <span>PARAMETER TUNING</span>
          </div>

          {/* Tracker Selection */}
          <div className="control-section">
            <div className="control-label">
              <span>Tracking Engine</span>
              <span className="label-value">{trackerType.toUpperCase()}</span>
            </div>
            <div className="radio-pill-group">
              <button 
                className={`radio-pill-btn ${trackerType === 'sort' ? 'active' : ''}`}
                onClick={() => handleTrackerChange('sort')}
              >
                SORT (Motion)
              </button>
              <button 
                className={`radio-pill-btn ${trackerType === 'deepsort' ? 'active' : ''}`}
                onClick={() => handleTrackerChange('deepsort')}
              >
                Deep SORT (Deep Visual)
              </button>
            </div>
          </div>

          {/* Video Feed Source */}
          <div className="control-section">
            <div className="control-label">
              <span>Video Feed Source</span>
              <span className="label-value">
                {videoSource === '0' ? 'Webcam Feed' : 
                 videoSource === 'videos/traffic_highway.mp4' ? 'Highway Traffic' :
                 videoSource === 'videos/traffic_pedestrians.mp4' ? 'Pedestrians & Cars' :
                 videoSource === 'videos/traffic_crowd.mp4' ? 'Crowded Street' : 'Store Flow'}
              </span>
            </div>
            
            <div className="source-selector-container" style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
              <div className="radio-pill-group" style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.4rem', background: 'none', border: 'none', padding: 0 }}>
                <button 
                  className={`radio-pill-btn ${videoSource === 'videos/traffic_highway.mp4' ? 'active' : ''}`}
                  onClick={() => handleSourceChange('videos/traffic_highway.mp4')}
                  style={{ borderRadius: '8px' }}
                >
                  Highway Traffic
                </button>
                <button 
                  className={`radio-pill-btn ${videoSource === 'videos/traffic_pedestrians.mp4' ? 'active' : ''}`}
                  onClick={() => handleSourceChange('videos/traffic_pedestrians.mp4')}
                  style={{ borderRadius: '8px' }}
                >
                  Pedestrians & Cars
                </button>
                <button 
                  className={`radio-pill-btn ${videoSource === 'videos/traffic_crowd.mp4' ? 'active' : ''}`}
                  onClick={() => handleSourceChange('videos/traffic_crowd.mp4')}
                  style={{ borderRadius: '8px' }}
                >
                  Crowded Street
                </button>
                <button 
                  className={`radio-pill-btn ${videoSource === 'videos/traffic_store.mp4' ? 'active' : ''}`}
                  onClick={() => handleSourceChange('videos/traffic_store.mp4')}
                  style={{ borderRadius: '8px' }}
                >
                  Store Flow
                </button>
              </div>
              
              <button 
                className={`radio-pill-btn ${videoSource === '0' ? 'active' : ''}`}
                onClick={() => handleSourceChange('0')}
                style={{ width: '100%', padding: '0.6rem', border: '1px solid rgba(0, 240, 255, 0.25)' }}
              >
                📹 Start Live Webcam Feed
              </button>
            </div>
          </div>

          {/* Confidence Slider */}
          <div className="control-section">
            <div className="control-label">
              <span>Detection Confidence</span>
              <span className="label-value">{Math.round(confThreshold * 100)}%</span>
            </div>
            <input 
              type="range" 
              min="0.1" 
              max="0.9" 
              step="0.05" 
              value={confThreshold}
              onChange={handleConfChange}
              className="custom-range"
            />
          </div>

          {/* Class Filters Checkboxes */}
          <div className="control-section">
            <div className="control-label">
              <span>Target Class Filters</span>
              <span className="label-value">{selectedClasses.length} selected</span>
            </div>
            <div className="class-tags-grid">
              {AVAILABLE_CLASSES.map(cls => (
                <div 
                  key={cls}
                  className={`class-tag ${selectedClasses.includes(cls) ? 'active' : ''}`}
                  onClick={() => toggleClass(cls)}
                >
                  {cls}
                </div>
              ))}
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}

export default App;
