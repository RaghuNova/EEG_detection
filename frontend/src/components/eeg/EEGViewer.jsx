import React, { useRef, useEffect, useState, useMemo } from 'react';
import { ZoomIn, ZoomOut, RotateCcw, ChevronLeft, ChevronRight, Activity, Sliders, Eye } from 'lucide-react';

export default function EEGViewer({
  signalData,
  windowInfos = [],
  events = [],
  activeWindowIndex = 0,
  onSelectWindow,
  recordingDuration = 0,
}) {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);

  const [timeOffset, setTimeOffset] = useState(0); // in seconds
  const [windowDuration, setWindowDuration] = useState(20); // 10s, 20s, or 30s visible
  const [amplitudeScale, setAmplitudeScale] = useState(1.0);
  const [selectedChannels, setSelectedChannels] = useState([]);
  const [hoveredInfo, setHoveredInfo] = useState(null);

  const channelNames = signalData?.channel_names || [];
  const rawData = signalData?.data_uv || [];
  const sfreq = signalData?.sampling_rate_hz || 256;

  // Initialize selected channels if not set
  useEffect(() => {
    if (channelNames.length > 0 && selectedChannels.length === 0) {
      setSelectedChannels(channelNames.slice(0, 16));
    }
  }, [channelNames]);

  // Sync with active window index when changed from outside
  useEffect(() => {
    if (windowInfos[activeWindowIndex]) {
      setTimeOffset(windowInfos[activeWindowIndex].start_seconds);
    }
  }, [activeWindowIndex, windowInfos]);

  // Render stacked traces on canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !rawData.length || !channelNames.length) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    // Clear background
    ctx.fillStyle = '#080E1B';
    ctx.fillRect(0, 0, width, height);

    const activeIndices = channelNames
      .map((name, idx) => (selectedChannels.includes(name) ? idx : -1))
      .filter((idx) => idx !== -1);

    if (activeIndices.length === 0) return;

    const nChannels = activeIndices.length;
    const channelHeight = height / (nChannels + 1);

    // Draw horizontal channel baseline grids
    ctx.strokeStyle = '#1E293B';
    ctx.lineWidth = 0.5;
    for (let i = 0; i < nChannels; i++) {
      const y = (i + 1) * channelHeight;
      ctx.beginPath();
      ctx.moveTo(70, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Time calculations
    const startSample = Math.max(0, Math.floor(timeOffset * sfreq));
    const totalSamplesToRender = Math.floor(windowDuration * sfreq);
    const endSample = Math.min(startSample + totalSamplesToRender, rawData[0]?.length || 0);
    const renderedSamples = endSample - startSample;

    if (renderedSamples <= 0) return;

    // Draw event highlight zones
    const startTimeSec = timeOffset;
    const endTimeSec = timeOffset + windowDuration;

    events.forEach((evt) => {
      if (evt.end_seconds > startTimeSec && evt.start_seconds < endTimeSec) {
        const x1 = 70 + ((Math.max(startTimeSec, evt.start_seconds) - startTimeSec) / windowDuration) * (width - 70);
        const x2 = 70 + ((Math.min(endTimeSec, evt.end_seconds) - startTimeSec) / windowDuration) * (width - 70);
        ctx.fillStyle = 'rgba(251, 113, 133, 0.15)'; // Rose glow for seizure event
        ctx.fillRect(x1, 0, Math.max(x2 - x1, 4), height);

        ctx.strokeStyle = '#FB7185';
        ctx.lineWidth = 1;
        ctx.strokeRect(x1, 0, Math.max(x2 - x1, 4), height);

        // Event label
        ctx.fillStyle = '#FB7185';
        ctx.font = '10px JetBrains Mono';
        ctx.fillText(`${evt.event_id || 'SEIZURE'} (${evt.duration_seconds}s)`, x1 + 4, 15);
      }
    });

    // Draw time scale bar at bottom
    ctx.fillStyle = '#64748B';
    ctx.font = '10px JetBrains Mono';
    const numTimeTicks = 5;
    for (let t = 0; t <= numTimeTicks; t++) {
      const frac = t / numTimeTicks;
      const x = 70 + frac * (width - 70);
      const tickTime = startTimeSec + frac * windowDuration;
      ctx.fillText(`${tickTime.toFixed(1)}s`, x - 10, height - 8);

      ctx.beginPath();
      ctx.moveTo(x, height - 25);
      ctx.lineTo(x, height - 20);
      ctx.strokeStyle = '#334155';
      ctx.stroke();
    }

    // Draw stacked EEG signal traces
    activeIndices.forEach((chIdx, renderIdx) => {
      const chName = channelNames[chIdx];
      const yCenter = (renderIdx + 1) * channelHeight;

      // Channel Label
      ctx.fillStyle = '#94A3B8';
      ctx.font = '10px JetBrains Mono';
      ctx.textAlign = 'right';
      ctx.fillText(chName, 62, yCenter + 3);
      ctx.textAlign = 'left';

      // Waveform trace
      const channelSignal = rawData[chIdx];
      if (!channelSignal) return;

      ctx.beginPath();
      ctx.strokeStyle = '#36C5E8'; // Cyan accent
      ctx.lineWidth = 1.0;

      const pxPerSample = (width - 70) / renderedSamples;
      const uVtoPx = (channelHeight * 0.45 * amplitudeScale) / 100.0; // 100 uV scale

      for (let s = 0; s < renderedSamples; s++) {
        const sampleIdx = startSample + s;
        const val = channelSignal[sampleIdx] || 0;
        const x = 70 + s * pxPerSample;
        const y = yCenter - val * uVtoPx;

        if (s === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
    });
  }, [timeOffset, windowDuration, amplitudeScale, selectedChannels, rawData, channelNames, sfreq, events]);

  // Handle canvas sizing
  useEffect(() => {
    const handleResize = () => {
      if (containerRef.current && canvasRef.current) {
        canvasRef.current.width = containerRef.current.clientWidth;
        canvasRef.current.height = Math.max(500, selectedChannels.length * 35 + 50);
      }
    };
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [selectedChannels]);

  // Navigate time
  const handleScrollTime = (deltaSeconds) => {
    setTimeOffset((prev) => Math.max(0, prev + deltaSeconds));
  };

  const toggleChannel = (name) => {
    setSelectedChannels((prev) =>
      prev.includes(name) ? prev.filter((ch) => ch !== name) : [...prev, name]
    );
  };

  return (
    <div className="space-y-4">
      {/* Viewer Toolbar */}
      <div className="p-3.5 rounded-xl bg-background-panel border border-border flex flex-wrap items-center justify-between gap-3 text-xs">
        {/* Navigation & Zoom */}
        <div className="flex items-center space-x-2">
          <div className="flex items-center space-x-1 bg-background rounded-lg border border-border p-1">
            <button
              onClick={() => handleScrollTime(-windowDuration)}
              disabled={timeOffset <= 0}
              className="p-1 rounded hover:bg-background-panel text-text-muted hover:text-text-primary disabled:opacity-30 cursor-pointer"
              title="Previous Window"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <span className="px-2 font-mono text-[11px] text-text-primary">
              {timeOffset.toFixed(1)}s – {(timeOffset + windowDuration).toFixed(1)}s
            </span>
            <button
              onClick={() => handleScrollTime(windowDuration)}
              className="p-1 rounded hover:bg-background-panel text-text-muted hover:text-text-primary cursor-pointer"
              title="Next Window"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>

          {/* Time Window Scale */}
          <div className="flex items-center space-x-1 bg-background rounded-lg border border-border p-1 text-[11px] font-mono">
            {[10, 20, 30].map((dur) => (
              <button
                key={dur}
                onClick={() => setWindowDuration(dur)}
                className={`px-2 py-0.5 rounded cursor-pointer ${
                  windowDuration === dur ? 'bg-cyan-accent text-background font-bold' : 'text-text-muted hover:text-text-primary'
                }`}
              >
                {dur}s
              </button>
            ))}
          </div>
        </div>

        {/* Amplitude Gain Controls */}
        <div className="flex items-center space-x-2">
          <span className="text-[11px] text-text-muted">Gain:</span>
          <div className="flex items-center space-x-1 bg-background rounded-lg border border-border p-1 text-[11px] font-mono">
            {[0.5, 1.0, 2.0, 4.0].map((scale) => (
              <button
                key={scale}
                onClick={() => setAmplitudeScale(scale)}
                className={`px-2 py-0.5 rounded cursor-pointer ${
                  amplitudeScale === scale ? 'bg-cyan-accent text-background font-bold' : 'text-text-muted hover:text-text-primary'
                }`}
              >
                {scale}x
              </button>
            ))}
          </div>

          <button
            onClick={() => { setTimeOffset(0); setAmplitudeScale(1.0); }}
            className="p-1.5 rounded-lg bg-background border border-border text-text-muted hover:text-text-primary"
            title="Reset View"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>
        </div>

        {/* Channel Selection Toggle */}
        <div className="flex items-center space-x-1.5 text-[11px] font-mono">
          <span className="text-text-muted">Channels:</span>
          <span className="text-cyan-accent font-bold">{selectedChannels.length} active</span>
        </div>
      </div>

      {/* Stacked EEG Trace Canvas */}
      <div ref={containerRef} className="rounded-xl border border-border bg-[#080E1B] overflow-hidden shadow-inner relative">
        <canvas ref={canvasRef} className="w-full block" />
      </div>

      {/* Channel Filters Pill Bar */}
      <div className="p-3 rounded-xl bg-background-panel border border-border">
        <span className="text-[10px] text-text-muted block mb-2 font-mono font-medium uppercase tracking-wider">
          EEG Montages & Channels Filter (Click to toggle)
        </span>
        <div className="flex flex-wrap gap-1.5">
          {channelNames.map((ch) => {
            const isSelected = selectedChannels.includes(ch);
            return (
              <button
                key={ch}
                onClick={() => toggleChannel(ch)}
                className={`px-2 py-0.5 rounded text-[10px] font-mono transition-colors cursor-pointer ${
                  isSelected
                    ? 'bg-cyan-accent/15 border border-cyan-accent/40 text-cyan-accent'
                    : 'bg-background border border-border text-text-muted hover:text-text-primary'
                }`}
              >
                {ch}
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
