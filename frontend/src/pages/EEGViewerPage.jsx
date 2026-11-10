import React, { useEffect, useState } from 'react';
import { Eye, Sliders, Brain, Cpu, Layers, AlertCircle, ArrowRight } from 'lucide-react';
import EEGViewer from '../components/eeg/EEGViewer';
import { useAnalysis } from '../components/layout/MainLayout';
import { api } from '../services/api';

export default function EEGViewerPage() {
  const { activeAnalysis, getActiveRecording, setHeaderInfo } = useAnalysis();
  const recording = getActiveRecording();

  const [activeWindowIdx, setActiveWindowIdx] = useState(0);
  const [signalSegment, setSignalSegment] = useState(null);
  const [loadingSignal, setLoadingSignal] = useState(false);

  useEffect(() => {
    setHeaderInfo({
      title: 'Interactive Multi-Channel EEG Viewer',
      subtitle: 'High-resolution stacked channel waveform inspection with multi-model probability overlay.',
    });
  }, [setHeaderInfo]);

  // Load preview signal or fetch initial segment
  useEffect(() => {
    if (recording) {
      if (recording.preview_signal) {
        setSignalSegment(recording.preview_signal);
      } else if (activeAnalysis?.analysis_id && recording.recording_id) {
        setLoadingSignal(true);
        api
          .getSignalSegment(activeAnalysis.analysis_id, recording.recording_id, 0, 40)
          .then((sig) => setSignalSegment(sig))
          .catch((err) => console.error('Error fetching signal segment:', err))
          .finally(() => setLoadingSignal(false));
      }
    }
  }, [recording, activeAnalysis]);

  if (!activeAnalysis || !recording) {
    return (
      <div className="p-8 text-center bg-background-panel rounded-xl border border-border">
        <p className="text-xs text-text-muted">No patient recording loaded. Please upload or select an analysis session.</p>
      </div>
    );
  }

  const windows = recording.windows || [];
  const activeWindow = windows[activeWindowIdx] || windows[0] || {};
  const allEvents = [
    ...(recording.models?.ml?.events || []),
    ...(recording.models?.cnn_bilstm?.events || []),
    ...(recording.models?.attention?.events || []),
  ];

  return (
    <div className="space-y-6">
      {/* Active Window Multi-Model Inspector */}
      {activeWindow && (
        <div className="p-4 rounded-xl bg-background-panel border border-border shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="flex items-center space-x-3">
            <div className="w-9 h-9 rounded-lg bg-cyan-accent/10 border border-cyan-accent/30 flex items-center justify-center text-cyan-accent">
              <Eye className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="text-xs font-bold font-mono text-text-primary">
                  {activeWindow.window_id || 'Window 0'}
                </span>
                <span className="text-[10px] text-text-muted font-mono">
                  ({activeWindow.start_seconds?.toFixed(1)}s – {activeWindow.end_seconds?.toFixed(1)}s)
                </span>
              </div>
              <p className="text-[10px] text-text-muted">
                Inspecting window {activeWindow.window_index + 1} of {windows.length} in {recording.filename}
              </p>
            </div>
          </div>

          {/* Model Probabilities for this window */}
          <div className="flex items-center space-x-4 font-mono text-xs">
            {/* ML */}
            <div className="px-3 py-1.5 rounded-lg bg-background border border-border">
              <span className="text-[10px] text-text-muted block">ML (XGB)</span>
              <span className="font-bold text-blue-accent">
                {(activeWindow.ml?.probability || 0).toFixed(3)}{' '}
                <span className="text-[9px] font-normal text-text-muted">
                  ({activeWindow.ml?.prediction ? 'SEIZURE' : 'Normal'})
                </span>
              </span>
            </div>

            {/* CNN */}
            <div className="px-3 py-1.5 rounded-lg bg-background border border-border">
              <span className="text-[10px] text-text-muted block">CNN-BiLSTM</span>
              <span className="font-bold text-cyan-accent">
                {(activeWindow.cnn_bilstm?.probability || 0).toFixed(3)}{' '}
                <span className="text-[9px] font-normal text-text-muted">
                  ({activeWindow.cnn_bilstm?.prediction ? 'SEIZURE' : 'Normal'})
                </span>
              </span>
            </div>

            {/* Attention */}
            <div className="px-3 py-1.5 rounded-lg bg-background border border-border">
              <span className="text-[10px] text-text-muted block">Attention (Dynamic)</span>
              <span className="font-bold text-purple-accent">
                {(activeWindow.attention?.probability || 0).toFixed(3)}{' '}
                <span className="text-[9px] font-normal text-text-muted">
                  ({activeWindow.attention?.prediction ? 'SEIZURE' : 'Normal'})
                </span>
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Main Stacked Canvas Waveform Viewer */}
      <EEGViewer
        signalData={signalSegment}
        windowInfos={windows}
        events={allEvents}
        activeWindowIndex={activeWindowIdx}
        onSelectWindow={setActiveWindowIdx}
        recordingDuration={recording.duration_seconds}
      />

      {/* Event Jump List */}
      {allEvents.length > 0 && (
        <div className="p-4 rounded-xl bg-background-panel border border-border">
          <h3 className="text-xs font-semibold text-text-primary mb-2">Jump to Detected Seizure Interval</h3>
          <div className="flex flex-wrap gap-2">
            {allEvents.map((evt, idx) => (
              <button
                key={idx}
                onClick={() => {
                  const targetWin = windows.findIndex((w) => w.start_seconds >= evt.start_seconds);
                  if (targetWin !== -1) setActiveWindowIdx(targetWin);
                }}
                className="px-3 py-1.5 rounded-lg text-[11px] font-mono bg-background border border-rose-accent/30 text-rose-accent hover:bg-rose-accent/10 transition-colors cursor-pointer flex items-center space-x-1"
              >
                <span>{evt.event_id || `Event ${idx+1}`}</span>
                <span className="text-text-muted font-normal">({evt.start_seconds}s - {evt.end_seconds}s)</span>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
