import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Loader2,
  CheckCircle2,
  AlertTriangle,
  ArrowRight,
  FileCheck
} from 'lucide-react';
import { api } from '../services/api';
import { useAnalysis } from '../components/layout/MainLayout';

const PIPELINE_STEPS = [
  { id: 'validation', label: 'EDF validation' },
  { id: 'preprocessing', label: 'Preprocessing' },
  { id: 'ml_inference', label: 'ML inference' },
  { id: 'cnn_bilstm_inference', label: 'CNN-BiLSTM inference' },
  { id: 'attention_inference', label: 'Attention inference' },
  { id: 'event_detection', label: 'Event detection' },
  { id: 'results_saved', label: 'Result storage' },
];

export default function Processing() {
  const { id: analysisId } = useParams();
  const navigate = useNavigate();
  const { setActiveAnalysis, setActiveRecordingId, setHeaderInfo } = useAnalysis();

  const [status, setStatus] = useState({
    status: 'processing',
    percent_complete: 0,
    total_recordings: 1,
    current_index: 1,
    current_filename: 'Initializing...',
  });
  const [error, setError] = useState(null);

  useEffect(() => {
    setHeaderInfo({
      title: 'Analysis Execution',
      subtitle: `Running multi-model inference pipeline for analysis session ${analysisId}`,
    });

    let isMounted = true;
    let pollTimer;
    let polling = false;
    let terminal = false;

    const pollStatus = async () => {
      if (polling) return;
      polling = true;
      try {
        const data = await api.getAnalysisStatus(analysisId);
        if (!isMounted) return;
        setStatus(data);
        setError(null);

        if (data.status === 'completed') {
          terminal = true;
          clearTimeout(pollTimer);
          const fullRes = await api.getAnalysisResults(analysisId);
          setActiveAnalysis(fullRes);
          if (fullRes.recordings && fullRes.recordings.length > 0) {
            setActiveRecordingId(fullRes.recordings[0].recording_id);
          }
          navigate(`/patient-analysis/${analysisId}`, { replace: true });
        } else if (data.status === 'failed') {
          terminal = true;
          clearTimeout(pollTimer);
          setError(data.error || 'Analysis failed. Check server logs.');
        }
      } catch (err) {
        if (!isMounted) return;
        setError(err.response?.data?.detail || err.message || 'Could not retrieve analysis status.');
      } finally {
        polling = false;
        if (isMounted && !terminal) pollTimer = setTimeout(pollStatus, 1500);
      }
    };

    pollStatus();

    return () => {
      isMounted = false;
      clearTimeout(pollTimer);
    };
  }, [analysisId, setActiveAnalysis, setActiveRecordingId, setHeaderInfo]);

  const isCompleted = status.status === 'completed';
  const currentStep = PIPELINE_STEPS.findIndex((step) => step.id === status.stage);

  return (
    <div className="max-w-2xl mx-auto py-8 space-y-6">
      {/* Central Progress Card */}
      <div className="p-8 rounded-2xl bg-background-panel border border-border shadow-lg relative overflow-hidden">
        <div className="flex items-center justify-between mb-6">
          <div>
            <span className="text-[10px] font-mono text-cyan-accent uppercase tracking-wider block">
              Session {analysisId}
            </span>
            <h2 className="text-base font-bold text-text-primary tracking-tight mt-0.5">
              {isCompleted ? (status.failed_recordings ? 'Analysis Finished with Recording Errors' : 'Analysis Completed Successfully') : 'Executing Neural Inference Pipeline'}
            </h2>
            <p className="text-xs text-text-muted mt-0.5">
              Patient: <span className="font-mono text-text-primary font-semibold">{status.patient_id || 'PATIENT'}</span>
            </p>
          </div>

          <div className="w-12 h-12 rounded-xl bg-background border border-border flex items-center justify-center">
            {isCompleted ? (
              <CheckCircle2 className="w-6 h-6 text-emerald-accent" />
            ) : error ? (
              <AlertTriangle className="w-6 h-6 text-rose-accent" />
            ) : (
              <Loader2 className="w-6 h-6 text-cyan-accent animate-spin" />
            )}
          </div>
        </div>

        {/* Progress Bar */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs font-mono">
            <span className="text-text-muted">
              {status.current_filename
                ? `${status.current_filename} (${status.current_index || 1}/${status.total_recordings || 1}) · ${status.stage || 'queued'}`
                : 'Waiting for analysis worker'}
            </span>
            <span className="text-cyan-accent font-bold">{status.completed_recordings || 0}/{status.total_recordings || 0} complete</span>
          </div>

          <div className="w-full h-2.5 rounded-full bg-background border border-border overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-blue-accent via-cyan-accent to-purple-accent transition-all duration-300 rounded-full"
              style={{ width: `${status.percent_complete || 0}%` }}
            />
          </div>
          <div className="flex flex-wrap gap-x-5 gap-y-1 text-[11px] text-text-muted">
            <span>Successful: {status.successful_recordings || 0}</span>
            <span>Failed: {status.failed_recordings || 0}</span>
            {status.current_model && <span>Current model: {status.current_model}</span>}
          </div>
        </div>

        {/* Step-by-Step Progress */}
        <div className="mt-8 pt-6 border-t border-border/60 space-y-2.5">
          <h3 className="text-[11px] font-semibold text-text-secondary uppercase tracking-wider mb-3">
            Inference Pipeline Stages
          </h3>
          {PIPELINE_STEPS.map((step, idx) => {
            const isCurrent = currentStep === idx;
            const isDone = isCompleted || (currentStep >= 0 && idx < currentStep);

            return (
              <div
                key={step.id}
                className={`flex items-center justify-between px-3.5 py-2 rounded-lg text-xs font-mono transition-colors ${
                  isDone
                    ? 'bg-emerald-accent/5 text-text-primary border border-emerald-accent/20'
                    : isCurrent
                    ? 'bg-cyan-accent/10 text-cyan-accent border border-cyan-accent/30 animate-pulse'
                    : 'text-text-muted bg-background/40 border border-border/40'
                }`}
              >
                <span>{step.label}</span>
                {isDone ? (
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-accent shrink-0" />
                ) : isCurrent ? (
                  <Loader2 className="w-3.5 h-3.5 text-cyan-accent animate-spin shrink-0" />
                ) : (
                  <div className="w-2 h-2 rounded-full bg-border" />
                )}
              </div>
            );
          })}
        </div>

        {status.stage_events?.length > 0 && <div className="mt-5 text-[11px] text-text-muted">
          <div className="font-semibold uppercase tracking-wider mb-2">Recent backend stages</div>
          <div className="flex flex-wrap gap-2">
            {status.stage_events.slice(-10).map((event, idx) => <span key={`${event.time}-${idx}`} className="px-2 py-1 rounded bg-background border border-border">
              {event.recording}: {event.stage}{event.model ? ` · ${event.model}` : ''}
            </span>)}
          </div>
        </div>}

        {status.recording_statuses?.length > 0 && <div className="mt-6 pt-5 border-t border-border/60">
          <h3 className="text-[11px] font-semibold text-text-secondary uppercase tracking-wider mb-3">Recordings</h3>
          <div className="max-h-48 overflow-auto space-y-1.5">
            {status.recording_statuses.map((recording) => <div key={recording.filename} className="flex justify-between text-xs px-3 py-2 rounded bg-background/50">
              <span className="font-mono truncate pr-3">{recording.filename}</span>
              <span className={recording.status === 'failed' ? 'text-rose-accent' : recording.status === 'completed' ? 'text-emerald-accent' : 'text-text-muted'}>{recording.status} · {recording.stage}</span>
            </div>)}
          </div>
        </div>}

        {error && (
          <div className="mt-6 p-4 rounded-xl bg-rose-accent/10 border border-rose-accent/30 text-rose-accent text-xs">
            <b>Pipeline Error:</b> {error}
          </div>
        )}

        {isCompleted && (
          <div className="mt-8 pt-6 border-t border-border flex items-center justify-between">
            <span className="text-xs text-emerald-accent font-medium flex items-center space-x-1.5">
              <CheckCircle2 className="w-4 h-4" />
              <span>{status.successful_recordings || 0} recordings succeeded; {status.failed_recordings || 0} failed. Opening results…</span>
            </span>

            <div className="flex items-center space-x-3">
              <button
                onClick={() => navigate('/comparison')}
                className="px-4 py-2 rounded-lg text-xs font-semibold bg-cyan-accent text-background hover:bg-cyan-accent/90 transition-all flex items-center space-x-1.5 shadow-md cursor-pointer"
              >
                <span>View Model Comparison</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
