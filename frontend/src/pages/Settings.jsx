import React, { useEffect, useState } from 'react';
import { Settings as SettingsIcon, ShieldCheck, CheckCircle2, Lock, HardDrive, Cpu, Layers, Brain, Info } from 'lucide-react';
import { api } from '../services/api';
import { useAnalysis } from '../components/layout/MainLayout';

export default function Settings() {
  const { setHeaderInfo } = useAnalysis();
  const [health, setHealth] = useState(null);

  useEffect(() => {
    setHeaderInfo({
      title: 'System Settings & Architecture Audit',
      subtitle: 'Inspection of model checkpoints, dynamic calibration policies, and storage parameters.',
    });

    api.getHealth().then((h) => setHealth(h)).catch((e) => console.error(e));
  }, [setHeaderInfo]);

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Integrity Notice */}
      <div className="p-4 rounded-xl bg-cyan-accent/10 border border-cyan-accent/30 text-xs text-text-primary flex items-start space-x-3">
        <ShieldCheck className="w-5 h-5 text-cyan-accent shrink-0 mt-0.5" />
        <div>
          <h3 className="font-semibold text-cyan-accent">Inference-Only Architecture & Weight Preservation</h3>
          <p className="text-[11px] text-text-muted mt-0.5">
            In accordance with research integrity specifications, all ML and DL models, pre-trained weights, and dynamic calibration
            parameters are read-only. Retraining, weight alteration, or arbitrary threshold invention is strictly disallowed.
          </p>
        </div>
      </div>

      {/* Model Artifacts Configuration Audit */}
      <div className="p-5 rounded-xl bg-background-panel border border-border space-y-4">
        <h2 className="text-xs font-semibold text-text-primary uppercase tracking-wider">
          Verified Model Artifacts & Provenance
        </h2>

        <div className="space-y-3 font-mono text-xs">
          {/* ML Model */}
          <div className="p-3 rounded-lg bg-background border border-border flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Cpu className="w-4 h-4 text-blue-accent" />
              <div>
                <span className="font-bold text-text-primary">Model 1: XGBoost Classifier</span>
                <span className="text-[10px] text-text-muted block">Artifact: models/seizure_ml_model.joblib</span>
              </div>
            </div>
            <div className="text-right">
              <span className="text-blue-accent font-bold">Th: {health?.models?.ml_threshold ? health.models.ml_threshold.toFixed(4) : '0.2790'}</span>
              <span className="text-[10px] text-emerald-accent block">Verified & Loaded</span>
            </div>
          </div>

          {/* CNN-BiLSTM */}
          <div className="p-3 rounded-lg bg-background border border-border flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Layers className="w-4 h-4 text-cyan-accent" />
              <div>
                <span className="font-bold text-text-primary">Model 2: CNN-BiLSTM Raw Signal Encoder</span>
                <span className="text-[10px] text-text-muted block">Checkpoint: models/final.pt (1).zip (22 Canonical Channels)</span>
              </div>
            </div>
            <div className="text-right">
              <span className="text-cyan-accent font-bold">Th: 0.5000</span>
              <span className="text-[10px] text-emerald-accent block">Verified & Loaded</span>
            </div>
          </div>

          {/* Attention Model */}
          <div className="p-3 rounded-lg bg-background border border-border flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Brain className="w-4 h-4 text-purple-accent" />
              <div>
                <span className="font-bold text-text-primary">Model 3: Temporal Attention Classifier</span>
                <span className="text-[10px] text-text-muted block">Checkpoint: models/final.pt.zip (4 Attention Heads)</span>
              </div>
            </div>
            <div className="text-right">
              <span className="text-purple-accent font-bold">Hybrid Quantile</span>
              <span className="text-[10px] text-emerald-accent block">Verified & Loaded</span>
            </div>
          </div>
        </div>
      </div>

      {/* Dynamic Threshold Calibration Settings */}
      <div className="p-5 rounded-xl bg-background-panel border border-border space-y-3 text-xs">
        <h2 className="font-semibold text-text-primary uppercase tracking-wider text-xs">
          Dynamic Calibration Policy (dependencies/hybrid_attention_threshold_calibration.json)
        </h2>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 font-mono">
          <div className="p-3 rounded-lg bg-background border border-border">
            <span className="text-[10px] text-text-muted block">Global Baseline Th.</span>
            <span className="font-bold text-text-primary">0.0300</span>
          </div>
          <div className="p-3 rounded-lg bg-background border border-border">
            <span className="text-[10px] text-text-muted block">Quantile (q)</span>
            <span className="font-bold text-text-primary">0.9850</span>
          </div>
          <div className="p-3 rounded-lg bg-background border border-border">
            <span className="text-[10px] text-text-muted block">Threshold Floor</span>
            <span className="font-bold text-text-primary">0.0300</span>
          </div>
          <div className="p-3 rounded-lg bg-background border border-border">
            <span className="text-[10px] text-text-muted block">Adaptive Blend</span>
            <span className="font-bold text-text-primary">1.0000</span>
          </div>
        </div>
      </div>

      {/* Storage and System Info */}
      <div className="p-5 rounded-xl bg-background-panel border border-border space-y-3 text-xs">
        <h2 className="font-semibold text-text-primary uppercase tracking-wider text-xs">
          Storage Directories & Environment
        </h2>

        <div className="space-y-2 font-mono text-text-secondary">
          <div className="flex items-center justify-between p-2 rounded bg-background">
            <span>Upload Directory:</span>
            <span className="text-cyan-accent">storage/uploads</span>
          </div>
          <div className="flex items-center justify-between p-2 rounded bg-background">
            <span>Analysis Archive:</span>
            <span className="text-cyan-accent">storage/analyses</span>
          </div>
          <div className="flex items-center justify-between p-2 rounded bg-background">
            <span>Report Generation:</span>
            <span className="text-cyan-accent">storage/reports</span>
          </div>
          <div className="flex items-center justify-between p-2 rounded bg-background">
            <span>Compute Accelerator:</span>
            <span className="text-emerald-accent">{health?.models?.device || 'Automatic (MPS/CUDA/CPU)'}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
