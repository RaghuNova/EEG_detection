import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Users,
  FileSpreadsheet,
  CheckCircle2,
  AlertTriangle,
  Upload,
  Clock,
  ArrowRight,
  TrendingUp,
  Brain,
  Cpu,
  Layers,
  FileDown
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  AreaChart,
  Area,
} from 'recharts';
import StatCard from '../components/dashboard/StatCard';
import ModelStatusCard from '../components/dashboard/ModelStatusCard';
import { api } from '../services/api';
import { useAnalysis } from '../components/layout/MainLayout';

export default function Dashboard() {
  const navigate = useNavigate();
  const { setActiveAnalysis, setHeaderInfo } = useAnalysis();
  const [history, setHistory] = useState([]);
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(true);
  const [analysisError, setAnalysisError] = useState('');

  useEffect(() => {
    setHeaderInfo({
      title: 'EEG Intelligence Overview',
      subtitle: 'Monitor recordings, model inference, and seizure analysis across patients.',
    });

    const fetchData = async () => {
      try {
        const [histRes, healthRes] = await Promise.all([
          api.getAnalysesHistory(),
          api.getHealth(),
        ]);
        setHistory(histRes || []);
        setHealth(healthRes || {});
      } catch (err) {
        console.error('Error loading dashboard data:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [setHeaderInfo]);

  // Aggregate stats across stored sessions
  const totalPatients = new Set(history.map((h) => h.patient_id)).size;
  const totalRecordings = history.reduce((acc, h) => acc + (h.total_recordings || 0), 0);
  const totalWindows = history.reduce((acc, h) => acc + (h.total_windows || 0), 0);
  const totalSuccessful = history.filter((h) => h.status === 'completed').length;
  const totalFailed = history.filter((h) => h.status === 'failed').length;

  const totalMLPos = history.reduce((acc, h) => acc + (h.summary?.ml_positive_windows || 0), 0);
  const totalCNNPos = history.reduce((acc, h) => acc + (h.summary?.cnn_positive_windows || 0), 0);
  const totalAttnPos = history.reduce((acc, h) => acc + (h.summary?.attention_positive_windows || 0), 0);

  // Prepare chart comparison data
  const comparisonData = history.slice(0, 6).map((h) => ({
    patient: h.patient_id,
    ML: h.summary?.ml_positive_windows || 0,
    CNN: h.summary?.cnn_positive_windows || 0,
    Attention: h.summary?.attention_positive_windows || 0,
  }));

  const handleSelectAnalysis = async (analysisId) => {
    setAnalysisError('');
    try {
      const full = await api.getAnalysisResults(analysisId);
      setActiveAnalysis(full);
      navigate('/comparison');
    } catch (err) {
      setAnalysisError(err.response?.data?.detail || err.message || 'Could not load this analysis.');
    }
  };

  return (
    <div className="space-y-6">
      {analysisError && <div className="rounded-lg border border-amber-accent/30 bg-amber-accent/10 px-4 py-3 text-xs text-amber-accent">{analysisError}</div>}
      {/* KPI Cards Grid */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3.5">
        <StatCard
          title="Patients Analyzed"
          value={totalPatients}
          subtitle="Unique patient IDs"
          icon={Users}
          color="cyan"
        />
        <StatCard
          title="EDF Recordings"
          value={totalRecordings}
          subtitle="Processed recordings"
          icon={FileSpreadsheet}
          color="blue"
        />
        <StatCard
          title="Windows Evaluated"
          value={totalWindows.toLocaleString()}
          subtitle="20-second segments"
          icon={Clock}
          color="purple"
        />
        <StatCard
          title="Seizure Windows"
          value={totalAttnPos}
          subtitle="Attention Alarms"
          icon={Brain}
          color="rose"
        />
        <StatCard
          title="Completed Runs"
          value={totalSuccessful}
          subtitle="100% evaluated"
          icon={CheckCircle2}
          color="emerald"
        />
        <StatCard
          title="Failed Recordings"
          value={totalFailed}
          subtitle="Corrupted/rejected"
          icon={AlertTriangle}
          color="amber"
        />
      </div>

      {/* Model Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <ModelStatusCard
          type="ml"
          modelName="XGBoost Classifier"
          architecture="122 Invariant Tabular Features"
          threshold={health?.models?.ml_threshold ? `${health.models.ml_threshold.toFixed(4)}` : '0.2790'}
          positiveWindows={totalMLPos}
          totalWindows={totalWindows}
        />
        <ModelStatusCard
          type="cnn"
          modelName="CNN-BiLSTM"
          architecture="22-Ch Raw 1D Conv + Recurrent"
          threshold="0.5000"
          positiveWindows={totalCNNPos}
          totalWindows={totalWindows}
        />
        <ModelStatusCard
          type="attention"
          modelName="AttentionClassifier"
          architecture="4-Head Attention + Residual Fusion"
          threshold="Dynamic Adaptive"
          positiveWindows={totalAttnPos}
          totalWindows={totalWindows}
        />
      </div>

      {/* Charts and Activity */}
      {history.length > 0 ? (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Multi-Model Detections Chart */}
          <div className="lg:col-span-2 p-5 rounded-xl bg-background-panel border border-border">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="text-xs font-semibold text-text-primary tracking-tight">
                  Seizure Detections by Model
                </h3>
                <p className="text-[11px] text-text-muted">
                  Positive 20-second windows across recent analyses
                </p>
              </div>
              <div className="flex items-center space-x-3 text-[11px]">
                <span className="flex items-center space-x-1">
                  <div className="w-2.5 h-2.5 rounded-sm bg-blue-accent" />
                  <span className="text-text-muted">ML</span>
                </span>
                <span className="flex items-center space-x-1">
                  <div className="w-2.5 h-2.5 rounded-sm bg-cyan-accent" />
                  <span className="text-text-muted">CNN</span>
                </span>
                <span className="flex items-center space-x-1">
                  <div className="w-2.5 h-2.5 rounded-sm bg-purple-accent" />
                  <span className="text-text-muted">Attention</span>
                </span>
              </div>
            </div>

            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={comparisonData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#23324B" vertical={false} />
                  <XAxis dataKey="patient" stroke="#64748B" fontSize={11} tickLine={false} />
                  <YAxis stroke="#64748B" fontSize={11} tickLine={false} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#111D32', borderColor: '#23324B', borderRadius: '8px', fontSize: '11px' }}
                    itemStyle={{ color: '#F1F5F9' }}
                  />
                  <Bar dataKey="ML" fill="#4D8DFF" radius={[3, 3, 0, 0]} />
                  <Bar dataKey="CNN" fill="#36C5E8" radius={[3, 3, 0, 0]} />
                  <Bar dataKey="Attention" fill="#A78BFA" radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Quick Analysis History List */}
          <div className="p-5 rounded-xl bg-background-panel border border-border flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-xs font-semibold text-text-primary tracking-tight">Recent Sessions</h3>
                <button
                  onClick={() => navigate('/records')}
                  className="text-[11px] text-cyan-accent hover:underline flex items-center space-x-0.5"
                >
                  <span>View All</span>
                  <ArrowRight className="w-3 h-3" />
                </button>
              </div>

              <div className="space-y-2">
                {history.slice(0, 5).map((item) => (
                  <div
                    key={item.analysis_id}
                    onClick={() => handleSelectAnalysis(item.analysis_id)}
                    className="p-3 rounded-lg bg-background-secondary/60 border border-border/80 hover:border-cyan-accent/50 cursor-pointer transition-all flex items-center justify-between"
                  >
                    <div>
                      <div className="text-xs font-semibold text-text-primary font-mono">{item.patient_id}</div>
                      <div className="text-[10px] text-text-muted">
                        {item.total_recordings} recordings • {item.total_windows} windows
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-xs font-bold text-purple-accent font-mono">
                        {item.summary?.attention_positive_windows || 0} <span className="text-[9px] text-text-muted font-normal">seiz</span>
                      </div>
                      <div className="text-[9px] text-emerald-accent">Completed</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <button
              onClick={() => navigate('/upload')}
              className="mt-4 w-full py-2 rounded-lg text-xs font-medium bg-background border border-border hover:border-cyan-accent text-text-primary flex items-center justify-center space-x-2 transition-colors"
            >
              <Upload className="w-3.5 h-3.5 text-cyan-accent" />
              <span>Upload New Patient Folder</span>
            </button>
          </div>
        </div>
      ) : (
        /* Empty State */
        <div className="p-12 rounded-xl bg-background-panel border border-dashed border-border text-center flex flex-col items-center justify-center space-y-4">
          <div className="w-14 h-14 rounded-2xl bg-cyan-accent/10 border border-cyan-accent/30 flex items-center justify-center text-cyan-accent glow-cyan">
            <Upload className="w-6 h-6" />
          </div>
          <div className="max-w-md">
            <h3 className="text-sm font-semibold text-text-primary">No Patient Analyses Yet</h3>
            <p className="text-xs text-text-muted mt-1">
              Upload your first patient EEG folder containing one or more EDF recordings to begin automated multi-model seizure inference.
            </p>
          </div>
          <button
            onClick={() => navigate('/upload')}
            className="px-4 py-2 rounded-lg text-xs font-semibold bg-cyan-accent text-background hover:bg-cyan-accent/90 transition-all shadow-md flex items-center space-x-2"
          >
            <Upload className="w-4 h-4" />
            <span>Select Patient EEG Folder</span>
          </button>
        </div>
      )}
    </div>
  );
}
