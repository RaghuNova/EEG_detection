import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { FolderOpen, FileDown, Eye, Clock, CheckCircle2, AlertTriangle, ArrowRight, Upload } from 'lucide-react';
import { api } from '../services/api';
import { useAnalysis } from '../components/layout/MainLayout';

export default function PatientRecords() {
  const navigate = useNavigate();
  const { setActiveAnalysis, setHeaderInfo } = useAnalysis();
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [actionError, setActionError] = useState('');

  useEffect(() => {
    setHeaderInfo({
      title: 'Patient Analysis Records',
      subtitle: 'Audit past EEG analysis sessions, inspect recordings, and retrieve PDF reports.',
    });

    api
      .getAnalysesHistory()
      .then((data) => setHistory(data || []))
      .catch((err) => console.error(err))
      .finally(() => setLoading(false));
  }, [setHeaderInfo]);

  const handleOpenAnalysis = async (analysisId) => {
    setActionError('');
    try {
      const full = await api.getAnalysisResults(analysisId);
      setActiveAnalysis(full);
      navigate('/comparison');
    } catch (err) {
      setActionError(err.response?.data?.detail || err.message || 'Could not load this analysis.');
    }
  };

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold text-text-primary">Historical Analysis Archive</h2>
          <p className="text-xs text-text-muted">Total of {history.length} patient analysis sessions recorded</p>
        </div>

        <button
          onClick={() => navigate('/upload')}
          className="px-3.5 py-1.5 rounded-lg text-xs font-semibold bg-cyan-accent text-background hover:bg-cyan-accent/90 transition-all flex items-center space-x-1.5 shadow-sm cursor-pointer"
        >
          <Upload className="w-3.5 h-3.5" />
          <span>New Analysis</span>
        </button>
      </div>

      {actionError && <div className="rounded-lg border border-amber-accent/30 bg-amber-accent/10 px-4 py-3 text-xs text-amber-accent">{actionError}</div>}

      {history.length > 0 ? (
        <div className="rounded-xl border border-border bg-background-panel overflow-hidden shadow-sm">
          <table className="w-full text-left text-xs">
            <thead className="bg-background-secondary text-text-muted border-b border-border text-[11px]">
              <tr>
                <th className="p-3">Patient ID</th>
                <th className="p-3">Session ID</th>
                <th className="p-3">Recordings</th>
                <th className="p-3">Windows</th>
                <th className="p-3">Attention Detections</th>
                <th className="p-3">Date</th>
                <th className="p-3 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/60 font-mono">
              {history.map((item) => (
                <tr key={item.analysis_id} className="hover:bg-background-secondary/30 transition-colors">
                  <td className="p-3 font-bold text-text-primary">{item.patient_id}</td>
                  <td className="p-3 text-text-muted">{item.analysis_id}</td>
                  <td className="p-3 text-text-secondary">{item.total_recordings} files</td>
                  <td className="p-3 text-text-secondary">{item.total_windows}</td>
                  <td className="p-3 font-bold text-purple-accent">
                    {item.summary?.attention_positive_windows || 0} windows
                  </td>
                  <td className="p-3 text-text-muted text-[11px] font-sans">
                    {item.created_at ? new Date(item.created_at).toLocaleString() : 'N/A'}
                  </td>
                  <td className="p-3 text-right">
                    <div className="flex items-center justify-end space-x-2">
                      <button
                        onClick={() => handleOpenAnalysis(item.analysis_id)}
                        className="px-2.5 py-1 rounded text-[11px] font-medium bg-background border border-border hover:border-cyan-accent text-cyan-accent transition-colors flex items-center space-x-1 cursor-pointer"
                      >
                        <Eye className="w-3 h-3" />
                        <span>Inspect</span>
                      </button>

                      <a
                        href={api.getPdfReportUrl(item.analysis_id)}
                        target="_blank"
                        rel="noreferrer"
                        className="p-1 rounded text-text-muted hover:text-text-primary hover:bg-background border border-transparent hover:border-border transition-colors"
                        title="Download PDF"
                      >
                        <FileDown className="w-3.5 h-3.5" />
                      </a>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="p-10 rounded-xl bg-background-panel border border-dashed border-border text-center space-y-3">
          <p className="text-xs text-text-muted">No historical analysis records found.</p>
          <button
            onClick={() => navigate('/upload')}
            className="px-4 py-2 rounded-lg text-xs font-semibold bg-cyan-accent text-background hover:bg-cyan-accent/90"
          >
            Upload Patient EEG
          </button>
        </div>
      )}
    </div>
  );
}
