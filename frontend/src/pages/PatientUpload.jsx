import React, { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  FolderUp,
  FileSpreadsheet,
  CheckCircle2,
  AlertCircle,
  Trash2,
  ArrowRight,
  ShieldAlert,
  HardDrive,
  Clock,
  Play
} from 'lucide-react';
import { api } from '../services/api';
import { useAnalysis } from '../components/layout/MainLayout';

export default function PatientUpload() {
  const navigate = useNavigate();
  const { setHeaderInfo } = useAnalysis();
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [annotationFiles, setAnnotationFiles] = useState([]);
  const [patientName, setPatientName] = useState('');
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(null);
  const [uploadError, setUploadError] = useState(null);

  const fileInputRef = useRef(null);
  const folderInputRef = useRef(null);

  useEffect(() => {
    setHeaderInfo({
      title: 'Patient Folder Upload',
      subtitle: 'Select or drag-and-drop a patient directory with multi-channel EDF recordings.',
    });
  }, [setHeaderInfo]);

  const handleFilesSelected = (fileList) => {
    const rawFiles = Array.from(fileList);
    const edfFiles = [];
    const sidecars = [];
    let detectedPatient = '';

    rawFiles.forEach((file) => {
      const relPath = file.webkitRelativePath || file.name;
      const parts = relPath.split('/');

      if (parts.length > 1 && !detectedPatient) {
        detectedPatient = parts[0].toUpperCase();
      }

      const lowerName = file.name.toLowerCase();
      if (lowerName.endsWith('.edf.seizures')) {
        sidecars.push({ file, relativePath: relPath });
      } else if (lowerName.endsWith('.edf')) {
        edfFiles.push({
          file, name: file.name, sizeBytes: file.size, relativePath: relPath,
          isValid: file.size > 0, error: file.size > 0 ? null : 'Empty EDF file',
        });
      }
    });

    if (detectedPatient && !patientName) {
      setPatientName(detectedPatient);
    }

    setSelectedFiles(edfFiles);
    setAnnotationFiles(sidecars);
    setUploadError(null);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);

    if (e.dataTransfer.items) {
      const files = [];
      const items = e.dataTransfer.items;
      for (let i = 0; i < items.length; i++) {
        const item = items[i];
        if (item.kind === 'file') {
          const file = item.getAsFile();
          if (file) files.push(file);
        }
      }
      handleFilesSelected(files);
    } else if (e.dataTransfer.files) {
      handleFilesSelected(e.dataTransfer.files);
    }
  };

  const removeFile = (indexToRemove) => {
    setSelectedFiles((prev) => prev.filter((_, idx) => idx !== indexToRemove));
  };

  const validFiles = selectedFiles.filter((f) => f.isValid);
  const invalidFiles = selectedFiles.filter((f) => !f.isValid);
  const totalSizeBytes = validFiles.reduce((acc, f) => acc + f.sizeBytes, 0);

  const formatBytes = (bytes) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const handleStartAnalysis = async () => {
    if (validFiles.length === 0) return;

    setUploading(true);
    setUploadError(null);

    try {
      const filesToUpload = validFiles.map((vf) => vf.file);
      const relativePaths = validFiles.map((vf) => vf.relativePath);

      // 1. Upload files
      const uploadRes = await api.uploadFolder(
        filesToUpload,
        relativePaths,
        patientName,
        annotationFiles.map((item) => item.file),
        annotationFiles.map((item) => item.relativePath),
        (event) => setUploadProgress(event),
      );
      const analysisId = uploadRes.analysis_id;

      // 2. Start background inference
      await api.startAnalysis(analysisId);

      // 3. Navigate to progress tracking screen
      navigate(`/processing/${analysisId}`);
    } catch (err) {
      console.error('Upload failed:', err);
      setUploadError(
        err.response?.data?.detail?.message ||
        err.response?.data?.detail ||
        'Failed to upload and start analysis. Verify server connection.'
      );
      setUploading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Upload Zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        className={`p-10 rounded-2xl border-2 border-dashed transition-all duration-200 text-center flex flex-col items-center justify-center space-y-4 ${
          isDragging
            ? 'border-cyan-accent bg-cyan-accent/10 glow-cyan'
            : 'border-border hover:border-cyan-accent/50 bg-background-panel/60'
        }`}
      >
        <div className="w-16 h-16 rounded-2xl bg-cyan-accent/10 border border-cyan-accent/30 flex items-center justify-center text-cyan-accent shadow-sm">
          <FolderUp className="w-8 h-8" />
        </div>

        <div>
          <h2 className="text-base font-semibold text-text-primary">
            Upload Patient EEG Directory
          </h2>
          <p className="text-xs text-text-muted mt-1 max-w-md">
            Drag and drop a complete patient folder containing multi-channel <code className="text-cyan-accent font-mono">.edf</code> recordings, or choose below.
          </p>
        </div>

        <div className="flex items-center space-x-3 pt-2">
          {/* Native Folder Selector */}
          <button
            type="button"
            onClick={() => folderInputRef.current?.click()}
            className="px-4 py-2 rounded-lg text-xs font-semibold bg-cyan-accent text-background hover:bg-cyan-accent/90 transition-all shadow-sm flex items-center space-x-2 cursor-pointer"
          >
            <FolderUp className="w-4 h-4" />
            <span>Select Folder</span>
          </button>

          {/* Individual Files Fallback */}
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="px-4 py-2 rounded-lg text-xs font-medium bg-background border border-border text-text-secondary hover:text-text-primary hover:border-text-muted transition-colors cursor-pointer"
          >
            <span>Select EDF Files</span>
          </button>
        </div>

        <input
          ref={folderInputRef}
          type="file"
          webkitdirectory="true"
          directory="true"
          multiple
          className="hidden"
          onChange={(e) => e.target.files && handleFilesSelected(e.target.files)}
        />
        <input
          ref={fileInputRef}
          type="file"
          accept=".edf"
          multiple
          className="hidden"
          onChange={(e) => e.target.files && handleFilesSelected(e.target.files)}
        />
      </div>

      {uploadError && (
        <div className="p-4 rounded-xl bg-rose-accent/10 border border-rose-accent/30 text-rose-accent text-xs flex items-center space-x-2">
          <ShieldAlert className="w-4 h-4 shrink-0" />
          <span>{uploadError}</span>
        </div>
      )}

      {/* Selected Recordings List */}
      {selectedFiles.length > 0 && (
        <div className="space-y-4">
          {/* Summary Bar */}
          <div className="p-4 rounded-xl bg-background-panel border border-border flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div className="flex items-center space-x-4">
              <div>
                <span className="text-[10px] text-text-muted uppercase tracking-wider block font-medium">Patient Identifier</span>
                <input
                  type="text"
                  value={patientName}
                  onChange={(e) => setPatientName(e.target.value)}
                  placeholder="e.g. CHB01"
                  className="mt-0.5 text-xs font-mono font-bold bg-background border border-border rounded px-2.5 py-1 text-cyan-accent focus:outline-none focus:border-cyan-accent uppercase"
                />
              </div>

              <div className="border-l border-border pl-4">
                <span className="text-[10px] text-text-muted uppercase tracking-wider block font-medium">Valid Recordings</span>
          <span className="text-xs font-mono font-bold text-text-primary">{validFiles.length} EDF files ({formatBytes(totalSizeBytes)})</span>
              </div>
        {annotationFiles.length > 0 && <div className="border-l border-border pl-4">
          <span className="text-[10px] text-text-muted uppercase tracking-wider block font-medium">Annotations kept separately</span>
          <span className="text-xs font-mono font-bold text-text-primary">{annotationFiles.length} sidecars</span>
        </div>}

              {invalidFiles.length > 0 && (
                <div className="border-l border-border pl-4">
                  <span className="text-[10px] text-rose-accent uppercase tracking-wider block font-medium">Rejected</span>
                  <span className="text-xs font-mono font-bold text-rose-accent">{invalidFiles.length} invalid</span>
                </div>
              )}
            </div>

            <button
              onClick={handleStartAnalysis}
              disabled={uploading || validFiles.length === 0}
              className="px-5 py-2.5 rounded-lg text-xs font-bold bg-cyan-accent text-background hover:bg-cyan-accent/90 disabled:opacity-50 transition-all flex items-center justify-center space-x-2 shadow-md shrink-0 cursor-pointer"
            >
              <Play className="w-3.5 h-3.5 fill-current" />
              <span>{uploading ? 'Uploading EDF recordings…' : 'Start EEG Analysis'}</span>
            </button>
          </div>

          {uploading && <div className="px-4 py-3 rounded-xl border border-cyan-accent/30 bg-cyan-accent/5 space-y-2">
            <div className="flex justify-between text-xs"><span className="text-text-secondary">Uploading and validating EDFs before analysis starts</span>
              {uploadProgress?.total > 0 && <span className="font-mono text-cyan-accent">{Math.floor(uploadProgress.loaded / uploadProgress.total * 100)}%</span>}
            </div>
            {uploadProgress?.total > 0 && <div className="h-1.5 rounded-full bg-background overflow-hidden"><div className="h-full bg-cyan-accent" style={{ width: `${Math.floor(uploadProgress.loaded / uploadProgress.total * 100)}%` }}/></div>}
          </div>}

          {/* Files List Table */}
          <div className="rounded-xl border border-border bg-background-panel overflow-hidden">
            <div className="px-4 py-3 border-b border-border bg-background-secondary/50 flex items-center justify-between text-xs font-semibold text-text-secondary">
              <span>Detected EDF Recordings ({selectedFiles.length})</span>
              <span>Validation</span>
            </div>

            <div className="divide-y divide-border/60 max-h-80 overflow-y-auto">
              {selectedFiles.map((fileObj, idx) => (
                <div key={idx} className="px-4 py-2.5 flex items-center justify-between hover:bg-background-secondary/30 text-xs">
                  <div className="flex items-center space-x-3 truncate">
                    <FileSpreadsheet className={`w-4 h-4 shrink-0 ${fileObj.isValid ? 'text-cyan-accent' : 'text-rose-accent'}`} />
                    <div className="truncate">
                      <span className="font-mono text-text-primary truncate block">{fileObj.name}</span>
                      <span className="text-[10px] text-text-muted">{formatBytes(fileObj.sizeBytes)} • {fileObj.relativePath}</span>
                    </div>
                  </div>

                  <div className="flex items-center space-x-3 shrink-0">
                    {fileObj.isValid ? (
                      <span className="flex items-center space-x-1 text-[11px] text-emerald-accent">
                        <CheckCircle2 className="w-3.5 h-3.5" />
                        <span>Ready</span>
                      </span>
                    ) : (
                      <span className="flex items-center space-x-1 text-[11px] text-rose-accent">
                        <AlertCircle className="w-3.5 h-3.5" />
                        <span>{fileObj.error}</span>
                      </span>
                    )}

                    <button
                      onClick={() => removeFile(idx)}
                      className="p-1 rounded text-text-muted hover:text-rose-accent hover:bg-background transition-colors"
                      title="Remove file"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
