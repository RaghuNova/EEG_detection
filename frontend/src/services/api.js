import axios from 'axios';

const API_BASE = '/api';

export const api = {
  // Health
  getHealth: async () => {
    const res = await axios.get(`${API_BASE}/health`);
    return res.data;
  },

  // Upload
  uploadFolder: async (files, relativePaths = [], patientId = '', annotationFiles = [], annotationPaths = [], onProgress) => {
    const formData = new FormData();
    files.forEach((file, index) => {
      formData.append('files', file);
      if (relativePaths[index]) {
        formData.append('relative_paths', relativePaths[index]);
      }
    });
    annotationFiles.forEach((file, index) => {
      formData.append('annotation_files', file);
      if (annotationPaths[index]) formData.append('annotation_relative_paths', annotationPaths[index]);
    });
    if (patientId) {
      formData.append('patient_id_override', patientId);
    }

    const res = await axios.post(`${API_BASE}/patients/upload`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: onProgress ? (event) => onProgress({ loaded: event.loaded, total: event.total }) : undefined,
    });
    return res.data;
  },

  // Analysis Control & Status
  startAnalysis: async (analysisId) => {
    const res = await axios.post(`${API_BASE}/analysis/start/${analysisId}`);
    return res.data;
  },

  getAnalysisStatus: async (analysisId) => {
    const res = await axios.get(`${API_BASE}/analysis/${analysisId}/status`);
    return res.data;
  },

  getAnalysisResults: async (analysisId) => {
    const res = await axios.get(`${API_BASE}/analysis/${analysisId}/results`);
    return res.data;
  },

  getAnalysisSummary: async (analysisId) => {
    const res = await axios.get(`${API_BASE}/analysis/${analysisId}/summary`);
    return res.data;
  },

  getRecordings: async (analysisId) => {
    const res = await axios.get(`${API_BASE}/analysis/${analysisId}/recordings`);
    return res.data;
  },

  getRecordingDetail: async (analysisId, recordingId) => {
    const res = await axios.get(`${API_BASE}/analysis/${analysisId}/recordings/${recordingId}`);
    return res.data;
  },

  getModelResults: async (analysisId, recordingId, modelKey) => {
    const res = await axios.get(`${API_BASE}/analysis/${analysisId}/recordings/${recordingId}/models/${modelKey}`);
    return res.data;
  },

  getSignalSegment: async (analysisId, recordingId, startSeconds = 0, durationSeconds = 20, maxChannels = 16) => {
    const res = await axios.get(`${API_BASE}/analysis/${analysisId}/recordings/${recordingId}/signal`, {
      params: { start_seconds: startSeconds, duration_seconds: durationSeconds, max_channels: maxChannels },
    });
    return res.data;
  },

  getAnalysesHistory: async () => {
    const res = await axios.get(`${API_BASE}/analyses`);
    return res.data;
  },

  getPdfReportUrl: (analysisId) => {
    return `${API_BASE}/analysis/${analysisId}/report/pdf`;
  },
};
