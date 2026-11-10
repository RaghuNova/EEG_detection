import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import MainLayout from './components/layout/MainLayout';
import Dashboard from './pages/Dashboard';
import PatientUpload from './pages/PatientUpload';
import Processing from './pages/Processing';
import PatientRecords from './pages/PatientRecords';
import MLAnalysis from './pages/MLAnalysis';
import CNNAnalysis from './pages/CNNAnalysis';
import AttentionAnalysis from './pages/AttentionAnalysis';
import ModelComparison from './pages/ModelComparison';
import EEGViewerPage from './pages/EEGViewerPage';
import Settings from './pages/Settings';
import PatientAnalysis from './pages/PatientAnalysis';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<MainLayout />}>
          <Route index element={<Dashboard />} />
          <Route path="upload" element={<PatientUpload />} />
          <Route path="records" element={<PatientRecords />} />
          <Route path="processing/:id" element={<Processing />} />
          <Route path="patient-analysis/:id" element={<PatientAnalysis />} />
          <Route path="analysis/ml" element={<MLAnalysis />} />
          <Route path="analysis/cnn" element={<CNNAnalysis />} />
          <Route path="analysis/attention" element={<AttentionAnalysis />} />
          <Route path="comparison" element={<ModelComparison />} />
          <Route path="viewer" element={<EEGViewerPage />} />
          <Route path="settings" element={<Settings />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
