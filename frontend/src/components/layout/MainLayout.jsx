import React, { createContext, useContext, useState } from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import Sidebar from './Sidebar';
import Header from './Header';

export const AnalysisContext = createContext(null);

export const useAnalysis = () => useContext(AnalysisContext);

export default function MainLayout() {
  const [activeAnalysis, setActiveAnalysis] = useState(null);
  const [activeRecordingId, setActiveRecordingId] = useState(null);
  const [headerInfo, setHeaderInfo] = useState({ title: 'Dashboard', subtitle: 'EEG Intelligence & Inference' });
  const location = useLocation();

  const selectRecording = (recId) => {
    setActiveRecordingId(recId);
  };

  const getActiveRecording = () => {
    if (!activeAnalysis?.recordings) return null;
    return (
      activeAnalysis.recordings.find((r) => r.recording_id === activeRecordingId) ||
      activeAnalysis.recordings[0] ||
      null
    );
  };

  return (
    <AnalysisContext.Provider
      value={{
        activeAnalysis,
        setActiveAnalysis,
        activeRecordingId,
        setActiveRecordingId,
        getActiveRecording,
        setHeaderInfo,
      }}
    >
      <div className="flex h-screen bg-background overflow-hidden">
        {/* Persistent Sidebar */}
        <Sidebar />

        {/* Main Content Area */}
        <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
          <Header
            title={headerInfo.title}
            subtitle={headerInfo.subtitle}
            activeAnalysis={activeAnalysis}
            activeRecordingId={activeRecordingId}
            onSelectRecording={selectRecording}
          />

          <main className="flex-1 overflow-y-auto p-6 bg-background">
            <div className="max-w-7xl mx-auto">
              <Outlet />
            </div>
          </main>
        </div>
      </div>
    </AnalysisContext.Provider>
  );
}
