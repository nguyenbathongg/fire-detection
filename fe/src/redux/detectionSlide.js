// detectionSlice.js
import { createSlice } from '@reduxjs/toolkit';

const detectionSlice = createSlice({
  name: 'detection',
  initialState: {
    frames: [],       
    statusText: '',
    processedUrl: '',
    detectionLogs: [],
  },
  reducers: {
    addFrame(state, action) {
      state.frames.push(action.payload);
    },
    setStatus(state, action) {
      state.statusText = action.payload;
    },
    setProcessedUrl(state, action) {
      state.processedUrl = action.payload;
    },
    addLog(state, action) {
      state.detectionLogs.push(action.payload);
    },
    resetDetection(state) {
      state.frames = [];
      state.statusText = '';
      state.processedUrl = '';
      state.detectionLogs = [];
    }
  },
});

export const { addFrame, setStatus, setProcessedUrl, addLog, resetDetection } = detectionSlice.actions;
export default detectionSlice.reducer;
