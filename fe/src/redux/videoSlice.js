import { createSlice } from '@reduxjs/toolkit';

const videoSlice = createSlice({
  name: 'video',
  initialState: {
    videoFile: null,
    videoUrl: '',
    mode: 'video', 
  },
  reducers: {
    setVideoData: (state, action) => {
      const { videoFile, videoUrl, mode } = action.payload;
      state.videoFile = videoFile;
      state.videoUrl = videoUrl;
      state.mode = mode;
    },
    clearVideoData: (state) => {
      state.videoFile = null;
      state.videoUrl = '';
      state.mode = 'video';
    },
  },
});

export const { setVideoData, clearVideoData } = videoSlice.actions;
export default videoSlice.reducer;
