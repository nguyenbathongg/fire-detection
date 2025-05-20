import { createSlice } from '@reduxjs/toolkit';

const initialState = {
  user: null,
  loading: false,
  error: null,
};

const userSlice = createSlice({
  name: 'user',
  initialState,
  reducers: {
    getUserStart: (state) => {
      state.loading = true;
      state.error = null;
    },
    getUserSuccess: (state, action) => {
      state.user = action.payload;
      state.loading = false;
    },
    getUserFailure: (state, action) => {
      state.loading = false;
      state.error = action.payload;
    },
    logout: () => initialState,
  },
});

export const {
  getUserStart,
  getUserSuccess,
  getUserFailure,
  logout,
} = userSlice.actions;

export default userSlice.reducer;
