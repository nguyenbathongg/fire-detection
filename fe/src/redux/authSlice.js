import { createSlice } from "@reduxjs/toolkit";

const authSlice = createSlice({
  name: "auth",
  initialState: {
    login: {
      currentUser: null,
      isFetching: false, 
      error: false 
    }, 
    register: {
      isFetching: false, 
      error: false, 
      success: false
    },
    logout: {
      isFetching: false, 
      error: false, 
    }
  },
  reducers: {
    loginStart: (state) => {
      state.login.isFetching = true;
    },
    loginSuccess: (state, action) => {
      state.login.isFetching = false;
      state.login.currentUser = action.payload;
      state.login.error = false;
      localStorage.setItem("access_token", action.payload.access_token);
      localStorage.setItem("isAuthenticated", "true");
    },
    loginFailed: (state) => {
      state.login.isFetching = false;
      state.login.error = true;
    },
    registerStart: (state) => {
      state.register.isFetching = true;
    },
    registerSuccess: (state) => {
      state.register.isFetching = false;
      state.register.error = false;
      state.register.success = true;
    },
    registerFailed: (state) => {
      state.register.isFetching = false;
      state.register.error = true;
      state.register.success = false;
    },
    logOutSuccess: (state) => {
      state.logout.isFetching = true;
      state.login.currentUser = null;
      state.logout.error = false;
       localStorage.removeItem("access_token");
       localStorage.setItem("isAuthenticated", "false");
    },
    logOutFailed: (state) => {
      state.logout.isFetching = false;
      state.logout.error = true;
    },
    logOutStart: (state) =>{
      state.logout.isFetching = true;
    }
  }
})

export const {
  loginStart, 
  loginFailed,
  loginSuccess,
  registerStart, 
  registerSuccess, 
  registerFailed,
  logOutStart,
  logOutSuccess,
  logOutFailed
} = authSlice.actions;

export default authSlice.reducer;