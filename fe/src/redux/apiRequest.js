import axios from "axios";
import {
  loginFailed,
  loginStart,
  loginSuccess,
} from "./authSlice";
import {
  getUserStart,
  getUserSuccess,
  getUserFailure,
} from './userSlice';

import { baseURL } from "../api/api";
import SummaryApi from "../api/api";

export const loginUser = (email, password, dispatch, navigate) => {
  dispatch(loginStart()); 

 axios({
    method: SummaryApi.login.method,
    url: baseURL + SummaryApi.login.url,
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
    },
    data: new URLSearchParams({
      username: email,
      password: password,
    }),
  })
    .then((response) => {
      const data = response.data;
      dispatch(loginSuccess(data));
      localStorage.setItem("access_token", data.access_token);
      navigate("/home");
    })
    .catch((err) => {
      dispatch(loginFailed()); 
      console.error("Login failed:", err);
    });
};

export const fetchCurrentUser = async (dispatch) => {
  dispatch(getUserStart());
  try {
    const res = await axios({
      method: SummaryApi.getCurrentUser.method,
      url: baseURL + SummaryApi.getCurrentUser.url,
      headers: {
        Authorization: `Bearer ${localStorage.getItem("access_token")}`,
      },
    });
    dispatch(getUserSuccess(res.data));
  } catch (err) {
    console.error('Lỗi khi lấy thông tin người dùng:', err);
    dispatch(getUserFailure(err.response?.data || err.message));
  }
};