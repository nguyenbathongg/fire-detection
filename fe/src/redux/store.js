import { configureStore } from "@reduxjs/toolkit";
// import { thunk } from "redux-thunk";
import authReducer from "./authSlice";
import videoReducer from "./videoSlice"
import userReducer from "./userSlice"
const store = configureStore({
  reducer: {
    auth: authReducer,
    video: videoReducer,
    user: userReducer,
   
  },
//   middleware: (getDefaultMiddleware) => getDefaultMiddleware().concat(thunk),
});

export default store;
