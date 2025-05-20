import React, { useState, useEffect } from "react";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
} from "react-router-dom";
import "./App.css";
import Sidebar from "./pages/global/Sidebar/Sidebar";
import Topbar from "./pages/global/Topbar/Topbar";
import Login from "./pages/LoginPage/LoginPage";
import Register from "./pages/RegisterPage/Register";
import Home from "./pages/HomePage/Home";
import Dashboard from "./pages/Dashboard/Dashboard";
import Video from "./pages/Video/Video";
import Bottombar from "./pages/global/Bottombar/Bottombar";
import Analyze from "./pages/Video/Analyze/Analyze";
import DetectionResult from "./pages/DetectionResult/DetectionResult";
import CameraResult from "./pages/CameraResult/CameraResult";
import ReviewResult from "./pages/ReviewPage/ReviewPage";
import ReviewStreamVideo from "./pages/ReviewPage/ReviewStreamVideo";
import Account from "./pages/Account/Account";
import Setting from "./pages/SettingPage/Setting";
import Manage from "./pages/Manage/Manage";
import UsersPage from "./pages/UserPages/UserPages";
import UserDetailPage from "./pages/UserPages/UserDetail/UserDetail";
import UserHistoryPage from "./pages/UserHistory/UserHistory";
import GetUserHistoryPage from "./pages/UserHistory/GetUserHistory";
import { ColorModeContext, useMode } from "./theme";
import { ThemeProvider, CssBaseline } from "@mui/material";
import { useDispatch, useSelector } from "react-redux";
import { fetchCurrentUser } from "./redux/apiRequest";
import ProtectedRoute from "./components/ProtectedRoute/ProtectedRoute";
import { ConfigProvider } from "antd";

function App() {
  const [theme, colorMode] = useMode();
  const [isSidebar, setIsSidebar] = useState(true);
  const dispatch = useDispatch();
  
  useEffect(() => {
    fetchCurrentUser(dispatch);
  }, [dispatch]);

  const user = useSelector((state) => state.user.user);

  return (
    <ColorModeContext.Provider value={colorMode}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <ConfigProvider>
          <Routes>
            <Route path="/" element={<Navigate to="/login" />} />
            <Route path="/login" element={<Login />} />
            <Route path="/register" element={<Register />} />
            <Route
              path="/*"
              element={
                <ProtectedRoute>
                  <div className="app-container">
                    <Topbar setIsSidebar={setIsSidebar} />
                    <div className="body-layout">
                      <Sidebar isSidebar={isSidebar} />
                      <main
                        className={`main-content ${
                          isSidebar ? "" : "collapsed"
                        }`}
                      >
                        <Routes>
                         {/* Chỉ User */}
                          {user?.role === "user" && (
                            <>
                              <Route path="/home" element={<Home />} />
                              <Route path="/video" element={<Video />} />
                              <Route path="/video/analyze" element={<Analyze />} />
                              <Route path="/video/detectionresult" element={<DetectionResult />} />
                                <Route path="/video/stream/review" element={<ReviewStreamVideo />} />
                              <Route path="/video/cameraresult" element={<CameraResult />} />
                              <Route path="/video/review" element={<ReviewResult />} />
                              <Route path="/settings" element={<Setting />} />
                            </>
                          )}

                          {/* Chỉ Admin */}
                          {user?.role === "admin" && (
                            <>
                              <Route path="/dashboard" element={<Dashboard />} />
                              <Route path="/manage" element={<Manage />} />
                              <Route path="/manage/review" element={<ReviewResult />} />
                              <Route path="/user-page" element={<UsersPage />} />
                              <Route path="/user-page/user-detail/:userId" element={<UserDetailPage />} />
                              <Route path="/user-page/user-detail/:userId/review" element={<ReviewResult />} />
                              <Route path="/user-page/user-detail/:userId/history" element={<GetUserHistoryPage />} />
                            </>
                          )}

                          {/* Cả hai vai trò */}
                          <Route path="/account" element={<Account />} />
                          <Route path="/account/history" element={<UserHistoryPage />} />
                          <Route path="/account/review" element={<ReviewResult />} />
                          <Route path="/account/history/review" element={<ReviewResult />} />
                        </Routes>
                      </main>
                    </div>
                    <Bottombar />
                  </div>
                </ProtectedRoute>
              }
            />
          </Routes>
        </ConfigProvider>
      </ThemeProvider>
    </ColorModeContext.Provider>
  );
}

export default App;
