
import React, { useState, useEffect, useRef } from "react"; 
import { useLocation, useNavigate } from "react-router-dom";
import { useSelector } from "react-redux";
import { baseURL } from "../../../api/api";
import SummaryApi from "../../../api/api";

const Analyze = () => {
  const { state } = useLocation();
  const navigate = useNavigate();
  const { videoFile, videoUrl, mode } = useSelector((state) => state.video);
  const [activeTab, setActiveTab] = useState("video");
  const [localVideoSrc, setLocalVideoSrc] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);
  const [status, setStatus] = useState("");
  const [progressPercent, setProgressPercent] = useState(0);
  const [processedVideoUrl, setProcessedVideoUrl] = useState(null);
  const token = localStorage.getItem("access_token");
  const user = useSelector((state) => state.user.user);
  const [notificationEnabled, setNotificationEnabled] = useState(false);

  useEffect(() => {
    const fetchNotificationSettings = async () => {
      try {
         const res = await fetch(baseURL + SummaryApi.notificationSettings.url, {
          headers: {
            accept: "application/json",
            Authorization: `Bearer ${localStorage.getItem("access_token")}`,
          },
        });

        if (!res.ok) {
          console.warn("Không lấy được cài đặt thông báo:", res.status);
          return;
        }

        const data = await res.json();
        setNotificationEnabled(data.enable_website_notification === true);
      } catch (error) {
        console.error("Lỗi khi lấy cài đặt thông báo:", error);
      }
    };

    fetchNotificationSettings();
  }, []);

  const socketRef = useRef(null);

  const selectedFile = videoFile || null;
  const youtubeUrl = videoUrl || "";
  const authToken = token || "";

  const addLog = (message, type = "info") => {
    console.log(`[${type}] ${message}`);
  };

  useEffect(() => {
    if (videoFile) {
      const objectUrl = URL.createObjectURL(videoFile);
      setLocalVideoSrc(objectUrl);
      return () => {
        URL.revokeObjectURL(objectUrl);
      };
    } else {
      setLocalVideoSrc(null);
    }
  }, [videoFile]);

  const extractYouTubeID = (url) => {
    const regex = /(?:\?v=|\/embed\/|\.be\/|\/shorts\/)([a-zA-Z0-9_-]{11})/;
    const match = url?.match(regex);
    return match ? match[1] : "";
  };

  const startCamera = () => {
    navigate("/video/cameraresult", { state: { mode: "camera" } });
  };

  const handleBackToUpload = () => {
    navigate("/video");
  };

  const handleAnalyze = () => {
  setErrorMsg(null);

  // Nếu mode là camera thì đi thẳng
  if (mode === "camera") {
    navigate("/video/cameraresult", { state: { mode } });
    return;
  }

  // Kiểm tra input
  if (!selectedFile && !youtubeUrl) {
    setErrorMsg("Chưa chọn video hoặc nhập link YouTube.");
    return;
  }

  if (selectedFile && selectedFile.size > 15 * 1024 * 1024) {
    setErrorMsg("File quá lớn! Kích thước tối đa 15MB.");
    return;
  }

 navigate("/video/detectionresult", {
    state: {
      mode: "video",
      videoFile: selectedFile,
      videoUrl: youtubeUrl,
      notificationEnabled
    }
  });
};


  return (
    <div className="video-page">
      <div className="tabs">
        <button
          className={`tab-button ${activeTab === "video" ? "active" : ""}`}
          onClick={handleBackToUpload}
        >
          Video
        </button>
        <button
          className={`tab-button ${activeTab === "camera" ? "active" : ""}`}
          onClick={() => setActiveTab("camera")}
        >
          Camera
        </button>
      </div>

      {activeTab === "video" && (
        <div className="video-preview-section">
          <div className="video-frame">
            {videoFile || videoUrl ? (
              videoFile ? (
                <video
                  src={localVideoSrc}
                  controls
                  className="analyze-video-player"
                />
              ) : (
                <iframe
                  src={`https://www.youtube.com/embed/${extractYouTubeID(
                    videoUrl
                  )}`}
                  title="YouTube video player"
                  frameBorder="0"
                  allowFullScreen
                  className="analyze-video-player"
                />
              )
            ) : (
              <p>Không có video được chọn.</p>
            )}
          </div>

          <div style={{ marginTop: "16px" }}>
            <button className="analyze-button" onClick={handleAnalyze}>
              Phân tích phát hiện cháy từ Video
            </button>
          </div>
          {errorMsg && (
            <p style={{ color: "red", marginTop: "10px" }}>{errorMsg}</p>
          )}
          {status === "loading" && (
            <div style={{ marginTop: "10px" }}>
              <div className="spinner" />
              <p>Đang xử lý, vui lòng chờ...</p>
            </div>
          )}

          {progressPercent > 0 && (
            <p>Tiến độ: {progressPercent} %</p>
          )}
        </div>
      )}

      {activeTab === "camera" && (
        <div className="tab-content camera-tab">
          <video
            id="cameraFeed"
            autoPlay
            playsInline
            muted
            className="camera-video"
          />
          <button className="camera-button" onClick={startCamera}>
            Bắt đầu Camera
          </button>
        </div>
      )}
      <style>{`
        .spinner {
          margin: 0 auto;
          width: 40px;
          height: 40px;
          border: 4px solid #ccc;
          border-top-color: #1d72b8;
          border-radius: 50%;
          animation: spin 1s linear infinite;
        }
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
};

export default Analyze;