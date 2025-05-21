import React, { useEffect, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import Snackbar from "@mui/material/Snackbar";
import Alert from "@mui/material/Alert";
import SummaryApi from "../../api/api";
import "./ResultDisplay.css";

const ResultDisplay = () => {
  const { state } = useLocation();
  const navigate = useNavigate();
  const audioRef = useRef(null);
  // const { volume } = useContext(AlertSettingsContext);

  // States
  const [currentFrame, setCurrentFrame] = useState(null);
  const [isPlaying, setIsPlaying] = useState(true);
  const [firePercent, setFirePercent] = useState(0);
  const [backgroundPercent, setBackgroundPercent] = useState(100);
  const [fireDetectionTimes, setFireDetectionTimes] = useState([]);
  const [showFireAlert, setShowFireAlert] = useState(false);
  const [status, setStatus] = useState("Đang kết nối...");
  const [progressPercent, setProgressPercent] = useState(0);
  const [videoEnded, setVideoEnded] = useState(false);
  const [isCameraStarting, setIsCameraStarting] = useState(true);
  const [hasShownFireAlert, setHasShownFireAlert] = useState(false);
  const [processedVideoUrl, setProcessedVideoUrl] = useState(null);
  const [currentTime, setCurrentTime] = useState("--:--");
  const [framesQueue, setFramesQueue] = useState([]);
  const [frameProcessing, setFrameProcessing] = useState(false);
  const [frameRate, setFrameRate] = useState(0);

  // Refs
  const wsRef = useRef(null);
  const lastFrameTimeRef = useRef(Date.now());
  const frameCountRef = useRef(0);
  const frameRateIntervalRef = useRef(null);

  const {
    mode,
    videoFile,
    videoUrl,
    notificationEnabled = false,
  } = state || {};

  // Kết nối WebSocket khi mode === 'video'
  useEffect(() => {
    if (mode !== "video") return;

    setStatus("Đang kết nối WebSocket...");
    // const wsUrl = "ws://localhost:8000/api/v1/ws/direct-process";
    const wsUrl = SummaryApi.directProcessWS;
    wsRef.current = new WebSocket(wsUrl);

    // Đo frameRate
    frameRateIntervalRef.current = setInterval(() => {
      const fps = frameCountRef.current;
      setFrameRate(fps);
      frameCountRef.current = 0;
    }, 1000);

    wsRef.current.onopen = () => {
      setStatus("Đã kết nối, đang xử lý...");

      const token = localStorage.getItem("access_token");
      if (token) {
        wsRef.current.send(JSON.stringify({ token }));
      }

      const metadata = {
        type: videoFile ? "upload" : "youtube",
        ...(videoFile ? {} : { youtube_url: videoUrl }),
      };
      wsRef.current.send(JSON.stringify(metadata));

      if (videoFile) {
        const reader = new FileReader();

        reader.onprogress = (e) => {
          if (e.lengthComputable) {
            setProgressPercent(Math.round((e.loaded / e.total) * 50));
          }
        };

        reader.onload = () => {
          wsRef.current.send(reader.result);
          setProgressPercent(50);
        };

        reader.readAsArrayBuffer(videoFile);
      }
    };

    wsRef.current.onmessage = (event) => {
      if (event.data instanceof Blob) {
        // Đếm số frame nhận được để tính FPS
        frameCountRef.current++;

        // Thêm frame mới vào queue
        const frameUrl = URL.createObjectURL(event.data);
        setFramesQueue(prev => [...prev, frameUrl]);
      } else {
        try {
          const data = JSON.parse(event.data);

          if (data.status === "frame") {
            const frameInfo = data.frame_info;
            const roundedFirePercent =
              Math.round((frameInfo.total_area || 0) * 100) / 100;
            const roundedBackgroundPercent =
              Math.round((100 - roundedFirePercent) * 100) / 100;

            setFirePercent(roundedFirePercent);
            setBackgroundPercent(roundedBackgroundPercent);

            if (frameInfo.fire_detected) {
              setFireDetectionTimes((prev) => [...prev, frameInfo.video_time]);
            }
          } else if (data.status === "progress") {
            setProgressPercent(100);
          } else if (data.status === "completed") {
            setStatus("Xử lý hoàn thành");
            if (data.processed_url) {
              setProcessedVideoUrl(data.processed_url);
            }
            setProgressPercent(100);
            setVideoEnded(true);
          } else if (data.status === "error") {
            setStatus(`Lỗi: ${data.message}`);
          }
        } catch (e) {
          console.error("Lỗi xử lý dữ liệu từ server:", e);
        }
      }
    };

    wsRef.current.onerror = (error) => {
      setStatus("Lỗi kết nối WebSocket");
      console.error("WebSocket error:", error);
    };

    wsRef.current.onclose = () => {
      setStatus("Kết nối đã đóng");
    };

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }

      // Dọn dẹp interval đo framerate
      if (frameRateIntervalRef.current) {
        clearInterval(frameRateIntervalRef.current);
      }

      // Dọn dẹp URL objects
      if (currentFrame) {
        URL.revokeObjectURL(currentFrame);
      }

      // Dọn dẹp queue
      framesQueue.forEach(url => URL.revokeObjectURL(url));
    };
  }, [mode, videoFile, videoUrl, notificationEnabled]);

  // Xử lý queue frames
  useEffect(() => {
    if (!isPlaying || framesQueue.length === 0 || frameProcessing) return;

    // Đánh dấu đang xử lý để tránh xử lý nhiều frames cùng lúc
    setFrameProcessing(true);

    // Lấy frame tiếp theo từ queue
    const nextFrame = framesQueue[0];
    const remainingFrames = framesQueue.slice(1);

    // Giải phóng frame cũ nếu có
    if (currentFrame) {
      URL.revokeObjectURL(currentFrame);
    }

    // Hiển thị frame mới
    setCurrentFrame(nextFrame);
    setFramesQueue(remainingFrames);

    // Đánh dấu đã xử lý xong
    setFrameProcessing(false);

    // Nếu queue quá dài (>10 frames), giữ lại 10 frames gần nhất để tránh memory leak
    if (remainingFrames.length > 10) {
      const framesToKeep = remainingFrames.slice(-10);
      const framesToRemove = remainingFrames.slice(0, -10);

      // Giải phóng frames cũ
      framesToRemove.forEach(url => URL.revokeObjectURL(url));
      setFramesQueue(framesToKeep);
    }
  }, [isPlaying, framesQueue, frameProcessing, currentFrame]);

  // Các state bổ sung cho camera mode
  const [cameraStatus, setCameraStatus] = useState("disconnected");
  const [cameraOn, setCameraOn] = useState(false);
  const [frameBase64, setFrameBase64] = useState(null);

  // Kết nối WebSocket khi mode === 'camera'
  useEffect(() => {
    if (mode !== "camera") return;
    setCameraStatus("connecting");
    setIsCameraStarting(true);
    console.log("Đang kết nối đến WebSocket...");

    // const wsUrl =
    //   process.env.REACT_APP_WS_URL || "ws://localhost:8000/api/v1/ws/fire";
    const wsUrl = SummaryApi.fireCameraWS;
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    // Đo frameRate cho camera mode
    frameRateIntervalRef.current = setInterval(() => {
      const fps = frameCountRef.current;
      setFrameRate(fps);
      frameCountRef.current = 0;
    }, 1000);

    ws.onopen = () => {
      console.log("Kết nối WebSocket thành công");
      setCameraStatus("connected");
      setCameraOn(true);
      // setIsCameraStarting(true);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.status) {
          if (data.status === "ready") {
            console.log("Camera đã sẵn sàng");
            setCameraStatus("connected");
            setCameraOn(true);
            setIsCameraStarting(false);
          } else if (data.status === "error") {
            console.error("Lỗi camera:", data.message);
            setCameraStatus("error");
            setCameraOn(false);
            setIsCameraStarting(false);
          }
          return;
        }

        // Đếm frame để tính FPS
        frameCountRef.current++;

        // Xử lý frame data
        setFrameBase64(data.frame);

        setFirePercent(Number(data.total_area.toFixed(2)));
        setBackgroundPercent(Number((100 - data.total_area).toFixed(2)));

        if (data.fire_detected) {
          console.log("Phát hiện cháy tại:", data.time);
          setFireDetectionTimes((prev) => [...prev, data.time]);
        }
      } catch (error) {
        console.error("Lỗi phân tích dữ liệu:", error);
      }
    };

    ws.onerror = (error) => {
      console.error("Lỗi WebSocket:", error);
      setCameraStatus("error");
      setCameraOn(false);
      setIsCameraStarting(false);
    };

    ws.onclose = () => {
      console.log("WebSocket đã đóng");
      setCameraStatus("disconnected");
      setCameraOn(false);
      setIsCameraStarting(false);
    };

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }

      if (frameRateIntervalRef.current) {
        clearInterval(frameRateIntervalRef.current);
      }
    };
  }, [mode, notificationEnabled]);


  useEffect(() => {
    if (mode !== "camera") return;

    const interval = setInterval(() => {
      const now = new Date();
      const formattedTime = now.toLocaleTimeString("vi-VN", {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      });
      setCurrentTime(formattedTime);
    }, 1000);

    return () => clearInterval(interval);
  }, [mode]);

  // Hàm tắt camera
  const handleTurnOffCamera = () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      console.log("Đang gửi lệnh tắt camera...");
      wsRef.current.send("stop");
      wsRef.current.close();
    }
    setCameraStatus("disconnected");
    setCameraOn(false);
    setFrameBase64(null);
    setIsCameraStarting(false);
    console.log("Camera đã tắt");
  };

  // Cảnh báo bằng giọng nói
  const speakAlert = (text) => {
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "vi-VN";
    // utterance.volume = volume / 100;
    utterance.rate = 1;
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utterance);
  };

  // Hiện cảnh báo âm thanh khi phát hiện cháy
  useEffect(() => {
    if (
      fireDetectionTimes.length === 0 || // Chưa phát hiện cháy nào
      !notificationEnabled || // Thông báo không được bật
      hasShownFireAlert // Đã hiển thị cảnh báo rồi
    ) {
      return;
    }

    // Chỉ hiển thị khi có lần phát hiện cháy đầu tiên
    if (fireDetectionTimes.length === 1) {
      setShowFireAlert(true);
      setHasShownFireAlert(true);
      speakAlert("Cảnh báo phát hiện cháy, vui lòng kiểm tra ngay!");

      const timer = setTimeout(() => {
        setShowFireAlert(false);
      }, 5000);

      return () => clearTimeout(timer);
    }
  }, [fireDetectionTimes, notificationEnabled, hasShownFireAlert]);

  const handleTurnOnCamera = () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) return;

    setCameraStatus("connecting");
    setIsCameraStarting(true);

    const wsUrl =
      process.env.REACT_APP_WS_URL || "ws://localhost:8000/api/v1/ws/fire";
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log("Bật lại WebSocket thành công");
      setCameraStatus("connected");
      setCameraOn(true);
      setIsCameraStarting(false);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.status === "ready") {
          console.log("Camera đã sẵn sàng");
          setCameraStatus("connected");
          setCameraOn(true);
          setIsCameraStarting(false);
        } else if (data.status === "error") {
          console.error("Lỗi camera:", data.message);
          setCameraStatus("error");
          setCameraOn(false);
          setIsCameraStarting(false);
        }

        if (data.frame) {
          frameCountRef.current++;
          setFrameBase64(data.frame);
          setFirePercent(Number(data.total_area.toFixed(2)));
          setBackgroundPercent(Number((100 - data.total_area).toFixed(2)));

          if (data.fire_detected) {
            console.log("Phát hiện cháy tại:", data.time);
            setFireDetectionTimes((prev) => [...prev, data.time]);
          }
        }
      } catch (error) {
        console.error("Lỗi phân tích dữ liệu:", error);
      }
    };

    ws.onerror = (error) => {
      console.error("Lỗi WebSocket:", error);
      setCameraStatus("error");
      setCameraOn(false);
      setIsCameraStarting(false);
    };

    ws.onclose = () => {
      console.log("WebSocket đã đóng");
      setCameraStatus("disconnected");
      setCameraOn(false);
      setIsCameraStarting(false);
    };
  };

  const handlePlayPause = () => setIsPlaying((prev) => !prev);

  const handleCreateNew = () => {
    if (mode === "camera") {
      setTimeout(() => {
        navigate("/video");
      }, 10);
    } else {
      navigate("/video");
    }
  };

  const lastDetectionTime =
    fireDetectionTimes.length > 0
      ? formatTime(fireDetectionTimes[fireDetectionTimes.length - 1])
      : null;

  return (
    <div className="result-page">
      <Snackbar
        open={showFireAlert}
        autoHideDuration={5000}
        onClose={() => setShowFireAlert(false)}
        anchorOrigin={{ vertical: "top", horizontal: "center" }}
      >
        <Alert severity="error" variant="filled" sx={{ width: "100%" }}>
          Cảnh báo phát hiện cháy!
        </Alert>
      </Snackbar>

      <div className="video-result-frame" style={{ position: "relative" }}>
        {mode === "camera" ? (
          frameBase64 ? (
            <img
              src={`data:image/jpeg;base64,${frameBase64}`}
              alt="Camera Frame"
              style={{ width: "100%", height: "62vh", objectFit: "cover" }}
            />
          ) : isCameraStarting ? (
            <div
              className="loading-placeholder"
              style={{
                width: "100%",
                height: "62vh",
                display: "flex",
                justifyContent: "center",
                alignItems: "center",
              }}
            >
              {/* Chờ khởi động cam... */}
            </div>
          ) : (
            <div
              className="loading-placeholder"
              style={{
                width: "100%",
                height: "62vh",
                display: "flex",
                justifyContent: "center",
                alignItems: "center",
              }}
            >
              {/* Camera đã tắt */}
            </div>
          )
        ) : (
          <>
            {currentFrame ? (
              <img
                src={currentFrame}
                alt="Processed Frame"
                style={{ width: "100%", height: "62vh", objectFit: "contain" }}
              />
            ) : (
              <div className="loading-placeholder">
                <div className="spinner"></div>
                <div className="loading-placeholders">
                  Đang xử lý, vui lòng chờ...
                </div>
              </div>
            )}

            {videoEnded && (
              <div className="video-overlay">
                <p>Streaming kết thúc !</p>
                <button
                  className="review-button"
                  onClick={() =>
                    navigate("/video/stream/review", {
                      state: { processedVideoUrl },
                    })
                  }
                >
                  Xem lại kết quả
                </button>
              </div>
            )}
          </>
        )}
      </div>
      {mode === "video" && (
        <div
          className="custom-control-progress"
          style={{
            visibility: currentFrame ? "visible" : "hidden",
          }}
        >
          <div className="progress-fill"></div>
          <div className="progress-thumb"></div>
        </div>
      )}

      <div className="stats-bar">
        <div className="left-column">
          <div className="progress-text">Fire</div>
          <div className="progress-container">
            <div className="progress-bar-wrapper">
              <div
                className="progress-fire"
                style={{ width: `${firePercent}%` }}
              ></div>
            </div>
            <span className="progress-percent">{firePercent}%</span>
          </div>

          <div className="progress-text">Background</div>
          <div className="progress-container">
            <div className="progress-bar-wrapper">
              <div
                className="progress-bg"
                style={{ width: `${backgroundPercent}%` }}
              ></div>
            </div>
            <span className="progress-percent">{backgroundPercent}%</span>
          </div>


        </div>

        <div className="right-column">
          <div className="detection-time">
            <strong>
              {mode === "camera"
                ? "Thời gian hiện tại"
                : "Thời điểm phát hiện cháy"}
              : {mode === "camera" ? currentTime : lastDetectionTime || "--:--"}
            </strong>
          </div>

          <div className="turnoff-button">
            {mode === "camera" && (
              <div style={{ display: "flex", gap: "10px", marginTop: "16px" }}>
                <button className="new-button" onClick={handleTurnOffCamera} disabled={!cameraOn}>
                  Tắt
                </button>
                <button className="new-button" onClick={handleTurnOnCamera} disabled={cameraOn}>
                  Bật
                </button>
              </div>
            )}

            <button className="new-button" onClick={handleCreateNew}>
              Tạo mới
            </button>
          </div>
        </div>
      </div>

      {/* <audio ref={audioRef} src="/fire-alert.mp3" preload="auto" /> */}
    </div>
  );
};

// Format thời gian dạng mm:ss từ giây
function parseTimeToSeconds(time) {
  if (typeof time === "number") return time;

  if (typeof time === "string") {
    const parts = time.split(":").map(Number);
    if (parts.length === 3) {
      // HH:MM:SS
      return parts[0] * 3600 + parts[1] * 60 + parts[2];
    } else if (parts.length === 2) {
      // MM:SS
      return parts[0] * 60 + parts[1];
    }
  }

  return 0; // mặc định nếu không parse được
}

function formatTime(time) {
  const seconds = parseTimeToSeconds(time);

  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins.toString().padStart(2, "0")}:${secs
    .toString()
    .padStart(2, "0")}`;
}

export default ResultDisplay;
