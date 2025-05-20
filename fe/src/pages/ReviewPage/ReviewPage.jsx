import React, { useEffect, useRef, useState } from "react";
import ReactPlayer from "react-player";
import { useLocation, useNavigate } from "react-router-dom";
import { useDispatch, useSelector } from "react-redux";
import { fetchCurrentUser } from "../../redux/apiRequest";
import axios from "axios";
import Slider from "@mui/material/Slider";
import { Button } from "antd";
import { ArrowLeftOutlined } from "@ant-design/icons";
import ReplayIcon from "@mui/icons-material/Replay";
import PauseOutlinedIcon from "@mui/icons-material/PauseOutlined";
import SaveAltOutlinedIcon from "@mui/icons-material/SaveAltOutlined";
import PlayArrowOutlinedIcon from "@mui/icons-material/PlayArrowOutlined";
import FastRewindOutlinedIcon from "@mui/icons-material/FastRewindOutlined";
import FastForwardOutlinedIcon from "@mui/icons-material/FastForwardOutlined";
import { baseURL } from "../../api/api";
import SummaryApi from "../../api/api";
import "../../components/ResultDisplay/ResultDisplay.css";
import "./ReviewPage.css";

const ReviewPage = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const dispatch = useDispatch();
  const { video_id, from } = location.state || {};
  const playerRef = useRef(null);
  const [playing, setPlaying] = useState(false);
  const [playedSeconds, setPlayedSeconds] = useState(0);
  const [duration, setDuration] = useState(0);
  const [videoUrl, setVideoUrl] = useState(null);

  useEffect(() => {
    fetchCurrentUser(dispatch);
  }, [dispatch]);

  const user = useSelector((state) => state.user.user);

  useEffect(() => {
    if (!video_id) {
      navigate("/manage");
      return;
    }

    const fetchVideoDetail = async () => {
      try {
          const api = SummaryApi.getVideoDetail(video_id);
        const res = await axios({
          baseURL: baseURL,
          url: api.url,
          method: api.method,
          headers: {
            Authorization: `Bearer ${localStorage.getItem("access_token")}`,
          },
        });
        console.log(res.data);
        setVideoUrl(res.data.processed_video_url);
        console.log("Cloudinary video URL:", res.data.processed_video_url);
      } catch (error) {
        console.error("Lỗi lấy video chi tiết:", error);
      }
    };

    fetchVideoDetail();
  }, [video_id, navigate]);

  const handleBack = () => {
    if (from) {
      navigate(from);
    } else {
      navigate(-1);
    }
  };

  // const handleSeek = (e) => {
  //   const newTime = parseFloat(e.target.value);
  //   playerRef.current.seekTo(newTime);
  //   setPlayedSeconds(newTime);
  // };

  return (
    <>
      <div className="review-header">
        <Button
          onClick={handleBack}
          icon={<ArrowLeftOutlined />}
          style={{
            // marginBottom: "1rem",
            padding: "0 0.8rem",
            borderRadius: "50%",
            backgroundColor: "#4CAF50",
            color: "white",
            boxShadow: "0 1px 4px rgba(0, 0, 0, 0.1)",
          }}
        />
        <h3 className="review-title">Xem lại video</h3>
      </div>
      <div className="review-page">
        <div className="video-review-frame">
          <ReactPlayer
            ref={playerRef}
            url={videoUrl}
            playing={playing}
            onProgress={({ playedSeconds }) => setPlayedSeconds(playedSeconds)}
            onDuration={(d) => setDuration(d)}
            width="100%"
            height="61vh"
          />
        </div>

        {/* Thanh tiến trình video */}
        <div className="custom-control">
          <Slider
            value={playedSeconds}
            min={0}
            max={duration}
            step={0.1}
            onChange={(e, newValue) => {
              playerRef.current.seekTo(newValue);
              setPlayedSeconds(newValue);
            }}
            sx={{
              color: "#F14E4E;",
              height: 4,
              "& .MuiSlider-thumb": {
                width: 12,
                height: 12,
                backgroundColor: "#fff",
                border: "2px solid currentColor",
              },
              "& .MuiSlider-rail": {
                opacity: 0.28,
              },
            }}
          />
        </div>

        <div className="custom-controls">
          <div className="left-control">
            <button
              onClick={() => {
                playerRef.current.seekTo(0);
                setPlaying(false);
              }}
            >
              <ReplayIcon className="control-button" />
            </button>
            <span className="control-time">
              {new Date(playedSeconds * 1000).toISOString().substr(14, 5)} /{" "}
              {new Date(duration * 1000).toISOString().substr(14, 5)}
            </span>
          </div>

          <div className="center-button">
            <button
              onClick={() => {
                playerRef.current.seekTo(playedSeconds - 5);
              }}
            >
              <FastRewindOutlinedIcon className="control-button" />
            </button>

            <button onClick={() => setPlaying(!playing)}>
              {playing ? (
                <PauseOutlinedIcon className="control-button" />
              ) : (
                <PlayArrowOutlinedIcon className="control-button" />
              )}
            </button>

            <button
              onClick={() => {
                playerRef.current.seekTo(playedSeconds + 5);
              }}
            >
              <FastForwardOutlinedIcon className="control-button" />
            </button>
          </div>

          <div className="right-control">
            <button
              onClick={() => {
                const downloadUrl = videoUrl.replace(
                  "/upload/",
                  "/upload/fl_attachment/"
                );
                const a = document.createElement("a");
                a.href = downloadUrl;
                a.setAttribute("download", "result.mp4");
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
              }}
            >
              <SaveAltOutlinedIcon className="control-button" />
            </button>
          </div>
        </div>

        {user?.role === "user" && (
          <button
            className="new-review-button"
            onClick={() => navigate("/video")}
          >
            Tạo mới
          </button>
        )}
      </div>
    </>
  );
};

export default ReviewPage;
