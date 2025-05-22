import React, { useRef, useState, useEffect } from "react";
import ReactPlayer from "react-player";
import { useNavigate, useLocation } from "react-router-dom";
import { useSelector, useDispatch } from "react-redux";
import { fetchCurrentUser } from "../../redux/apiRequest";
import Slider from "@mui/material/Slider";
import { Button } from "antd";
import ReplayIcon from "@mui/icons-material/Replay";
import PauseOutlinedIcon from "@mui/icons-material/PauseOutlined";
import SaveAltOutlinedIcon from "@mui/icons-material/SaveAltOutlined";
import PlayArrowOutlinedIcon from "@mui/icons-material/PlayArrowOutlined";
import FastRewindOutlinedIcon from "@mui/icons-material/FastRewindOutlined";
import FastForwardOutlinedIcon from "@mui/icons-material/FastForwardOutlined";
import "../../components/ResultDisplay/ResultDisplay.css";
import "./ReviewPage.css";

const ReviewStreamVideo = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const dispatch = useDispatch();
  const playerRef = useRef(null);
  const processedVideoUrl = location.state?.processedVideoUrl;

  const [playing, setPlaying] = useState(false);
  const [playedSeconds, setPlayedSeconds] = useState(0);
  const [duration, setDuration] = useState(0);

  useEffect(() => {
    fetchCurrentUser(dispatch);
  }, [dispatch]);

  const user = useSelector((state) => state.user.user);
  console.log(processedVideoUrl);
  return (
    <>
      <div className="review-header"></div>
      <div className="review-page">
        <div className="video-review-frame">
          <ReactPlayer
            ref={playerRef}
            url={processedVideoUrl}
            playing={playing}
            onProgress={({ playedSeconds }) => setPlayedSeconds(playedSeconds)}
            onDuration={(d) => setDuration(d)}
            style={{
              maxWidth: "100%",
              maxHeight: "100%",
              height: "100%",
              width: "auto",
              objectFit: "contain",
              // margin: "auto", 
            }}
          />
        </div>

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
              color: "#F14E4E",
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
                playerRef.current.seekTo(Math.max(playedSeconds - 5, 0));
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
                playerRef.current.seekTo(Math.min(playedSeconds + 5, duration));
              }}
            >
              <FastForwardOutlinedIcon className="control-button" />
            </button>
          </div>

          <div className="right-control">
            <button
              onClick={() => {
                if (!processedVideoUrl) return;
                const downloadUrl = processedVideoUrl.replace(
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

export default ReviewStreamVideo;
