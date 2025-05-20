import React, { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useDispatch } from 'react-redux';
import { setVideoData } from '../../redux/videoSlice';
import { Upload, Button } from 'antd';
import { DeleteOutlined } from '@ant-design/icons';
import FileUploadOutlinedIcon from '@mui/icons-material/FileUploadOutlined';
import VideoLibraryOutlinedIcon from '@mui/icons-material/VideoLibraryOutlined';
import { RiLinkM } from 'react-icons/ri';
import Snackbar from '@mui/material/Snackbar';
import Alert from '@mui/material/Alert';
import './Video.css';

const MAX_FILE_SIZE_MB = 15;
const MAX_VIDEO_DURATION_SEC = 3600; // 1 hour

const isValidVideoFormat = (fileName) => {
  return /\.(mp4|avi)$/i.test(fileName);
};

const isValidYoutubeUrl = (url) => {
  if (!url) return 'URL trống, vui lòng nhập URL.';
  const youtubeRegex = /^(https?:\/\/)?(www\.)?(youtube\.com\/(watch\?v=|shorts\/)|youtu\.be\/)[\w-]{11}([?&=a-zA-Z0-9_-]*)?$/;
  if (!youtubeRegex.test(url)) {
    return 'URL không đúng định dạng Youtube.';
  }

  if (url.length > 1000) return 'URL quá dài (vượt quá 1000 ký tự).';
  if (!/^(https?:\/\/)?(www\.)?(youtube\.com|youtu\.be)\//.test(url)) 
    return 'URL không phải trang Youtube.';
  if (/private/.test(url)) 
    return 'Video ở chế độ riêng tư, không thể truy cập.';
  return null; // hợp lệ
};


const Video = () => {
  const navigate = useNavigate();
  const dispatch = useDispatch();

  const [videoUrl, setVideoUrl] = useState('');
  const [videoFile, setVideoFile] = useState(null);
  const [activeTab, setActiveTab] = useState('video');
  const videoRef = useRef(null);

  // Snackbar state
  const [snackbar, setSnackbar] = useState({
    open: false,
    message: '',
    severity: 'info', // 'error', 'warning', 'success', 'info'
  });

  const showMessage = (message, severity = 'info') => {
    setSnackbar({ open: true, message, severity });
  };

  const handleCloseSnackbar = (event, reason) => {
    if (reason === 'clickaway') return;
    setSnackbar({ ...snackbar, open: false });
  };

  const checkVideoDuration = (file) => {
    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(file);
      const video = document.createElement('video');
      video.preload = 'metadata';
      video.src = url;

      video.onloadedmetadata = () => {
        URL.revokeObjectURL(url);
        if (video.duration > MAX_VIDEO_DURATION_SEC) {
          reject(new Error('Video dài hơn 1 giờ, không được phép tải lên.'));
        } else {
          resolve(true);
        }
      };

      video.onerror = () => {
        URL.revokeObjectURL(url);
        reject(new Error('File video bị lỗi hoặc không thể đọc metadata.'));
      };
    });
  };

  const handleBeforeUpload = async (file) => {
    if (file.size === 0) {
      showMessage('File video rỗng hoặc không hợp lệ.', 'error');
      return Upload.LIST_IGNORE;
    }

    if (!isValidVideoFormat(file.name)) {
      showMessage('Chỉ hỗ trợ định dạng MP4 và AVI.', 'error');
      return Upload.LIST_IGNORE;
    }

    if (file.size / 1024 / 1024 > MAX_FILE_SIZE_MB) {
      showMessage(`Video phải nhỏ hơn ${MAX_FILE_SIZE_MB}MB!`, 'error');
      return Upload.LIST_IGNORE;
    }

    try {
      await checkVideoDuration(file);
    } catch (error) {
      showMessage(error.message, 'error');
      return Upload.LIST_IGNORE;
    }

    setVideoFile(file);
    setVideoUrl('');
    return false; // chặn auto upload
  };

  const handleUpload = () => {
    if (videoFile) {
      dispatch(setVideoData({ videoFile, videoUrl: '', mode: 'video' }));
      navigate('/video/analyze');
    } else if (videoUrl) {
        const errorMsg = isValidYoutubeUrl(videoUrl);
        if (errorMsg) {
          showMessage(errorMsg, 'error');
          return;
        }
      dispatch(setVideoData({ videoFile: null, videoUrl, mode: 'video' }));
      navigate('/video/analyze');
    } else {
      showMessage('Vui lòng chọn file hoặc nhập URL video.', 'warning');
    }
  };

  const handleRemoveVideo = () => {
    setVideoFile(null);
    setVideoUrl('');
  };


  const startCamera = () => {
  navigate("/video/cameraresult", { state: { mode: "camera" } });
};

  return (
    <div className="video-page">
      <div className="tabs">
        <button className={`tab-button ${activeTab === 'video' ? 'active' : ''}`} onClick={() => setActiveTab('video')}>
          Video
        </button>
        <button className={`tab-button ${activeTab === 'camera' ? 'active' : ''}`} onClick={() => setActiveTab('camera')}>
          Camera
        </button>
      </div>

      {activeTab === 'video' && (
        <div className="tab-content video-tab">
          <div className="upload-section">
            <div className="upload-header">
              <FileUploadOutlinedIcon className="file-upload-icon" />
              <p className="upload-title">Tải lên video</p>
            </div>

            <div className="upload-box">
              {videoFile ? (
                <div
                  style={{
                    width: '100%',
                    height: '100%',
                    position: 'relative',
                    display: 'flex',
                    justifyContent: 'center',
                    alignItems: 'center',
                  }}
                >
                  <p style={{ marginBottom: 0 }}>{videoFile.name}</p>
                  <Button
                    icon={<DeleteOutlined />}
                    danger
                    onClick={handleRemoveVideo}
                    size="small"
                    style={{ position: 'absolute', top: 8, right: 8 }}
                  />
                </div>
              ) : (
                <Upload
                  accept="video/mp4,video/avi"
                  showUploadList={false}
                  beforeUpload={handleBeforeUpload}
                  disabled={!!videoUrl}
                >
                  <div>
                    <div className="upload-icon">
                      <VideoLibraryOutlinedIcon className="file-upload-icon-2" />
                    </div>
                    <p>Hỗ trợ các định dạng: MP4, AVI (tối đa 15MB)</p>
                  </div>
                </Upload>
              )}
            </div>

            <div className="upload-title">Hoặc</div>

            <div className="upload-link">
              <RiLinkM className="link-icon" />
              <p className="upload-title">Đường dẫn video</p>
            </div>

            <div className="url-input">
              <input
                type="text"
                placeholder="Nhập URL video Youtube..."
                value={videoUrl}
                disabled={!!videoFile}
                onChange={(e) => {
                  setVideoUrl(e.target.value);
                  setVideoFile(null);
                }}
              />
            </div>

            <button className="upload-button" onClick={handleUpload}>
              Xem trước video
            </button>
          </div>
        </div>
      )}

      {activeTab === 'camera' && (
        <div className="tab-content camera-tab">
          <video id="cameraFeed" autoPlay playsInline muted className="camera-video" />
          <button className="camera-button" onClick={startCamera}>
            Bắt đầu Camera
          </button>
        </div>
      )}

      {/* Snackbar chung */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={1500}
        onClose={handleCloseSnackbar}
        anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
      >
        <Alert severity={snackbar.severity} onClose={handleCloseSnackbar} sx={{ width: '100%' }}>
          {snackbar.message}
        </Alert>
      </Snackbar>
    </div>
  );
};

export default Video;
