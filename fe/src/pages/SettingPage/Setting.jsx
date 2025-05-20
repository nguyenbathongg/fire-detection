import React, { useState, useEffect } from "react";
import "./Setting.css";
import axios from "axios";
import SettingsOutlinedIcon from "@mui/icons-material/SettingsOutlined";
import VolumeUpIcon from "@mui/icons-material/VolumeUp";
import WarningIcon from "@mui/icons-material/Warning";
import MoreVertIcon from "@mui/icons-material/MoreVert";
import { baseURL } from "../../api/api";
import SummaryApi from "../../api/api";
import {
  Slider,
  Box,
  IconButton,
  Button,
  Snackbar,
  Dialog,
  DialogTitle,
  DialogActions,
  Menu,
  MenuItem,
  Alert,
} from "@mui/material";

export default function Setting() {
  const [volume, setVolume] = useState(70);
  const [alert1, setAlert1] = useState(true);
  const [alert2, setAlert2] = useState(true);
  const [isEditing, setIsEditing] = useState(false);
  const [openSnackbar, setOpenSnackbar] = useState(false);
  const [openConfirm, setOpenConfirm] = useState(false);
  const [anchorEl, setAnchorEl] = useState(null);

  const handleVolumeChange = (event, newValue) => {
    if (isEditing) setVolume(newValue);
  };

  const handleMenuOpen = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };
const token = localStorage.getItem("access_token");

  useEffect(() => {
    if (token) {
      axios({
        method: SummaryApi.notificationSettings.method.get,
        url: baseURL + SummaryApi.notificationSettings.url,
        headers: {
          Authorization: `Bearer ${token}`,
        },
      })
        .then((response) => {
          setAlert1(response.data.enable_website_notification);
          setAlert2(response.data.enable_email_notification);
        })
        .catch((error) => {
          console.error("Lỗi khi lấy cài đặt thông báo:", error);
        });
    }
  }, [token]);

  const handleEditClick = () => {
    setIsEditing(true);
    handleMenuClose();
  };

  // const handleSave = () => {
  //   setIsEditing(false);
  //   setOpenConfirm(false);
  //   setOpenSnackbar(true);
  // };
 const handleSave = () => {
  
   if (token) {
      axios({
        method: SummaryApi.notificationSettings.method.post,
        url: baseURL + SummaryApi.notificationSettings.url,
        headers: {
          Authorization: `Bearer ${token}`,
        },
        data: {
          enable_website_notification: alert1,
          enable_email_notification: alert2,
        },
      })
      .then((res) => {
        setIsEditing(false);
        setOpenConfirm(false);
        setOpenSnackbar(true);
        return axios.get("http://localhost:8000/api/v1/notifications/settings", {
          headers: {
            Authorization: `Bearer ${token}`, 
          },
        });
      })
      .then((response) => {
        setAlert1(response.data.enable_website_notification);
        setAlert2(response.data.enable_email_notification);
      })
      .catch((error) => {
        console.error("Lỗi khi lưu hoặc tải lại cài đặt:", error);
        setOpenConfirm(false);
      });
  } else {
    console.error("Không tìm thấy access_token");
  }
};


  const handleSaveConfirm = () => {
    setOpenConfirm(true);
  };

  return (
    <div className="settings-container">
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <h2 className="settings-title">
          <SettingsOutlinedIcon
            sx={{ marginRight: "0.5rem" }}
            className="setting-icon"
          />
          Cài đặt
        </h2>
        <IconButton sx={{ marginRight: "-1rem" }} onClick={handleMenuOpen}>
          <MoreVertIcon />
        </IconButton>
        <Menu
          anchorEl={anchorEl}
          open={Boolean(anchorEl)}
          onClose={handleMenuClose}
        >
          <MenuItem onClick={handleEditClick}>Chỉnh sửa</MenuItem>
        </Menu>
      </div>

      {/* <div className="section-a">
        <button className="section-settingbutton">Âm lượng cảnh báo</button>
        <Box sx={{ width: "50%", display: "flex", alignItems: "center" }}>
          <VolumeUpIcon />
          <Slider
            value={volume}
            onChange={handleVolumeChange}
            aria-label="Volume"
            valueLabelDisplay="auto"
            min={0}
            max={100}
            disabled={!isEditing}
            sx={{ marginLeft: 1, flex: 1 }}
          />
          <span className="volume-value" style={{ marginLeft: "1rem" }}>
            {volume}
          </span>
        </Box>
      </div>

      <hr className="divider" /> */}

      <div className="section-b">
        <button className="section-button-danger">
          <WarningIcon style={{ marginRight: "8px" }} />
          Chế độ cảnh báo
        </button>
        <div className="alert-option">
          <label>Hiển thị báo động ra màn hình kèm chuông báo</label>
          <input
            type="checkbox"
            checked={alert1}
            onChange={() => isEditing && setAlert1(!alert1)}
            disabled={!isEditing}
          />
        </div>
        <div className="alert-option">
          <label>Gửi cảnh báo về email kèm hình ảnh cháy trích từ video</label>
          <input
            type="checkbox"
            checked={alert2}
            onChange={() => isEditing && setAlert2(!alert2)}
            disabled={!isEditing}
          />
        </div>
      </div>

      {isEditing && (
        <div style={{ marginTop: "1rem", textAlign: "right" }}>
          <Button
            variant="contained"
            color="primary"
            onClick={handleSaveConfirm}
            sx={{
              "&:hover": {
                backgroundColor: "#ff6666",
              },
            }}
          >
            Lưu
          </Button>
        </div>
      )}

      <Snackbar
        open={openSnackbar}
        autoHideDuration={1500}
        onClose={() => setOpenSnackbar(false)}
        anchorOrigin={{ vertical: "top", horizontal: "center" }}
      >
        {/* <Box
          sx={{
            backgroundColor: "#4caf50",
            color: "white",
            px: 2,
            py: 1,
            borderRadius: 1,
            boxShadow: 3,
            fontSize: "1rem",
          }}
        >
          Lưu cài đặt thành công
        </Box> */}
        <Alert
          onClose={() => setOpenSnackbar(false)}
          severity="success"
          variant="filled"
        >
          Lưu cài đặt thành công!
        </Alert>
      </Snackbar>
      <Dialog open={openConfirm} onClose={() => setOpenConfirm(false)}>
        <DialogTitle>Bạn có chắc chắn muốn lưu thay đổi?</DialogTitle>
        <DialogActions>
          <Button onClick={() => setOpenConfirm(false)} color="inherit">
            Hủy
          </Button>
          <Button
            onClick={handleSave}
            color="primary"
            variant="contained"
            sx={{
              "&:hover": {
                backgroundColor: "#ff6666",
              },
            }}
          >
            Xác nhận
          </Button>
        </DialogActions>
      </Dialog>
    </div>
  );
}
