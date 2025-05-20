import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { FaRegUserCircle } from "react-icons/fa";
import { Logout } from "@mui/icons-material";
import { IconButton, Snackbar, Alert as MuiAlert } from "@mui/material";
import { MdOutlineLocalFireDepartment } from "react-icons/md";
import {
  Avatar,
  Dropdown,
  Menu,
  Modal,
  Input,
  message,
} from "antd";
import "./Topbar.css";
import { useDispatch, useSelector } from "react-redux";
import {
  logOutStart,
  logOutSuccess,
  logOutFailed,
} from "../../../redux/authSlice";
import { logout } from "../../../redux/userSlice";
import { fetchCurrentUser } from "../../../redux/apiRequest";
import { baseURL } from "../../../api/api";
import SummaryApi from "../../../api/api";
import axios from "axios";

const Topbar = () => {
  const navigate = useNavigate();
  const dispatch = useDispatch();

  const [isModalVisible, setIsModalVisible] = useState(false);
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [passwordError, setPasswordError] = useState("");
  const [alert, setAlert] = useState(null); // { type: "success" | "error", msg: string }
  const [openSnackbar, setOpenSnackbar] = useState(false);

  useEffect(() => {
    fetchCurrentUser(dispatch);
  }, [dispatch]);

  const user = useSelector((state) => state.user.user);

  const validatePassword = (value) => {
    const strongPasswordRegex =
      /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{6,}$/;
    if (!strongPasswordRegex.test(value)) {
      setPasswordError(
        "Ít nhất 6 ký tự, gồm chữ hoa, chữ thường, số và ký tự đặc biệt"
      );
    } else {
      setPasswordError("");
    }
  };

  const handleLogout = () => {
    dispatch(logOutStart());
    try {
      localStorage.removeItem("user");
      localStorage.removeItem("access_token");
      localStorage.removeItem("user_id");
      localStorage.removeItem("role");
      localStorage.removeItem("isAuthenticated");
      dispatch(logout());
      dispatch(logOutSuccess());
      navigate("/login");
    } catch (error) {
      dispatch(logOutFailed());
      console.error("Lỗi khi logout:", error);
    }
  };

  const showChangePasswordModal = () => {
    setIsModalVisible(true);
    setAlert(null);
    setCurrentPassword("");
    setNewPassword("");
    setConfirmPassword("");
    setPasswordError("");
  };

  const handlePasswordChange = async () => {
    if (!currentPassword || !newPassword || !confirmPassword) {
      message.warning("Vui lòng nhập đầy đủ thông tin.");
      return;
    }

    if (newPassword !== confirmPassword) {
      setAlert({ type: "error", msg: "Mật khẩu mới không khớp." });
      setOpenSnackbar(true);
      return;
    }

    validatePassword(newPassword);
    if (passwordError) {
      return;
    }

    try {
      const token = localStorage.getItem("access_token");

     const { url, method } = SummaryApi.changePassword;

      await axios({
        method,
        url: `${baseURL}${url}`,
        data: {
          current_password: currentPassword,
          new_password: newPassword,
        },
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
        },
      });

      setIsModalVisible(false);
      setCurrentPassword("");
      setNewPassword("");
      setConfirmPassword("");

      setAlert({ type: "success", msg: "Đổi mật khẩu thành công." });
      setOpenSnackbar(true);
    } catch (error) {
      const errMsg =
        error?.response?.data?.detail?.[0]?.msg ||
        error?.response?.data?.detail ||
        "Đổi mật khẩu thất bại";
      setAlert({ type: "error", msg: errMsg });
      setOpenSnackbar(true);
    }
  };

  const handleSnackbarClose = (_, reason) => {
    if (reason === "clickaway") return;
    setOpenSnackbar(false);
  };

  const menu = (
    <Menu>
      <Menu.Item key="account" onClick={() => navigate("/account")}>
        Tài khoản
      </Menu.Item>
      <Menu.Item key="changePassword" onClick={showChangePasswordModal}>
        Đổi mật khẩu
      </Menu.Item>
    </Menu>
  );

  return (
    <div className="topbar">
      {/* Snackbar MUI */}
      <Snackbar
        open={openSnackbar}
        autoHideDuration={2000}
        onClose={handleSnackbarClose}
        anchorOrigin={{ vertical: "top", horizontal: "center" }}
      >
        {alert && (
          <MuiAlert
            elevation={6}
            variant="filled"
            onClose={handleSnackbarClose}
            severity={alert.type}
          >
            {alert.msg}
          </MuiAlert>
        )}
      </Snackbar>

      <div className="topbar-left">
        <MdOutlineLocalFireDepartment className="topbar-logo" />
        <h1 className="topbar-title">Phát hiện đám cháy</h1>
      </div>

      <div className="topbar-right">
        <Dropdown overlay={menu} trigger={["click"]}>
          <div className="topbar-user" style={{ cursor: "pointer" }}>
            <Avatar
              size="1.5rem"
              icon={<FaRegUserCircle className="topbar-user-logo" />}
            />
            <div className="topbar-username-container">
              <span className="topbar-username">{user?.username}</span>
            </div>
          </div>
        </Dropdown>
        <div className="topbar-logout">
          <IconButton onClick={handleLogout}>
            <Logout style={{ color: "white" }} />
          </IconButton>
        </div>
      </div>

      <Modal
        className="custom-modal"
        title={<div className="modal-title"> Đổi mật khẩu </div>}
        open={isModalVisible}
        onOk={handlePasswordChange}
        onCancel={() => setIsModalVisible(false)}
        okText="Xác nhận"
        cancelText="Hủy"
      >
        <Input.Password
          placeholder="Mật khẩu hiện tại"
          value={currentPassword}
          onChange={(e) => setCurrentPassword(e.target.value)}
          style={{ marginBottom: 10 }}
        />
        <Input.Password
          placeholder="Mật khẩu mới"
          value={newPassword}
          onChange={(e) => {
            setNewPassword(e.target.value);
            validatePassword(e.target.value);
          }}
          style={{ marginBottom: 10 }}
        />
        {passwordError && (
          <div style={{ color: "red", marginBottom: 10 }}>{passwordError}</div>
        )}
        <Input.Password
          placeholder="Xác nhận mật khẩu mới"
          value={confirmPassword}
          onChange={(e) => setConfirmPassword(e.target.value)}
        />
      </Modal>
    </div>
  );
};

export default Topbar;
