import React, { useState } from "react";
import axios from "axios";
import "../LoginPage/Login.css";
import { useNavigate } from "react-router-dom";
import { FaEye, FaEyeSlash } from "react-icons/fa";
import { MdOutlineLocalFireDepartment } from "react-icons/md";
import Snackbar from "@mui/material/Snackbar";
import MuiAlert from "@mui/material/Alert";
import fireImage from "../../assets/img/firelogin.jpg";
import { baseURL } from "../../api/api";
import SummaryApi from "../../api/api";
import "./Register.css";

const Alert = React.forwardRef(function Alert(props, ref) {
  return <MuiAlert elevation={6} ref={ref} variant="filled" {...props} />;
});

const Register = () => {
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [passwordError, setPasswordError] = useState("");
  const [confirm, setConfirm] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const navigate = useNavigate();

  // Snackbar
  const [openSnackbar, setOpenSnackbar] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState("");
  const [snackbarSeverity, setSnackbarSeverity] = useState("success");

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

  const handleRegister = async (e) => {
    e.preventDefault();

    if (passwordError) {
      setSnackbarMessage("Mật khẩu không hợp lệ");
      setSnackbarSeverity("error");
      setOpenSnackbar(true);
      return;
    }

    if (password !== confirm) {
      setSnackbarMessage("Mật khẩu không khớp");
      setSnackbarSeverity("error");
      setOpenSnackbar(true);
      return;
    }

    try {
       const response = await axios({
        method: SummaryApi.register.method,
        url: SummaryApi.register.url,
        baseURL: baseURL, 
        data: {
          username: username,
          email: email,
          password: password,
        }
      });

      if (response.status === 200) {
        setSnackbarMessage("Đăng ký thành công!");
        setSnackbarSeverity("success");
        setOpenSnackbar(true);
        setTimeout(() => {
          navigate("/login");
        }, 1500);
      }
    } catch (err) {
      if (err.response) {
        const detail = err.response.data?.detail;
        let msg = "Đăng ký thất bại, vui lòng thử lại";

        if (Array.isArray(detail)) {
          msg = "Lỗi: " + (detail[0]?.msg || msg);
        } else if (typeof detail === "string") {
          msg = "Lỗi: " + detail;
        }

        setSnackbarMessage(msg);
        setSnackbarSeverity("error");
        setOpenSnackbar(true);
      } else {
        setSnackbarMessage("Lỗi kết nối máy chủ, vui lòng thử lại");
        setSnackbarSeverity("error");
        setOpenSnackbar(true);
      }
    }
  };

  return (
    <div className="register-container">
      <div className="register-box">
        <div className="register-form-section">
          <div className="register-fire-header1">
            <MdOutlineLocalFireDepartment className="fire-icon-home" />
          </div>
          <h2 className="register-title">Đăng ký tài khoản</h2>
          <form onSubmit={handleRegister}>
            <input
              type="text"
              placeholder="Tên đăng nhập"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="register-input"
              required
            />
            <input
              type="email"
              placeholder="Email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="register-input"
              required
            />
            <div className="password-input-wrapper">
              <input
                type={showPassword ? "text" : "password"}
                placeholder="Mật khẩu"
                value={password}
                onChange={(e) => {
                  setPassword(e.target.value);
                  validatePassword(e.target.value);
                }}
                className="register-input"
                required
              />
              <span
                className="toggle-password"
                onClick={() => setShowPassword(!showPassword)}
              >
                {showPassword ? <FaEyeSlash /> : <FaEye />}
              </span>
            </div>
            <div className="register-error">
              {passwordError && <p className="register-error">{passwordError}</p>}
            </div>
            <div className="password-input-wrapper">
              <input
                type={showConfirmPassword ? "text" : "password"}
                placeholder="Xác nhận mật khẩu"
                value={confirm}
                onChange={(e) => setConfirm(e.target.value)}
                className="register-input"
                required
              />
              <span
                className="toggle-password"
                onClick={() => setShowConfirmPassword(!showConfirmPassword)}
              >
                {showConfirmPassword ? <FaEyeSlash /> : <FaEye />}
              </span>
            </div>
            <button type="submit" className="register-button">
              Đăng ký
            </button>
          </form>
          <div className="register-divider">
            <hr className="divider-line" />
            <span className="divider-text">Hoặc</span>
            <hr className="divider-line" />
          </div>
          <div className="register-register">
            Đã có tài khoản?{" "}
            <span className="register-link" onClick={() => navigate("/login")}>
              Đăng nhập
            </span>
          </div>
        </div>
        <div className="register-image-section">
          <img
            src={fireImage}
            alt="Register Visual"
            className="register-image"
          />
        </div>
      </div>

      <Snackbar
        open={openSnackbar}
        autoHideDuration={3000}
        onClose={() => setOpenSnackbar(false)}
        anchorOrigin={{ vertical: "top", horizontal: "center" }}
      >
        <Alert
          onClose={() => setOpenSnackbar(false)}
          severity={snackbarSeverity}
        >
          {snackbarMessage}
        </Alert>
      </Snackbar>
    </div>
  );
};

export default Register;
