import React, { useState} from "react";
import "./Login.css"; 
import axios from "axios";
import { useNavigate } from "react-router-dom";
import { useDispatch} from "react-redux";
import { loginStart, loginSuccess, loginFailed } from "../../redux/authSlice";
import { FaEye, FaEyeSlash } from "react-icons/fa";
// import fireImage from "../../assets/img/firefighters.jpg"; 
import fireImage from "../../assets/img/firelogin.jpg"; 
import { MdOutlineLocalFireDepartment } from "react-icons/md";
import { baseURL } from "../../api/api";
import SummaryApi from "../../api/api";
const Login = () => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [errorMsg, setErrorMsg] = useState("");
  const [showPassword, setShowPassword] = useState(false);
 
  const dispatch = useDispatch();
  const navigate = useNavigate();
  
 const handleLogin = async (e) => {
  e.preventDefault();
  dispatch(loginStart());

   try {
    const { url: loginUrl, method: loginMethod } = SummaryApi.login;

    const response = await axios({
      method: loginMethod,
      url: `${baseURL}${loginUrl}`,
      data: new URLSearchParams({
        username: email,
        password: password,
      }),
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
      },
    });

    const data = response.data;
    dispatch(loginSuccess(data));
    const { url: userUrl, method: userMethod } = SummaryApi.getCurrentUser;

    const userRes = await axios({
      method: userMethod,
      url: `${baseURL}${userUrl}`,
      headers: {
        Authorization: `Bearer ${data.access_token}`,
      },
    });

    const userData = userRes.data;
    if (userData.role === "admin") {
      navigate("/dashboard");
    } else {
      navigate("/home");
    }

    
  } catch (err) {
    dispatch(loginFailed());
    setErrorMsg("Email hoặc mật khẩu không đúng.");
    console.error("Login failed:", err);
  }
};



  return (
    <div className="login-container">
      <div className="login-box">
        <div className="login-form-section">
            <div className="fire-header1">
                <MdOutlineLocalFireDepartment className="fire-icon-home"/>
            </div>
          <h2 className="login-title">Hệ thống phát hiện đám cháy</h2>
          <form onSubmit={handleLogin}>
            {/* <label className="login-label">Email</label> */}
            <input
              type="email"
              placeholder="Email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="login-input"
              required
            />
            {/* <label className="login-label">Mật khẩu</label> */}
            <div className="password-input-wrapper">
                <input
                    type={showPassword ? "text" : "password"}
                    placeholder="Mật khẩu"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className="login-input"
                    required
                />
                <span
                    className="toggle-password"
                    onClick={() => setShowPassword(!showPassword)}
                >
                    {showPassword ? <FaEyeSlash /> : <FaEye />}
                </span>
                </div>

            {/* <div className="login-forgot">Quên mật khẩu?</div> */}
            {errorMsg && <p className="login-error">{errorMsg}</p>}
            <button type="submit" className="login-button">
              Đăng nhập
            </button>
          </form>
          <div className="login-divider">
            <hr className="divider-line" />
            <span className="divider-text">Hoặc</span>
            <hr className="divider-line" />
          </div>
          <div className="login-register">
            Chưa có tài khoản?{" "}
            <span className="register-link" onClick={() => navigate("/register")}>Đăng ký</span>
          </div>
        </div>

        <div className="login-image-section">
          <img
            src={fireImage}
            alt="Login Visual"
            className="login-image"
          />
        </div>
      </div>
    </div>
  );
};

export default Login;
