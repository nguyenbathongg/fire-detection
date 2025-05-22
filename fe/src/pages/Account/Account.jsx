import React, { useState, useEffect } from "react";
import "./Account.css";
import { IconButton, Snackbar, Alert } from "@mui/material";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import { Edit, Logout } from "@mui/icons-material";
import { FaHistory } from "react-icons/fa";
import AccountCircleOutlinedIcon from "@mui/icons-material/AccountCircleOutlined";
import SaveAltOutlinedIcon from "@mui/icons-material/SaveAltOutlined";
import EditUserModal from "../../components/EditAccountModal/EditAccountModal";
import CheckCircleOutlineIcon from "@mui/icons-material/CheckCircleOutline";
import CloseIcon from "@mui/icons-material/Close";
import { Tooltip, message, Table, Pagination } from "antd";
import { PlayCircleOutlined } from "@ant-design/icons";
import { useDispatch } from "react-redux";
import { baseURL } from "../../api/api";
import SummaryApi from "../../api/api";
import {
  logOutStart,
  logOutSuccess,
  logOutFailed,
} from "../../redux/authSlice";
import { logout } from "../../redux/userSlice";

const Account = () => {
  const [user, setUser] = useState({});
  const dispatch = useDispatch();
  const [userInfo, setUserInfo] = useState(user);
  const [editModalVisible, setEditModalVisible] = useState(false);
  const [snackbarOpen, setSnackbarOpen] = useState(false);
  const navigate = useNavigate();

  const limit = 5;
  const [page, setPage] = useState(1);
  const [allVideos, setAllVideos] = useState([]);
  const [videos, setVideos] = useState([]);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [total, setTotal] = useState(0);
  const pageSize = 10;

  const fetchCurrentUser = async () => {
    const token = localStorage.getItem("access_token");
    if (!token) {
      navigate("/login");
      return;
    }

    try {
      const response = await axios({
        method: SummaryApi.getCurrentUser.method,
        url: baseURL + SummaryApi.getCurrentUser.url,
        headers: {
          Authorization: `Bearer ${token}`,
          Accept: "application/json",
        },
      });

      const userData = response.data;
      setUser({
        name: userData.username || "Người dùng",
        email: userData.email || "",
        phone: userData.phone_number || "",
        address: userData.address || "",
        role: userData.role || "user",
      });
      setUserInfo({
        name: userData.username || "Người dùng",
        email: userData.email || "",
        phone: userData.phone_number || "",
        address: userData.address || "",
      });
    } catch (error) {
      console.error("Không lấy được thông tin người dùng:", error);
      if (error.response && error.response.status === 401) {
        dispatch(logout());
        navigate("/login");
      }
    }
  };

  useEffect(() => {
    fetchCurrentUser();
  }, []);

  // Lấy toàn bộ video (không phân trang)
  useEffect(() => {
    const fetchVideos = async () => {
      try {
        const resVideos = await axios({
          method: SummaryApi.fetchVideos.method,
          url: baseURL + SummaryApi.fetchVideos.url,
          headers: {
            Authorization: `Bearer ${localStorage.getItem("access_token")}`,
          },
        });
        setAllVideos(resVideos.data);
      } catch (err) {
        console.error("Lỗi khi lấy video:", err);
      }
    };
    fetchVideos();
  }, []);

  // Cập nhật video theo trang khi allVideos hoặc page thay đổi
  useEffect(() => {
    const start = (page - 1) * limit;
    const end = start + limit;
    setVideos(allVideos.slice(start, end));
  }, [allVideos, page]);

  // Tổng số trang tính từ tổng video có được
  const totalPages = Math.ceil(allVideos.length / limit) || 1;
  const startIndex = (page - 1) * limit + 1;
  const endIndex = Math.min(page * limit, allVideos.length);

  const fetchHistory = async () => {
    setLoading(true);
    try {
      const res = await axios({
        method: SummaryApi.fetchHistory.method,
        url: baseURL + SummaryApi.fetchHistory.url,
        headers: {
          Authorization: `Bearer ${localStorage.getItem("access_token")}`,
        },
        params: { skip: 0, limit: 900 },
      });

      const allHistory = Array.isArray(res.data) ? res.data : [];
      setHistory(allHistory);
      setTotal(allHistory.length);
    } catch (err) {
      if (err.response?.status === 401) {
        message.error("Vui lòng đăng nhập để xem lịch sử.");
        navigate("/login");
      } else {
        message.error("Không thể tải lịch sử hoạt động.");
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  const cleanDescription = (desc) =>
    desc
      .replace(/VideoTypeEnum\.YOUTUBE/g, "Youtube")
      .replace(/VideoTypeEnum\.UPLOAD/g, "Upload")
      .replace(/True/g, "Có")
      .replace(/False/g, "Không")
      .replace(/,+/g, ", ")
      .replace(/\s+/g, " ")
      .trim();

  const columns = [
    {
      title: <div style={{ textAlign: "center" }}>Ngày</div>,
      dataIndex: "created_at",
      key: "date",
      render: (text) => {
        const date = new Date(text);
        return (
          <div style={{ textAlign: "center" }}>
            {date.toLocaleDateString("vi-VN")}
          </div>
        );
      },
    },
    {
      title: <div style={{ textAlign: "center" }}>Giờ</div>,
      dataIndex: "created_at",
      key: "time",
      render: (text) => {
        const date = new Date(text);
        // date.setHours(date.getHours() + 7);
        return (
          <div style={{ textAlign: "center" }}>

            {date.toLocaleTimeString("vi-VN")}
          </div>
        );
      },
    },
    {
      title: <div style={{ textAlign: "center" }}>Hành động</div>,
      dataIndex: "description",
      key: "description",
      render: (_, item) => (
        <div
          style={{
            textAlign: "left",
            display: "flex",
            alignItems: "center",
            gap: 8,
          }}
        >
          <span>{cleanDescription(item.description)}.</span>

          {item.video_id && (
            <span
              className="watch-link"
              style={{
                display: "flex",
                alignItems: "center",
                color: "#000000",
                cursor: "pointer",
              }}
              onClick={() =>
                navigate("/account/review", {
                  state: {
                    video_id: item.video_id,
                    from: "/account",
                  },
                })
              }
            >
              <PlayCircleOutlined style={{ fontSize: 18, marginRight: 4 }} />
              <span>Xem video liên quan</span>
            </span>
          )}
        </div>
      ),
    },
  ];

  const handleEditSave = async (newData) => {
    setUser(newData);
    setUserInfo(newData);
    setSnackbarOpen(true);
    setEditModalVisible(false);
    await fetchCurrentUser();
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

  return (
    <div className="user-container">
      <div className="account-box">
        <div className="account-header">
          <h2>Tài khoản</h2>
          <IconButton onClick={handleLogout}>
            <Logout />
          </IconButton>
        </div>

        <div className="account-content">
          <div className="account-name">
            <AccountCircleOutlinedIcon className="account-avatar" />
            <span className="name-input">{user.name}</span>
            <IconButton
              onClick={() => setEditModalVisible(true)}
              style={{ color: "white" }}
            >
              <Edit />
            </IconButton>
          </div>

          <div className="account-details">
            <p>
              <strong>Email:</strong>&nbsp;{user.email}
            </p>
            <p>
              <strong>SĐT:</strong>&nbsp;{user.phone}
            </p>
            <p>
              <strong>Địa chỉ:</strong> &nbsp;{user.address}
            </p>
            {user.role === "admin" && (
              <p>
                <strong>Vai trò:</strong> &nbsp;{user.role}
              </p>
            )}
          </div>
        </div>
      </div>
      {user.role === "user" && (
        <div className="history-box">
          <div
            className="history-header"
            style={{
              position: "relative",
              textAlign: "center",
            }}
          >
            <h3 style={{ margin: 0 }}>Lịch sử phân tích video</h3>
            <Tooltip title="Xem lịch sử người dùng">
              <IconButton
                onClick={() => navigate("/account/history")}
                style={{
                  position: "absolute",
                  top: "50%",
                  right: 0,
                  transform: "translateY(-50%)",
                }}
              >
                <FaHistory style={{ color: "#60477D", fontSize: "1.5rem" }} />
              </IconButton>
            </Tooltip>
          </div>

          <div className="history-table">
            <div className="table-header">
              <span>Ngày</span>
              <span>Giờ</span>
              <span>Phát hiện cháy</span>
              <span>Video</span>
            </div>
            {videos.length > 0 ? (
              videos.map((video, index) => {
                const date = new Date(video.created_at);
                // date.setHours(date.getHours() + 7);

                const formattedDate = date.toLocaleDateString("vi-VN");
                const formattedTime = date.toLocaleTimeString("vi-VN", {
                  hour: "2-digit",
                  minute: "2-digit",
                });

                return (
                  <div key={video.video_id} className="table-row">
                    <span>{formattedDate}</span>
                    <span>{formattedTime}</span>
                    <span
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: "0.375rem",
                        justifyContent: "center",
                      }}
                    >
                      {video.fire_detected ? (
                        <>
                          <CheckCircleOutlineIcon
                            style={{
                              color: "green",
                              fontSize: "1rem",
                              fontWeight: "bold",
                            }}
                          />
                          {/* <span>Có phát hiện lửa</span> */}
                        </>
                      ) : (
                        <>
                          <CloseIcon
                            style={{
                              color: "red",
                              fontSize: "1rem",
                              fontWeight: "bold",
                            }}
                          />
                          {/* <span>Không phát hiện lửa</span> */}
                        </>
                      )}
                    </span>

                    <span
                      className="video-name"
                      style={{
                        display: "flex",
                        cursor: "pointer",
                        color: "#000000",
                        fontWeight: 500,
                        alignItems: "center",
                        whiteSpace: "normal",
                        wordBreak: "break-word",
                        maxWidth: "100%",
                        flexDirection: "row",
                        flexWrap: "wrap",
                      }}
                    >
                      <Tooltip title="Nhấn để xem lại video">
                        <span
                          onClick={() => {
                            navigate("/account/review", {
                              state: {
                                video_id: video.video_id,
                                from: "/account",
                              },
                            });
                          }}
                          style={{
                            display: "inline-flex",
                            alignItems: "center",
                          }}
                        >
                          <PlayCircleOutlined
                            style={{ marginRight: "0.5rem" }}
                          />
                          <span
                            style={{
                              wordBreak: "break-word",
                              whiteSpace: "normal",
                              maxWidth: "200px",
                              width: "100%",
                              display: "block",
                              textAlign: "justify",
                            }}
                          >
                            {video.file_name}
                          </span>
                        </span>
                      </Tooltip>
                    </span>
                    <IconButton
                      className="download-button"
                      onClick={() => {
                        const downloadUrl = video.processed_video_url.replace(
                          "/upload/",
                          "/upload/fl_attachment/"
                        );
                        const a = document.createElement("a");
                        a.href = downloadUrl;
                        a.setAttribute("download", "processed_video.mp4");
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                      }}
                    >
                      <SaveAltOutlinedIcon style={{ color: "#60477D" }} />
                    </IconButton>
                  </div>
                );
              })
            ) : (
              <div
                style={{
                  display: "flex",
                  justifyContent: "center",
                  alignItems: "center",
                  padding: "1.25rem",
                }}
              >
                <p style={{ margin: 0, fontSize: "0.8rem" }}>
                  Chưa có video nào.
                </p>
              </div>
            )}
          </div>

          <div className="video-pagination">
            <Pagination
              current={page}
              pageSize={limit}
              total={allVideos.length}
              onChange={(newPage) => setPage(newPage)}
              showSizeChanger={false}
              showLessItems={true}

            />
          </div>

          {/*   Phân trang */}
          {/* {totalPages >= 1 && (
            <div
              style={{
                display: "flex",
                alignItems: "center",
                flexWrap: "wrap",
                justifyContent: "center",
                position: "relative",
              }}
            >
              <div
                style={{
                  position: "absolute",
                  right: 0,
                  fontSize: "0.8rem",
                  color: "#000000",
                  whiteSpace: "nowrap",
                  marginTop: "0.75rem",
                }}
              >
                {startIndex}–{endIndex} trong tổng số {allVideos.length} videos
              </div>

              <div className="video-pagination" style={{ textAlign: "center" }}>
                {[...Array(totalPages)].map((_, i) => (
                  <button
                    key={i}
                    onClick={() => setPage(i + 1)}
                    style={{
                      borderRadius: "50%",
                      backgroundColor: page === i + 1 ? "#4caf50" : "#ffffff",
                      color: page === i + 1 ? "white" : "black",
                      border: "none",
                      cursor: "pointer",
                      fontSize: "0.8rem",
                    }}
                    aria-current={page === i + 1 ? "page" : undefined}
                  >
                    {i + 1}
                  </button>
                ))}
              </div>
            </div>
          )} */}
        </div>
      )}

      {user?.role === "admin" && (
        <div className="history-box">
          <div
            className="history-header"
            style={{
              position: "relative",
              textAlign: "center",
            }}
          >
            <h3>Lịch sử hoạt động</h3>
          </div>
          {loading ? (
            <p>Đang tải dữ liệu...</p>
          ) : history.length === 0 ? (
            <p>Không có hoạt động nào.</p>
          ) : (
            <Table
              columns={columns}
              dataSource={history.map((item) => ({
                key: item.history_id,
                ...item,
              }))}
              pagination={{
                current: page,
                pageSize,
                total,
                onChange: (page) => setPage(page),
                showSizeChanger: false,
                 showLessItems:true,
              }}
              bordered
              //  scroll={{ x: "max-content" }}
            />
          )}
        </div>
      )}

      {/* Modal chỉnh sửa */}
      <EditUserModal
        isVisible={editModalVisible}
        onClose={() => setEditModalVisible(false)}
        user={userInfo}
        userInfo={userInfo}
        onSave={handleEditSave}
      />

      <Snackbar
        open={snackbarOpen}
        autoHideDuration={1500}
        onClose={() => setSnackbarOpen(false)}
        anchorOrigin={{ vertical: "top", horizontal: "center" }}
      >
        <Alert
          onClose={() => setSnackbarOpen(false)}
          severity="success"
          variant="filled"
        >
          Cập nhật thành công!
        </Alert>
      </Snackbar>
    </div>
  );
};

export default Account;
