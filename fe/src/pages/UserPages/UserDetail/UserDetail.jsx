import React, { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { IconButton } from "@mui/material";
import axios from "axios";
import { Button, Table, Tooltip } from "antd";
import { ArrowLeftOutlined } from "@ant-design/icons";
import { FaHistory } from "react-icons/fa";
import CheckCircleOutlineIcon from "@mui/icons-material/CheckCircleOutline";
import CloseIcon from "@mui/icons-material/Close";
import SaveAltOutlinedIcon from "@mui/icons-material/SaveAltOutlined";
import { PlayCircleOutlined } from "@ant-design/icons";
import { baseURL } from "../../../api/api";
import SummaryApi from "../../../api/api";
import "./UserDetail.css";

const UserDetailPage = () => {
  const { userId } = useParams();
  const [userDetails, setUserDetails] = useState(null);
  const [userVideos, setUserVideos] = useState([]);
  const navigate = useNavigate();
const [pagination, setPagination] = useState({
  current: 1,
  pageSize: 5,
});

useEffect(() => {
  const fetchUserDetails = async () => {
    try {
      const { url, method } = SummaryApi.getUserById(userId);
      const response = await axios({
        method,
        url: baseURL + url,
        headers: {
          Authorization: `Bearer ${localStorage.getItem("access_token")}`,
        },
      });
      setUserDetails(response.data);
    } catch (error) {
      console.error("Error fetching user details:", error);
    }
  };

  if (userId) {
    fetchUserDetails();
  }
}, [userId]);

// Gọi video của user đó
useEffect(() => {
  const fetchVideos = async () => {
    try {
      const { url, method } = SummaryApi.fetchAllVideos;
      const resVideos = await axios({
        method,
        url: baseURL + url,
        headers: {
          Authorization: `Bearer ${localStorage.getItem("access_token")}`,
        },
        params: {
          skip: 0,
          limit: 1000,
        },
      });

      if (userDetails?.user_id) {
        const filteredVideos = resVideos.data.filter(
          (video) => video.user_id === userDetails.user_id
        );

        const videos = filteredVideos.map((video) => {
          const dateObj = new Date(video.updated_at);
          // dateObj.setHours(dateObj.getHours() + 7);
          return {
            key: video.video_id,
            date: dateObj.toLocaleDateString("vi-VN"),
            time: dateObj.toLocaleTimeString("vi-VN", {
              hour: "2-digit",
              minute: "2-digit",
            }),
            fireDetected: video.fire_detected ? "Có" : "Không",
            file_name: video.file_name,
            videoName: video.processed_video_url ? (
              <a
                href={video.processed_video_url}
                target="_blank"
                rel="noreferrer"
              >
                Xem video
              </a>
            ) : (
              "Không có"
            ),
            processed_video_url: video.processed_video_url,
          };
        });

        setUserVideos(videos);
      }
    } catch (error) {
      console.error("Lỗi khi lấy dữ liệu video:", error);
    }
  };

  if (userDetails) {
    fetchVideos();
  }
}, [userDetails]);

  const handleBack = () => {
    navigate("/user-page");
  };

  if (!userDetails) return <div>Loading...</div>;

  function formatToVietnamTime(dateString) {
    const date = new Date(dateString);
    // Cộng thêm 7 giờ (theo giờ Việt Nam)
    // date.setHours(date.getHours() + 7);

    return date.toLocaleString("vi-VN", {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      // hour: "2-digit",
      // minute: "2-digit",
    });
  }

  const createdAt = formatToVietnamTime(userDetails.created_at);
  const updatedAt = formatToVietnamTime(userDetails.updated_at);

  const totalVideos = userVideos.length;
  const fireDetectedCount = userVideos.filter(
    (video) => video.fireDetected === "Có"
  ).length;

  const columns = [
    {
      title: <span>Ngày</span>,
      dataIndex: "date",
      key: "date",
      align: "center",
    },
    {
      title: <span>Giờ</span>,
      dataIndex: "time",
      key: "time",
      width: "10%",
      align: "center",
    },
    {
      title: <span>Phát hiện cháy</span>,
      dataIndex: "fireDetected",
      key: "fire",
      width: "25%",
      align: "center",
      render: (fireDetected) =>
        fireDetected === "Có" ? (
          <CheckCircleOutlineIcon
            style={{ color: "green", fontSize: "1rem" }}
          />
        ) : (
          <CloseIcon style={{ color: "red", fontSize: "1rem" }} />
        ),
    },
    {
      title: (
        <span style={{ display: "block", textAlign: "center" }}>Video</span>
      ),
      dataIndex: "file_name",
      key: "video",
      width: "35%",
      render: (fileName, record) => (
        <Tooltip title="Nhấn để xem lại video">
          <span
            onClick={() => {
              navigate(`/user-page/user-detail/${userId}/review`, {
                state: {
                  video_id: record.key,
                  from: `/user-page/user-detail/${userId}`,
                },
              });
            }}
            style={{
              textAlign: "justify",
              wordBreak: "break-word",
              whiteSpace: "normal",
              maxWidth: "100%",
              display: "block",
              cursor: "pointer",
              color: "#000000",
              fontWeight: 500,
            }}
          >
            <PlayCircleOutlined style={{ marginRight: "0.4rem" }} />
            {fileName || ""}
          </span>
        </Tooltip>
      ),
    },
    {
      title: "",
      key: "download",
      width: "10%",
      align: "center",
      render: (_, record) => (
        <Button
          type="text"
          icon={<SaveAltOutlinedIcon style={{ color: "#60477D" }} />}
          onClick={() => {
            if (record.processed_video_url) {
              const downloadUrl = record.processed_video_url.replace(
                "/upload/",
                "/upload/fl_attachment/"
              );
              const a = document.createElement("a");
              a.href = downloadUrl;
              a.setAttribute("download", "processed_video.mp4");
              document.body.appendChild(a);
              a.click();
              document.body.removeChild(a);
            }
          }}
        />
      ),
    },
  ];
  return (
    <div className="user-container">
      <div className="account-box">
        <div className="user-header">
          <Button
            onClick={handleBack}
            icon={<ArrowLeftOutlined />}
            style={{
              marginRight: "1rem",
              padding: "0 0.8rem",
              borderRadius: "50%",
            }}
          />
          <h3>Chi tiết người dùng</h3>
        </div>

        <div className="account-content">
          <div className="account-name">
            <span className="name-input">{userDetails.username}</span>
          </div>

          <div className="account-details">
            <p>
              <strong>Email:</strong> &nbsp;{userDetails.email}
            </p>
            <p>
              <strong>Địa chỉ:</strong> &nbsp;{userDetails.address || "Chưa có"}
            </p>
            <p>
              <strong>Vai trò:</strong>&nbsp; {userDetails.role}
            </p>
            <p>
              <strong>Ngày tạo:</strong>&nbsp; {createdAt}
            </p>
            <p>
              <strong>Ngày cập nhật:</strong> &nbsp;{updatedAt}
            </p>
            <p>
              <strong>Tổng số video:</strong> &nbsp;{totalVideos}
            </p>
            <p>
              <strong>Số video phát hiện cháy:</strong>&nbsp; {fireDetectedCount}
            </p>
          </div>
        </div>
      </div>

      <div className="history-box">
        <div
          className="history-header"
          style={{
            position: "relative",
            textAlign: "center",
          }}
        >
          <h3 style={{ margin: 0 }}>Lịch sử phân tích video</h3>
          <Tooltip title="Xem lịch sử chi tiết người dùng">
            <IconButton
              onClick={() =>
                navigate(`/user-page/user-detail/${userId}/history`)
              }
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
        <div className="table-wrapper">
        <Table
          columns={columns}
          dataSource={userVideos}
           pagination={{
            current: pagination.current,
            pageSize: pagination.pageSize,
            showSizeChanger: true,
            pageSizeOptions: ["5", "10", "20", "50"],
            onChange: (page, pageSize) => {
              setPagination({
                current: page,
                pageSize: pageSize,
              });
            },
            showLessItems: true,
            locale: {
                items_per_page: "bản ghi / trang",
                jump_to: "Đi tới",
                jump_to_confirm: "Xác nhận",
                page: "Trang",
                prev_page: "Trang trước",
                next_page: "Trang sau",
                prev_5: "Lùi 5 trang",
                next_5: "Tiến 5 trang",
                prev_3: "Lùi 3 trang",
                next_3: "Tiến 3 trang",
              },
          }}
          locale={{ emptyText: "Không có video lịch sử" }}
          style={{ marginTop: 20 }}
          
          // scroll={{ x: "max-content" }}
        />
        </div>
      </div>
    </div>
  );
};

export default UserDetailPage;
