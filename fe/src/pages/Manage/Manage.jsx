import React, { useEffect, useState } from "react";
import axios from "axios";
import { Table, Button, Tooltip } from "antd";
import SaveAltOutlinedIcon from "@mui/icons-material/SaveAltOutlined";
import CheckCircleOutlineIcon from "@mui/icons-material/CheckCircleOutline";
import CloseIcon from "@mui/icons-material/Close";
import { PlayCircleOutlined } from '@ant-design/icons';
import { useNavigate } from "react-router-dom"; 
import { baseURL } from "../../api/api";
import SummaryApi from "../../api/api";
import "./Manage.css";

const Manage = () => {
   const navigate = useNavigate();
  const [data, setData] = useState([]);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(5);
  const limit = pageSize;

  useEffect(() => {
    const fetchVideos = async () => {
      try {
        const resVideos = await axios({
          method: SummaryApi.fetchAllVideos.method,
          url: baseURL + SummaryApi.fetchAllVideos.url,
          headers: {
            Authorization: `Bearer ${localStorage.getItem("access_token")}`,
          },
        });
        console.log(resVideos.data);
        const videos = resVideos.data.map((video, idx) => {
          const dateObj = new Date(video.updated_at);
          // dateObj.setHours(dateObj.getHours() + 7);
          const formattedDate = dateObj.toLocaleDateString("vi-VN");
          const formattedTime = dateObj.toLocaleTimeString("vi-VN", {
            hour: "2-digit",
            minute: "2-digit",
          });

          return {
            key: video.video_id,
            user: video.username || "Unknown",
            date: formattedDate,
            time: formattedTime,
            videoName: video.file_name,
            fireDetected: video.fire_detected,
            processedVideoUrl: video.processed_video_url,
          };
        });
        setData(videos);
      } catch (error) {
        console.error("Lỗi khi lấy dữ liệu video:", error);
      }
    };

    fetchVideos();
  }, []);

  const columns = [
    {
      title: "Người dùng",
      dataIndex: "user",
      key: "user",
      sorter: (a, b) => a.user.localeCompare(b.user),
      align: "center",
      render: (text) => (
        <span style={{ textAlign: "left", display: "block" }}>{text}</span>
      ),
      width: 150,
    },
    {
      title: "Ngày",
      dataIndex: "date",
      key: "date",
      sorter: (a, b) => {
        const [dayA, monthA, yearA] = a.date.split("/").map(Number);
        const [dayB, monthB, yearB] = b.date.split("/").map(Number);

        const dateA = new Date(yearA, monthA - 1, dayA);
        const dateB = new Date(yearB, monthB - 1, dayB);

        return dateA - dateB;
      },
      align: "center",
      width: 120,
    },
    {
      title: "Giờ",
      dataIndex: "time",
      key: "time",
      // sorter: (a, b) => a.time.localeCompare(b.time),
      align: "center",
      width: 90,
    },
    {
      title: "Phát hiện cháy",
      dataIndex: "fireDetected",
      key: "fireDetected",
      align: "center",
      render: (detected) =>
        detected ? (
          <CheckCircleOutlineIcon
            style={{ color: "green", fontSize: "1rem", fontWeight: "bold" }}
          />
        ) : (
          <CloseIcon
            style={{ color: "red", fontSize: "1rem", fontWeight: "bold" }}
          />
        ),
      width: 170,
    },
    {
      title: "Video",
      dataIndex: "videoName",
      key: "videoName",
      align: "center",
      render: (text, record, index) => (
    <div
      className="video-container"
      style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
      }}
    >
      <Tooltip title="Nhấn để xem lại video">
        <span
          className="video-name"
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
          onClick={() => {
            navigate("/manage/review", {
              state: { video_id: record.key, from: "/manage" },
            });
          }}
        >
          <PlayCircleOutlined style={{ marginRight: "0.5rem" }} />
          {record.videoName}
        </span>
      </Tooltip>
          <Button
            type="link"
            icon={<SaveAltOutlinedIcon className="custom-download-icon" />}
            onClick={() => {
              const downloadUrl = record.processedVideoUrl.replace(
                "/upload/",
                "/upload/fl_attachment/"
              );
              const a = document.createElement("a");
              a.href = downloadUrl;
              a.setAttribute("download", record.videoName);
              document.body.appendChild(a);
              a.click();
              document.body.removeChild(a);
            }}
            className="buttonicon-upload"
            // style={{ paddingLeft: 10 }}
          />
        </div>
      ),
    },
  ];

  const handlePageSizeChange = (current, size) => {
    setPage(current);
    setPageSize(size);
  };

  return (
    <div className="manage-container">
      <div className="model-info">
        <h2 className="model-title">
          THÔNG TIN CHI TIẾT MÔ HÌNH PHÁT HIỆN ĐÁM CHÁY
        </h2>
        <p className="model-name">
          Mô hình: <span className="highlight">YOLOv11</span>
        </p>
        <div className="model-metrics">
          <div>Precision : 0,854(B) / 0,859(M)</div>
          <div>mAP@0.5: 0,776(B) / 0,769(M) </div>
          <div>FPS: 15</div>
        </div>
      </div>

      <div className="history-section">
        <h3 className="history-title">Lịch sử hệ thống</h3>
        <div className="table-wrapper">
          <Table
            columns={columns}
            dataSource={data}
            pagination={{
              current: page,
              pageSize,
              total: data.length,
              onChange: (page) => setPage(page),
              showQuickJumper: true,
              showSizeChanger: true,
              pageSizeOptions: ["5", "10", "15", "20", "30"],
              onShowSizeChange: handlePageSizeChange,
              showLessItems: true,
              showTotal: (total, range) =>
                `${range[0]}-${range[1]} trong tổng số ${total} bản ghi`,
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
            bordered
              // scroll={{ x: "max-content" }}
          />
        </div>
      </div>
    </div>
  );
};

export default Manage;
