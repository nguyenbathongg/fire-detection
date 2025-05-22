import React, { useEffect, useState } from "react";
import { Card, Row, Col, Statistic, Typography } from "antd";
import axios from "axios";
import { baseURL } from "../../api/api";
import SummaryApi from "../../api/api";
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  Legend,
  Title,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
} from "chart.js";
import { Doughnut } from "react-chartjs-2";
import "./Dashboard.css";

ChartJS.register(
  ArcElement,
  Tooltip,
  Legend,
  Title,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement
);

const rootFontSize = parseFloat(getComputedStyle(document.documentElement).fontSize);

const fontSizeRem = 0.7; // 0.75rem
const fontSizeCenterRem = 1.5;
const centerTextPlugin = {
  id: "centerTextPlugin",
  beforeDraw: (chart) => {
    const { ctx, chartArea } = chart;
    const fireCount = chart.data.datasets[0].data[1];
    const fontSize = fontSizeCenterRem * rootFontSize;
    const fontFamily = "'Montserrat', sans-serif";

    ctx.save();
    ctx.font = `bold ${fontSize}px ${fontFamily}`;
    ctx.fillStyle = "#ff7a3a";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    const centerX = (chartArea.left + chartArea.right) / 2;
    const centerY = (chartArea.top + chartArea.bottom) / 2;
    ctx.fillText(`${fireCount}`, centerX, centerY);
    ctx.restore();
  },
};
ChartJS.register(centerTextPlugin);

const Dashboard = () => {
  const [users, setUsers] = useState([]);
  const [videos, setVideos] = useState([]);

  useEffect(() => {
     const fetchData = async () => {
      try {
        const token = localStorage.getItem("access_token");

        const [userRes, videoRes] = await Promise.all([
          axios({
            method: SummaryApi.fetchAllUsers.method,
            url: `${baseURL}${SummaryApi.fetchAllUsers.url}`,
            headers: {
              Authorization: `Bearer ${token}`,
            },
            params: {
              skip: 0,
              limit: 1000,
            },
          }),
          axios({
            method: SummaryApi.fetchAllVideos.method,
            url: `${baseURL}${SummaryApi.fetchAllVideos.url}`,
            headers: {
              Authorization: `Bearer ${token}`,
            },
             params: {
              skip: 0,
              limit: 1000,
            },
          }),
        ]);

        setUsers(userRes.data);
        setVideos(videoRes.data);
      } catch (error) {
        console.error("Lỗi khi fetch dữ liệu:", error);
      }
    };

    fetchData();
  }, []);

  const fireCount = videos.filter((v) => v.fire_detected).length;
  const noFireCount = videos.length - fireCount;

  const pieData = {
    labels: ["Không cháy", "Có cháy"],
    datasets: [
      {
        data: [noFireCount, fireCount],
        backgroundColor: ["#4caf50", "#ff7a3a"],
      },
    ],
  };

  

  const pieOptions = {
    responsive: true,
    maintainAspectRatio: false,
    cutout: "70%",
    plugins: {
      legend: {
        reverse: true,
        position: "bottom",
        labels: {
          font: {
            family: "Montserrat",
            size: fontSizeRem * rootFontSize, 
            weight: "bold",
          },
          usePointStyle: true,
          pointStyle: "circle",
          padding: 5,
          boxWidth: 15,
          color: "#000000",
        },
      },
    },
  };

  const isToday = (dateStr) => {
    const date = new Date(dateStr);
    const today = new Date();
    return (
      date.getDate() === today.getDate() &&
      date.getMonth() === today.getMonth() &&
      date.getFullYear() === today.getFullYear()
    );
  };

 const isThisWeek = (dateStr) => {
  const date = new Date(dateStr);
  const today = new Date();

  const day = today.getDay();
  const diffToMonday = (day === 0 ? 6 : day - 1);
  const monday = new Date(today);
  monday.setDate(today.getDate() - diffToMonday);
  monday.setHours(0, 0, 0, 0);

  const sunday = new Date(monday);
  sunday.setDate(monday.getDate() + 6);
  sunday.setHours(23, 59, 59, 999);

  return date >= monday && date <= sunday;
};


  const todayAlerts = videos.filter(
    (v) => v.fire_detected && isToday(v.created_at)
  ).length;

  const weekAlerts = videos.filter(
    (v) => v.fire_detected && isThisWeek(v.created_at)
  ).length;

  return (
    <div
      className="dashboard-container"
      style={{ fontFamily: "Montserrat, sans-serif" }}
    >
      <Row gutter={[24, 24]}>
        <Col span={12}>
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              height: "100%",
              justifyContent: "space-between",
            }}
          >
            <div
              style={{
                flex: 1,
                marginBottom: "1.5rem",
                display: "flex",
                alignItems: "center",
                width: "100%",
              }}
            >
              <Card
                className="dashboard-card"
                style={{
                  width: "100%",
                  display: "flex",
                  justifyContent: "center",
                  alignItems: "center",
                }}
              >
                <div
                  style={{
                    textAlign: "center",
                    fontWeight: "600",
                    marginBottom: "0.5rem",
                    fontFamily: "Montserrat",
                    fontSize: "1.1rem",
                    color: "#000000",
                  }}
                >
                  Tổng số người dùng
                </div>
                <Statistic
                  value={users.length}
                  valueStyle={{
                    fontSize: "1rem",
                    color: "#000000",
                    fontFamily: "Montserrat",
                    display: "flex",
                    justifyContent: "center",
                    alignItems: "center",
                  }}
                />
              </Card>
            </div>
            <div style={{ flex: 1, display: "flex", alignItems: "center" }}>
              <Card
                className="dashboard-card"
                style={{
                  width: "100%",
                  display: "flex",
                  justifyContent: "center",
                  alignItems: "center",
                }}
              >
                <div
                  style={{
                    textAlign: "center",
                    fontWeight: "600",
                    marginBottom: "0.5rem",
                    fontFamily: "Montserrat",
                    fontSize: "1.1rem",
                    color: "#000000",
                  }}
                >
                  Số video đã tải lên
                </div>
                <Statistic
                  value={videos.length}
                  valueStyle={{
                    fontSize: "1rem",
                    color: "#000000",
                    fontFamily: "Montserrat",
                    display: "flex",
                    justifyContent: "center",
                    alignItems: "center",
                  }}
                />
              </Card>
            </div>
          </div>
        </Col>
        <Col span={12}>
          <Card
            className="dashboard-card"
            style={{ height: "100%", textAlign: "center" }}
          >
            <div
              className="chart-title"
              style={{
                textAlign: "center",
                fontWeight: "600",
                marginBottom: "1rem",
                fontFamily: "Montserrat",
                fontSize: "1.1rem",
              }}
            >
              Tỉ lệ video có cháy / không cháy
            </div>
            <div
             className="donut-chart"
            >
              <Doughnut data={pieData} options={pieOptions} />
              
            </div>
            <span style={{ fontSize: "1rem" }}>
                Tỷ lệ cháy / tổng video:{" "}
                {videos.length > 0
                  ? `${((fireCount / videos.length) * 100).toFixed(2)}%`
                  : "0%"}
              </span>
          </Card>
        </Col>
      </Row>

      <Row gutter={[24, 24]} style={{ marginTop: "1.5rem" }}>
        <Col span={24}>
          <Card className="dashboard-card" style={{ textAlign: "center" }}>
            <Typography.Title
              level={4}
              style={{
                marginBottom: 16,
                fontFamily: "Montserrat",
                textAlign: "center",
                margin: 0,
                fontSize: "1.1rem",
              }}
            >
                Thống kê hoạt động hệ thống trong tuần
            </Typography.Title>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                fontSize: "1rem",
                padding: "0 2rem",
                color: "#000000",
                marginTop: "0.5rem",
              }}
            >
               <span>Người dùng mới trong tuần: {users.filter(u => isThisWeek(u.created_at)).length}</span>
              <span>Video tải lên trong tuần: {videos.filter(v => isThisWeek(v.created_at)).length}</span>
              <span>Video phát hiện cháy trong tuần: {weekAlerts}</span>
               
            </div>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default Dashboard;
