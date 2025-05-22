import React, { useEffect, useState } from "react";
import { List, Card, Typography, message, Empty, Button, Tooltip } from "antd";
import axios from "axios";
import { useNavigate, useParams } from "react-router-dom";
import { PlayCircleOutlined, ArrowLeftOutlined } from "@ant-design/icons";
import { baseURL } from "../../api/api";
import SummaryApi from "../../api/api";
import "./UserHistory.css";

const { Title, Text } = Typography;
const pageSize = 10;

const GetUserHistoryPage = () => {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const navigate = useNavigate();
  const { userId } = useParams(); // Lấy user_id từ URL

  const fetchHistory = async () => {
    setLoading(true);
    try {
      const token = localStorage.getItem("access_token");
       const api = SummaryApi.fetchUserHistory(userId);
    const res = await axios({
      method: api.method,
      url: baseURL + api.url, 
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });
      const allHistory = Array.isArray(res.data) ? res.data : [];
      setHistory(allHistory);
      setTotal(allHistory.length);
    } catch (err) {
      if (err.response?.status === 401) {
        message.error("Vui lòng đăng nhập.");
        navigate("/login");
      } else {
        message.error("Không thể tải lịch sử người dùng.");
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (userId) fetchHistory();
  }, [userId]);

  const cleanDescription = (desc) =>
    desc
      .replace(/VideoTypeEnum\.YOUTUBE/g, "Youtube")
      .replace(/VideoTypeEnum\.UPLOAD/g, "Upload")
      .replace(/True/g, "Có")
      .replace(/False/g, "Không")
      .replace(/,+/g, ", ")
      .replace(/\s+/g, " ")
      .trim();

  return (
    <div className="history-container">
      <div className="user-history-header">
        <Tooltip title="Quay về chi tiết người dùng">
          <Button
            type="primary"
            icon={<ArrowLeftOutlined />}
            style={{
              backgroundColor: "#4CAF50",
              color: "white",
              borderRadius: "50%",
              width: 40,
              height: 40,
              padding: 0,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              marginRight: 16,
            }}
            onClick={() => navigate(`/user-page/user-detail/${userId}`)}
          />
        </Tooltip>

        <Title level={2} style={{ flexGrow: 1, textAlign: "center", margin: 0, fontSize: "1.75rem" }}>
          Lịch sử chi tiết hoạt động người dùng
        </Title>
      </div>

      <List
        loading={loading}
        dataSource={history.slice((page - 1) * pageSize, page * pageSize)}
        pagination={{
          current: page,
          pageSize,
          total: total,
          onChange: (p) => setPage(p),
          showSizeChanger: false,
          showLessItems: true,
        }}
        locale={{
          emptyText: <Empty description="Không có lịch sử hoạt động nào" />,
        }}
        renderItem={(item) => {
          const date = new Date(item.created_at);
          // date.setHours(date.getHours() + 7);
          const formattedDate =
            date.toLocaleTimeString("vi-VN", { hour12: false }) +
            " " +
            date.toLocaleDateString("vi-VN");

          return (
            <List.Item>
              <Card style={{ width: "100%" }}>
                <div>
                  <Text type="secondary" style={{ color: "#000000", fontWeight: "600", fontSize: "1rem" }}>
                    {formattedDate} - {cleanDescription(item.description)}
                  </Text>
                </div>

                {item.video_id && (
                  <div
                    className="watch-video-link"
                    onClick={() =>
                      navigate(`/user-page/user-detail/${userId}/review`, {
                        state: {
                          video_id: item.video_id,
                          from: `/user-page/user-detail/${userId}/history`,
                        },
                      })
                    }
                  >
                    <PlayCircleOutlined style={{ fontSize: "1.25rem" }} />
                    <span style={{ fontSize: "1rem" }}>Xem video liên quan</span>
                  </div>
                )}
              </Card>
            </List.Item>
          );
        }}
      />
    </div>
  );
};

export default GetUserHistoryPage;
