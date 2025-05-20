import React, { useEffect, useState } from "react";
import { List, Card, Typography, message, Empty, Button ,Tooltip } from "antd";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import { PlayCircleOutlined, ArrowLeftOutlined } from "@ant-design/icons";
import "./UserHistory.css"
import { baseURL } from "../../api/api";
import SummaryApi from "../../api/api";
const { Title, Text, Paragraph } = Typography;
const pageSize = 10;

const UserHistoryPage = () => {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const navigate = useNavigate();

  const fetchHistory = async () => {
    setLoading(true);
    try {
      const token = localStorage.getItem("access_token");
      const { url, method } = SummaryApi.fetchHistory;

    const res = await axios({
      method,
      url: `${baseURL}${url}?skip=0&limit=1000`, 
      headers: {
        Authorization: `Bearer ${token}`,
      },
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

  return (
    <div className="history-container">
        <div className="user-history-header">
        <Tooltip title="Quay về tài khoản">
    <Button
      type="primary"
      icon={<ArrowLeftOutlined />}
      style={{
        backgroundColor: "#4CAF50",
        color: "white",
        borderColor: "#4CAF50",
        // color: "#4CAF50",
        borderRadius: "50%",   
        width: 40,
        height: 40,
        padding: 0,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        marginRight: 16,
      }}
      onClick={() => navigate("/account")}
    />
  </Tooltip>

      <Title level={2} style={{flexGrow: 1,
      textAlign: "center",
      margin: 0,}}>
        Lịch sử chi tiết hoạt động
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
        }}
        locale={{
          emptyText: <Empty description="Không có lịch sử hoạt động nào" />,
        }}
        renderItem={(item) => {
          const date = new Date(item.created_at);
          date.setHours(date.getHours() + 7);
          const formattedDate =
            date.toLocaleTimeString("vi-VN", { hour12: false }) +
            " " +
            date.toLocaleDateString("vi-VN");

          return (
            <List.Item>
              <Card style={{ width: "100%" }}>
                <div>
                  <Text type="secondary" style={{ color: "#000000", fontWeight: "600" }}>
                    {formattedDate} - {cleanDescription(item.description)}
                  </Text>
                </div>

                {item.video_id && (
                  <div
                    className="watch-video-link"
                    
                    onClick={() =>
                      navigate("/account/history/review", {
                        state: {
                          video_id: item.video_id,
                          from: "/account/history",
                        },
                      })
                    }
                  >
                    <PlayCircleOutlined style={{ fontSize: 18 }} />
                    <span>Xem video liên quan</span>
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

export default UserHistoryPage;
