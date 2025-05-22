import React, { useState, useEffect } from "react";
import {
  Table,
  Button,
  Modal,
  Form,
  Input,
  Select,
  Popconfirm,
  Tooltip,
  Tabs,
} from "antd";
import { EditOutlined, DeleteOutlined, PlusOutlined } from "@ant-design/icons";
import axios from "axios";
import { baseURL } from "../../api/api";
import SummaryApi from "../../api/api";
import { useNavigate } from "react-router-dom";
import Snackbar from "@mui/material/Snackbar";
import MuiAlert from "@mui/material/Alert";
import "./UserPage.css";

const { TabPane } = Tabs;

const Alert = React.forwardRef(function Alert(props, ref) {
  return <MuiAlert elevation={6} ref={ref} variant="filled" {...props} />;
});

const UsersPage = () => {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [form] = Form.useForm();
  const [activeTab, setActiveTab] = useState("user");
  const navigate = useNavigate();

  const [snackbarOpen, setSnackbarOpen] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState("");
  const [snackbarSeverity, setSnackbarSeverity] = useState("success");

  const showSnackbar = (message, severity = "success") => {
    setSnackbarMessage(message);
    setSnackbarSeverity(severity);
    setSnackbarOpen(true);
  };

  const handleSnackbarClose = (_, reason) => {
    if (reason === "clickaway") return;
    setSnackbarOpen(false);
  };

  useEffect(() => {
    fetchUsers();
  }, []);

  const fetchUsers = () => {
    setLoading(true);
    axios({
      method: SummaryApi.fetchAllUsers.method,
      url: baseURL + SummaryApi.fetchAllUsers.url,
      headers: {
        Authorization: `Bearer ${localStorage.getItem("access_token")}`,
      },
      params: { skip: 0, limit: 100 },
    })
      .then((response) => {
        setUsers(response.data);
        setLoading(false);
      })
      .catch(() => {
        showSnackbar("Lỗi khi tải danh sách người dùng", "error");
        setLoading(false);
      });
  };

  const handleDeleteUser = (userId) => {
    axios({
      method: "delete",
      url: `${baseURL}/api/v1/users/${userId}`,
      headers: {
        Authorization: `Bearer ${localStorage.getItem("access_token")}`,
      },
    })
      .then(() => {
        showSnackbar("Đã xóa người dùng");
        fetchUsers();
      })
      .catch(() => {
        showSnackbar("Xóa người dùng thất bại", "error");
      });
  };

  const handleEditUser = (userId) => {
    navigate(`/user-page/user-detail/${userId}`);
  };

  const handleAddUser = async () => {
    try {
      const values = await form.validateFields();
      const payload = { ...values, role: "admin" };

     axios({
        method: SummaryApi.addUser.method,
        url: baseURL + SummaryApi.addUser.url,
        headers: {
          Authorization: `Bearer ${localStorage.getItem("access_token")}`,
        },
        data: payload,
      })
        .then(() => {
          showSnackbar("Đã thêm người dùng thành công!");
          fetchUsers();
          setIsModalVisible(false);
          form.resetFields();
        })
        .catch((error) => {
          // showSnackbar("Thêm người dùng thất bại", "error");
          const errorMessage =
            error.response?.data?.message ||
            error.message ||
            "Thêm người dùng thất bại";
          showSnackbar(errorMessage, "error");
        });
    } catch (error) {
      console.error("Form validation failed:", error);
    }
  };

  const handleModalCancel = () => {
    setIsModalVisible(false);
    form.resetFields();
  };

  const columns = [
    {
      title: "Tên người dùng",
      dataIndex: "username",
      key: "username",
      align: "center",
      render: (text) => (
        <span style={{ textAlign: "left", display: "block" }}>{text}</span>
      ),
    },
    {
      title: "Email",
      dataIndex: "email",
      key: "email",
      align: "center",
      render: (text) => (
        <span style={{ textAlign: "left", display: "block" }}>{text}</span>
      ),
    },
    {
      title: "Vai trò",
      dataIndex: "role",
      key: "role",
      align: "center",
    },
    {
      title: "Hành động",
      key: "action",
      align: "center",
      render: (_, record) => (
        <>
          <Tooltip title="Chỉnh sửa">
            <Button
              icon={<EditOutlined />}
              onClick={() => handleEditUser(record.user_id)}
              style={{ marginRight: "0.3rem", fontSize: "0.8rem" }}
            />
          </Tooltip>
          <Popconfirm
            title="Bạn có chắc chắn muốn xóa người dùng này?"
            onConfirm={() => handleDeleteUser(record.user_id)}
            okText="Xóa"
            cancelText="Hủy"
          >
            <Tooltip title="Xóa">
              <Button danger icon={<DeleteOutlined />} />
            </Tooltip>
          </Popconfirm>
        </>
      ),
    },
  ];

  const filteredUsers = users.filter((u) => u.role === activeTab);

  return (
    <div className="users-page">
      <div className="header">
        <h2>Quản lý người dùng</h2>
        {/* {activeTab === "admin" && (
          <Button
            type="primary"
            icon={<PlusOutlined />}
            onClick={() => setIsModalVisible(true)}
            className="add-user-button"
          >
            Thêm admin
          </Button>
        )} */}
      </div>

      {/* <Tabs
        defaultActiveKey="user"
        onChange={setActiveTab}
        items={[
          { label: "User", key: "user" },
          { label: "Admin", key: "admin" },
        ]}
      /> */}

      <Tabs
        defaultActiveKey="user"
        onChange={setActiveTab}
        items={[{ label: "User", key: "user" }]}
      />

      <Table
        columns={columns}
        dataSource={filteredUsers}
        loading={loading}
        rowKey="user_id"
        pagination={{
          pageSize: 10,
          // showSizeChanger: true,
          // pageSizeOptions: ['5', '10', '20', '50'],
          showTotal: (total, range) => `${range[0]}-${range[1]} trong ${total} người dùng`,
          showLessItems: true,
        }}
        className="users-table"
          // scroll={{ x: "max-content" }}
      />

      <Modal
        className="custom-modal"
        title={<div className="modal-title">Thêm admin</div>}
        open={isModalVisible}
        onCancel={handleModalCancel}
        footer={null}
      >
        <Form form={form} layout="vertical" onFinish={handleAddUser}>
          <Form.Item
            label="Username"
            name="username"
            rules={[
              { required: true, message: "Vui lòng nhập tên người dùng!" },
            ]}
          >
            <Input />
          </Form.Item>
          <Form.Item
            label="Email"
            name="email"
            rules={[
              {
                required: true,
                type: "email",
                message: "Vui lòng nhập email hợp lệ!",
              },
            ]}
          >
            <Input />
          </Form.Item>
          <Form.Item
            label="Mật khẩu"
            name="password"
            rules={[
              { required: true, message: "Vui lòng nhập mật khẩu!" },
              {
                pattern:
                  /^(?=.*[A-Za-z])(?=.*\d)(?=.*[!@#$%^&*])[A-Za-z\d!@#$%^&*]{6,}$/,
                message:
                  "Mật khẩu phải có ít nhất 6 ký tự, bao gồm chữ cái, số và ký tự đặc biệt.",
              },
            ]}
          >
            <Input.Password />
          </Form.Item>

          {activeTab !== "admin" && (
            <Form.Item
              label="Role"
              name="role"
              rules={[{ required: true, message: "Vui lòng chọn vai trò!" }]}
            >
              <Select>
                <Select.Option value="admin">Admin</Select.Option>
                <Select.Option value="user">User</Select.Option>
              </Select>
            </Form.Item>
          )}

          <div className="save">
            <Button type="primary" htmlType="submit">
              Lưu
            </Button>
          </div>
        </Form>
      </Modal>

      <Snackbar
        open={snackbarOpen}
        autoHideDuration={3000}
        onClose={handleSnackbarClose}
        anchorOrigin={{ vertical: "top", horizontal: "center" }}
      >
        <Alert onClose={handleSnackbarClose} severity={snackbarSeverity}>
          {snackbarMessage}
        </Alert>
      </Snackbar>
    </div>
  );
};

export default UsersPage;
