import React, { useState, useEffect } from "react";
import { Modal, Form, Input, Button, message } from "antd";
import axios from "axios";
import { baseURL } from "../../api/api";
import SummaryApi from "../../api/api";
import "./EditAccountModal.css";

const EditUserModal = ({ isVisible, onClose, userInfo, onSave }) => {
  const [form] = Form.useForm();
  const [isSaving, setIsSaving] = useState(false);
  useEffect(() => {
  if (isVisible && userInfo) {
    form.setFieldsValue(userInfo);
  }
}, [isVisible, userInfo, form]);


  const handleSave = async (values) => {
   try {
      setIsSaving(true);
      const payload = {
        username: values.name,
        address: values.address,
        phone_number: values.phone,
      };
       const response = await axios({
        method: SummaryApi.update_user_info.method,             
        url: baseURL + SummaryApi.update_user_info.url,         
        headers: {
          Authorization: `Bearer ${localStorage.getItem("access_token")}`,
          Accept: "application/json",
        },
        data: payload,
      });
      onSave({
        name: response.data.username,
        email: response.data.email,
        phone: response.data.phone_number,
        address: response.data.address,
      });

      message.success("Cập nhật thông tin thành công!");
      setIsSaving(false);
      onClose();
    } catch (error) {
      message.error("Cập nhật thất bại. Vui lòng thử lại!");
      console.error("Lỗi cập nhật user:", error);
      setIsSaving(false);
     }
  };


  return (
    <Modal
      className="custom-modal"
      title={<div className="modal-title"> Chỉnh sửa thông tin </div>}
      open={isVisible}
      onCancel={onClose}
      footer={null}
    >
      <Form
        form={form}
        layout="vertical"
        onFinish={handleSave}
        initialValues={userInfo}
      >
        <div className="form-row1">
          <Form.Item
            label="Họ tên"
            name="name"
            rules={[{ required: true, message: "Vui lòng nhập họ tên!" }]}
          >
            <Input placeholder="Nhập họ tên" />
          </Form.Item>
        </div>

        {/* <div className="form-row2">
          <Form.Item
            label="Email"
            name="email"
            rules={[
              { required: true, message: "Vui lòng nhập email!" },
              { type: "email", message: "Email không hợp lệ!" },
            ]}
          >
            <Input placeholder="Nhập email" />
          </Form.Item>
        </div> */}

        <div className="form-row2">
          <Form.Item
            label="Số điện thoại"
            name="phone"
            rules={[
            { required: true, message: "Vui lòng nhập số điện thoại!" },
            {
              pattern: /^[0-9]+$/,
              message: "Số điện thoại chỉ được chứa số!"
            }
          ]}
          >
            <Input placeholder="Nhập số điện thoại" />
          </Form.Item>
        </div>

        <div className="form-row2">
          <Form.Item
            label="Địa chỉ"
            name="address"
            rules={[{ required: true, message: "Vui lòng nhập địa chỉ!" }]}
          >
            <Input placeholder="Nhập địa chỉ" />
          </Form.Item>
        </div>

        <div className="save">
          <Button type="primary" htmlType="submit" loading={isSaving}>
            Lưu thông tin
          </Button>
        </div>
      </Form>
    </Modal>
  );
};

export default EditUserModal;
