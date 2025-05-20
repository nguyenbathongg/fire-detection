import React from "react";
import WildFire from "../../assets/img/Wild_fire.png";
import { MdOutlineLocalFireDepartment } from "react-icons/md";
import "./Home.css";

const Home = () => {
  return (
    <div className="fire-detection-wrapper">
      <div className="fire-container">
      <div className="fire-header">
        <MdOutlineLocalFireDepartment className="fire-icon-home"/>
        <h1>HỆ THỐNG PHÁT HIỆN ĐÁM CHÁY QUA VIDEO</h1>
      </div>
      <div className="fire-detection-container">
        <div className="fire-text">
          <div className="fire-text-header">
          <h5>Giới thiệu chung</h5>
          </div>
          <p>
            Với sự phát triển của trí tuệ nhân tạo và thị giác máy tính, việc
            khai thác dữ liệu từ video để phát hiện sớm các hiện tượng bất
            thường, như đám cháy, đang trở thành một hướng đi đầy tiềm năng.
          </p>
          <p>
            Ứng dụng này triển khai một hệ thống có khả năng nhận diện và đánh
            dấu vùng cháy trong video bằng các kỹ thuật xử lý ảnh, học sâu và
            khai phá dữ liệu. Người dùng có thể tải lên hoặc nhận kết quả từ hệ
            thống để tự động phân tích và hiển thị cảnh báo thực quan.
          </p>
          <p>
            Đây là một ứng dụng thực tiễn của khai phá dữ liệu trong lĩnh vực an
            toàn và giám sát, nhằm hỗ trợ cảnh báo kịp thời trước các tình
            huống nguy hiểm.
          </p>
        </div>
        <div className="fire-image">
          <img
            src={WildFire} 
            alt="Wildfire in forest"
            className="fire-img"
          />
        </div>
      </div>
      </div>
    </div>
  );
};

export default Home;
