import React, { useState, useEffect } from "react";
import axios from "axios";
import { FaBars } from "react-icons/fa";
import SmartDisplayOutlinedIcon from "@mui/icons-material/SmartDisplayOutlined";
import SettingsOutlinedIcon from "@mui/icons-material/SettingsOutlined";
import ManageAccountsOutlinedIcon from "@mui/icons-material/ManageAccountsOutlined";
import VideoSettingsOutlinedIcon from "@mui/icons-material/VideoSettingsOutlined";
import PersonOutlineOutlinedIcon from "@mui/icons-material/PersonOutlineOutlined";
import { MdOutlineLocalFireDepartment } from "react-icons/md";
import HomeOutlinedIcon from "@mui/icons-material/HomeOutlined";
import { NavLink } from "react-router-dom";
import { useLocation } from "react-router-dom";
import { useDispatch, useSelector } from "react-redux";
import { fetchCurrentUser } from "../../../redux/apiRequest";
import "./Sidebar.css";

const Sidebar = () => {
  const [collapsed, setCollapsed] = useState(false);
  const location = useLocation();
  const dispatch = useDispatch();
  const isVideoActive = ["/video", "/detectionresult", "/cameraresult"].some(
    (path) => location.pathname.startsWith(path)
  );

  useEffect(() => {
    fetchCurrentUser(dispatch);
  }, [dispatch]);

  const user = useSelector((state) => state.user.user);
  return (
    <div className={`sidebar ${collapsed ? "collapsed" : "expanded"}`}>
      <div className="sidebar-header">
        <button
          className="collapse-btn"
          onClick={() => setCollapsed(!collapsed)}
        >
          <FaBars />
        </button>
      </div>
      <div
        className="fire-icon"
        style={{ display: collapsed ? "none" : "block" }}
      >
        <NavLink to={user?.role === "admin" ? "/dashboard" : "/home"}>
          <MdOutlineLocalFireDepartment />
        </NavLink>
      </div>
      <nav className="sidebar-nav">
        <ul>
          {user?.role === "user" && (
            <>
              <li>
                <NavLink
                  to="/video"
                  className={({ isActive }) =>
                    `sidebar-item ${isVideoActive ? "active" : ""}`
                  }
                >
                  <SmartDisplayOutlinedIcon className="sidebar-icon" />
                  {!collapsed && "Video"}
                </NavLink>
              </li>
              <li>
                <NavLink
                  to="/account"
                  className={({ isActive }) =>
                    `sidebar-item ${isActive ? "active" : ""}`
                  }
                >
                  <PersonOutlineOutlinedIcon className="sidebar-icon" />
                  {!collapsed && "Tài khoản"}
                </NavLink>
              </li>
              <li>
                <NavLink
                  to="/settings"
                  className={({ isActive }) =>
                    `sidebar-item ${isActive ? "active" : ""}`
                  }
                >
                  <SettingsOutlinedIcon className="sidebar-icon" />
                  {!collapsed && "Cài đặt"}
                </NavLink>
              </li>
            </>
          )}

          {user?.role === "admin" && (
            <>
              <li>
                <NavLink
                  to="/dashboard"
                  className={({ isActive }) =>
                    `sidebar-item ${isActive ? "active" : ""}`
                  }
                >
                  <HomeOutlinedIcon className="sidebar-icon" />
                  {!collapsed && "Tổng quan"}
                </NavLink>
              </li>
               <li>
                <NavLink
                  to="/account"
                  className={({ isActive }) =>
                    `sidebar-item ${isActive ? "active" : ""}`
                  }
                >
                  <PersonOutlineOutlinedIcon className="sidebar-icon" />
                  {!collapsed && "Tài khoản"}
                </NavLink>
              </li>
              <li>
                <NavLink
                  to="/manage"
                  className={({ isActive }) =>
                    `sidebar-item ${isActive ? "active" : ""}`
                  }
                >
                  <VideoSettingsOutlinedIcon className="sidebar-icon" />
                  {!collapsed && "Hệ thống"}
                </NavLink>
              </li>
              <li>
                <NavLink
                  to="/user-page"
                  className={({ isActive }) =>
                    `sidebar-item ${isActive ? "active" : ""}`
                  }
                >
                  <ManageAccountsOutlinedIcon className="sidebar-icon" />
                  {!collapsed && "Người dùng"}
                </NavLink>
              </li>
             
            </>
          )}
        </ul>
      </nav>
    </div>
  );
};

export default Sidebar;
