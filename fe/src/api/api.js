export const baseURL =
  process.env.REACT_APP_BASE_API || "http://localhost:8000";

const SummaryApi = {
  login: {
    url: "/api/v1/auth/login",
    method: "post",
  },
  register: {
    url: "/api/v1/auth/register",
    method: "post",
  },
  getCurrentUser: {
    url: "/api/v1/users/me",
    method: "get",
  },
  update_user_info: {
    url: "/api/v1/users/me",
    method: "put",
  },
  directProcessWS: `${baseURL.replace(/^http/, "ws")}/api/v1/ws/direct-process`,
  fireCameraWS: `${baseURL.replace(/^http/, "ws")}/api/v1/ws/fire`,
  fetchVideos: {
    url: "/api/v1/videos",
    method: "get",
  },
  fetchHistory: {
    url: "/api/v1/history/me",
    method: "get",
  },
  fetchAllUsers: {
    url: "/api/v1/users",
    method: "get",
  },
  fetchAllVideos: {
    url: "/api/v1/videos/all",
    method: "get",
  },
  getVideoDetail: (video_id) => ({
    url: `/api/v1/videos/${video_id}`,
    method: "get",
  }),
  getUserById: (userId) => ({ 
    url: `/api/v1/users/${userId}`, 
    method: "get" 
  }),
  changePassword: {
    url: "/api/v1/auth/change-password",
    method: "post",
  },
  notificationSettings: {
    url: "/api/v1/notifications/settings",
    method: {
      get: "get",
      post: "post",
    },
  },
  fetchUserHistory: (userId) => ({
    url: `/api/v1/history/${userId}?skip=0&limit=1000`,
    method: "get",
  }),
  addUser: {
    url: "/api/v1/users",
    method: "post",
  },
};

export default SummaryApi;
