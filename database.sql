-- Tạo kiểu ENUM
CREATE TYPE role_enum AS ENUM ('user', 'admin');
CREATE TYPE video_type_enum AS ENUM ('upload', 'youtube');
CREATE TYPE status_enum AS ENUM ('pending', 'processing', 'completed', 'failed');

-- Tạo bảng users
CREATE TABLE users (
    user_id UUID PRIMARY KEY,
    username VARCHAR(50) NOT NULL,
    email VARCHAR(100) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    address VARCHAR(255),
    phone_number VARCHAR(20),
    role role_enum NOT NULL DEFAULT 'user',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tạo bảng videos
CREATE TABLE videos (
    video_id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(user_id) ON DELETE SET NULL,
    original_video_url VARCHAR(255) NOT NULL,
    processed_video_url VARCHAR(255),
    video_type video_type_enum NOT NULL,
    youtube_url VARCHAR(255),
    status status_enum NOT NULL DEFAULT 'pending',
    fire_detected BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tạo bảng fire_detections
CREATE TABLE fire_detections (
    detection_id UUID PRIMARY KEY,
    video_id UUID REFERENCES videos(video_id) ON DELETE CASCADE,
    fire_start_time FLOAT,
    fire_end_time FLOAT,
    confidence FLOAT NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    max_fire_frame INTEGER,
    max_fire_frame_image_path VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT check_time_order CHECK (fire_start_time < fire_end_time OR fire_end_time IS NULL)
);

-- Tạo bảng notifications
CREATE TABLE notifications (
    notification_id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(user_id) ON DELETE SET NULL,
    video_id UUID REFERENCES videos(video_id) ON DELETE SET NULL,
    title VARCHAR(100) NOT NULL,
    message TEXT NOT NULL,
    enable_email_notification BOOLEAN DEFAULT FALSE,
    enable_website_notification BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tạo bảng user_history
CREATE TABLE user_history (
    history_id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(user_id) ON DELETE SET NULL,
    action_type VARCHAR(50) NOT NULL,
    video_id UUID REFERENCES videos(video_id) ON DELETE SET NULL,
    notification_id UUID REFERENCES notifications(notification_id) ON DELETE SET NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tạo hàm và trigger cho updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_users_updated_at
BEFORE UPDATE ON users
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_videos_updated_at
BEFORE UPDATE ON videos
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_notifications_updated_at
BEFORE UPDATE ON notifications
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- Tạo index để tối ưu hiệu suất
CREATE INDEX idx_users_role ON users(role);
CREATE INDEX idx_videos_user_id ON videos(user_id);
CREATE INDEX idx_videos_status ON videos(status);
CREATE INDEX idx_fire_detections_video_id ON fire_detections(video_id);
CREATE INDEX idx_notifications_user_id ON notifications(user_id);
CREATE INDEX idx_notifications_enable_email ON notifications(enable_email_notification);
CREATE INDEX idx_notifications_enable_website ON notifications(enable_website_notification);
CREATE INDEX idx_user_history_user_id ON user_history(user_id);
CREATE INDEX idx_user_history_action_type ON user_history(action_type);
CREATE INDEX idx_user_history_created_at ON user_history(created_at);