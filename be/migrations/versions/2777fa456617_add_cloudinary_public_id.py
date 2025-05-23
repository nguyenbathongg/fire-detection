"""Add_cloudinary_public_id

Revision ID: 2777fa456617
Revises: 
Create Date: 2025-05-10 00:38:26.895406

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '2777fa456617'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index('idx_fire_detections_video_id', table_name='fire_detections')
    op.drop_index('idx_notifications_enable_email', table_name='notifications')
    op.drop_index('idx_notifications_enable_website', table_name='notifications')
    op.drop_index('idx_notifications_user_id', table_name='notifications')
    op.drop_index('idx_user_history_action_type', table_name='user_history')
    op.drop_index('idx_user_history_created_at', table_name='user_history')
    op.drop_index('idx_user_history_user_id', table_name='user_history')
    op.alter_column('users', 'role',
               existing_type=postgresql.ENUM('user', 'admin', name='role_enum'),
               type_=sa.String(length=10),
               existing_nullable=False,
               existing_server_default=sa.text("'user'::role_enum"))
    op.drop_index('idx_users_role', table_name='users')
    op.add_column('videos', sa.Column('cloudinary_public_id', sa.String(length=255), nullable=True))
    op.alter_column('videos', 'video_type',
               existing_type=postgresql.ENUM('upload', 'youtube', name='video_type_enum'),
               type_=sa.String(length=10),
               existing_nullable=False)
    op.alter_column('videos', 'status',
               existing_type=postgresql.ENUM('pending', 'processing', 'completed', 'failed', name='status_enum'),
               type_=sa.String(length=10),
               existing_nullable=False,
               existing_server_default=sa.text("'pending'::status_enum"))
    op.alter_column('videos', 'fire_detected',
               existing_type=sa.BOOLEAN(),
               nullable=False,
               existing_server_default=sa.text('false'))
    op.drop_index('idx_videos_status', table_name='videos')
    op.drop_index('idx_videos_user_id', table_name='videos')
    op.drop_constraint('videos_user_id_fkey', 'videos', type_='foreignkey')
    op.create_foreign_key(None, 'videos', 'users', ['user_id'], ['user_id'])
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_constraint(None, 'videos', type_='foreignkey')
    op.create_foreign_key('videos_user_id_fkey', 'videos', 'users', ['user_id'], ['user_id'], ondelete='SET NULL')
    op.create_index('idx_videos_user_id', 'videos', ['user_id'], unique=False)
    op.create_index('idx_videos_status', 'videos', ['status'], unique=False)
    op.alter_column('videos', 'fire_detected',
               existing_type=sa.BOOLEAN(),
               nullable=True,
               existing_server_default=sa.text('false'))
    op.alter_column('videos', 'status',
               existing_type=sa.String(length=10),
               type_=postgresql.ENUM('pending', 'processing', 'completed', 'failed', name='status_enum'),
               existing_nullable=False,
               existing_server_default=sa.text("'pending'::status_enum"))
    op.alter_column('videos', 'video_type',
               existing_type=sa.String(length=10),
               type_=postgresql.ENUM('upload', 'youtube', name='video_type_enum'),
               existing_nullable=False)
    op.drop_column('videos', 'cloudinary_public_id')
    op.create_index('idx_users_role', 'users', ['role'], unique=False)
    op.alter_column('users', 'role',
               existing_type=sa.String(length=10),
               type_=postgresql.ENUM('user', 'admin', name='role_enum'),
               existing_nullable=False,
               existing_server_default=sa.text("'user'::role_enum"))
    op.create_index('idx_user_history_user_id', 'user_history', ['user_id'], unique=False)
    op.create_index('idx_user_history_created_at', 'user_history', ['created_at'], unique=False)
    op.create_index('idx_user_history_action_type', 'user_history', ['action_type'], unique=False)
    op.create_index('idx_notifications_user_id', 'notifications', ['user_id'], unique=False)
    op.create_index('idx_notifications_enable_website', 'notifications', ['enable_website_notification'], unique=False)
    op.create_index('idx_notifications_enable_email', 'notifications', ['enable_email_notification'], unique=False)
    op.create_index('idx_fire_detections_video_id', 'fire_detections', ['video_id'], unique=False)
    # ### end Alembic commands ### 