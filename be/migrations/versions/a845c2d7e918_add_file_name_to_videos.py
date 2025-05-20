"""Add file_name to videos

Revision ID: a845c2d7e918
Revises: c70b4780ea77
Create Date: 2025-05-20 10:10:53

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a845c2d7e918'
down_revision: Union[str, None] = 'c70b4780ea77'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Thêm cột file_name vào bảng videos
    op.add_column('videos', sa.Column('file_name', sa.String(255), nullable=True))


def downgrade() -> None:
    # Xóa cột file_name khỏi bảng videos
    op.drop_column('videos', 'file_name')
