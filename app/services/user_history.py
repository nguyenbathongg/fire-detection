from sqlalchemy.orm import Session
from sqlalchemy import desc
import uuid
from app.models import User, UserHistory


def get_user_history(db: Session, user_id: uuid.UUID, skip: int = 0, limit: int = 100):
    return db.query(UserHistory).filter(UserHistory.user_id == user_id) \
        .order_by(desc(UserHistory.created_at)).offset(skip).limit(limit).all()


def add_view_history(db: Session, user_id: uuid.UUID, action_type: str, description: str):
    history = UserHistory(
        history_id=uuid.uuid4(),
        user_id=user_id,
        action_type=action_type,
        description=description,
    )
    db.add(history)
    db.commit()
