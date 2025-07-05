from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.app.database.models import TortillaMeasurement
from datetime import datetime

SQLALCHEMY_DATABASE_URL = "sqlite:///./tortillas.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)



def save_measurement(track_id: int, width: float, height: float):
    db = SessionLocal()
    try:
        measurement = TortillaMeasurement(
            track_id=track_id,
            width_cm=width,
            height_cm=height,
            timestamp=datetime.now(),
        )
        db.add(measurement)
        db.commit()
        db.refresh(measurement)
        return measurement
    finally:
        db.close()
