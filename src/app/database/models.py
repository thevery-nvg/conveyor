from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from src.app.database.base import Base


class TortillaMeasurement(Base):
    id = Column(Integer, primary_key=True, index=True)
    track_id = Column(Integer, index=True)
    width_cm = Column(Float)
    height_cm = Column(Float)
    timestamp = Column(DateTime)

    def __repr__(self):
        return f"<Tortilla {self.track_id} (W: {self.width_cm}, H: {self.height_cm})>"