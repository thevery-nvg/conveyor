from pydantic import BaseModel
from datetime import date

class TortillaStatsResponse(BaseModel):
    date: date
    valid: int
    invalid_size: int
    invalid_oval: int

    class Config:
        orm_mode = True