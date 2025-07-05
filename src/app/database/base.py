from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase, declared_attr
from sqlalchemy import Column, Integer, Float, DateTime

naming_convention: dict[str, str] = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


def camel_case_to_snake_case(input_str: str) -> str:
    """
    >>> camel_case_to_snake_case("SomeSDK")
    'some_sdk'
    >>> camel_case_to_snake_case("RServoDrive")
    'r_servo_drive'
    >>> camel_case_to_snake_case("SDKDemo")
    'sdk_demo'
    """
    chars = []
    for c_idx, char in enumerate(input_str):
        if c_idx and char.isupper():
            nxt_idx = c_idx + 1
            flag = nxt_idx >= len(input_str) or input_str[nxt_idx].isupper()
            prev_char = input_str[c_idx - 1]
            if prev_char.isupper() and flag:
                pass
            else:
                chars.append("_")
        chars.append(char.lower())
    return "".join(chars)


class Base(DeclarativeBase):
    __abstract__ = True
    metadata = MetaData(
        naming_convention=naming_convention
    )

    @declared_attr
    def __tablename__(cls):
        return f"{camel_case_to_snake_case(cls.__name__)}s"




class TortillaMeasurement(Base):
    id = Column(Integer, primary_key=True, index=True)
    track_id = Column(Integer, index=True)
    width_cm = Column(Float)
    height_cm = Column(Float)
    timestamp = Column(DateTime)

    def __repr__(self):
        return f"<Tortilla {self.track_id} (W: {self.width_cm}, H: {self.height_cm})>"
