from sqlalchemy import Column, Integer, String
from database import Base

class User(Base):
  __tablename__ = "users"

  id = Column(Integer, primary_key=True)
  full_name = Column(String(50), index=True)
  email = Column(String(50), unique=True, index=True)