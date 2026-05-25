"""Seed database with sample stations and routes."""

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend.database import Base, engine


def seed():
    Base.metadata.create_all(bind=engine)
    print("Seeded database with sample data")


if __name__ == "__main__":
    seed()
