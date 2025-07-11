from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = "sqlite+aiosqlite:///./tortillas.db"

from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine, async_sessionmaker



class DatabaseHelper:
    def __init__(
            self,
            url: str = str(SQLALCHEMY_DATABASE_URL),  # Дефолтный URL для SQLite
            echo: bool = False,
            pool_size: int = 5,
            max_overflow: int = 10,
            connect_args: dict = None
    ):
        # Для SQLite нужно добавить специальные connect_args
        default_connect_args = {"check_same_thread": False}
        if connect_args:
            default_connect_args.update(connect_args)

        self.engine: AsyncEngine = create_async_engine(
            url,
            echo=echo,
            pool_size=pool_size,
            max_overflow=max_overflow,
            connect_args=default_connect_args  # Добавляем параметры для SQLite
        )
        self.session_factory = async_sessionmaker(
            bind=self.engine,
            expire_on_commit=False,
            autoflush=False,
            autocommit=False,
        )

    async def dispose(self):
        await self.engine.dispose()

    async def session_getter(self):
        async with self.session_factory() as session:
            yield session


# Инициализация для SQLite
db_helper = DatabaseHelper(
    url=str(SQLALCHEMY_DATABASE_URL),  # Или берем из settings, если там указано
    echo=False,
    pool_size= 5,
    max_overflow= 10,
    connect_args={"check_same_thread": False}  # Обязательно для SQLite
)