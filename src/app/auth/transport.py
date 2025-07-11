from fastapi_users.authentication import BearerTransport
from fastapi_users.authentication import CookieTransport

bearer_transport = BearerTransport(tokenUrl="/api/auth/login",)
cookie_transport = CookieTransport(cookie_max_age=300,cookie_name='fastapi_session')
