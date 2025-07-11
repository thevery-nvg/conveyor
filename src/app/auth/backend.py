from fastapi_users.authentication import AuthenticationBackend
from fastapi_users.authentication import CookieTransport
from fastapi_users.authentication import JWTStrategy

SECRET = "SECRET"


def get_jwt_strategy() -> JWTStrategy:
    return JWTStrategy(secret=SECRET, lifetime_seconds=3600)


cookie_transport = CookieTransport(
    cookie_name="fastapiusersauth",
    cookie_secure=False,  # важно для HTTP, если не используете HTTPS
    cookie_path="/",
    cookie_httponly=True,
    cookie_samesite="lax",  # или "none" + Secure=True, если через HTTPS и с кросс-доменом
    cookie_max_age=3600,
)

auth_cookie_backend = AuthenticationBackend(
    name="fastapi-users-auth",
    transport=cookie_transport,
    get_strategy=get_jwt_strategy,
)
