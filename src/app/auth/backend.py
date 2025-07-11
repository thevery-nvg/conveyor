from fastapi import Request
from fastapi_users.authentication import AuthenticationBackend
from .transport import bearer_transport, cookie_transport
from .strategy import get_jwt_strategy


auth_cookie_backend = AuthenticationBackend(
    name="fastapi-users-auth",
    transport=cookie_transport,
    get_strategy=get_jwt_strategy,
)


