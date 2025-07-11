from fastapi import APIRouter
from fastapi.security import HTTPBearer
from fastapi_users import FastAPIUsers

from app.auth.backend import auth_cookie_backend
from app.auth.models import User
from app.auth.user_manager import get_user_manager
from app.auth.schemas import UserRead, UserCreate, UserUpdate

fastapi_users = FastAPIUsers[User, int](
    get_user_manager,
    [auth_cookie_backend],
)

http_bearer = HTTPBearer(auto_error=False)
auth_router = APIRouter(
    prefix="/auth",
    tags=["Auth"],
    #dependencies=[Depends(http_bearer)],
)

# \login \logout
auth_router.include_router(
    fastapi_users.get_auth_router(auth_cookie_backend),
)
# \register
auth_router.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate),
)
# \reset-password \forgot-password
auth_router.include_router(
    fastapi_users.get_reset_password_router(),
    )
# \verify \request-verify-token
auth_router.include_router(
    fastapi_users.get_verify_router(UserRead),
)

users_router = APIRouter(
    prefix="/users",
    tags=["Users"],
)
# \me \ {id}
users_router.include_router(
    fastapi_users.get_users_router(
        UserRead,
        UserUpdate,
    )
)
current_user = fastapi_users.current_user(active=True,optional=True)
current_user_strict = fastapi_users.current_user(active=True)
current_superuser = fastapi_users.current_user(
    active=True,
    superuser=True,
)
