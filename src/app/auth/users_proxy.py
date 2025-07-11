import os
from pathlib import Path
from typing import Annotated  # For Python3.10+...

import httpx

from fastapi import APIRouter, Form, Request, status
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

fastapi_users_proxy_router = APIRouter(
    tags=["Auth proxy"]
)

fastapi_users_proxy_router.mount(
    "/static", StaticFiles(directory="static"), name="static"
)
current_dir = Path(__file__).resolve().parent.parent
templates_dir = os.path.join(current_dir, "templates")
static_dir = os.path.join(current_dir, "static")

templates = Jinja2Templates(directory=templates_dir)
fastapi_users_proxy_router.mount("/static", StaticFiles(directory=static_dir), name="static")


@fastapi_users_proxy_router.get("/login_proxy")
async def login_proxy_get(request: Request):
    return templates.TemplateResponse(
        "login.html",
        {"request": request},
    )


@fastapi_users_proxy_router.get("/register_proxy")
async def login_proxy_get(request: Request):
    return templates.TemplateResponse(
        "register.html",
        {"request": request},
    )


@fastapi_users_proxy_router.get("/logout_proxy")
async def logout_proxy_get(request: Request):
    return templates.TemplateResponse(
        "logout.html",
        {"request": request},
    )


@fastapi_users_proxy_router.post("/login_proxy")
async def login_proxy(
        request: Request,
        email: Annotated[str, Form()],
        password: Annotated[str, Form()],
) -> RedirectResponse:
    """WIP: HTML Form based solution to pass credentials to /auth/jwt/login..."""

    async with httpx.AsyncClient() as client:
        response = await client.post(
            url=f"{request.base_url}api/auth/login",
            headers={
                "accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={"username": email, "password": password},
        )

    # Also add an Auth Token we received to the redirect...
    redirect = RedirectResponse(
        url=f"{request.base_url}", status_code=status.HTTP_302_FOUND
    )
    redirect.set_cookie(key="auth", value=response.cookies.get("auth"), httponly=True)

    return redirect


@fastapi_users_proxy_router.post("/logout_proxy")
async def logout_proxy(
        request: Request,
        email: Annotated[str, Form()],
        password: Annotated[str, Form()],
) -> RedirectResponse:
    """WIP: HTML Form based for /auth/jwt/logout..."""

    async with httpx.AsyncClient() as client:
        await client.post(
            url=f"{request.base_url}api/auth/logout",
            headers={
                "accept": "application/json",
            },
            cookies={"auth": request.cookies.get("auth")},
        )

    redirect_response = RedirectResponse(
        url=f"{request.base_url}", status_code=status.HTTP_302_FOUND
    )
    redirect_response.delete_cookie("auth")

    return redirect_response
