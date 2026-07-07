from fastapi import HTTPException, Request

def make_auth_dependency(token: str | None):
    async def check(request: Request):
        if token is None:
            return
        got = request.headers.get("authorization", "")
        if got != f"Bearer {token}":
            raise HTTPException(401, "invalid or missing bearer token")
    return check
