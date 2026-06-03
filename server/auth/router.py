from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from auth.database import get_db
from auth.dependencies import get_current_active_user
from auth.models import User
from auth.schemas import Token, UserCreate, UserResponse
from auth.service import authenticate_user, create_access_token, hash_password

router = APIRouter()


@router.post("/register", response_model=UserResponse, status_code=201)
def register(user_in: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == user_in.email).first():
        raise HTTPException(status_code=400, detail="Email deja înregistrat")
    user = User(email=user_in.email, hashed_password=hash_password(user_in.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post("/login", response_model=Token)
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form.username, form.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email sau parolă incorectă",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token({"sub": user.email})
    return Token(access_token=token, token_type="bearer")


@router.get("/me", response_model=UserResponse)
def me(current_user: User = Depends(get_current_active_user)):
    return current_user
