package com.photomatch.api;

public class UserCreate {
    public String email;
    public String password;

    public UserCreate(String email, String password) {
        this.email    = email;
        this.password = password;
    }
}
