package com.photomatch.api;

import com.google.gson.annotations.SerializedName;

public class UserProfile {
    public int     id;
    public String  email;
    @SerializedName("is_active") public boolean isActive;
}
