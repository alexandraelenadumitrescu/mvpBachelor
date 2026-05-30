package com.photomatch.lite.api;

import java.util.List;
import okhttp3.MultipartBody;
import okhttp3.ResponseBody;
import retrofit2.Call;
import retrofit2.http.Field;
import retrofit2.http.FormUrlEncoded;
import retrofit2.http.Multipart;
import retrofit2.http.POST;
import retrofit2.http.Part;

public interface ApiService {

    @FormUrlEncoded
    @POST("auth/login")
    Call<TokenResponse> login(
        @Field("username") String email,
        @Field("password") String password
    );

    @Multipart
    @POST("blur-sensitive")
    Call<ResponseBody> blurSensitive(@Part MultipartBody.Part image);

    @Multipart
    @POST("delivery/run")
    Call<DeliveryResponse> deliveryRun(@Part List<MultipartBody.Part> photos);
}
