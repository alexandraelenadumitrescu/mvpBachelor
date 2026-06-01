package com.photomatch.lite.api;

import java.util.List;
import okhttp3.MultipartBody;
import okhttp3.RequestBody;
import okhttp3.ResponseBody;
import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.Field;
import retrofit2.http.FormUrlEncoded;
import retrofit2.http.Multipart;
import retrofit2.http.POST;
import retrofit2.http.Part;
import retrofit2.http.Query;

public interface ApiService {

    @FormUrlEncoded
    @POST("auth/login")
    Call<TokenResponse> login(
        @Field("username") String email,
        @Field("password") String password
    );

    @Multipart
    @POST("blur-sensitive")
    Call<ResponseBody> blurSensitive(
        @Part  MultipartBody.Part image,
        @Query("detector") String detector
    );

    @Multipart
    @POST("delivery/run")
    Call<DeliveryResponse> deliveryRun(
        @Part("employees_url") RequestBody employeesUrl,
        @Part List<MultipartBody.Part> photos
    );

    @Multipart
    @POST("cluster/faces")
    Call<FaceClusterResponse> clusterFaces(@Part List<MultipartBody.Part> photos);

    @POST("delivery/match-embeddings")
    Call<MatchResponse> matchEmbeddings(@Body MatchRequest request);

    @Multipart
    @POST("delivery/send-matched")
    Call<DeliveryResponse> sendMatched(
        @Part("to_email") RequestBody toEmail,
        @Part List<MultipartBody.Part> photos
    );
}
