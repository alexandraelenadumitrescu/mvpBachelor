package com.photomatch.api;

import java.util.List;
import java.util.Map;

import okhttp3.MultipartBody;
import okhttp3.RequestBody;
import okhttp3.ResponseBody;
import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.Field;
import retrofit2.http.FormUrlEncoded;
import retrofit2.http.GET;
import retrofit2.http.Multipart;
import retrofit2.http.POST;
import retrofit2.http.Part;
import retrofit2.http.Path;
import retrofit2.http.Query;

public interface ApiService {

    @GET("health")
    Call<Map<String, Object>> health();

    @GET("auth/me")
    Call<UserProfile> getMe();

    // ── Auth ─────────────────────────────────────────────────────────────────

    @FormUrlEncoded
    @POST("auth/login")
    Call<TokenResponse> login(
        @Field("username") String email,
        @Field("password") String password
    );

    @POST("auth/register")
    Call<UserProfile> register(@Body RegisterRequest body);

    // ── Blur (auth required on server) ───────────────────────────────────────

    @Multipart
    @POST("blur-detect")
    Call<BlurDetectResponse> blurDetect(
        @Part  MultipartBody.Part file,
        @Query("detector") String detector
    );

    @Multipart
    @POST("blur-sensitive")
    Call<ResponseBody> blurSensitive(
        @Part  MultipartBody.Part file,
        @Query("detector") String detector
    );

    // ── Delivery (auth required on server) ───────────────────────────────────

    @Multipart
    @POST("delivery/run")
    Call<DeliveryResponse> deliveryRun(
        @Part("employees_url") RequestBody employeesUrl,
        @Part List<MultipartBody.Part> photos
    );

    // ── Vector-only retrieval ─────────────────────────────────────────────────

    @POST("search_and_correct")
    Call<SearchAndCorrectResponse> searchAndCorrect(
        @Body  SearchAndCorrectRequest request,
        @Query("aesthetic_weight") float   aestheticWeight,
        @Query("include_images")   boolean includeImages
    );

    @POST("style/search")
    Call<StyleSearchResponse> styleSearch(@Body StyleSearchRequest request);

    @POST("style/vectors")
    Call<StyleVectorsResponse> styleVectors(@Body StyleVectorsRequest request);

    // ── LUT download ──────────────────────────────────────────────────────────

    @GET("lut/{basename}")
    Call<LutResponse> getLut(@Path("basename") String basename);

    // ── Cluster ───────────────────────────────────────────────────────────────

    @POST("cluster")
    Call<ClusterResponse> cluster(@Body ClusterRequest request);
}
