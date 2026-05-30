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

/**
 * V2 API — images NEVER leave the device.
 * All server calls use pre-computed 517-dim hybrid vectors or metadata only.
 *
 * Auth: after login, store the token and pass it via AuthInterceptor
 * (OkHttpClient interceptor that adds "Authorization: Bearer <token>" to every request).
 */
public interface ApiService {

    @GET("health")
    Call<Map<String, Object>> health();

    // ── Authentication ───────────────────────────────────────────────────────

    @POST("auth/register")
    Call<UserResponse> register(@Body UserCreate request);

    /** OAuth2 form-encoded login. Returns JWT token. */
    @FormUrlEncoded
    @POST("auth/login")
    Call<TokenResponse> login(
        @Field("username") String email,
        @Field("password") String password
    );

    @GET("auth/me")
    Call<UserResponse> me();

    // ── Vector-only retrieval ────────────────────────────────────────────────

    /** FAISS retrieval using pre-computed hybrid vector. Returns basename + similarity only. */
    @POST("search_and_correct")
    Call<SearchAndCorrectResponse> searchAndCorrect(
        @Body  SearchAndCorrectRequest request,
        @Query("aesthetic_weight") float   aestheticWeight,
        @Query("include_images")   boolean includeImages
    );

    /** Style-constrained FAISS retrieval using pre-computed hybrid vector. */
    @POST("style/search")
    Call<StyleSearchResponse> styleSearch(@Body StyleSearchRequest request);

    /** Store pre-computed style vectors (no images). Returns session_id. */
    @POST("style/vectors")
    Call<StyleVectorsResponse> styleVectors(@Body StyleVectorsRequest request);

    // ── LUT download ─────────────────────────────────────────────────────────

    /** Download a 3D LUT for a given reference image basename. */
    @GET("lut/{basename}")
    Call<LutResponse> getLut(@Path("basename") String basename);

    // ── Cluster (already vector-only) ────────────────────────────────────────

    @POST("cluster")
    Call<ClusterResponse> cluster(@Body ClusterRequest request);

    // ── Blur sensitive regions ───────────────────────────────────────────────

    /** Send image to server, Gemini detects sensitive regions, OpenCV blurs them. Returns JPEG bytes. */
    @Multipart
    @POST("blur-sensitive")
    Call<ResponseBody> blurSensitive(@Part MultipartBody.Part file);

    // ── Mail ─────────────────────────────────────────────────────────────────

    /**
     * Send photos by email. The server reads SMTP credentials from its environment.
     * Photos are attached as JPEG files.
     *
     * Usage:
     *   MultipartBody.Part photo = MultipartBody.Part.createFormData(
     *       "files", "photo.jpg", RequestBody.create(bytes, MediaType.parse("image/jpeg")));
     */
    @Multipart
    @POST("mail/send")
    Call<MailSendResponse> sendPhotos(
        @Part("to_email")  RequestBody toEmail,
        @Part("subject")   RequestBody subject,
        @Part("message")   RequestBody message,
        @Part List<MultipartBody.Part> files
    );
}
