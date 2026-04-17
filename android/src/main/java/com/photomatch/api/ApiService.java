package com.photomatch.api;

import java.util.Map;

import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.GET;
import retrofit2.http.POST;
import retrofit2.http.Path;
import retrofit2.http.Query;

/**
 * V2 API — images NEVER leave the device.
 * All server calls use pre-computed 517-dim hybrid vectors or metadata only.
 */
public interface ApiService {

    @GET("health")
    Call<Map<String, Object>> health();

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
}
