package com.photomatch.lite.ui;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.view.LayoutInflater;
import android.view.View;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import com.photomatch.lite.api.ApiClient;
import com.photomatch.lite.base.BaseApiActivity;
import com.photomatch.lite.databinding.ActivityBlurBinding;
import java.io.IOException;
import java.io.InputStream;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.RequestBody;
import okhttp3.ResponseBody;
import retrofit2.Call;

public class BlurActivity extends BaseApiActivity<Uri, ResponseBody> {

    private ActivityBlurBinding binding;
    private Uri selectedUri;

    private final ActivityResultLauncher<String> pickImage =
        registerForActivityResult(new ActivityResultContracts.GetContent(), uri -> {
            if (uri != null) {
                selectedUri = uri;
                binding.imgOriginal.setImageURI(uri);
            }
        });

    @Override
    protected View onCreateLayout(LayoutInflater inflater) {
        binding = ActivityBlurBinding.inflate(inflater);
        return binding.getRoot();
    }

    @Override
    protected void onBindViews() {
        binding.btnPick.setOnClickListener(v -> pickImage.launch("image/*"));
        binding.btnBlur.setOnClickListener(v -> {
            if (selectedUri != null) submit();
            else showError("Selectează o fotografie mai întâi");
        });
    }

    @Override
    protected Uri buildRequest() {
        return selectedUri;
    }

    @Override
    protected Call<ResponseBody> callApi(Uri uri) {
        try {
            InputStream is = getContentResolver().openInputStream(uri);
            byte[] bytes = is.readAllBytes();
            is.close();
            RequestBody body = RequestBody.create(bytes, MediaType.parse("image/jpeg"));
            MultipartBody.Part part = MultipartBody.Part.createFormData("file", "photo.jpg", body);
            return ApiClient.service().blurSensitive(part);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    protected void onSuccess(ResponseBody response) {
        try {
            byte[] bytes = response.bytes();
            Bitmap blurred = BitmapFactory.decodeByteArray(bytes, 0, bytes.length);
            binding.imgBlurred.setImageBitmap(blurred);
        } catch (IOException e) {
            showError("Nu s-a putut decoda imaginea");
        }
    }
}
