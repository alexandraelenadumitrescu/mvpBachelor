package com.photomatch.lite.ui;

import android.net.Uri;
import android.view.LayoutInflater;
import android.view.View;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import com.photomatch.lite.api.ApiClient;
import com.photomatch.lite.api.DeliveryDetail;
import com.photomatch.lite.api.DeliveryRequest;
import com.photomatch.lite.api.DeliveryResponse;
import com.photomatch.lite.base.BaseApiActivity;
import com.photomatch.lite.databinding.ActivityDeliveryBinding;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.RequestBody;
import retrofit2.Call;

public class DeliveryActivity extends BaseApiActivity<DeliveryRequest, DeliveryResponse> {

    private ActivityDeliveryBinding binding;
    private List<Uri> selectedUris = new ArrayList<>();

    private final ActivityResultLauncher<String> pickPhotos =
        registerForActivityResult(new ActivityResultContracts.GetMultipleContents(), uris -> {
            selectedUris = new ArrayList<>(uris);
            binding.tvCount.setText(uris.size() + " fotografii selectate");
        });

    @Override
    protected View onCreateLayout(LayoutInflater inflater) {
        binding = ActivityDeliveryBinding.inflate(inflater);
        return binding.getRoot();
    }

    @Override
    protected void onBindViews() {
        binding.btnPick.setOnClickListener(v -> pickPhotos.launch("image/*"));
        binding.btnSend.setOnClickListener(v -> {
            if (binding.etUrl.getText().toString().trim().isEmpty()) {
                showError("Introduceți URL-ul angajaților");
            } else if (selectedUris.isEmpty()) {
                showError("Selectează fotografii mai întâi");
            } else {
                submit();
            }
        });
    }

    @Override
    protected DeliveryRequest buildRequest() {
        return new DeliveryRequest(
            binding.etUrl.getText().toString().trim(),
            selectedUris
        );
    }

    @Override
    protected Call<DeliveryResponse> callApi(DeliveryRequest request) {
        RequestBody urlPart = RequestBody.create(
            request.employeesUrl, MediaType.parse("text/plain")
        );
        List<MultipartBody.Part> parts = new ArrayList<>();
        for (Uri uri : request.photos) {
            try {
                InputStream is    = getContentResolver().openInputStream(uri);
                byte[]      bytes = is.readAllBytes();
                is.close();
                RequestBody body = RequestBody.create(bytes, MediaType.parse("image/jpeg"));
                parts.add(MultipartBody.Part.createFormData("photos", "photo.jpg", body));
            } catch (IOException e) {
                showError("Eroare la citirea fotografiei");
            }
        }
        return ApiClient.service().deliveryRun(urlPart, parts);
    }

    @Override
    protected void onSuccess(DeliveryResponse response) {
        StringBuilder sb = new StringBuilder();
        sb.append("Potrivite: ").append(response.matched)
          .append(" | Trimise: ").append(response.emails_sent)
          .append(" | Eșuate: ").append(response.failed);
        if (response.details != null) {
            for (DeliveryDetail d : response.details) {
                sb.append("\n→ ").append(d.email)
                  .append(" (").append(d.photos_count).append(" poze)");
            }
        }
        binding.tvStatus.setText(sb.toString());
    }
}
