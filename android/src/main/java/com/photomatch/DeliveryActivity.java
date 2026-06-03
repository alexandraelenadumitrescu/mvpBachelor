package com.photomatch;

import android.net.Uri;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ProgressBar;
import android.widget.Switch;
import android.widget.TextView;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;

import com.photomatch.api.DeliveryResponse;
import com.photomatch.base.BaseServerActivity;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.RequestBody;
import retrofit2.Call;
import retrofit2.Response;

public class DeliveryActivity extends BaseServerActivity {

    private static final int    MAX_PHOTOS = 100;
    private static final MediaType JPEG    = MediaType.parse("image/jpeg");

    private List<Uri>   selectedUris = new ArrayList<>();
    private EditText    etEmployeesUrl;
    private Switch      switchMode;
    private TextView    tvModeV1;
    private TextView    tvModeV2;
    private Button      btnPick;
    private Button      btnSend;
    private TextView    tvCount;
    private TextView    tvStatus;
    private TextView    tvError;
    private ProgressBar progressBar;

    private final ActivityResultLauncher<String> pickPhotos =
        registerForActivityResult(new ActivityResultContracts.GetMultipleContents(), uris -> {
            if (uris != null && !uris.isEmpty()) {
                selectedUris = new ArrayList<>(uris.subList(0, Math.min(uris.size(), MAX_PHOTOS)));
                int n = selectedUris.size();
                String suffix = uris.size() > MAX_PHOTOS ? " (limitat la " + MAX_PHOTOS + ")" : "";
                tvCount.setText(n + " fotografi" + (n == 1 ? "e" : "i") + " selectat" + (n == 1 ? "a" : "e") + suffix);
                tvStatus.setVisibility(View.GONE);
                tvError.setVisibility(View.GONE);
            }
        });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_delivery);

        etEmployeesUrl = findViewById(R.id.etEmployeesUrl);
        switchMode     = findViewById(R.id.switchMode);
        tvModeV1       = findViewById(R.id.tvModeV1);
        tvModeV2       = findViewById(R.id.tvModeV2);
        btnPick        = primaryButton = findViewById(R.id.btnPick);
        btnSend        = findViewById(R.id.btnSend);
        tvCount        = findViewById(R.id.tvCount);
        tvStatus       = findViewById(R.id.tvStatus);
        tvError        = findViewById(R.id.tvError);
        progressBar    = findViewById(R.id.progressBar);

        switchMode.setOnCheckedChangeListener((btn, checked) -> updateModeLabels(checked));
        updateModeLabels(false);

        btnPick.setOnClickListener(v -> pickPhotos.launch("image/*"));
        btnSend.setOnClickListener(v -> {
            String url = etEmployeesUrl.getText().toString().trim();
            if (url.isEmpty()) { showError("Introdu URL-ul paginii cu angajati"); return; }
            if (selectedUris.isEmpty()) { showError("Selecteaza fotografii mai intai"); return; }
            startDelivery(url, switchMode.isChecked());
        });
    }

    private void updateModeLabels(boolean v2) {
        int amber = getResources().getColor(R.color.color_amber, null);
        int dim   = getResources().getColor(R.color.color_text_dim, null);
        tvModeV1.setTextColor(v2 ? dim   : amber);
        tvModeV2.setTextColor(v2 ? amber : dim);
    }

    private void startDelivery(String employeesUrl, boolean useV2) {
        setProcessing(true);
        btnSend.setEnabled(false);
        tvStatus.setVisibility(View.GONE);
        tvError.setVisibility(View.GONE);

        executor.execute(() -> {
            try {
                RequestBody urlBody = RequestBody.create(employeesUrl, MediaType.parse("text/plain"));
                List<MultipartBody.Part> parts = new ArrayList<>();
                for (Uri uri : selectedUris) {
                    RequestBody body = new RequestBody() {
                        @Override public MediaType contentType() { return JPEG; }
                        @Override public void writeTo(okio.BufferedSink sink) throws IOException {
                            try (InputStream is = getContentResolver().openInputStream(uri)) {
                                if (is == null) throw new IOException("Cannot open: " + uri);
                                sink.writeAll(okio.Okio.source(is));
                            }
                        }
                    };
                    parts.add(MultipartBody.Part.createFormData("photos", "photo.jpg", body));
                }

                Call<DeliveryResponse> call = useV2
                    ? api().deliveryRunV2(urlBody, parts)
                    : api().deliveryRun(urlBody, parts);

                Response<DeliveryResponse> resp = call.execute();

                runOnUiThread(() -> {
                    setProcessing(false);
                    btnSend.setEnabled(true);
                    if (resp.isSuccessful() && resp.body() != null) {
                        DeliveryResponse r = resp.body();
                        String status;
                        if (useV2) {
                            status = "Clustere: " + r.clustersFound
                                   + "  |  Potrivite: " + r.matched
                                   + "  |  Trimise: " + r.sent
                                   + "  |  Esecuri: " + r.failed;
                        } else {
                            status = "Potrivite: " + r.matched
                                   + "  |  Trimise: " + r.sent
                                   + "  |  Esecuri: " + r.failed;
                        }
                        tvStatus.setText(status);
                        tvStatus.setVisibility(View.VISIBLE);
                    } else {
                        showError("Eroare server: HTTP " + resp.code());
                    }
                });

            } catch (IOException e) {
                runOnUiThread(() -> {
                    setProcessing(false);
                    btnSend.setEnabled(true);
                    showError("Eroare: " + e.getMessage());
                });
            }
        });
    }
}
