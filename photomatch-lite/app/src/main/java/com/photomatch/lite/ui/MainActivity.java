package com.photomatch.lite.ui;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import androidx.appcompat.app.AppCompatActivity;
import com.photomatch.lite.api.ApiClient;
import com.photomatch.lite.databinding.ActivityMainBinding;

public class MainActivity extends AppCompatActivity {

    private ActivityMainBinding binding;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        if (!ApiClient.isLoggedIn()) {
            startActivity(new Intent(this, LoginActivity.class));
        }

        binding.btnLogin.setOnClickListener(v ->
            startActivity(new Intent(this, LoginActivity.class)));
        binding.btnLogout.setOnClickListener(v -> {
            ApiClient.clearToken();
            updateAuthButtons();
        });
        binding.btnBlur.setOnClickListener(v ->
            startActivity(new Intent(this, BlurActivity.class)));
        binding.btnDelivery.setOnClickListener(v ->
            startActivity(new Intent(this, DeliveryActivity.class)));
        binding.btnCluster.setOnClickListener(v ->
            startActivity(new Intent(this, ClusterActivity.class)));
    }

    @Override
    protected void onResume() {
        super.onResume();
        updateAuthButtons();
    }

    private void updateAuthButtons() {
        boolean loggedIn = ApiClient.isLoggedIn();
        binding.btnLogin.setVisibility(loggedIn ? View.GONE : View.VISIBLE);
        binding.btnLogout.setVisibility(loggedIn ? View.VISIBLE : View.GONE);
    }
}
