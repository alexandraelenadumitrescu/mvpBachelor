package com.photomatch.lite.ui;

import android.content.Intent;
import android.os.Bundle;
import androidx.appcompat.app.AppCompatActivity;
import com.photomatch.lite.databinding.ActivityMainBinding;

public class MainActivity extends AppCompatActivity {

    private ActivityMainBinding binding;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        binding.btnLogin.setOnClickListener(v ->
            startActivity(new Intent(this, LoginActivity.class)));
        binding.btnBlur.setOnClickListener(v ->
            startActivity(new Intent(this, BlurActivity.class)));
        binding.btnDelivery.setOnClickListener(v ->
            startActivity(new Intent(this, DeliveryActivity.class)));
    }
}
