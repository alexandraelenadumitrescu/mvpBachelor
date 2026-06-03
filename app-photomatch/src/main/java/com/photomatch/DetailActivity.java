package com.photomatch;

import android.graphics.Rect;
import android.os.Bundle;
import android.view.MotionEvent;
import android.view.ViewTreeObserver;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import com.bumptech.glide.Glide;
import com.photomatch.api.ProcessResponse;

import java.util.Locale;

public class DetailActivity extends AppCompatActivity {

    private FrameLayout flCompare;
    private ImageView   ivBefore;
    private android.view.View vDivider;

    private float dividerFraction = 0.5f;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_detail);

        ProcessResponse resp = ResponseCache.current;
        if (resp == null) { finish(); return; }

        flCompare = findViewById(R.id.flCompare);
        ivBefore  = findViewById(R.id.ivBefore);
        vDivider  = findViewById(R.id.vDivider);
        ImageView ivAfter     = findViewById(R.id.ivAfter);
        TextView  tvMatchInfo = findViewById(R.id.tvMatchInfo);

        // Load images via Glide so bitmap lifecycle is managed
        Glide.with(this).load(decodeBase64(resp.originalB64)).fitCenter().into(ivBefore);
        Glide.with(this).load(decodeBase64(resp.finalB64)).fitCenter().into(ivAfter);

        int similarityPct = Math.round(resp.similarity * 100);
        tvMatchInfo.setText(String.format(Locale.US,
            "MATCHED  %s  ·  %d%% SIMILAR", resp.retrieved, similarityPct));

        flCompare.setOnTouchListener((v, event) -> {
            int action = event.getAction();
            if (action == MotionEvent.ACTION_DOWN || action == MotionEvent.ACTION_MOVE) {
                int width = flCompare.getWidth();
                if (width > 0) {
                    dividerFraction = Math.max(0.05f, Math.min(0.95f, event.getX() / width));
                    updateDivider();
                }
                return true;
            }
            return false;
        });

        flCompare.getViewTreeObserver().addOnGlobalLayoutListener(
            new ViewTreeObserver.OnGlobalLayoutListener() {
                @Override
                public void onGlobalLayout() {
                    flCompare.getViewTreeObserver().removeOnGlobalLayoutListener(this);
                    updateDivider();
                }
            });
    }

    private void updateDivider() {
        int width  = flCompare.getWidth();
        int height = flCompare.getHeight();
        if (width == 0 || height == 0) return;

        int divX = (int) (dividerFraction * width);
        vDivider.setTranslationX(divX - vDivider.getWidth() / 2f);
        ivBefore.setClipBounds(new Rect(0, 0, divX, height));
        ivBefore.invalidate();
    }

    private static byte[] decodeBase64(String b64) {
        if (b64 == null) return null;
        return android.util.Base64.decode(b64, android.util.Base64.DEFAULT);
    }
}
