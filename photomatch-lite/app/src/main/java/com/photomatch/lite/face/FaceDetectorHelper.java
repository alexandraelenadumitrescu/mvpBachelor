package com.photomatch.lite.face;

import android.graphics.Bitmap;
import android.graphics.Rect;
import com.google.android.gms.tasks.Tasks;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;
import java.util.ArrayList;
import java.util.List;

public class FaceDetectorHelper {

    private final FaceDetector client;

    public FaceDetectorHelper() {
        FaceDetectorOptions options = new FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .setMinFaceSize(0.1f)
            .build();
        client = FaceDetection.getClient(options);
    }

    public List<Rect> detect(Bitmap bmp) throws Exception {
        InputImage image = InputImage.fromBitmap(bmp, 0);
        List<Face> faces = Tasks.await(client.process(image));
        List<Rect> boxes = new ArrayList<>(faces.size());
        for (Face face : faces) boxes.add(face.getBoundingBox());
        return boxes;
    }

    public void close() {
        client.close();
    }
}
