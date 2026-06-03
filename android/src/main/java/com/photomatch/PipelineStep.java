package com.photomatch;

import android.net.Uri;

import java.util.ArrayList;
import java.util.List;

public class PipelineStep {

    public enum Type {
        BLUR_FILTER, COLOR_CORRECT, BURST_DETECT, FACE_GROUP, STYLE_TRANSFER, BATCH_EDIT
    }

    public enum Status { IDLE, RUNNING, DONE, ERROR }

    public final Type   type;
    public       Status status       = Status.IDLE;
    public       String summary      = "";
    public       String resultDetail = "";          // secondary info line shown in detail panel

    // Populated after execution
    public final List<Uri>    resultUris   = new ArrayList<>();
    public final List<String> resultBase64 = new ArrayList<>();

    // Adjustable strength parameter (used by the settings slider)
    public float strengthParam;

    public PipelineStep(Type type) {
        this.type          = type;
        this.strengthParam = defaultStrength();
    }

    // ── Display strings ───────────────────────────────────────────────────────

    public String name() {
        switch (type) {
            case BLUR_FILTER:    return "Blur Filter";
            case COLOR_CORRECT:  return "Color Correct";
            case BURST_DETECT:   return "Burst Detect";
            case FACE_GROUP:     return "Face Group";
            case STYLE_TRANSFER: return "Style Transfer";
            default:             return "Batch Edit";
        }
    }

    public String subtitle() {
        switch (type) {
            case BLUR_FILTER:    return "removes out-of-focus photos";
            case COLOR_CORRECT:  return "applies AI color grading";
            case BURST_DETECT:   return "groups near-identical shots";
            case FACE_GROUP:     return "clusters photos by person";
            case STYLE_TRANSFER: return "applies your personal style";
            default:             return "corrects all remaining photos";
        }
    }

    // ── Slider configuration ─────────────────────────────────────────────────

    /** SeekBar max value (progress range is 0..sliderMax()). */
    public int sliderMax() {
        switch (type) {
            case BLUR_FILTER:    return 150;   // 50-200
            case BURST_DETECT:   return 19;    // 0.80-0.99
            case FACE_GROUP:     return 6;     // 0.60-0.90
            default:             return 9;     // 0.1-1.0
        }
    }

    public String sliderLabel() {
        switch (type) {
            case BLUR_FILTER:    return "Blur threshold  (Strict - Lenient)";
            case COLOR_CORRECT:  return "Correction strength  (Subtle - Strong)";
            case BURST_DETECT:   return "Similarity threshold  (Similar - Identical)";
            case FACE_GROUP:     return "Match strictness  (Loose - Strict)";
            case STYLE_TRANSFER: return "Style strength  (Subtle - Strong)";
            default:             return "Correction strength  (Subtle - Strong)";
        }
    }

    public String sliderMinLabel() {
        switch (type) {
            case BLUR_FILTER:  return "50";
            case BURST_DETECT: return "0.80";
            case FACE_GROUP:   return "0.60";
            default:           return "0.1";
        }
    }

    public String sliderMaxLabel() {
        switch (type) {
            case BLUR_FILTER:  return "200";
            case BURST_DETECT: return "0.99";
            case FACE_GROUP:   return "0.90";
            default:           return "1.0";
        }
    }

    /** Convert strengthParam → SeekBar progress integer. */
    public int sliderProgress() {
        switch (type) {
            case BLUR_FILTER:  return Math.round(strengthParam) - 50;
            case BURST_DETECT: return Math.round((strengthParam - 0.80f) * 100);
            case FACE_GROUP:   return Math.round((strengthParam - 0.60f) * 20);
            default:           return Math.round((strengthParam - 0.1f) * 10);
        }
    }

    /** Convert SeekBar progress integer → strengthParam. */
    public float progressToParam(int progress) {
        switch (type) {
            case BLUR_FILTER:  return progress + 50f;
            case BURST_DETECT: return 0.80f + progress * 0.01f;
            case FACE_GROUP:   return 0.60f + progress * 0.05f;
            default:           return 0.1f  + progress * 0.1f;
        }
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    private float defaultStrength() {
        switch (type) {
            case BLUR_FILTER:    return 100f;
            case BURST_DETECT:   return 0.92f;
            case FACE_GROUP:     return 0.75f;
            case STYLE_TRANSFER: return 0.5f;
            default:             return 0.3f;
        }
    }
}
