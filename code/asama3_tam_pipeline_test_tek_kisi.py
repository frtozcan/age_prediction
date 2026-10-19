# -*- coding: utf-8 -*-
"""
AŞAMA 3 — Tam Pipeline Testi (UZMAN HACİM VERSİYONU)
======================================================
Yeni bir hasta için:
  1. Segmentasyon modeli  → sağ / sol / toplam sinüs hacmi tahmin eder
  2. ML yaş modeli        → bu hacimlerden yaş tahmin eder
  3. ML cinsiyet modeli   → bu hacimlerden cinsiyet tahmin eder

Bağlantı:
  Excel'deki uzman ölçümleri     → Aşama 2'de model eğitildi
  Segmentasyon modeli tahmini    → Aşama 3'te aynı özellikler üretilir
  Hasta eşleştirme               → Hasta_ADI sütunu ("37-14") ile dosya adı ("37-14.nii.gz")

FEATURE_COLS = ['maxilla_R', 'maxilla_L', 'maxilla_T']
  maxilla_R : Sağ sinüs hacmi  (cm³) — segmentasyon ile tahmin
  maxilla_L : Sol sinüs hacmi  (cm³) — segmentasyon ile tahmin
  maxilla_T : Toplam sinüs hacmi (cm³) = R + L
"""

import os
import re
import numpy as np
import nibabel as nib
import cv2
import tensorflow as tf
import tensorflow.keras.backend as K
import joblib
import pandas as pd

# ─── PARAMETRELER ─────────────────────────────────────────
SEG_MODEL_PATH = r"E:\3D_sinus\models\best_sinus_model.h5"
MODEL_YAS_PATH = r"E:\3D_sinus\ml_data\model_yas.joblib"
MODEL_CIN_PATH = r"E:\3D_sinus\ml_data\model_cinsiyet.joblib"
EXCEL_PATH     = r"E:\3D_sinus\UZMAN_hacim_bilgileri_sag_sol_toplam_yas.xlsx"

# Test hastası — değiştirin
TEST_IMG_PATH  = r"E:\3D_sinus\test_nifti\img\37-14.nii.gz"
TEST_MASK_PATH = r"E:\3D_sinus\test_nifti\mask\37-14.nii.gz"  # opsiyonel

OFFSET_H, OFFSET_W = 83, 93
TARGET_H, TARGET_W = 261, 324
MODEL_H, MODEL_W   = 256, 256
W_CENTER, W_WIDTH  = 400, 2000

# !! KRİTİK: Aşama 2 ile birebir aynı !!
FEATURE_COLS = ['maxilla_R', 'maxilla_L', 'maxilla_T']
# ──────────────────────────────────────────────────────────


def f1_score(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    inter = K.sum(y_true_f * y_pred_f)
    return (2. * inter + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())


def apply_windowing(image, wc, ww):
    lo = wc - ww / 2.0
    hi = wc + ww / 2.0
    return (((np.clip(image, lo, hi) - lo) / (hi - lo)) * 255.0).astype(np.uint8)


def calculate_metrics(y_true, y_pred):
    yt = y_true.flatten().astype(np.float32)
    yp = y_pred.flatten().astype(np.float32)
    inter = np.sum(yt * yp)
    dice  = (2. * inter) / (np.sum(yt) + np.sum(yp) + 1e-7)
    iou   = inter / (np.sum(yt) + np.sum(yp) - inter + 1e-7)
    prec  = inter / (np.sum(yp) + 1e-7)
    rec   = inter / (np.sum(yt) + 1e-7)
    return dice, iou, prec, rec


def parse_filename(filename):
    base = os.path.basename(filename).replace('.nii.gz', '').replace('.nii', '')
    m = re.match(r'^(\d+)-(\d+)$', base)
    if m:
        return int(m.group(1)), int(m.group(2))
    m2 = re.match(r'^(\d+)$', base)
    if m2:
        return int(m2.group(1)), None
    return None, None


def get_excel_info(excel_path, hasta_adi_key):
    """
    Excel'den bu hastanın uzman ölçümlerini ve gerçek etiketleri döner.
    hasta_adi_key: "37-14" gibi dosya adı kısmı (uzantısız, sayısal)
    """
    try:
        df = pd.read_excel(excel_path)
        df = df.rename(columns={
            'maxilla R': 'maxilla_R', 'maxilla L': 'maxilla_L',
            'maxilla T': 'maxilla_T', 'CİNSİYET': 'cinsiyet',
            'YAS': 'yas', 'Hasta_ADI': 'hasta_adi',
        })
        # Hasta_ADI sütunundaki değer "37-14" gibi — dosya adıyla eşleştir
        row = df[df['hasta_adi'].astype(str).str.strip() == hasta_adi_key]
        if len(row) == 0:
            return None
        r = row.iloc[0]
        return {
            'uzman_R'   : float(r['maxilla_R']),
            'uzman_L'   : float(r['maxilla_L']),
            'uzman_T'   : float(r['maxilla_T']),
            'gercek_yas': int(r['yas']),
            'cinsiyet'  : str(r['cinsiyet']).strip(),
        }
    except Exception as e:
        print(f"[UYARI] Excel okunamadi: {e}")
        return None


def segment_sag_sol(seg_model, img_path, mask_path=None):
    """
    Segmentasyon yapar.
    Maske varsa sağ/sol ayrımı için maskeyi orta eksenden böler.
    Döner: {'pred_R_cm3', 'pred_L_cm3', 'pred_T_cm3', + metrikler}
    """
    img_can  = nib.as_closest_canonical(nib.load(img_path))
    img_data = img_can.get_fdata()
    vx, vy, vz   = img_can.header.get_zooms()[:3]
    unit_vol_mm3 = vx * vy * vz

    mask_data = None
    if mask_path and os.path.exists(mask_path):
        mask_data = nib.as_closest_canonical(nib.load(mask_path)).get_fdata()

    pred_list, true_list = [], []

    for i in range(img_data.shape[2]):
        if mask_data is not None:
            raw_mask = (mask_data[:, :, i] > 0).astype(np.uint8)
            if not np.any(raw_mask):
                continue
        else:
            raw_mask = None

        aligned  = np.rot90(img_data[:, :, i], k=1)
        windowed = apply_windowing(aligned, W_CENTER, W_WIDTH)
        cropped  = windowed[OFFSET_H:OFFSET_H+TARGET_H, OFFSET_W:OFFSET_W+TARGET_W]
        inp = cv2.resize(cropped, (MODEL_W, MODEL_H), interpolation=cv2.INTER_AREA)
        inp = np.expand_dims(inp.astype(np.float32) / 255.0, axis=(0, -1))

        pred256 = (seg_model.predict(inp, verbose=0)[0, :, :, 0] > 0.5).astype(np.uint8)

        if raw_mask is None and not np.any(pred256):
            continue

        canvas = np.zeros((512, 512), dtype=np.uint8)
        canvas[OFFSET_H:OFFSET_H+TARGET_H, OFFSET_W:OFFSET_W+TARGET_W] = \
            cv2.resize(pred256, (TARGET_W, TARGET_H), interpolation=cv2.INTER_NEAREST)

        pred_list.append(canvas)
        if raw_mask is not None:
            true_list.append(np.rot90(raw_mask, k=1))

    if not pred_list:
        return None

    pred_vol = np.stack(pred_list, axis=-1)   # (512, 512, n_slices)

    # Sağ / Sol ayrımı: görüntünün orta sütunundan böl
    # rot90 sonrası: sol anatomik = sağ görüntü yarısı (sütun > 256)
    # Not: Bu yaklaşım sol/sağı yaklaşık olarak ayırır.
    mid = pred_vol.shape[1] // 2
    pred_R = pred_vol[:, mid:, :]   # Görüntüde sağ yarı → anatomik SOL
    pred_L = pred_vol[:, :mid, :]   # Görüntüde sol yarı → anatomik SAĞ

    pred_R_cm3 = np.sum(pred_R) * unit_vol_mm3 / 1000.0
    pred_L_cm3 = np.sum(pred_L) * unit_vol_mm3 / 1000.0
    pred_T_cm3 = np.sum(pred_vol) * unit_vol_mm3 / 1000.0

    result = {
        'pred_R_cm3'   : round(pred_R_cm3, 4),
        'pred_L_cm3'   : round(pred_L_cm3, 4),
        'pred_T_cm3'   : round(pred_T_cm3, 4),
        'n_aktif_kesit': len(pred_list),
        'slice_thick_mm': round(float(vz), 4),
    }

    if true_list:
        true_vol = np.stack(true_list, axis=-1)
        true_T_cm3 = np.sum(true_vol) * unit_vol_mm3 / 1000.0
        dice, iou, prec, rec = calculate_metrics(true_vol, pred_vol)
        result.update({
            'true_T_cm3'    : round(true_T_cm3, 4),
            'hacim_hata_pct': round(abs(pred_T_cm3 - true_T_cm3) / (true_T_cm3 + 1e-7) * 100, 2),
            'dice'          : round(dice, 4),
            'iou'           : round(iou,  4),
            'precision'     : round(prec, 4),
            'recall'        : round(rec,  4),
        })

    return result


def predict_age_gender(seg_result, yas_model, cin_model):
    """
    Segmentasyondan elde edilen R/L/T hacimleri ile yaş ve cinsiyet tahmin eder.
    FEATURE_COLS = ['maxilla_R', 'maxilla_L', 'maxilla_T']
    """
    X = np.array([[
        seg_result['pred_R_cm3'],   # maxilla_R
        seg_result['pred_L_cm3'],   # maxilla_L
        seg_result['pred_T_cm3'],   # maxilla_T
    ]])

    yas_tahmin = float(yas_model.predict(X)[0])
    cin_enc    = int(cin_model.predict(X)[0])
    cin_prob   = cin_model.predict_proba(X)[0]
    cinsiyet   = "Erkek" if cin_enc == 1 else "Kadin"
    cin_guven  = max(cin_prob) * 100

    return {
        'yas_tahmin'   : round(yas_tahmin, 1),
        'cinsiyet'     : cinsiyet,
        'cin_guven_pct': round(cin_guven, 1),
    }


def main():
    # ── Model yükle ───────────────────────────────────────
    print("=" * 62)
    print("MODEL YUKLEME")
    print("=" * 62)
    seg_model = tf.keras.models.load_model(
        SEG_MODEL_PATH, custom_objects={'f1_score': f1_score}
    )
    print("  + Segmentasyon modeli hazir")

    yas_model = joblib.load(MODEL_YAS_PATH)
    cin_model = joblib.load(MODEL_CIN_PATH)
    print(f"  + Yas modeli    — {yas_model.n_features_in_} ozellik ({FEATURE_COLS})")
    print(f"  + Cinsiyet mod. — {cin_model.n_features_in_} ozellik ({FEATURE_COLS})")

    assert yas_model.n_features_in_ == len(FEATURE_COLS), \
        f"Yas modeli {yas_model.n_features_in_} ozellik bekliyor ama {len(FEATURE_COLS)} tanimlandi!"
    assert cin_model.n_features_in_ == len(FEATURE_COLS), \
        f"Cinsiyet modeli {cin_model.n_features_in_} ozellik bekliyor ama {len(FEATURE_COLS)} tanimlandi!"

    # ── Hasta bilgisi ─────────────────────────────────────
    fname         = os.path.basename(TEST_IMG_PATH)
    hasta_adi_key = fname.replace('.nii.gz', '').replace('.nii', '')
    hasta_no, yas_dosya = parse_filename(fname)
    mask_mevcut   = os.path.exists(TEST_MASK_PATH)

    print("\n" + "=" * 62)
    print("HASTA BILGISI")
    print("=" * 62)
    print(f"  Dosya          : {fname}")
    print(f"  Hasta No       : {hasta_no}")
    print(f"  Yas (dosyadan) : {yas_dosya}")
    print(f"  Maske          : {'Mevcut' if mask_mevcut else 'Yok'}")

    # Excel'den uzman ölçümleri al
    excel_info = get_excel_info(EXCEL_PATH, hasta_adi_key)
    if excel_info:
        print(f"\n  Uzman Olcumleri (referans):")
        print(f"    maxilla R  : {excel_info['uzman_R']:.4f} cm3")
        print(f"    maxilla L  : {excel_info['uzman_L']:.4f} cm3")
        print(f"    maxilla T  : {excel_info['uzman_T']:.4f} cm3")
        print(f"    Gercek Yas : {excel_info['gercek_yas']}")
        print(f"    Cinsiyet   : {excel_info['cinsiyet']}")
    else:
        print(f"  [UYARI] Excel'de '{hasta_adi_key}' bulunamadi.")

    # ── Segmentasyon ──────────────────────────────────────
    print("\n[1] Segmentasyon yapiliyor...")
    seg_result = segment_sag_sol(
        seg_model, TEST_IMG_PATH,
        TEST_MASK_PATH if mask_mevcut else None
    )

    if seg_result is None:
        print("[HATA] Sinus bolgesi bulunamadi!")
        return

    print(f"\n  Tahmin Hacimler (segmentasyondan):")
    print(f"    maxilla R (pred) : {seg_result['pred_R_cm3']:.4f} cm3")
    print(f"    maxilla L (pred) : {seg_result['pred_L_cm3']:.4f} cm3")
    print(f"    maxilla T (pred) : {seg_result['pred_T_cm3']:.4f} cm3")
    print(f"    Aktif Kesit      : {seg_result['n_aktif_kesit']}")

    if 'true_T_cm3' in seg_result:
        print(f"\n  Segmentasyon Metrikleri:")
        print(f"    GT Hacim  : {seg_result['true_T_cm3']:.4f} cm3")
        print(f"    Hata      : %{seg_result['hacim_hata_pct']:.2f}")
        print(f"    Dice      : {seg_result['dice']:.4f}")
        print(f"    IoU       : {seg_result['iou']:.4f}")
        print(f"    Precision : {seg_result['precision']:.4f}")
        print(f"    Recall    : {seg_result['recall']:.4f}")

    # Uzman vs Segmentasyon karşılaştırması
    if excel_info:
        print(f"\n  Uzman vs Segmentasyon Karsilastirma:")
        for key, uzman_val, pred_val in [
            ('R', excel_info['uzman_R'], seg_result['pred_R_cm3']),
            ('L', excel_info['uzman_L'], seg_result['pred_L_cm3']),
            ('T', excel_info['uzman_T'], seg_result['pred_T_cm3']),
        ]:
            fark = abs(uzman_val - pred_val)
            pct  = fark / uzman_val * 100
            print(f"    maxilla {key}: Uzman={uzman_val:.3f}  Pred={pred_val:.3f}"
                  f"  Fark={fark:.3f} cm3 (%{pct:.1f})")

    # ── ML tahmin ─────────────────────────────────────────
    print("\n[2] Yas ve cinsiyet tahmin ediliyor...")
    print(f"  Kullanilan ozellikler: {FEATURE_COLS}")
    print(f"  Degerler: R={seg_result['pred_R_cm3']:.3f}  "
          f"L={seg_result['pred_L_cm3']:.3f}  T={seg_result['pred_T_cm3']:.3f}")

    ml_result = predict_age_gender(seg_result, yas_model, cin_model)

    # ── Final Rapor ───────────────────────────────────────
    print("\n" + "=" * 62)
    print("TAHMIN SONUCLARI")
    print("=" * 62)
    print(f"  Sag Sinus Hacmi  : {seg_result['pred_R_cm3']:.4f} cm3", end="")
    if excel_info:
        print(f"   (Uzman: {excel_info['uzman_R']:.4f})", end="")
    print()
    print(f"  Sol Sinus Hacmi  : {seg_result['pred_L_cm3']:.4f} cm3", end="")
    if excel_info:
        print(f"   (Uzman: {excel_info['uzman_L']:.4f})", end="")
    print()
    print(f"  Toplam Hacim     : {seg_result['pred_T_cm3']:.4f} cm3", end="")
    if excel_info:
        print(f"   (Uzman: {excel_info['uzman_T']:.4f})", end="")
    print()
    print(f"  Tahmini Yas      : {ml_result['yas_tahmin']:.1f} yil", end="")
    if excel_info:
        hata = abs(ml_result['yas_tahmin'] - excel_info['gercek_yas'])
        print(f"   (Gercek: {excel_info['gercek_yas']}  |  Hata: {hata:.1f} yil)", end="")
    elif yas_dosya:
        hata = abs(ml_result['yas_tahmin'] - yas_dosya)
        print(f"   (Dosyadan: {yas_dosya}  |  Hata: {hata:.1f} yil)", end="")
    print()
    print(f"  Tahmini Cinsiyet : {ml_result['cinsiyet']}"
          f"  (Guven: %{ml_result['cin_guven_pct']:.1f})", end="")
    if excel_info:
        eslesme = "DOGRU" if (
            (ml_result['cinsiyet'] == 'Erkek' and excel_info['cinsiyet'].lower() == 'erkek') or
            (ml_result['cinsiyet'] == 'Kadin' and excel_info['cinsiyet'].lower() == 'kadin')
        ) else "YANLIS"
        print(f"   (Gercek: {excel_info['cinsiyet']}  |  {eslesme})", end="")
    print()
    print("=" * 62)


if __name__ == "__main__":
    main()