# -*- coding: utf-8 -*-


import os
import re
import numpy as np
import nibabel as nib
import cv2
import tensorflow as tf
import tensorflow.keras.backend as K
from glob import glob
from docx import Document as DocxDocument
import pandas as pd

# ─── PARAMETRELER ─────────────────────────────────────────
MODEL_PATH    = r"E:\3D_sinus\models\best_sinus_model.h5"
IMG_DIR       = r"E:\3D_sinus\img"           # Tüm hastalar (eğitim + test)
MASK_DIR      = r"E:\3D_sinus\mask"
DOCX_PATH     = r"E:\3D_sinus\ct yas_cinsiyet.docx"
OUTPUT_CSV    = r"E:\3D_sinus\ml_data\egitim_verisi.csv"

OFFSET_H, OFFSET_W = 83, 93
TARGET_H, TARGET_W = 261, 324
MODEL_H, MODEL_W   = 256, 256
W_CENTER, W_WIDTH  = 400, 2000
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


def parse_filename(filename):
    base = os.path.basename(filename).replace('.nii.gz', '').replace('.nii', '')
    m = re.match(r'^(\d+)-(\d+)$', base)
    if m:
        return int(m.group(1)), int(m.group(2))
    m2 = re.match(r'^(\d+)$', base)
    if m2:
        return int(m2.group(1)), None
    return None, None


def read_docx(docx_path):
    data = {}
    doc = DocxDocument(docx_path)
    for table in doc.tables:
        for i, row in enumerate(table.rows):
            cells = [c.text.strip() for c in row.cells]
            if i == 0 and not cells[0].isdigit():
                continue
            try:
                no  = int(cells[0])
                ct  = float(re.sub(r'[^\d.]', '', cells[1])) if len(cells) > 1 else None
                yas = int(cells[2])                           if len(cells) > 2 else None
                cin = cells[3].lower().strip()                if len(cells) > 3 else 'bilinmiyor'
                data[no] = {'ct_mm': ct, 'yas': yas, 'cinsiyet': cin}
            except (ValueError, IndexError):
                continue
    print(f"[BİLGİ] DOCX'ten {len(data)} hasta okundu.")
    return data


def compute_volume(model, img_path, mask_path):
    """Bir hasta için segmentasyon yapar, hacmi (cm³) döner."""
    img_nii  = nib.load(img_path)
    mask_nii = nib.load(mask_path)
    img_can  = nib.as_closest_canonical(img_nii)
    mask_can = nib.as_closest_canonical(mask_nii)
    img_data  = img_can.get_fdata()
    mask_data = mask_can.get_fdata()

    vx, vy, vz  = img_can.header.get_zooms()[:3]
    unit_vol_mm3 = vx * vy * vz

    pred_list, true_list = [], []

    for i in range(img_data.shape[2]):
        raw_mask = (mask_data[:, :, i] > 0).astype(np.uint8)
        if not np.any(raw_mask):
            continue

        aligned  = np.rot90(img_data[:, :, i], k=1)
        windowed = apply_windowing(aligned, W_CENTER, W_WIDTH)
        cropped  = windowed[OFFSET_H:OFFSET_H+TARGET_H, OFFSET_W:OFFSET_W+TARGET_W]
        inp = cv2.resize(cropped, (MODEL_W, MODEL_H), interpolation=cv2.INTER_AREA)
        inp = np.expand_dims(inp.astype(np.float32) / 255.0, axis=(0, -1))

        pred256 = (model.predict(inp, verbose=0)[0, :, :, 0] > 0.5).astype(np.uint8)
        canvas  = np.zeros((512, 512), dtype=np.uint8)
        canvas[OFFSET_H:OFFSET_H+TARGET_H, OFFSET_W:OFFSET_W+TARGET_W] = \
            cv2.resize(pred256, (TARGET_W, TARGET_H), interpolation=cv2.INTER_NEAREST)

        pred_list.append(canvas)
        true_list.append(np.rot90(raw_mask, k=1))

    if not pred_list:
        return None

    pred_vol = np.stack(pred_list, axis=-1)
    true_vol = np.stack(true_list, axis=-1)

    pred_cm3 = np.sum(pred_vol) * unit_vol_mm3 / 1000.0
    true_cm3 = np.sum(true_vol) * unit_vol_mm3 / 1000.0

    return {
        'pred_vol_cm3'   : round(pred_cm3, 4),
        'gercek_vol_cm3' : round(true_cm3, 4),
        'slice_thick_mm' : round(vz, 4),
        'n_aktif_kesit'  : len(pred_list),
    }


def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    print("Segmentasyon modeli yükleniyor...")
    model = tf.keras.models.load_model(
        MODEL_PATH, custom_objects={'f1_score': f1_score}
    )
    print("Model yüklendi.\n")

    docx_data = read_docx(DOCX_PATH)
    img_paths = sorted(glob(os.path.join(IMG_DIR, "*.nii*")))
    print(f"{len(img_paths)} hasta işlenecek.\n")

    rows = []
    for idx, img_path in enumerate(img_paths, 1):
        fname     = os.path.basename(img_path)
        mask_path = os.path.join(MASK_DIR, fname)
        if not os.path.exists(mask_path):
            print(f"[{idx:03d}] {fname}  → Maske yok, atlanıyor.")
            continue

        hasta_no, _ = parse_filename(fname)
        info        = docx_data.get(hasta_no, {})
        yas         = info.get('yas', None)
        cinsiyet    = info.get('cinsiyet', None)
        ct_mm       = info.get('ct_mm', None)

        if yas is None or cinsiyet is None:
            print(f"[{idx:03d}] {fname}  → DOCX'te kayıt yok, atlanıyor.")
            continue

        print(f"[{idx:03d}/{len(img_paths)}] {fname}  Yaş:{yas}  Cinsiyet:{cinsiyet}", end="  →  ")

        result = compute_volume(model, img_path, mask_path)
        if result is None:
            print("Aktif maske yok, atlanıyor.")
            continue

        rows.append({
            'hasta_no'       : hasta_no,
            'dosya_adi'      : fname,
            'yas'            : yas,
            'cinsiyet'       : cinsiyet,           # 'erkek' / 'kadın'
            'cinsiyet_enc'   : 1 if cinsiyet == 'erkek' else 0,  # 1=erkek, 0=kadın
            'ct_kalinlik_mm' : ct_mm,
            'pred_vol_cm3'   : result['pred_vol_cm3'],
            'gercek_vol_cm3' : result['gercek_vol_cm3'],
            'slice_thick_mm' : result['slice_thick_mm'],
            'n_aktif_kesit'  : result['n_aktif_kesit'],
        })
        print(f"Pred Hacim: {result['pred_vol_cm3']:.3f} cm³  |  GT: {result['gercek_vol_cm3']:.3f} cm³")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    print(f"\n{'='*55}")
    print(f"Toplam {len(df)} hasta kaydedildi → {OUTPUT_CSV}")
    print(f"Yaş aralığı   : {int(df['yas'].min())} – {int(df['yas'].max())}")
    print(f"Erkek / Kadın : {(df['cinsiyet']=='erkek').sum()} / {(df['cinsiyet']=='kadın').sum()}")
    print(f"Ort. Pred Hacim : {df['pred_vol_cm3'].mean():.3f} ± {df['pred_vol_cm3'].std():.3f} cm³")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
