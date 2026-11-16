# -*- coding: utf-8 -*-

import os, glob
import numpy as np, pandas as pd, nibabel as nib, cv2
from scipy import ndimage, stats
import tensorflow as tf
import tensorflow.keras.backend as K

# ─── YOLLAR ───────────────────────────────────────────────
SEG_MODEL_PATH = r"E:\3D_sinus\models\best_sinus_model.h5"
EXCEL_REF_PATH = r"E:\3D_sinus\github_upload\results\segmentasyon_raporu_uzman.xlsx"   # 30 hasta
TEST_IMG_DIR   = r"E:\3D_sinus\test_nifti\img"
TEST_MASK_DIR  = r"E:\3D_sinus\test_nifti\mask"
OUT_XLSX       = r"E:\3D_sinus\orta_hat_duzeltme\anatomik_orta_hat_BB_duzenli_cikti.xlsx"

# ─── PARAMETRELER ─────────────────────────────────────────
OFFSET_H, OFFSET_W = 83, 93          # kirpma baslangici (satir, sutun)
TARGET_H, TARGET_W = 261, 324        # kirpma boyutu
MODEL_H, MODEL_W   = 256, 256        # model girdi boyutu
W_CENTER, W_WIDTH  = 400, 2000       # pencere merkezi / genisligi (HU)
CANVAS             = 512             # kesit boyutu
ESIK               = 0.5             # olasilik esigi
ACILAR             = np.arange(-15, 15.01, 1.0)   # orta hat aci aramasi (derece)
SUTUNLAR           = range(176, 337)              # orta hat sutun aramasi
SERIT_MAX          = 160                          # simetri seridi yari genisligi (piksel)
KOMSULUK_26        = np.ones((3, 3, 3), dtype=bool)
METRIKLER          = ["dice", "iou", "precision", "recall"]
# Elle secilen kesit araligi: hasta -> (ilk, son) NIfTI kesit no (1'den baslar, uclar dahil).
# Yalniz bu kesitler tahmine ve degerlendirmeye girer; listede olmayan hastalarda tum kesitler kullanilir.
KESIT_ARALIGI      = {"k4-5": (62, 73)}
# Elle secilen asagi kaydirma: hasta -> piksel (+ = ekranda asagi). Bas goruntude yukarida kaldiginda
# tahminden once goruntu asagi kaydirilir; degerlendirmede uzman maskesi de ayni miktarda kaydirilir.
KAYDIRMA_ASAGI     = {"k9-4": 100}
# ──────────────────────────────────────────────────────────


def f1_score(y_true, y_pred):
    """Modeli yuklemek icin gereken ozel metrik."""
    a, b = K.flatten(y_true), K.flatten(y_pred)
    i = K.sum(a * b)
    return (2. * i + K.epsilon()) / (K.sum(a) + K.sum(b) + K.epsilon())


def windowing(img):
    lo, hi = W_CENTER - W_WIDTH / 2.0, W_CENTER + W_WIDTH / 2.0
    return (((np.clip(img, lo, hi) - lo) / (hi - lo)) * 255.0).astype(np.uint8)


def predict(model, data):
    """ADIM 1 — Tum kesitler icin tahmin. Doner: (3B tahmin maskesi, pencerelenmis 3B goruntu)."""
    n = data.shape[2]
    win = np.empty((CANVAS, CANVAS, n), dtype=np.uint8)
    inp = np.empty((n, MODEL_H, MODEL_W, 1), dtype=np.float32)
    for i in range(n):
        w = windowing(np.rot90(data[:, :, i], k=1))
        win[:, :, i] = w
        c = w[OFFSET_H:OFFSET_H + TARGET_H, OFFSET_W:OFFSET_W + TARGET_W]
        inp[i, :, :, 0] = cv2.resize(c, (MODEL_W, MODEL_H), interpolation=cv2.INTER_AREA) / 255.0
    prob = model.predict(inp, batch_size=16, verbose=0)[..., 0]
    pred = np.zeros((CANVAS, CANVAS, n), dtype=bool)
    for i in range(n):
        p = (prob[i] > ESIK).astype(np.uint8)
        if p.any():
            pred[OFFSET_H:OFFSET_H + TARGET_H, OFFSET_W:OFFSET_W + TARGET_W, i] = \
                cv2.resize(p, (TARGET_W, TARGET_H), interpolation=cv2.INTER_NEAREST) > 0
    return pred, win


def find_midline(win, pred):
    """ADIM 2 — Ayna simetrisi ile hastaya ozgu orta hat.
    Doner: 512x512 mantiksal harita (True = orta hattin sagi = hastanin sagi)."""
    n = win.shape[2]
    sl = pred.reshape(-1, n).any(axis=0)             # tahminde sinus gorulen kesitler
    if not sl.any():
        sl[:] = True
    M = win[:, :, sl].astype(np.float32).mean(axis=2)
    merkez = (CANVAS / 2.0, CANVAS / 2.0)
    en_iyi = (-2.0, 0.0, CANVAS // 2)                # (r, aci, sutun)
    for aci in ACILAR:
        A = cv2.getRotationMatrix2D(merkez, float(aci), 1.0)
        R = cv2.warpAffine(M, A, (CANVAS, CANVAS), flags=cv2.INTER_LINEAR, borderValue=0)
        for c in SUTUNLAR:
            k = min(c, CANVAS - c, SERIT_MAX)
            a = R[:, c - k:c].ravel()                # sol serit
            b = R[:, c:c + k][:, ::-1].ravel()       # sag seridin ayna goruntusu
            a = a - a.mean(); b = b - b.mean()
            r = float(a @ b / (np.sqrt((a @ a) * (b @ b)) + 1e-9))
            if r > en_iyi[0]:
                en_iyi = (r, float(aci), c)
    _, aci, c = en_iyi
    A = cv2.getRotationMatrix2D(merkez, aci, 1.0)
    yy, xx = np.mgrid[0:CANVAS, 0:CANVAS]
    return (A[0, 0] * xx + A[0, 1] * yy + A[0, 2]) >= c


def largest_cc(mask):
    """ADIM 4 — En buyuk 3B bagli bilesen (26-komsuluk)."""
    lab, nl = ndimage.label(mask, structure=KOMSULUK_26)
    if nl <= 1:
        return mask.copy()
    return lab == (int(np.argmax(np.bincount(lab.ravel())[1:])) + 1)


def metrics(gt, pr):
    """3B Dice / IoU / Precision / Recall."""
    gt = gt.astype(bool); pr = pr.astype(bool)
    tp = np.sum(gt & pr); fp = np.sum(~gt & pr); fn = np.sum(gt & ~pr)
    d = lambda a, b: float(a) / float(b) if b > 0 else np.nan
    return {"dice": d(2 * tp, 2 * tp + fp + fn), "iou": d(tp, tp + fp + fn),
            "precision": d(tp, tp + fp), "recall": d(tp, tp + fn)}


def load_ref():
    """Uzman (ITK-SNAP) hacimleri: {hasta: {uzman_R, uzman_L, uzman_T}}."""
    df = pd.read_excel(EXCEL_REF_PATH, sheet_name=0, header=1)
    df = df[df.iloc[:, 0].astype(str).str.match(r"^[kK]?\d")]
    return {str(r.iloc[0]).strip(): {"uzman_R": float(r["Uzman SAĞ (cm³)"]),
                                     "uzman_L": float(r["Uzman SOL (cm³)"]),
                                     "uzman_T": float(r["Uzman TOPLAM (cm³)"])}
            for _, r in df.iterrows()}


def kaydir(vol, asagi, dolgu):
    """Hacmi ekranda 'asagi' piksel asagi kaydirir (standart yonde y ekseni ekranda yukari bakar,
    bu yuzden y boyunca geri kaydirilir); bosalan yer 'dolgu' degeriyle doldurulur."""
    out = np.full(vol.shape, dolgu, dtype=vol.dtype)
    if asagi > 0:
        out[:, :vol.shape[1] - asagi] = vol[:, asagi:]
    else:
        out[:, -asagi:] = vol[:, :vol.shape[1] + asagi]
    return out


def process_case(model, img_path, mask_path, aralik=None, asagi=0):
    img = nib.as_closest_canonical(nib.load(img_path))
    data = img.get_fdata()
    vx, vy, vz = [float(z) for z in img.header.get_zooms()[:3]]
    voksel_cm3 = vx * vy * vz / 1000.0
    if data.shape[:2] != (CANVAS, CANVAS):
        raise ValueError(f"Beklenmeyen kesit boyutu {data.shape[:2]} (512x512 olmali)")
    kesit = slice(aralik[0] - 1, aralik[1]) if aralik else slice(None)   # elle secilen kesitler
    data = data[:, :, kesit]
    if asagi:                                              # elle secilen asagi kaydirma
        data = kaydir(data, asagi, float(data.min()))

    pred, win = predict(model, data)                       # 1) tahmin
    sag3 = find_midline(win, pred)[:, :, None]             # 2) orta hat
    predR = largest_cc(pred & sag3)                        # 3-4) SAG, en buyuk bilesen
    predL = largest_cc(pred & ~sag3)                       # 3-4) SOL, en buyuk bilesen
    res = {"pred_R": predR.sum() * voksel_cm3,             # 5) hacimler
           "pred_L": predL.sum() * voksel_cm3,
           "pred_T": (predR.sum() + predL.sum()) * voksel_cm3}

    # --- degerlendirme: uzman maskesi yalniz burada kullanilir ---
    res.update({f"{m}_{s}": np.nan for m in METRIKLER for s in "TRL"})
    if os.path.exists(mask_path):
        gt = np.asarray(nib.as_closest_canonical(nib.load(mask_path)).dataobj)[:, :, kesit]
        if asagi:
            gt = kaydir(gt, asagi, 0)
        gt = np.rot90(gt, k=1, axes=(0, 1)).astype(np.int16)
        gR, gL = gt == 1, gt == 2
        for s, g, p in [("T", gR | gL, predR | predL), ("R", gR, predR), ("L", gL, predL)]:
            for m, v in metrics(g, p).items():
                res[f"{m}_{s}"] = v
    return res


def icc(y1, y2):
    """ICC(2,1) mutlak uyum ve ICC(3,1) tutarlilik (Shrout & Fleiss)."""
    Y = np.column_stack([y1, y2]); n, k = Y.shape
    gm = Y.mean()
    msr = k * np.sum((Y.mean(1) - gm) ** 2) / (n - 1)
    msc = n * np.sum((Y.mean(0) - gm) ** 2) / (k - 1)
    mse = np.sum((Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + gm) ** 2) / ((n - 1) * (k - 1))
    return ((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n),
            (msr - mse) / (msr + (k - 1) * mse))


def agreement(ref, pred):
    """Uzman ve model hacimleri arasindaki uyum istatistikleri."""
    ref, pred = np.asarray(ref, float), np.asarray(pred, float)
    d = pred - ref
    egim, kesisim, r, p_r, _ = stats.linregress(ref, pred)
    icc2, icc3 = icc(ref, pred)
    rve = np.abs(d) / ref * 100
    return {"n": len(ref),
            "Uzman ort (cm3)": ref.mean(), "Uzman SS": ref.std(ddof=1),
            "Model ort (cm3)": pred.mean(), "Model SS": pred.std(ddof=1),
            "Fark ort (Model-Uzman)": d.mean(), "Fark SS": d.std(ddof=1),
            "BA alt sinir": d.mean() - 1.96 * d.std(ddof=1),
            "BA ust sinir": d.mean() + 1.96 * d.std(ddof=1),
            "Eslestirilmis t p": stats.ttest_rel(pred, ref).pvalue,
            "Wilcoxon p": stats.wilcoxon(pred, ref).pvalue,
            "Pearson r": r, "Pearson p": p_r, "R2": r ** 2,
            "Regresyon egim": egim, "Regresyon kesisim": kesisim,
            "ICC(2,1) mutlak uyum": icc2, "ICC(3,1) tutarlilik": icc3,
            "RVE ort (%)": rve.mean(), "RVE SS (%)": rve.std(ddof=1)}


def main():
    ref = load_ref()
    print(f"Uzman dosyasi: {len(ref)} hasta", flush=True)
    model = tf.keras.models.load_model(SEG_MODEL_PATH, custom_objects={"f1_score": f1_score})

    rows = []
    for i, hasta in enumerate(sorted(ref), 1):
        bulunan = glob.glob(os.path.join(TEST_IMG_DIR, hasta + ".nii*"))
        if not bulunan:
            print(f"[{i:02d}/{len(ref)}] {hasta}: GORUNTU YOK", flush=True)
            continue
        img_path = bulunan[0]
        r = process_case(model, img_path, os.path.join(TEST_MASK_DIR, os.path.basename(img_path)),
                         KESIT_ARALIGI.get(hasta), KAYDIRMA_ASAGI.get(hasta, 0))
        r.update({"hasta": hasta, **ref[hasta]})
        rows.append(r)
        print(f"[{i:02d}/{len(ref)}] {hasta:7s} pred R={r['pred_R']:.2f} L={r['pred_L']:.2f} T={r['pred_T']:.2f}"
              f" | uzman R={r['uzman_R']:.2f} L={r['uzman_L']:.2f} T={r['uzman_T']:.2f}"
              f" | Dice T={r['dice_T']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    for s in "RLT":
        df[f"fark_{s}"] = df[f"pred_{s}"] - df[f"uzman_{s}"]
        df[f"RVE_{s}_%"] = df[f"fark_{s}"].abs() / df[f"uzman_{s}"] * 100

    # --- Sayfa 1: Hasta_bazinda ---
    sutunlar = (["hasta"]
                + [f"{a}_{s}" for s in "RLT" for a in ("uzman", "pred", "fark")]
                + [f"{m}_{s}" for m in METRIKLER for s in "TRL"]
                + [f"RVE_{s}_%" for s in "TRL"])
    hasta_bazinda = df[sutunlar].round(4)

    # --- Sayfa 2: hacim_uyum ---
    hacim_uyum = pd.DataFrame({ad: agreement(df[f"uzman_{s}"], df[f"pred_{s}"])
                               for s, ad in [("R", "SAG"), ("L", "SOL"), ("T", "TOPLAM")]}).round(4)

    # --- Sayfa 3: segmentasyon ---
    segmentasyon = pd.DataFrame({ad: {m: f"{df[f'{m}_{s}'].mean():.4f} ± {df[f'{m}_{s}'].std(ddof=1):.4f}"
                                      for m in METRIKLER}
                                 for s, ad in [("T", "TOPLAM"), ("R", "SAG"), ("L", "SOL")]})

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as xw:
        hasta_bazinda.to_excel(xw, sheet_name="Hasta_bazinda", index=False)
        hacim_uyum.to_excel(xw, sheet_name="hacim_uyum")
        segmentasyon.to_excel(xw, sheet_name="segmentasyon")

    print("\n=== hacim_uyum ===\n" + hacim_uyum.to_string())
    print("\n=== segmentasyon ===\n" + segmentasyon.to_string())
    print(f"\nKaydedildi: {OUT_XLSX}")


if __name__ == "__main__":
    main()
