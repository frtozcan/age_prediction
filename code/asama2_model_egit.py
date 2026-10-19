# -*- coding: utf-8 -*-
"""
AŞAMA 2 — ML Model Eğitimi (UZMAN HACİM VERSİYONU)
=====================================================
Eğitim verisi olarak uzman ölçümlerini kullanır:
  maxilla R  → sağ sinüs hacmi (cm³)
  maxilla L  → sol sinüs hacmi (cm³)
  maxilla T  → toplam sinüs hacmi (cm³)

Model A — Yaş Tahmini (Regresyon)
  Giriş : maxilla_R, maxilla_L, maxilla_T
  Çıkış : yaş

Model B — Cinsiyet Tahmini (Sınıflandırma)
  Giriş : maxilla_R, maxilla_L, maxilla_T
  Çıkış : erkek (1) / kadın (0)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import LeaveOneOut, cross_val_score, KFold
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score, accuracy_score, classification_report,
                             confusion_matrix)

# ─── PARAMETRELER ─────────────────────────────────────────
EXCEL_PATH     = r"E:\3D_sinus\UZMAN_hacim_bilgileri_sag_sol_toplam_yas.xlsx"
MODEL_YAS_PATH = r"E:\3D_sinus\ml_data\model_yas.joblib"
MODEL_CIN_PATH = r"E:\3D_sinus\ml_data\model_cinsiyet.joblib"
PLOT_PATH      = r"E:\3D_sinus\ml_data\model_performans.png"

# !! KRİTİK: Aşama 3 ile birebir aynı sıra ve isimler !!
FEATURE_COLS = ['maxilla_R', 'maxilla_L', 'maxilla_T']
# ──────────────────────────────────────────────────────────


def load_data(excel_path):
    df = pd.read_excel(excel_path)
    df = df.rename(columns={
        'maxilla R' : 'maxilla_R',
        'maxilla L' : 'maxilla_L',
        'maxilla T' : 'maxilla_T',
        'CİNSİYET'  : 'cinsiyet',
        'YAS'       : 'yas',
        'Hasta_ADI' : 'hasta_adi',
        'SAYI NO'   : 'sayi_no',
        'CTKESİT'   : 'ct_kesit',
    })
    df['cinsiyet_enc'] = df['cinsiyet'].apply(
        lambda x: 1 if str(x).lower().strip() == 'erkek' else 0
    )
    df = df.dropna(subset=FEATURE_COLS + ['yas', 'cinsiyet_enc'])
    return df


def evaluate_regressor(name, model, X, y, cv):
    mae = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
    r2  = cross_val_score(model, X, y, cv=cv, scoring='r2')
    return {'name': name, 'mae_cv': -mae.mean(), 'mae_std': mae.std(), 'r2_cv': r2.mean()}


def evaluate_classifier(name, model, X, y, cv):
    acc = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    return {'name': name, 'acc_cv': acc.mean(), 'acc_std': acc.std()}


def plot_results(y_true_age, y_pred_age, y_true_gen, y_pred_gen, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("ML Model Performansi -- Uzman Hacim Verileri", fontsize=14, fontweight='bold')

    ax = axes[0]
    ax.scatter(y_true_age, y_pred_age, c='steelblue', edgecolors='navy', alpha=0.7, s=60)
    lim = [min(y_true_age.min(), y_pred_age.min()) - 0.5,
           max(y_true_age.max(), y_pred_age.max()) + 0.5]
    ax.plot(lim, lim, 'r--', lw=1.5, label='Ideal')
    mae = mean_absolute_error(y_true_age, y_pred_age)
    r2  = r2_score(y_true_age, y_pred_age)
    ax.annotate(f"MAE={mae:.2f} yil\nR2={r2:.3f}",
                xy=(0.05, 0.88), xycoords='axes fraction', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow'))
    ax.set_xlabel("Gercek Yas"); ax.set_ylabel("Tahmin Yas")
    ax.set_title("Yas: Gercek vs Tahmin"); ax.legend(fontsize=9)

    ax2 = axes[1]
    errors = y_pred_age - y_true_age
    ax2.hist(errors, bins=15, color='steelblue', edgecolor='navy', alpha=0.8)
    ax2.axvline(0, color='red', linestyle='--', lw=1.5)
    ax2.axvline(errors.mean(), color='orange', linestyle='--', lw=1.5,
                label=f"Ort. Hata={errors.mean():.2f}")
    ax2.set_xlabel("Tahmin Hatasi (yil)"); ax2.set_ylabel("Hasta Sayisi")
    ax2.set_title("Yas Tahmin Hata Dagilimi"); ax2.legend(fontsize=9)

    ax3 = axes[2]
    cm = confusion_matrix(y_true_gen, y_pred_gen)
    im = ax3.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax3)
    ax3.set_xticks([0, 1]); ax3.set_yticks([0, 1])
    ax3.set_xticklabels(['Kadin', 'Erkek'])
    ax3.set_yticklabels(['Kadin', 'Erkek'])
    ax3.set_xlabel("Tahmin"); ax3.set_ylabel("Gercek")
    ax3.set_title("Cinsiyet Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax3.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=14)
    acc = accuracy_score(y_true_gen, y_pred_gen)
    ax3.annotate(f"Accuracy = {acc:.3f}", xy=(0.5, -0.12),
                 xycoords='axes fraction', ha='center', fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"[BILGI] Grafik kaydedildi: {save_path}")


def main():
    print(f"Excel okunuyor: {EXCEL_PATH}")
    df = load_data(EXCEL_PATH)
    n  = len(df)
    print(f"Toplam {n} hasta yuklendi.")
    print(f"Yas araligi   : {int(df['yas'].min())} - {int(df['yas'].max())}")
    print(f"Erkek / Kadin : {(df['cinsiyet_enc']==1).sum()} / {(df['cinsiyet_enc']==0).sum()}")
    print(f"\nKullanilan ozellikler: {FEATURE_COLS}")
    print(df[FEATURE_COLS].describe().round(3).to_string())

    X     = df[FEATURE_COLS].values
    y_yas = df['yas'].values.astype(float)
    y_cin = df['cinsiyet_enc'].values.astype(int)

    cv     = LeaveOneOut() if n <= 50 else KFold(n_splits=5, shuffle=True, random_state=42)
    cv_str = f"LOO (n={n})" if n <= 50 else f"5-Fold (n={n})"
    print(f"\nCapraz dogrulama: {cv_str}\n")

    # MODEL A — YAS
    print("=" * 58)
    print("MODEL A -- YAS TAHMINI (Regresyon)")
    print("=" * 58)
    candidates_reg = [
        ("Linear Regression",  LinearRegression()),
        ("Ridge (alpha=1)",    Ridge(alpha=1.0)),
        ("Ridge (alpha=10)",   Ridge(alpha=10.0)),
        ("Random Forest",      RandomForestRegressor(n_estimators=300,
                                   max_depth=8, min_samples_leaf=2, random_state=42)),
        ("Gradient Boosting",  GradientBoostingRegressor(n_estimators=200,
                                   max_depth=3, learning_rate=0.05, random_state=42)),
    ]
    best_reg, best_mae, best_reg_name = None, float('inf'), ""
    for name, model in candidates_reg:
        res  = evaluate_regressor(name, model, X, y_yas, cv)
        flag = "  <- EN IYI" if res['mae_cv'] < best_mae else ""
        print(f"  {name:<25}  MAE={res['mae_cv']:.2f} +/- {res['mae_std']:.2f} yil"
              f"  R2={res['r2_cv']:.3f}{flag}")
        if res['mae_cv'] < best_mae:
            best_mae = res['mae_cv']
            best_reg = model
            best_reg_name = name

    best_reg.fit(X, y_yas)
    y_pred_age = best_reg.predict(X)
    print(f"\n  Secilen    : {best_reg_name}")
    print(f"  Egitim MAE : {mean_absolute_error(y_yas, y_pred_age):.2f} yil")
    print(f"  Egitim RMSE: {np.sqrt(mean_squared_error(y_yas, y_pred_age)):.2f} yil")
    print(f"  Egitim R2  : {r2_score(y_yas, y_pred_age):.4f}")
    if hasattr(best_reg, 'feature_importances_'):
        print(f"\n  Ozellik Onemleri:")
        for col, imp in zip(FEATURE_COLS, best_reg.feature_importances_):
            print(f"    {col:<12}: {imp:.4f}")

    # MODEL B — CİNSİYET
    print("\n" + "=" * 58)
    print("MODEL B -- CINSIYET TAHMINI (Siniflandirma)")
    print("=" * 58)
    candidates_clf = [
        ("Random Forest",  RandomForestClassifier(n_estimators=300,
                               max_depth=8, min_samples_leaf=2, random_state=42)),
    ]
    best_clf, best_acc, best_clf_name = None, 0.0, ""
    for name, model in candidates_clf:
        res  = evaluate_classifier(name, model, X, y_cin, cv)
        flag = "  <- EN IYI" if res['acc_cv'] > best_acc else ""
        print(f"  {name:<25}  Acc={res['acc_cv']:.3f} +/- {res['acc_std']:.3f}{flag}")
        if res['acc_cv'] > best_acc:
            best_acc = res['acc_cv']
            best_clf = model
            best_clf_name = name

    best_clf.fit(X, y_cin)
    y_pred_gen = best_clf.predict(X)
    print(f"\n  Secilen      : {best_clf_name}")
    print(f"  Egitim Acc   : {accuracy_score(y_cin, y_pred_gen):.4f}")
    print(f"\n  Siniflandirma Raporu:")
    print(classification_report(y_cin, y_pred_gen, target_names=['Kadin (0)', 'Erkek (1)']))
    if hasattr(best_clf, 'feature_importances_'):
        print(f"  Ozellik Onemleri:")
        for col, imp in zip(FEATURE_COLS, best_clf.feature_importances_):
            print(f"    {col:<12}: {imp:.4f}")

    os.makedirs(os.path.dirname(MODEL_YAS_PATH), exist_ok=True)
    joblib.dump(best_reg, MODEL_YAS_PATH)
    joblib.dump(best_clf, MODEL_CIN_PATH)
    print(f"\n{'='*58}")
    print(f"Modeller kaydedildi:")
    print(f"  {MODEL_YAS_PATH}")
    print(f"  {MODEL_CIN_PATH}")
    print(f"  Ozellik sayisi: {len(FEATURE_COLS)}  ->  {FEATURE_COLS}")
    print(f"{'='*58}\n")

    plot_results(y_yas, y_pred_age, y_cin, y_pred_gen, PLOT_PATH)


if __name__ == "__main__":
    main()