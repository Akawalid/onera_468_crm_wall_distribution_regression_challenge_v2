# ATTENTION: vérifier si les points ne sont pas très informatifs sur la masse volumique
# Ce serait intéressant de laisser les candidats considère les simlulations non fiables dans leurs models
# https://www.codabench.org/competitions/9975/ (For custom librarieso
# croissance de mémpore entre 5 et 10% de données consimées

import numpy as np
import pandas as pd

epsilon = 1.e-6
nwallp  = 260774
ntrain_expected = 312 * nwallp
ntest_expected  = 156 * nwallp

BASE = "FILES_RHO_TRAINTESTSPLIT"
CSV  = BASE + "/traintest_splitting1_MinfAoAPi_with_scores.csv"

X_train = np.load(BASE + "/X9_TRAIN_POINT_fl32.npy")
X_test  = np.load(BASE + "/X9_TEST_POINT_fl32.npy")
Y_train = np.load(BASE + "/RHO_TRAIN_POINT_fl32.npy")
Y_test  = np.load(BASE + "/RHO_TEST_POINT_fl32.npy")
df      = pd.read_csv(CSV)

print("=" * 60)
print("1. DIMENSIONS")
print(f"   X_train : {X_train.shape}  attendu ({ntrain_expected}, 9)")
print(f"   X_test  : {X_test.shape}   attendu ({ntest_expected}, 9)")
print(f"   Y_train : {Y_train.shape}  attendu ({ntrain_expected}, 1)")
print(f"   Y_test  : {Y_test.shape}   attendu ({ntest_expected}, 1)")

ok_shapes = (
    X_train.shape == (ntrain_expected, 9) and
    X_test.shape  == (ntest_expected,  9) and
    Y_train.shape == (ntrain_expected, 1) and
    Y_test.shape  == (ntest_expected,  1)
)
print(f"   => {'OK' if ok_shapes else 'ERREUR'}")

print("=" * 60)
print("2. SPLIT 2/3 - 1/3 depuis le CSV")
n_train_csv = df['Train'].sum()
n_test_csv  = (~df['Train']).sum()
n_total     = len(df)
print(f"   Total conditions : {n_total}  (attendu 468)")
print(f"   Train            : {n_train_csv}  (attendu 312)")
print(f"   Test             : {n_test_csv}   (attendu 156)")
ok_split = (n_total == 468 and n_train_csv == 312 and n_test_csv == 156)
print(f"   => {'OK' if ok_split else 'ERREUR'}")

print("=" * 60)
print("3. PAR (Mach, Pi) : 8 train + 4 test = 12 AoA")
anomalies = []
for (mach, pi), grp in df.groupby(['Mach', 'Pi']):
    n_tr = grp['Train'].sum()
    n_te = (~grp['Train']).sum()
    total = len(grp)
    # exception article : Mach 0.3, 0.82, 0.96 => 2 AoA extremes forcees en train
    forced = mach in [0.30, 0.82, 0.96]
    if total != 12 or n_tr != 8 or n_te != 4:
        anomalies.append((mach, pi, total, n_tr, n_te, forced))

if not anomalies:
    print("   Toutes les paires (Mach, Pi) : 8 train / 4 test => OK")
else:
    print(f"   {len(anomalies)} anomalie(s) :")
    for a in anomalies:
        print(f"     Mach={a[0]:.2f} Pi={a[1]} | total={a[2]} train={a[3]} test={a[4]}"
              + (" [exception article]" if a[5] else " [INATTENDU]"))

print("=" * 60)
print("4. AoA EXTREMES EN TRAIN pour Mach 0.3, 0.82, 0.96 (article)")
for mach in [0.30, 0.82, 0.96]:
    for pi in df['Pi'].unique():
        grp = df[(df['Mach'] == mach) & (df['Pi'] == pi)]
        if grp.empty:
            continue
        aoa_min = grp['AoA'].min()
        aoa_max = grp['AoA'].max()
        row_min = grp[grp['AoA'] == aoa_min].iloc[0]
        row_max = grp[grp['AoA'] == aoa_max].iloc[0]
        ok_min = bool(row_min['Train'])
        ok_max = bool(row_max['Train'])
        status = "OK" if (ok_min and ok_max) else "ERREUR"
        print(f"   Mach={mach:.2f} Pi={pi} | AoA_min={aoa_min} train={ok_min} | "
              f"AoA_max={aoa_max} train={ok_max} => {status}")

print("=" * 60)
print("5. CONFIDENCE SCORES dans X (AoA >= 10 ou <= -10 => poids 0.5)")
aoa_train = X_train[:, 7]
aoa_test  = X_test[:, 7]
extreme_train = np.abs(aoa_train) >= 10. - epsilon
extreme_test  = np.abs(aoa_test)  >= 10. - epsilon
print(f"   Points train avec |AoA| >= 10 : {extreme_train.sum():,} "
      f"({100*extreme_train.mean():.1f}%)")
print(f"   Points test  avec |AoA| >= 10 : {extreme_test.sum():,}  "
      f"({100*extreme_test.mean():.1f}%)")

print("=" * 60)
print("6. PAS DE FUITE TRAIN/TEST (conditions Mach/AoA/Pi uniques)")
def extract_conditions(X, nwallp):
    conds = set()
    n_conds = X.shape[0] // nwallp
    for i in range(n_conds):
        row = X[i * nwallp]
        conds.add((round(float(row[6]), 4),
                   round(float(row[7]), 4),
                   round(float(row[8]), 4)))
    return conds

conds_train = extract_conditions(X_train, nwallp)
conds_test  = extract_conditions(X_test,  nwallp)
overlap = conds_train & conds_test
print(f"   Conditions train : {len(conds_train)} | test : {len(conds_test)}")
print(f"   Overlap          : {len(overlap)} => {'OK' if len(overlap)==0 else 'FUITE DETECTEE'}")

print("=" * 60)
print("7. VALEURS RHO (sanite)")
print(f"   Y_train min={Y_train.min():.4f} max={Y_train.max():.4f} "
      f"mean={Y_train.mean():.4f} nan={np.isnan(Y_train).sum()}")
print(f"   Y_test  min={Y_test.min():.4f} max={Y_test.max():.4f} "
      f"mean={Y_test.mean():.4f} nan={np.isnan(Y_test).sum()}")

print("=" * 60)
print("RESUME FINAL")
print(f"  Shapes     : {'OK' if ok_shapes else 'ERREUR'}")
print(f"  Split CSV  : {'OK' if ok_split  else 'ERREUR'}")
print(f"  Pas de fuite : {'OK' if len(overlap)==0 else 'ERREUR'}")