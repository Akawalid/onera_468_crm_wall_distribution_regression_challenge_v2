import h5py
import numpy as np

family_map = {
    'AIRCRAFT_BODY:FUSE':      0,
    'AIRCRAFT_BODY:TAILU':     1,
    'AIRCRAFT_BODY:TAILL':     1,
    'AIRCRAFT_BODY:TAILTE':    1,
    'AIRCRAFT_BODY:WINGU':     2,
    'AIRCRAFT_BODY:WINGL':     2,
    'AIRCRAFT_BODY:WINGTE':    2,
    'AIRCRAFT_BODY:WINGU2':    2,
    'AIRCRAFT_BODY:NACELLEOU': 3,
    'AIRCRAFT_BODY:NACELLEIN': 3,
    'AIRCRAFT_BODY:NACELLETE': 3,
    'AIRCRAFT_BODY:PYLON':     4,
    'AIRCRAFT_BODY:PYLON2':    4,
}

label_names = {0: 'Fuselage', 1: 'Empennage', 2: 'Aile', 3: 'Nacelle', 4: 'Pylone'}

all_xyz    = []
all_labels = []

with h5py.File("CFSE.Charbonnier.WBNP.ae2.75deg.Structured.T.cgns", 'r') as f:
    base = f['BASE#1']

    for zone_name in base.keys():
        zone = base[zone_name]
        if 'ZoneBC' not in zone:
            continue

        coords = zone['GridCoordinates']
        X = np.array(coords['CoordinateX'][' data'])
        Y = np.array(coords['CoordinateY'][' data'])
        Z = np.array(coords['CoordinateZ'][' data'])

        for bc_name in zone['ZoneBC'].keys():
            bc = zone['ZoneBC'][bc_name]
            if 'FamilyName' not in bc or 'PointRange' not in bc:
                continue

            raw    = bc['FamilyName'][' data']
            family = ''.join(chr(c) for c in raw).strip()
            if family not in family_map:
                continue

            zone_label = family_map[family]

            pr = np.array(bc['PointRange'][' data'])
            # shape (2, 3) → [[i1, j1, k1], [i2, j2, k2]] (1-indexed)
            i1, j1, k1 = pr[0, 0] - 1, pr[0, 1] - 1, pr[0, 2] - 1
            i2, j2, k2 = pr[1, 0],     pr[1, 1],     pr[1, 2]

            x_face = X[i1:i2, j1:j2, k1:k2].flatten()
            y_face = Y[i1:i2, j1:j2, k1:k2].flatten()
            z_face = Z[i1:i2, j1:j2, k1:k2].flatten()

            xyz = np.stack([x_face, y_face, z_face], axis=1)
            all_xyz.append(xyz)
            all_labels.append(np.full(len(x_face), zone_label))

all_xyz    = np.vstack(all_xyz)
all_labels = np.concatenate(all_labels)

print(f"Total points extraits : {len(all_xyz)}")
for k, name in label_names.items():
    print(f"  {name} : {(all_labels == k).sum()} points")

np.save('cgns_xyz.npy',    all_xyz)
np.save('cgns_labels.npy', all_labels)
