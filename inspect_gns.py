import h5py
import numpy as np
from collections import defaultdict

CGNS_FILE = "mesh_Charbonnier.cgns"

COMPONENT_KEYWORDS = {
    "nacelle":  ["nacelle", "nac"],
    "pylon":    ["pylon", "pylone", "mat"],
    "wing":     ["wing", "aile", "wng"],
    "fuselage": ["fuselage", "fus"],
    "flap":     ["flap", "volet"],
    "slat":     ["slat", "bec"],
    "tail":     ["tail", "empennage", "htail", "vtail", "horiz", "vert"],
}

def decode(val):
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace").strip()
    if isinstance(val, np.ndarray):
        return "".join(chr(c) for c in val if c != 0).strip()
    return str(val).strip()

def guess_component(name: str) -> str:
    patch = name.split(":")[-1] if ":" in name else name
    patch_lower = patch.lower()
    for component, keywords in COMPONENT_KEYWORDS.items():
        for kw in keywords:
            if kw in patch_lower:
                return component
    return "unknown"

def collect_families(root: h5py.Group) -> dict[str, str]:
    families = {}
    for base_name, base in root.items():
        if not isinstance(base, h5py.Group):
            continue
        for item_name, item in base.items():
            if not isinstance(item, h5py.Group):
                continue
            label = decode(item.attrs.get("label", b""))
            if label == "Family_t":
                families[item_name] = guess_component(item_name)
    return families

def read_zone_coordinates(zone: h5py.Group) -> np.ndarray | None:
    for child_name, child in zone.items():
        if not isinstance(child, h5py.Group):
            continue
        label = decode(child.attrs.get("label", b""))
        if label == "GridCoordinates_t":
            coords = {}
            for coord_name, coord_node in child.items():
                if not isinstance(coord_node, h5py.Group):
                    continue
                data_node = coord_node.get(" data")
                if data_node is None:
                    continue
                coords[coord_name] = np.array(data_node).ravel()
            if "CoordinateX" in coords and "CoordinateY" in coords and "CoordinateZ" in coords:
                x = coords["CoordinateX"]
                y = coords["CoordinateY"]
                z = coords["CoordinateZ"]
                return np.column_stack([x, y, z])
    return None

def get_zone_family(zone: h5py.Group) -> str:
    for child_name, child in zone.items():
        if not isinstance(child, h5py.Group):
            continue
        label = decode(child.attrs.get("label", b""))
        if label == "FamilyName_t":
            data_node = child.get(" data")
            if data_node is not None:
                return decode(np.array(data_node))
    return ""

def classify_zones(cgns_file: str) -> dict[str, dict]:
    results = defaultdict(lambda: {"points": [], "zones": []})

    with h5py.File(cgns_file, "r") as f:
        families = collect_families(f)

        for base_name, base in f.items():
            if not isinstance(base, h5py.Group):
                continue
            base_label = decode(base.attrs.get("label", b""))
            if base_label != "CGNSBase_t":
                continue

            for zone_name, zone in base.items():
                if not isinstance(zone, h5py.Group):
                    continue
                zone_label = decode(zone.attrs.get("label", b""))
                if zone_label != "Zone_t":
                    continue

                family_name = get_zone_family(zone)
                if family_name and family_name in families:
                    component = families[family_name]
                else:
                    component = guess_component(zone_name)
                    if component == "unknown" and family_name:
                        component = guess_component(family_name)

                coords = read_zone_coordinates(zone)
                if coords is not None:
                    results[component]["points"].append(coords)
                    results[component]["zones"].append(zone_name)

    return dict(results)

def summarize(results: dict[str, dict]):
    print(f"\n{'='*60}")
    print(f"  Classification des points — mesh_Charbonnier.cgns")
    print(f"{'='*60}")
    total_pts = 0
    for component in sorted(results.keys()):
        pts_list = results[component]["points"]
        zones = results[component]["zones"]
        n_pts = sum(p.shape[0] for p in pts_list)
        total_pts += n_pts
        print(f"\n  [{component.upper()}]")
        print(f"    Zones    : {len(zones)}")
        print(f"    Points   : {n_pts:,}")
        for i, (z, p) in enumerate(zip(zones, pts_list)):
            print(f"      zone[{i}] {z!r:40s}  {p.shape[0]:>8,} pts")
    print(f"\n{'='*60}")
    print(f"  TOTAL : {total_pts:,} points sur {sum(len(v['zones']) for v in results.values())} zones")
    print(f"{'='*60}\n")

def export_npy(results: dict[str, dict], out_prefix: str = "charbonnier"):
    for component, data in results.items():
        if not data["points"]:
            continue
        all_pts = np.vstack(data["points"])
        fname = f"{out_prefix}_{component}.npy"
        np.save(fname, all_pts)
        print(f"  Exporté : {fname}  ({all_pts.shape[0]:,} points)")

if __name__ == "__main__":
    import sys
    cgns_path = sys.argv[1] if len(sys.argv) > 1 else CGNS_FILE

    print(f"\nLecture de : {cgns_path}")
    results = classify_zones(cgns_path)
    summarize(results)

    export = input("Exporter les composants en .npy ? [o/N] ").strip().lower()
    if export == "o":
        export_npy(results)
        print("Export terminé.")