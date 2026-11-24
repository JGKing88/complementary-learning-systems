"""Apply an audit's corrections to a sweep table, in place.

    python apply_corrections.py SWEEP.csv CORRECTIONS.csv [PREFIX]

CORRECTIONS.csv has columns row_id, column, old_value, new_value, reason, evidence. A change is applied
only when the cell still holds old_value (both empty counts as a match), so a stale or mistyped
correction is reported instead of overwriting newer data. column=DROP removes the row.

One corrections file can target several tables by prefixing row ids, e.g. "sweep_kernel.csv:K26".
With PREFIX, only entries whose row_id starts with it are used (prefix stripped). Without PREFIX,
entries whose row_id contains ":" belong to another table and are skipped.
"""
import csv
import sys


def main():
    sweep_path, corr_path = sys.argv[1], sys.argv[2]
    prefix = sys.argv[3] if len(sys.argv) > 3 else None
    with open(sweep_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    by_id = {r["id"]: r for r in rows}
    with open(corr_path, newline="", encoding="utf-8") as f:
        corrections = list(csv.DictReader(f))

    applied, conflicts, missing, dropped, skipped = 0, [], [], set(), 0
    for c in corrections:
        rid, col = c["row_id"].strip(), c["column"].strip()
        if prefix:
            if not rid.startswith(prefix):
                skipped += 1
                continue
            rid = rid[len(prefix):]
        elif ":" in rid:
            skipped += 1
            continue
        if rid not in by_id:
            missing.append(rid)
            continue
        if col == "DROP":
            dropped.add(rid)
            continue
        if col not in fields:
            conflicts.append((rid, col, "no such column"))
            continue
        cur = (by_id[rid][col] or "").strip()
        if cur != (c["old_value"] or "").strip():
            conflicts.append((rid, col, f"cell holds {cur[:60]!r}, correction expects {c['old_value'][:60]!r}"))
            continue
        by_id[rid][col] = c["new_value"]
        applied += 1

    kept = [r for r in rows if r["id"] not in dropped]
    with open(sweep_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(kept)
    print(f"{applied} applied, {len(dropped)} rows dropped, {len(conflicts)} conflicts, {len(missing)} unknown ids, "
          f"{skipped} for other tables")
    for rid, col, why in conflicts:
        print(f"  CONFLICT {rid}.{col}: {why}")
    for rid in sorted(set(missing)):
        print(f"  UNKNOWN id {rid}")


if __name__ == "__main__":
    main()
