import pandas as pd

from demaml.data import audit_csv, preprocess_csv


def _write_csv(path):
    rows = []
    for i in range(20):
        rows.append(
            {
                "Timestamp": f"2022/09/01 {i:02d}:00",
                "From Bank": 1,
                "Account": f"A{i % 4}",
                "To Bank": 2,
                "Account.1": f"B{(i + 1) % 4}",
                "Amount Received": i + 1.0,
                "Receiving Currency": "USD",
                "Amount Paid": i + 1.0,
                "Payment Currency": "USD",
                "Payment Format": "ACH",
                "Is Laundering": int(i in (5, 18)),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_audit_and_chronological_split(tmp_path):
    csv = tmp_path / "tiny.csv"
    _write_csv(csv)
    report = audit_csv(csv)
    assert report["rows"] == 20
    assert report["positives"] == 2
    data = preprocess_csv(csv, tmp_path / "processed.pt")
    assert data.edge_index.shape == (2, 20)
    assert (data.split == 0).sum() == 12
    assert (data.split == 1).sum() == 2
    assert (data.split == 2).sum() == 6
    assert data.time.tolist() == sorted(data.time.tolist())
