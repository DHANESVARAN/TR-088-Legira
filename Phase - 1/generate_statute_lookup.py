import argparse
import os

import pandas as pd


def build_lookup(master_df: pd.DataFrame, life_years_proxy: float) -> pd.DataFrame:
    out = master_df.copy()
    out["section"] = out["section"].astype(str).str.strip().str.upper()
    out["punishment_type"] = out["punishment_type"].astype(str).str.lower().str.strip()
    out["max_sentence_years"] = pd.to_numeric(out["max_sentence_years"], errors="coerce")

    out["is_non_finite_sentence"] = out["punishment_type"].isin(["life", "death"])
    out["max_sentence_years_effective"] = out["max_sentence_years"]
    out.loc[out["punishment_type"].eq("life"), "max_sentence_years_effective"] = life_years_proxy
    out.loc[out["punishment_type"].eq("death"), "max_sentence_years_effective"] = float("nan")

    out["max_sentence_days"] = (out["max_sentence_years_effective"] * 365).round()

    cols = [
        "section",
        "law_code",
        "description",
        "punishment_type",
        "max_sentence_years",
        "max_sentence_years_effective",
        "max_sentence_days",
        "bailability",
        "is_non_finite_sentence",
    ]
    return out[cols]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate section lookup from statute master table")
    parser.add_argument(
        "--master-file",
        default=os.path.join("data", "ipc_bns_statutes_master.csv"),
        help="Master statute CSV",
    )
    parser.add_argument(
        "--output-file",
        default=os.path.join("data", "ipc_bns_max_sentence_lookup.csv"),
        help="Generated lookup CSV",
    )
    parser.add_argument(
        "--life-years-proxy",
        type=float,
        default=25.0,
        help="Proxy years for life imprisonment in ratio scoring",
    )
    args = parser.parse_args()

    master_df = pd.read_csv(args.master_file)
    lookup_df = build_lookup(master_df, args.life_years_proxy)
    lookup_df.to_csv(args.output_file, index=False)

    print(f"Generated lookup: {args.output_file}")
    print(f"Sections: {len(lookup_df)}")


if __name__ == "__main__":
    main()
