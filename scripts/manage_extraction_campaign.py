"""Initialize or refresh a full-dataset extraction campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from infobudget.rl_router.campaign import initialize_campaign, refresh_campaign
from infobudget.rl_router.config import load_rl_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("init", "refresh"))
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--dataset", default="longmemeval")
    parser.add_argument("--split", default="full")
    parser.add_argument("--method", required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    args = parser.parse_args()
    bundle = load_rl_bundle(args.config_dir)
    if args.action == "init":
        manifest = initialize_campaign(
            bundle,
            campaign_id=args.campaign_id,
            dataset_name=args.dataset,
            split=args.split,
            segmentation_method=args.method,
            run_prefix=args.run_prefix,
        )
    else:
        manifest = refresh_campaign(bundle, args.campaign_id)
    print(
        json.dumps(
            {
                "campaign_id": manifest["campaign_id"],
                "status": manifest["status"],
                "sample_count": manifest["sample_count"],
                "complete_samples": manifest.get("complete_samples", 0),
                "quality_violations": manifest.get("quality_violations", {}),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
