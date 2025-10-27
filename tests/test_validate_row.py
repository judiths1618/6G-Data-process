import os
import sys
import tempfile
import textwrap
import unittest


sys.path.append(os.path.dirname(os.path.dirname(__file__)))
print(sys.path)


from dq_local_beam import (
    DEFAULT_RULE,
    RowCtx,
    evaluate_metadata_prerequisites,
    load_config,
    validate_row_against_rule,
)


class ValidateRowRequiredColumnTest(unittest.TestCase):
    def test_required_column_uses_alias_for_value_lookup(self) -> None:
        rule = dict(DEFAULT_RULE)
        rule["required_cols"] = ["user id"]

        rc = RowCtx(
            file="example.csv",
            rownum=1,
            header=["User ID"],
            data={"User ID": ""},
        )
        setattr(rc, "_dq_header_alias", {"user id": "User ID"})

        valid, issues = validate_row_against_rule(rc, rule, None)

        self.assertIsNone(valid)
        self.assertEqual(len(issues), 1)
        self.assertEqual(
            issues[0],
            {
                "file": "example.csv",
                "row": 1,
                "reason": "null_required:User ID",
            },
        )


class MetadataPrerequisiteTest(unittest.TestCase):
    def test_missing_metadata_for_dataset_returns_single_notification(self) -> None:
        rules = [dict(DEFAULT_RULE)]
        metadata_ready, notifications = evaluate_metadata_prerequisites(
            rules, ["alpha.csv", "beta.csv"]
        )

        self.assertFalse(metadata_ready)
        self.assertEqual(len(notifications), 1)
        self.assertEqual(notifications[0]["scope"], "dataset")
        self.assertIn("alpha.csv", notifications[0]["files"])
        self.assertIn("beta.csv", notifications[0]["files"])

    def test_partial_metadata_reports_missing_files(self) -> None:
        rule = dict(DEFAULT_RULE)
        rule["metadata_by_file"] = {"alpha.csv": {"id": "Identifier"}}
        rules = [rule]

        metadata_ready, notifications = evaluate_metadata_prerequisites(
            rules, ["alpha.csv", "beta.csv"]
        )

        self.assertFalse(metadata_ready)
        self.assertEqual(len(notifications), 1)
        self.assertEqual(notifications[0]["scope"], "file")
        self.assertEqual(notifications[0]["file"], "beta.csv")

    def test_metadata_targets_fall_back_to_default_section(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_path = os.path.join(tmpdir, "metadata.txt")
            with open(metadata_path, "w", encoding="utf-8") as fp:
                fp.write(
                    textwrap.dedent(
                        """
                        Scenario description
                        Number of Data Samples: 2411
                        index: Sample identifier
                        time_stamp[UTC]: Capture timestamp in hh-mm-ss-ms format
                        unit2_num_sat: Number of connected satellites
                        unit2_fix_type: Fix type description
                        """
                    ).strip()
                )

            config_path = os.path.join(tmpdir, "config.yaml")
            with open(config_path, "w", encoding="utf-8") as fp:
                fp.write(
                    textwrap.dedent(
                        f"""
                        rules:
                          - patterns: [".*"]
                            metadata_path: "{metadata_path}"
                            metadata_targets:
                              - sample.csv
                        """
                    ).strip()
                )

            cfg = load_config(config_path)["rules"]
            rule = cfg[0]

            metadata_cols = rule.get("metadata_by_file", {}).get("sample.csv")
            self.assertIsNotNone(metadata_cols)
            self.assertIn("index", metadata_cols)
            self.assertIn("time_stamp[utc]", metadata_cols)
            self.assertNotIn("number of data samples", metadata_cols)

            metadata_ready, notifications = evaluate_metadata_prerequisites(
                cfg, ["sample.csv"]
            )

            self.assertTrue(metadata_ready)
            self.assertEqual(notifications, [])


if __name__ == "__main__":
    unittest.main()
