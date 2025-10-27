import json
import os
import sys
import tempfile
import textwrap
import unittest


sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dq_local_beam import (  # noqa: E402
    DEFAULT_RULE,
    RowCtx,
    _VISUALIZATION_TOTAL_LIMIT_ENV,
    _run_without_beam,
    _single_shard_path,
    collect_numeric_values_by_file,
    evaluate_metadata_prerequisites,
    load_config,
    summarize_numeric_values,
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

    def test_deepsense_scenario1_metadata_columns_are_clean(self) -> None:
        if load_config.__globals__.get("yaml") is None:
            self.skipTest("PyYAML is required to parse the DeepSense Scenario 1 rules")

        root = os.path.dirname(os.path.dirname(__file__))
        config_path = os.path.join(root, "deepsense_scen1_dq_rules.yaml")
        rules = load_config(config_path)["rules"]
        self.assertGreaterEqual(len(rules), 1)

        rule = rules[0]
        metadata_cols = rule.get("metadata_by_file", {}).get("scenario1.csv")
        self.assertIsNotNone(metadata_cols)

        column_names = set(metadata_cols.keys())
        expected = {
            "index",
            "unit1_rgb",
            "unit1_pwr_60ghz",
            "unit1_loc",
            "unit2_loc",
            "unit1_beam_index",
            "seq_index",
            "time_stamp[utc]",
            "unit2_direction",
            "unit2_num_sat",
            "unit2_sat_used",
            "unit2_fix_type",
            "unit2_dgps",
            "unit2_pdop",
            "unit2_hdop",
        }

        self.assertTrue(expected.issubset(column_names))
        self.assertNotIn("wireless sensor [phased array]", column_names)

    def test_deepsense_scenario1_numeric_columns_have_distribution_profiles(self) -> None:
        if load_config.__globals__.get("yaml") is None:
            self.skipTest("PyYAML is required to parse the DeepSense Scenario 1 rules")

        root = os.path.dirname(os.path.dirname(__file__))
        config_path = os.path.join(root, "deepsense_scen1_dq_rules.yaml")
        csv_path = os.path.join(
            root, "6GDALI_Datasets", "DeepSense", "Scenario1", "scenario1.csv"
        )

        rules = load_config(config_path)["rules"]
        per_file_values, _ = collect_numeric_values_by_file([csv_path], rules)
        numeric_values = per_file_values.get(csv_path)
        self.assertIsNotNone(numeric_values)

        self.assertIn("unit2_pdop", {k.lower() for k in numeric_values.keys()})
        pdop_values = numeric_values.get("unit2_PDOP")
        self.assertIsNotNone(pdop_values)
        self.assertGreater(len(pdop_values), 0)

        stats = summarize_numeric_values(pdop_values)
        histogram = stats.get("histogram") or {}
        self.assertTrue(histogram.get("edges"))
        self.assertTrue(histogram.get("counts"))

        outliers = stats.get("outlier_bounds") or {}
        self.assertIn("lower", outliers)
        self.assertIn("upper", outliers)


class NumericProfilingIntegrationTest(unittest.TestCase):
    def test_profiles_present_when_visualizations_are_disabled(self) -> None:
        if load_config.__globals__.get("yaml") is None:
            self.skipTest("PyYAML is required for this integration test")

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "metrics.csv")
            with open(csv_path, "w", encoding="utf-8") as fp:
                fp.write("id,value\n")
                for idx, value in enumerate([1, 2, 3, 4, 100], start=1):
                    fp.write(f"{idx},{value}\n")

            metadata_path = os.path.join(tmpdir, "metadata.txt")
            with open(metadata_path, "w", encoding="utf-8") as fp:
                fp.write(
                    textwrap.dedent(
                        """
                        File: metrics.csv
                        id: Record identifier
                        value: Observed measurement
                        """
                    ).strip()
                )

            config_path = os.path.join(tmpdir, "config.yaml")
            with open(config_path, "w", encoding="utf-8") as fp:
                fp.write(
                    textwrap.dedent(
                        """
                        rules:
                          - patterns: [".*metrics\\.csv$"]
                            metadata_path: "{metadata}"
                            metadata_targets:
                              - metrics.csv
                            required_cols: [id, value]
                            numeric_cols: [value]
                            primary_key: id
                        """
                    ).format(metadata=metadata_path.replace("\\", "/"))
                )

            good_out = os.path.join(tmpdir, "good", "output")
            bad_out = os.path.join(tmpdir, "bad", "issues")
            dq_out = os.path.join(tmpdir, "dq")
            input_pattern = os.path.join(tmpdir, "*.csv")

            original_limit = os.environ.get(_VISUALIZATION_TOTAL_LIMIT_ENV)
            os.environ[_VISUALIZATION_TOTAL_LIMIT_ENV] = "1B"
            try:
                _run_without_beam(
                    input_pattern,
                    good_out,
                    bad_out,
                    dq_out,
                    config_path,
                )
            finally:
                if original_limit is None:
                    os.environ.pop(_VISUALIZATION_TOTAL_LIMIT_ENV, None)
                else:
                    os.environ[_VISUALIZATION_TOTAL_LIMIT_ENV] = original_limit

            report_path = _single_shard_path(os.path.join(dq_out, "quality_report"), ".json")
            with open(report_path, "r", encoding="utf-8") as fp:
                report = json.load(fp)

            self.assertIn("per_file", report)
            self.assertEqual(len(report["per_file"]), 1)
            numeric_columns = report["per_file"][0].get("numeric_columns") or {}
            self.assertIn("value", numeric_columns)
            value_profile = numeric_columns["value"]
            histogram = value_profile.get("distribution") or {}
            self.assertTrue(histogram.get("edges"))
            self.assertTrue(histogram.get("counts"))
            outliers = value_profile.get("outliers") or {}
            self.assertIn("lower_fence", outliers)
            self.assertIn("upper_fence", outliers)
            viz_note = (report.get("visualizations") or {}).get("note", "")
            self.assertIn("Visualizations skipped", viz_note)


if __name__ == "__main__":
    unittest.main()
