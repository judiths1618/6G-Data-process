import os
import sys
import unittest


sys.path.append(os.path.dirname(os.path.dirname(__file__)))
print(sys.path)


from dq_local_beam import (
    DEFAULT_RULE,
    RowCtx,
    evaluate_metadata_prerequisites,
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


if __name__ == "__main__":
    unittest.main()
