import unittest

from dq_local_beam import DEFAULT_RULE, RowCtx, validate_row_against_rule


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


if __name__ == "__main__":
    unittest.main()
