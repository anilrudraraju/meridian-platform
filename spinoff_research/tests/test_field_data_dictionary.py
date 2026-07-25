import unittest

from spinoff_research.field_data_dictionary import (
    FIELD_DEFINITIONS,
    FIELD_BY_KEY,
    ExtractionCategory,
    fields_by_category,
    fields_by_extraction_category,
    get_field,
)


class TestFieldDataDictionary(unittest.TestCase):
    def test_all_47_fields_present(self):
        self.assertEqual(len(FIELD_DEFINITIONS), 47)

    def test_field_keys_are_unique(self):
        keys = [f.field_key for f in FIELD_DEFINITIONS]
        self.assertEqual(len(keys), len(set(keys)))

    def test_field_keys_are_snake_case(self):
        for f in FIELD_DEFINITIONS:
            self.assertRegex(f.field_key, r"^[a-z][a-z0-9_]*$",
                              f"field_key '{f.field_key}' is not snake_case")

    def test_display_names_preserve_original_spreadsheet_wording(self):
        ceo_tenure = get_field("ceo_tenure_at_parent")
        # internal key/description correct the "tenor" typo, but the
        # spreadsheet column name must stay exactly as Rich wrote it for
        # import/export compatibility.
        self.assertIn("tenure", ceo_tenure.description.lower())
        self.assertEqual(ceo_tenure.spreadsheet_column_name, "CEO tenure at the parent")

    def test_get_field_returns_definition(self):
        f = get_field("distribution_ratio")
        self.assertEqual(f.field_key, "distribution_ratio")

    def test_get_field_raises_on_unknown_key(self):
        with self.assertRaises(KeyError):
            get_field("does_not_exist")

    def test_fields_by_category_groups_correctly(self):
        mgmt_fields = fields_by_category("management_and_incentives")
        self.assertEqual(len(mgmt_fields), 5)
        keys = {f.field_key for f in mgmt_fields}
        self.assertIn("ceo_came_from_parent", keys)
        self.assertIn("tsr_based_incentive_plan", keys)

    def test_ebitda_derived_fields_are_not_structured(self):
        """
        Regression test for the brief's explicit rule: do not label EBITDA
        margin, ROIC, or debt-to-EBITDA as structured/deterministic just
        because they look numeric — none of these are standardized XBRL
        concepts.
        """
        for key in ("spinoff_ebitda_margin", "roic"):
            f = get_field(key)
            self.assertEqual(
                f.extraction_category, ExtractionCategory.AI_ASSISTED,
                f"{key} must not be classified as structured",
            )
        for key in ("spinoff_debt_to_ebitda", "dis_synergy_pct_of_prior_year_sales"):
            f = get_field(key)
            self.assertEqual(
                f.extraction_category, ExtractionCategory.CALCULATED,
                f"{key} must not be classified as purely structured (its inputs include an AI-assisted value)",
            )

    def test_calculated_fields_declare_required_inputs(self):
        for f in fields_by_extraction_category(ExtractionCategory.CALCULATED):
            self.assertTrue(f.required_inputs, f"{f.field_key} is calculated but declares no required_inputs")
            self.assertIsNotNone(f.calculation_formula, f"{f.field_key} is calculated but has no formula")

    def test_calculated_field_inputs_reference_real_field_keys(self):
        for f in fields_by_extraction_category(ExtractionCategory.CALCULATED):
            for input_key in f.required_inputs:
                self.assertIn(input_key, FIELD_BY_KEY,
                               f"{f.field_key} references unknown input field '{input_key}'")

    def test_scheduled_fields_have_observation_window_or_are_manual_only(self):
        scheduled = fields_by_extraction_category(ExtractionCategory.SCHEDULED_MANUAL)
        self.assertGreaterEqual(len(scheduled), 6)
        # windowed fields (things that must wait) must declare how long
        windowed_keys = {"dividend_initiated_within_12mo", "insider_buying_within_3mo",
                          "cluster_insider_buying_within_3mo", "insider_buyer_count", "tsr"}
        for f in scheduled:
            if f.field_key in windowed_keys:
                self.assertIsNotNone(
                    f.required_observation_window_days,
                    f"{f.field_key} should declare required_observation_window_days",
                )

    def test_tsr_is_not_a_bare_percentage_field(self):
        """Brief explicitly warns against implementing TSR without a methodology."""
        tsr = get_field("tsr")
        self.assertEqual(tsr.extraction_category, ExtractionCategory.SCHEDULED_MANUAL)
        self.assertIn("methodology", tsr.notes.lower())

    def test_every_field_has_a_description(self):
        for f in FIELD_DEFINITIONS:
            self.assertTrue(f.description and f.description.strip(),
                             f"{f.field_key} has no description")


if __name__ == "__main__":
    unittest.main()
