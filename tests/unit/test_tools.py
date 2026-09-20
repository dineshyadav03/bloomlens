"""The three agent tools in src/tools.py, called directly (no agent, no LLM)."""

import pytest

from src import tools


class TestMentionsUnnegatedDamage:
    @pytest.mark.parametrize(
        "note",
        [
            "Some browning on the outer petals.",
            "Visible wilting near the stem.",
            "There is slight discoloration and a blemish.",
            "Petals droop noticeably.",
        ],
    )
    def test_plain_damage_is_flagged(self, note):
        assert tools._mentions_unnegated_damage(note) is True

    @pytest.mark.parametrize(
        "note",
        [
            "Completely free of wilting, discoloration, or pest damage.",
            "No visible blemishes.",
            "No signs of browning or damage.",
            "Without any wilting.",
            "The bloom is fresh and healthy.",
            "",
        ],
    )
    def test_negated_or_absent_damage_is_not_flagged(self, note):
        """Regression for Milestone 3: a naive keyword match flagged 'free of wilting'."""
        assert tools._mentions_unnegated_damage(note) is False

    def test_each_sentence_is_judged_on_its_own(self):
        assert tools._mentions_unnegated_damage("No wilting. Some browning at the tips.") is True
        assert tools._mentions_unnegated_damage("Some browning at the tips. No wilting.") is True

    def test_a_negation_cue_must_be_a_whole_word(self):
        """'cannot' contains the substring 'not ', 'casino ' contains 'no ' -- neither
        negates anything. Regression for a substring-matching bug found by these tests."""
        assert tools._mentions_unnegated_damage("Visible browning on the petals; cannot rule out wilting.") is True
        assert tools._mentions_unnegated_damage("Browning is visible, as in a casino carpet.") is True


class TestAssessQuality:
    def call(self, grade, note):
        return tools.assess_quality.invoke({"quality_grade": grade, "quality_note": note})

    def test_grade_is_normalized(self):
        assert self.call(" b ", "fine")["quality_grade"] == "B"

    def test_invalid_grade_is_reported_not_raised(self):
        assert "error" in self.call("D", "fine")

    def test_grade_a_with_visible_damage_gets_a_consistency_warning(self):
        assert self.call("A", "Some browning on the edges.")["consistency_warning"]

    def test_grade_a_with_negated_damage_has_no_warning(self):
        assert self.call("A", "Completely free of wilting or damage.")["consistency_warning"] is None

    def test_only_grade_a_is_cross_checked(self):
        assert self.call("B", "Some browning on the edges.")["consistency_warning"] is None


class TestLookupTaxonomy:
    def test_known_species_returns_taxonomy_and_description(self, species_list):
        rose = next(s for s in species_list if s["common_name"] == "Rose")
        text = tools.lookup_taxonomy.invoke({"species": "Rose"})
        assert rose["taxonomy_string"] in text and rose["description"] in text

    def test_unknown_species_is_a_clear_message(self):
        text = tools.lookup_taxonomy.invoke({"species": "Unicorn flower"})
        assert "not in BloomLens's curated species list" in text


class TestCheckPrice:
    def test_delegates_to_the_pricing_module(self, prices_csv):
        result = tools.check_price.invoke({"species": "Rose", "grade": "A"})
        assert result["price_per_stem"] == pytest.approx(0.55)
        assert result["simulated"] is True
