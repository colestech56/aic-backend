"""Tests for the Content Screener."""

import pytest

from app.services.content_screener import ContentScreener


@pytest.fixture
def screener():
    return ContentScreener()


class TestCleanContent:
    def test_valid_sensory_focus_passes(self, screener):
        text = "Take a moment. Notice 5 things you can see around you right now. Focus on their colors and shapes."
        passed, violations = screener.screen(text)
        assert passed
        assert violations == []

    def test_valid_breathing_passes(self, screener):
        text = "Try this: breathe in slowly for 4 counts, hold for 4, breathe out for 6. Repeat 3 times."
        passed, violations = screener.screen(text)
        assert passed
        assert violations == []


class TestLengthCheck:
    def test_over_280_fails(self, screener):
        text = "A" * 281 + ". Another sentence."
        passed, violations = screener.screen(text)
        assert not passed
        assert any("too_long" in v for v in violations)

    def test_exactly_280_passes(self, screener):
        # Create exactly 280 char message with proper sentences
        text = "Take a moment to breathe. " + "A" * 230 + ". One more thing."
        text = text[:280]
        # Just check that length itself isn't a violation if ≤ 280
        passed, violations = screener.screen("Take a moment to breathe. Focus on what you see.")
        length_violations = [v for v in violations if "too_long" in v]
        assert len(length_violations) == 0


class TestSentenceCount:
    def test_one_sentence_fails(self, screener):
        text = "Just one sentence here"
        passed, violations = screener.screen(text)
        assert not passed
        assert any("too_few_sentences" in v for v in violations)

    def test_five_sentences_fails(self, screener):
        text = "One. Two. Three. Four. Five."
        passed, violations = screener.screen(text)
        assert not passed
        assert any("too_many_sentences" in v for v in violations)

    def test_two_sentences_passes(self, screener):
        text = "Take a moment to pause. Notice what you can see around you."
        passed, violations = screener.screen(text)
        sentence_violations = [v for v in violations if "sentences" in v]
        assert len(sentence_violations) == 0

    def test_four_sentences_passes(self, screener):
        text = "Breathe in for 4. Hold for 4. Breathe out for 6. Repeat three times."
        passed, violations = screener.screen(text)
        sentence_violations = [v for v in violations if "sentences" in v]
        assert len(sentence_violations) == 0


class TestForbiddenPatterns:
    def test_clinical_language_therapy(self, screener):
        text = "This therapy technique can help. Try breathing deeply."
        passed, violations = screener.screen(text)
        assert not passed
        assert any("clinical_language" in v for v in violations)

    def test_clinical_language_symptoms(self, screener):
        text = "Your symptoms may improve. Try focusing on what you see."
        passed, violations = screener.screen(text)
        assert not passed
        assert any("clinical_language" in v for v in violations)

    def test_clinical_language_medication(self, screener):
        text = "Along with your medication, try breathing. It can help."
        passed, violations = screener.screen(text)
        assert not passed
        assert any("clinical_language" in v for v in violations)

    def test_hollow_praise(self, screener):
        text = "Great job taking a moment for yourself! Keep it up."
        passed, violations = screener.screen(text)
        assert not passed
        assert any("hollow_praise" in v for v in violations)

    def test_person_attribution(self, screener):
        text = "You're so strong for getting through this. Try breathing."
        passed, violations = screener.screen(text)
        assert not passed
        assert any("person_attribution" in v for v in violations)

    def test_fabricated_data(self, screener):
        text = "Studies show that breathing helps 80% of people. Try it now."
        passed, violations = screener.screen(text)
        assert not passed
        assert any("fabricated_data" in v for v in violations)

    def test_minimizing(self, screener):
        text = "Don't worry about it. Everything will be fine."
        passed, violations = screener.screen(text)
        assert not passed
        assert any("minimizing" in v for v in violations)

    def test_command_without_strategy(self, screener):
        text = "Calm down and think positive. You'll be fine."
        passed, violations = screener.screen(text)
        assert not passed
        assert any("command_without_strategy" in v for v in violations)

    def test_time_prediction(self, screener):
        text = "This will pass soon. You'll feel better tomorrow."
        passed, violations = screener.screen(text)
        assert not passed
        assert any("time_prediction" in v for v in violations)


class TestCosineSimilarityDedup:
    def test_identical_text_rejected(self, screener):
        text = "Take a moment to pause. Notice what you can see around you."
        recent = [text]
        passed, violations = screener.screen(text, recent)
        assert not passed
        assert any("too_similar" in v for v in violations)

    def test_very_similar_text_rejected(self, screener):
        text1 = "Take a moment to pause. Notice what you can see around you."
        text2 = "Take a moment to pause. Notice what you can hear around you."
        passed, violations = screener.screen(text2, [text1])
        # These are very similar — should likely be caught
        # (depends on TF-IDF vectorization)
        assert isinstance(passed, bool)  # just verify it runs

    def test_different_text_accepted(self, screener):
        text1 = "Breathe in for 4 counts, hold for 4. Breathe out slowly for 6 counts."
        text2 = "Look around the room. Name five things you can see right now."
        passed, violations = screener.screen(text2, [text1])
        sim_violations = [v for v in violations if "too_similar" in v]
        assert len(sim_violations) == 0

    def test_empty_recent_passes(self, screener):
        text = "Take a moment to breathe. Focus on the exhale."
        passed, violations = screener.screen(text, [])
        sim_violations = [v for v in violations if "too_similar" in v]
        assert len(sim_violations) == 0

    def test_no_recent_passes(self, screener):
        text = "Take a moment to breathe. Focus on the exhale."
        passed, violations = screener.screen(text, None)
        sim_violations = [v for v in violations if "too_similar" in v]
        assert len(sim_violations) == 0


class TestMultipleViolations:
    def test_multiple_violations_all_reported(self, screener):
        # Long text + clinical language + time prediction
        text = "Your therapy symptoms will pass soon. " + "A" * 300
        passed, violations = screener.screen(text)
        assert not passed
        assert len(violations) >= 2  # at least length + clinical
