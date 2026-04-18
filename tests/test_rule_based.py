from src.models.rule_based import NegationAwareRuleModel, RuleBasedConfig


def test_negation_aware_rule_based():
    cfg = RuleBasedConfig(lexicons={"behavior_tobacco": ["smoker"]}, negation_cues=["denies", "no"], window_tokens=3)
    model = NegationAwareRuleModel(cfg, "behavior_tobacco")
    assert model.predict_one("Patient is a smoker") == 1
    assert model.predict_one("Patient denies smoker status") == 0
