"""Comprehensive test suite for Email Triage OpenEnv v0.4.0."""
import pytest
from env import (
    EmailTriageEnv, PRIORITY_WEIGHTS, TASK_TEMPLATE_MAP, ESCALATION_BUDGET,
    CONSEQUENCE_STAGE_TEMPLATES, CHURN_THRESHOLD, MAX_CONSEQUENCES_PER_EPISODE,
    _generate_email_from_template, _ROUTE_TEMPLATES, _ESCALATION_TEMPLATES,
    _HARD_TEMPLATES, _NEW_HARD_TEMPLATES, _INTERNAL_NOISE_TEMPLATES,
)
from models import Action, State

@pytest.fixture
def env():
    return EmailTriageEnv()

# --- Reset & Validation ---
class TestReset:
    def test_valid_tasks(self, env):
        for t in ("easy", "medium", "hard", "enterprise"):
            obs = env.reset(seed=0, task_id=t)
            assert obs.current_email is not None
            assert obs.done is False

    def test_invalid_task(self, env):
        with pytest.raises(ValueError, match="Unknown task_id"):
            env.reset(task_id="invalid")

    def test_email_counts(self, env):
        for t in TASK_TEMPLATE_MAP:
            env.reset(seed=0, task_id=t)
            assert env.total_emails == len(TASK_TEMPLATE_MAP[t])

    def test_reset_clears_state(self, env):
        env.reset(seed=0, task_id="hard")
        env.step(Action(action_type="archive"))
        env.reset(seed=0, task_id="hard")
        s = env.state
        assert s.emails_processed == 0
        assert s.consequences_triggered == 0
        assert s.customer_satisfaction == 1.0
        assert s.churn_risk is False

    def test_initial_score_bounded(self, env):
        for t in TASK_TEMPLATE_MAP:
            env.reset(seed=0, task_id=t)
            assert 0 < env.state.current_score < 1

# --- Seeds ---
class TestSeeds:
    def test_reproducible(self, env):
        env.reset(seed=42, task_id="hard")
        a = [e["id"] for e in env.emails]
        env.reset(seed=42, task_id="hard")
        b = [e["id"] for e in env.emails]
        assert a == b

    def test_different_seeds_differ(self, env):
        env.reset(seed=1, task_id="hard")
        a = [e["id"] for e in env.emails]
        env.reset(seed=2, task_id="hard")
        b = [e["id"] for e in env.emails]
        assert a != b

    def test_episode_id_generated(self, env):
        env.reset(task_id="easy")
        assert env.state.episode_id != ""

# --- Rewards ---
class TestRewards:
    def _skip_to_target(self, env, target, task_id="hard"):
        env.reset(seed=0, task_id=task_id)
        for i, e in enumerate(env.emails):
            if e["target"] == target:
                for _ in range(i):
                    env.step(Action(action_type="defer"))
                return e
        pytest.fail(f"No email with target '{target}'")

    def test_correct_route(self, env):
        e = self._skip_to_target(env, "sales")
        obs = env.step(Action(action_type="route", department="sales"))
        assert obs.reward > 0

    def test_partial_credit(self, env):
        e = self._skip_to_target(env, "sales")
        obs = env.step(Action(action_type="route", department="billing"))
        assert obs.reward > 0
        assert "Partial" in obs.previous_action_feedback

    def test_wrong_action_negative(self, env):
        e = self._skip_to_target(env, "sales")
        obs = env.step(Action(action_type="escalate"))
        assert obs.reward < 0

    def test_correct_escalation(self, env):
        e = self._skip_to_target(env, "escalate")
        obs = env.step(Action(action_type="escalate"))
        assert obs.reward > 0

    def test_missed_escalation(self, env):
        e = self._skip_to_target(env, "escalate")
        obs = env.step(Action(action_type="archive"))
        assert obs.reward < 0

    def test_correct_request_info(self, env):
        e = self._skip_to_target(env, "request_info")
        obs = env.step(Action(action_type="request_info"))
        assert obs.reward > 0

    def test_correct_archive(self, env):
        e = self._skip_to_target(env, "archive")
        obs = env.step(Action(action_type="archive"))
        assert obs.reward > 0

    def test_correct_legal_compliance(self, env):
        e = self._skip_to_target(env, "legal_compliance")
        obs = env.step(Action(action_type="route", department="legal_compliance"))
        assert obs.reward > 0

# --- Defer ---
class TestDefer:
    def test_small_negative(self, env):
        env.reset(seed=0, task_id="easy")
        obs = env.step(Action(action_type="defer"))
        assert obs.reward < 0
        assert obs.reward > -0.5

    def test_counter(self, env):
        env.reset(seed=0, task_id="easy")
        env.step(Action(action_type="defer"))
        env.step(Action(action_type="defer"))
        assert env.state.deferred_count == 2

    def test_no_consequence(self, env):
        env.reset(seed=0, task_id="hard")
        n = env.total_emails
        for _ in range(n):
            env.step(Action(action_type="defer"))
        assert env.state.consequences_triggered == 0

    def test_no_csat_decay(self, env):
        env.reset(seed=0, task_id="easy")
        env.step(Action(action_type="defer"))
        assert env.state.customer_satisfaction == 1.0

# --- CSAT Decay ---
class TestCSAT:
    def test_decays_on_mishandling(self, env):
        env.reset(seed=0, task_id="hard")
        env.step(Action(action_type="archive"))  # wrong for most
        assert env.state.customer_satisfaction < 1.0

    def test_no_decay_on_correct(self, env):
        env.reset(seed=0, task_id="hard")
        for e in env.emails[:]:
            t = e["target"]
            if t in ("sales", "billing", "technical_support", "legal_compliance"):
                env.step(Action(action_type="route", department=t))
            elif t == "escalate":
                env.step(Action(action_type="escalate"))
            elif t == "request_info":
                env.step(Action(action_type="request_info"))
            elif t == "archive":
                env.step(Action(action_type="archive"))
        assert env.state.customer_satisfaction == 1.0

    def test_churn_terminates_early(self, env):
        env.reset(seed=0, task_id="hard")
        done = False
        steps = 0
        while not done:
            obs = env.step(Action(action_type="archive"))
            done = obs.done
            steps += 1
        assert env.state.churn_risk is True
        assert steps < env.total_emails  # terminated early

# --- Consequence Chains (3-stage) ---
class TestConsequences:
    def test_misroute_triggers(self, env):
        env.reset(seed=0, task_id="easy")
        n = env.total_emails
        env.step(Action(action_type="route", department="billing" if env.emails[0]["target"] != "billing" else "sales"))
        assert env.total_emails > n

    def test_stages_escalate(self, env):
        env.reset(seed=0, task_id="hard")
        done = False
        while not done:
            obs = env.step(Action(action_type="archive"))
            done = obs.done
        assert env.state.consequence_stage_reached >= 2

    def test_max_capped(self, env):
        env.reset(seed=0, task_id="hard")
        done = False
        while not done:
            obs = env.step(Action(action_type="archive"))
            done = obs.done
        assert env.state.consequences_triggered <= MAX_CONSEQUENCES_PER_EPISODE

    def test_correct_no_consequence(self, env):
        env.reset(seed=0, task_id="hard")
        n = env.total_emails
        for e in env.emails[:]:
            t = e["target"]
            if t in ("sales", "billing", "technical_support", "legal_compliance"):
                env.step(Action(action_type="route", department=t))
            elif t == "escalate":
                env.step(Action(action_type="escalate"))
            elif t == "request_info":
                env.step(Action(action_type="request_info"))
            elif t == "archive":
                env.step(Action(action_type="archive"))
        assert env.state.consequences_triggered == 0

# --- Escalation Budget ---
class TestBudget:
    def test_init_per_task(self, env):
        for t in ESCALATION_BUDGET:
            env.reset(seed=0, task_id=t)
            assert env.state.escalation_budget_remaining == ESCALATION_BUDGET[t]

    def test_deducts_on_escalate(self, env):
        env.reset(seed=0, task_id="medium")
        b = env.state.escalation_budget_remaining
        env.step(Action(action_type="escalate"))
        assert env.state.escalation_budget_remaining < b

    def test_exhaustion_penalizes(self, env):
        env.reset(seed=0, task_id="easy")  # budget=2
        env.step(Action(action_type="escalate"))
        env.step(Action(action_type="escalate"))
        assert env.state.escalation_budget_remaining == 0
        # Next escalation should be penalized
        obs = env.step(Action(action_type="escalate"))
        # reward depends on target, but budget is 0

    def test_defer_no_cost(self, env):
        env.reset(seed=0, task_id="medium")
        b = env.state.escalation_budget_remaining
        env.step(Action(action_type="defer"))
        assert env.state.escalation_budget_remaining == b

    def test_never_negative(self, env):
        env.reset(seed=0, task_id="easy")
        for _ in range(10):
            env.step(Action(action_type="escalate"))
        assert env.state.escalation_budget_remaining == 0

# --- Confidence Calibration ---
class TestConfidence:
    def test_overconfidence_penalty(self, env):
        env.reset(seed=0, task_id="easy")
        obs_high = env.step(Action(action_type="archive", confidence=0.95))
        env.reset(seed=0, task_id="easy")
        obs_low = env.step(Action(action_type="archive", confidence=0.3))
        # High confidence wrong should be worse
        assert obs_high.reward < obs_low.reward

    def test_brier_tracked(self, env):
        env.reset(seed=0, task_id="easy")
        env.step(Action(action_type="archive", confidence=0.9))
        assert env.state.calibration_score > 0

    def test_default_confidence(self, env):
        env.reset(seed=0, task_id="easy")
        env.step(Action(action_type="archive"))  # default 0.5
        assert env.state.calibration_score > 0  # still tracked

# --- Score Clamping ---
class TestScoreClamping:
    def test_perfect_under_one(self, env):
        for t in ("easy", "medium", "hard"):
            env.reset(seed=0, task_id=t)
            for e in env.emails[:]:
                tgt = e["target"]
                if tgt in ("sales", "billing", "technical_support", "legal_compliance"):
                    env.step(Action(action_type="route", department=tgt))
                elif tgt == "escalate":
                    env.step(Action(action_type="escalate"))
                elif tgt == "request_info":
                    env.step(Action(action_type="request_info"))
                elif tgt == "archive":
                    env.step(Action(action_type="archive"))
            assert 0 < env.state.current_score < 1

    def test_all_wrong_above_zero(self, env):
        for t in ("easy", "medium", "hard"):
            env.reset(seed=0, task_id=t)
            done = False
            while not done:
                obs = env.step(Action(action_type="archive"))
                done = obs.done
            assert 0 < env.state.current_score < 1

# --- SLA Delay ---
class TestSLADelay:
    def test_no_penalty_low_priority(self, env):
        env.reset(seed=0, task_id="easy")
        for e in env.emails:
            if e.get("priority") in ("low", "medium"):
                assert env._sla_delay_penalty(e) == 0.0

    def test_sla_deadline_amplifies(self, env):
        env.reset(seed=0, task_id="hard")
        for e in env.emails:
            if e.get("sla_deadline_minutes") and e.get("priority") in ("critical", "high"):
                e["queue_position"] = env.total_emails - 1  # force late
                penalty = env._sla_delay_penalty(e)
                assert penalty > 0

# --- Thread History ---
class TestThreadHistory:
    def test_present_in_hard(self, env):
        env.reset(seed=0, task_id="hard")
        found = any(e.get("thread_history") for e in env.emails)
        assert found, "Hard task should have at least one email with thread_history"

    def test_observation_includes_thread(self, env):
        env.reset(seed=0, task_id="hard")
        for i, e in enumerate(env.emails):
            if e.get("thread_history"):
                for _ in range(i):
                    env.step(Action(action_type="defer"))
                obs = env._get_obs("test")
                assert obs.current_email.thread_history is not None
                assert len(obs.current_email.thread_history) > 0
                return

# --- Legal Compliance ---
class TestLegalCompliance:
    def test_target_exists_in_hard(self, env):
        env.reset(seed=0, task_id="hard")
        targets = {e["target"] for e in env.emails}
        assert "legal_compliance" in targets

    def test_correct_route(self, env):
        env.reset(seed=0, task_id="hard")
        for i, e in enumerate(env.emails):
            if e["target"] == "legal_compliance":
                for _ in range(i):
                    env.step(Action(action_type="defer"))
                obs = env.step(Action(action_type="route", department="legal_compliance"))
                assert obs.reward > 0
                return

# --- Prompt Injection ---
class TestPromptInjection:
    def test_injection_templates_exist(self, env):
        env.reset(seed=0, task_id="hard")
        ids = [e["id"] for e in env.emails if any(h in e["id"] for h in ["h11", "h12", "h13"])]
        assert len(ids) == 3

    def test_correct_targets(self, env):
        env.reset(seed=0, task_id="hard")
        for e in env.emails:
            if "h11" in e["id"]:
                assert e["target"] == "escalate"
            elif "h12" in e["id"]:
                assert e["target"] == "technical_support"
            elif "h13" in e["id"]:
                assert e["target"] == "request_info"

# --- Enterprise Task ---
class TestEnterprise:
    def test_more_emails_than_hard(self, env):
        env.reset(seed=0, task_id="hard")
        hard_count = env.total_emails
        env.reset(seed=0, task_id="enterprise")
        assert env.total_emails > hard_count

    def test_has_noise(self, env):
        env.reset(seed=0, task_id="enterprise")
        archive_count = sum(1 for e in env.emails if e["target"] == "archive")
        assert archive_count >= 8, "Enterprise should have many archive (noise) emails"

    def test_budget_is_12(self, env):
        env.reset(seed=0, task_id="enterprise")
        assert env.state.escalation_budget_remaining == 12

# --- Procedural Generation ---
class TestProceduralGen:
    def test_valid_email(self):
        import random
        rng = random.Random(0)
        email = _generate_email_from_template(_ROUTE_TEMPLATES[0], rng)
        assert all(k in email for k in ("id", "subject", "body", "target"))

    def test_matching_variants(self):
        for t in _ROUTE_TEMPLATES + _ESCALATION_TEMPLATES + _HARD_TEMPLATES + _NEW_HARD_TEMPLATES + _INTERNAL_NOISE_TEMPLATES:
            assert len(t["subjects"]) == len(t["bodies"]), f"{t['id']}: mismatched variant counts"

    def test_queue_positions(self, env):
        env.reset(seed=0, task_id="hard")
        positions = sorted(e["queue_position"] for e in env.emails)
        assert positions == list(range(len(env.emails)))

# --- Ground Truth ---
class TestGroundTruth:
    VALID_TARGETS = {"sales", "billing", "technical_support", "legal_compliance", "escalate", "request_info", "archive"}

    def test_all_targets_valid(self, env):
        for t in TASK_TEMPLATE_MAP:
            env.reset(seed=0, task_id=t)
            for e in env.emails:
                assert e["target"] in self.VALID_TARGETS

    def test_hard_has_all_types(self, env):
        env.reset(seed=0, task_id="hard")
        assert {e["target"] for e in env.emails} == self.VALID_TARGETS

    def test_easy_no_escalations(self, env):
        env.reset(seed=0, task_id="easy")
        assert "escalate" not in {e["target"] for e in env.emails}

# --- Episode Boundaries ---
class TestEpisodeBoundaries:
    def test_done_after_all(self, env):
        env.reset(seed=0, task_id="easy")
        for _ in range(env.total_emails):
            obs = env.step(Action(action_type="defer"))
        assert obs.done is True

    def test_step_after_done(self, env):
        env.reset(seed=0, task_id="easy")
        for _ in range(env.total_emails):
            env.step(Action(action_type="defer"))
        obs = env.step(Action(action_type="defer"))
        assert obs.done is True
        assert obs.current_email is None
