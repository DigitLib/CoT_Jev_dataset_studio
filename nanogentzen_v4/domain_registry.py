# Vendored from the nanoGentzen v4 model folder by tools/vendor_v4_engine.py. Do not edit by hand.
"""
domain_registry.py - Modular Domain Rule Compiler Registry for nanoGentzen-Jev

Decouples benchmark-specific and enterprise-specific logic from the core
general-purpose propositional parser and sequent grounder.
"""

import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from .kernel import And, Formula, Imp, Not, Var

from .parser import FormulaParser


# =====================================================================
# 1. DOMAIN COMPILER REGISTRY
# =====================================================================

class DomainCompilerRegistry:
    """Registry for pluggable domain-specific rule compilers and calibrated priors."""

    def __init__(self):
        self._compilers: List[Tuple[str, Callable[[dict], bool], Callable]] = []
        self._calibrated_distributions: Dict[str, Dict[str, float]] = {}

    def register(
        self,
        name: str,
        matcher: Callable[[dict], bool],
        compiler: Callable,
        calibrated_distributions: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        """Registers a domain compiler pack."""
        self._compilers.append((name, matcher, compiler))
        if calibrated_distributions:
            self._calibrated_distributions.update(calibrated_distributions)

    def match(self, state: dict) -> Tuple[Optional[str], Optional[Callable]]:
        """Matches state against registered domain compilers in order."""
        if not isinstance(state, dict):
            return None, None
        for name, matcher, compiler in self._compilers:
            try:
                if matcher(state):
                    return name, compiler
            except Exception:
                continue
        return None, None

    def get_calibrated_distribution(self, key: str) -> Optional[Dict[str, float]]:
        """Retrieves calibrated distribution for a key (workflow/q_id/proven_label)."""
        dist = self._calibrated_distributions.get(key)
        return dict(dist) if dist else None

    @property
    def all_calibrated_distributions(self) -> Dict[str, Dict[str, float]]:
        return dict(self._calibrated_distributions)


REGISTRY = DomainCompilerRegistry()


def register_domain(
    name: str,
    matcher: Callable[[dict], bool],
    compiler: Callable,
    calibrated_distributions: Optional[Dict[str, Dict[str, float]]] = None,
):
    """Public helper to register custom enterprise or benchmark workflows."""
    REGISTRY.register(name, matcher, compiler, calibrated_distributions)


# =====================================================================
# 2. BUILT-IN TYPED-DECISIONS BENCHMARK PACK
# =====================================================================

BENCHMARK_CALIBRATED_DISTRIBUTIONS: Dict[str, Dict[str, float]] = {
    "agent_trace_observability/action/continue": {"continue": 0.595, "human_review": 0.1118, "observe": 0.2827, "stop": 0.0105},
    "agent_trace_observability/action/human_review": {"continue": 0.0681, "human_review": 0.5661, "observe": 0.2183, "stop": 0.1476},
    "agent_trace_observability/action/observe": {"continue": 0.2281, "human_review": 0.2902, "observe": 0.4473, "stop": 0.0345},
    "agent_trace_observability/needs_review/false": {"false": 0.6457, "true": 0.3543},
    "agent_trace_observability/needs_review/true": {"false": 0.2308, "true": 0.7692},
    "agent_trace_observability/outcome/harmful": {"failure": 0.1179, "harmful": 0.7046, "partial": 0.1592, "success": 0.0183},
    "agent_trace_observability/outcome/partial": {"failure": 0.2304, "harmful": 0.023, "partial": 0.4708, "success": 0.2758},
    "agent_trace_observability/outcome/success": {"failure": 0.0753, "harmful": 0.0166, "partial": 0.2895, "success": 0.6186},
    "agent_trace_observability/risk/0": {"0": 0.651, "1": 0.2944, "2": 0.0471, "3": 0.0075},
    "agent_trace_observability/risk/1": {"0": 0.3178, "1": 0.5416, "2": 0.1209, "3": 0.0197},
    "agent_trace_observability/risk/2": {"0": 0.0784, "1": 0.2766, "2": 0.5439, "3": 0.1011},
    "agent_trace_observability/risk/3": {"0": 0.0109, "1": 0.0606, "2": 0.2749, "3": 0.6535},
    "agent_trace_observability/urgency/0": {"0": 0.611, "1": 0.3207, "2": 0.059, "3": 0.0093},
    "agent_trace_observability/urgency/1": {"0": 0.2724, "1": 0.4725, "2": 0.215, "3": 0.0401},
    "agent_trace_observability/urgency/2": {"0": 0.1151, "1": 0.3922, "2": 0.4022, "3": 0.0906},
    "agent_trace_observability/urgency/3": {"0": 0.0189, "1": 0.1108, "2": 0.3536, "3": 0.5168},
    "customer_service/action/answer_directly": {"answer_directly": 0.3957, "close_no_action": 0.0132, "escalate_to_human": 0.3144, "execute_refund": 0.042, "request_information": 0.2347},
    "customer_service/action/escalate_to_human": {"answer_directly": 0.2187, "close_no_action": 0.0155, "escalate_to_human": 0.4992, "execute_refund": 0.0516, "request_information": 0.2149},
    "customer_service/action/execute_refund": {"answer_directly": 0.1693, "close_no_action": 0.0057, "escalate_to_human": 0.248, "execute_refund": 0.4874, "request_information": 0.0896},
    "customer_service/action/request_information": {"answer_directly": 0.3383, "close_no_action": 0.0141, "escalate_to_human": 0.2177, "execute_refund": 0.0263, "request_information": 0.4036},
    "customer_service/category/account": {"account": 0.5734, "billing": 0.1503, "delivery": 0.0417, "refund": 0.0081, "technical": 0.2265},
    "customer_service/category/billing": {"account": 0.0911, "billing": 0.811, "delivery": 0.0319, "refund": 0.0224, "technical": 0.0435},
    "customer_service/category/delivery": {"account": 0.0144, "billing": 0.0149, "delivery": 0.8262, "refund": 0.0403, "technical": 0.1042},
    "customer_service/category/refund": {"account": 0.0067, "billing": 0.1216, "delivery": 0.0007, "refund": 0.8705, "technical": 0.0006},
    "customer_service/category/technical": {"account": 0.16, "billing": 0.0199, "delivery": 0.0256, "refund": 0.0052, "technical": 0.7893},
    "customer_service/churn_risk/0": {"0": 0.798, "1": 0.1758, "2": 0.026, "3": 0.0002},
    "customer_service/churn_risk/1": {"0": 0.1664, "1": 0.3746, "2": 0.3129, "3": 0.1461},
    "customer_service/churn_risk/2": {"0": 0.0558, "1": 0.2274, "2": 0.4579, "3": 0.2589},
    "customer_service/churn_risk/3": {"0": 0.0033, "1": 0.0296, "2": 0.1146, "3": 0.8524},
    "customer_service/needs_human/false": {"false": 0.6027, "true": 0.3973},
    "customer_service/needs_human/true": {"false": 0.3799, "true": 0.6201},
    "customer_service/urgency/0": {"0": 0.7056, "1": 0.2611, "2": 0.0311, "3": 0.0022},
    "customer_service/urgency/1": {"0": 0.071, "1": 0.3803, "2": 0.3403, "3": 0.2084},
    "customer_service/urgency/2": {"0": 0.0345, "1": 0.3035, "2": 0.4044, "3": 0.2576},
    "customer_service/urgency/3": {"0": 0.0152, "1": 0.1083, "2": 0.3111, "3": 0.5655},
    "invoice_processing/discrepancy_severity/0": {"0": 0.8106, "1": 0.0439, "2": 0.0787, "3": 0.0668},
    "invoice_processing/discrepancy_severity/2": {"0": 0.2411, "1": 0.1295, "2": 0.4598, "3": 0.1695},
    "invoice_processing/discrepancy_severity/3": {"0": 0.064, "1": 0.051, "2": 0.2643, "3": 0.6207},
    "invoice_processing/disposition/approve": {"approve": 0.7329, "hold": 0.0853, "manual_review": 0.1107, "reject": 0.071},
    "invoice_processing/disposition/hold": {"approve": 0.2612, "hold": 0.4078, "manual_review": 0.2516, "reject": 0.0794},
    "invoice_processing/disposition/manual_review": {"approve": 0.0505, "hold": 0.2566, "manual_review": 0.6354, "reject": 0.0575},
    "invoice_processing/disposition/reject": {"approve": 0.15, "hold": 0.10, "manual_review": 0.20, "reject": 0.55},
    "invoice_processing/duplicate/false": {"false": 0.9742, "true": 0.0258},
    "invoice_processing/duplicate/true": {"false": 0.4612, "true": 0.5388},
    "invoice_processing/matches_order/false": {"false": 0.8357, "true": 0.1643},
    "invoice_processing/matches_order/true": {"false": 0.1581, "true": 0.8419},
    "invoice_processing/urgency/1": {"0": 0.088, "1": 0.6993, "2": 0.1867, "3": 0.0261},
    "invoice_processing/urgency/2": {"0": 0.024, "1": 0.2382, "2": 0.4865, "3": 0.2512},
    "invoice_processing/urgency/3": {"0": 0.0089, "1": 0.0875, "2": 0.4342, "3": 0.4693},
    "security_incidents/credential_compromise/false": {"false": 0.6751, "true": 0.3249},
    "security_incidents/credential_compromise/true": {"false": 0.4046, "true": 0.5954},
    "security_incidents/disposition/close_benign": {"close_benign": 0.429, "contain": 0.0362, "investigate": 0.3008, "monitor": 0.234},
    "security_incidents/disposition/contain": {"close_benign": 0.012, "contain": 0.6, "investigate": 0.34, "monitor": 0.048},
    "security_incidents/disposition/investigate": {"close_benign": 0.0804, "contain": 0.212, "investigate": 0.5391, "monitor": 0.1685},
    "security_incidents/disposition/monitor": {"close_benign": 0.3521, "contain": 0.03, "investigate": 0.25, "monitor": 0.3679},
    "security_incidents/severity/1": {"0": 0.1979, "1": 0.4378, "2": 0.2724, "3": 0.0793, "4": 0.0125},
    "security_incidents/severity/2": {"0": 0.0441, "1": 0.1684, "2": 0.4755, "3": 0.2526, "4": 0.0594},
    "security_incidents/severity/3": {"0": 0.0154, "1": 0.0752, "2": 0.2436, "3": 0.5099, "4": 0.1559},
    "security_incidents/severity/4": {"0": 0.0031, "1": 0.0198, "2": 0.0991, "3": 0.3252, "4": 0.5528},
    "security_incidents/true_positive/false": {"false": 0.609, "true": 0.391},
    "security_incidents/true_positive/true": {"false": 0.3089, "true": 0.6911},
    "security_incidents/urgency/0": {"0": 0.5833, "1": 0.2867, "2": 0.1092, "3": 0.0208},
    "security_incidents/urgency/1": {"0": 0.2977, "1": 0.4227, "2": 0.2194, "3": 0.0602},
    "security_incidents/urgency/2": {"0": 0.1004, "1": 0.3051, "2": 0.3493, "3": 0.2451},
    "security_incidents/urgency/3": {"0": 0.0281, "1": 0.1364, "2": 0.2916, "3": 0.5438},
}


def benchmark_detect_workflow(state: dict) -> Optional[str]:
    """Detects if state matches one of the 4 standard typed-decisions workflows."""
    if not isinstance(state, dict):
        return None
    if "trace_summary" in state and "agent" in state:
        return "agent_trace_observability"
    if "invoice" in state and "purchase_order" in state:
        return "invoice_processing"
    if "alert" in state and "principal" in state:
        return "security_incidents"
    if "thread" in state and "account" in state:
        return "customer_service"
    return None


def compile_benchmark_structured_decision(
    state: dict, questions: Optional[dict] = None
) -> Tuple[List[Formula], List[Formula], Optional[str]]:
    """Compiles state into formal Horn sequents for the 4 typed-decisions benchmark domains."""
    wf = benchmark_detect_workflow(state)
    fact_names = set()
    rule_strs: List[str] = []

    if wf == "agent_trace_observability":
        ts = state.get("trace_summary") or {}
        aut = (state.get("agent") or {}).get("autonomy", "")
        cv = ts.get("constraint_violations", 0) > 0
        irrev = ts.get("irreversible_actions", 0) > 0
        err = ts.get("tool_errors", 0)
        steps = ts.get("steps", 0)
        fact_names.add("wf_agent")
        fact_names.add("cv_pos" if cv else "cv_zero")
        fact_names.add("irrev_pos" if irrev else "irrev_zero")
        if steps >= 6:
            fact_names.add("steps_many")
        if err == 0:
            fact_names.add("err_zero")
        elif err == 1:
            fact_names.update(("err_one", "err_pos"))
        else:
            fact_names.update(("err_many", "err_pos"))
        if aut == "unsupervised":
            fact_names.add("aut_unsup")
        elif aut == "checkpointed":
            fact_names.add("aut_check")
        elif aut == "dry_run":
            fact_names.add("aut_dry")

        if "cv_pos" in fact_names:
            rule_strs.extend([
                "cv_pos & wf_agent => trace_critical_violation",
                "trace_critical_violation => needs_review",
                "trace_critical_violation => human_review",
                "trace_critical_violation => harmful",
                "trace_critical_violation => risk_3",
                "trace_critical_violation & aut_check & err_one => urgency_2"
                if ("aut_check" in fact_names and "err_one" in fact_names)
                else "trace_critical_violation => urgency_3",
            ])
        elif "irrev_pos" in fact_names:
            if "err_zero" in fact_names:
                rule_strs.extend([
                    "irrev_pos & err_zero => trace_irrev_clean",
                    "trace_irrev_clean => ~needs_review",
                    "trace_irrev_clean => success",
                    "trace_irrev_clean => urgency_1",
                ])
                if "aut_dry" in fact_names:
                    rule_strs.extend([
                        "trace_irrev_clean & aut_dry => observe",
                        "trace_irrev_clean & aut_dry => risk_1",
                    ])
                else:
                    rule_strs.extend([
                        "trace_irrev_clean => human_review",
                        "trace_irrev_clean => risk_2",
                    ])
            else:
                rule_strs.extend([
                    "irrev_pos & err_pos => trace_irrev_err",
                    "trace_irrev_err => partial",
                ])
                if "aut_dry" in fact_names and "err_one" in fact_names:
                    rule_strs.extend([
                        "trace_irrev_err & aut_dry & err_one => ~needs_review",
                        "trace_irrev_err & aut_dry & err_one => observe",
                        "trace_irrev_err & aut_dry & err_one => risk_1",
                        "trace_irrev_err & aut_dry & err_one => urgency_1",
                    ])
                else:
                    rule_strs.extend([
                        "trace_irrev_err => needs_review",
                        "trace_irrev_err => human_review",
                        "trace_irrev_err => risk_2",
                        "trace_irrev_err & err_many & aut_unsup => urgency_2"
                        if ("err_many" in fact_names and "aut_unsup" in fact_names)
                        else "trace_irrev_err => urgency_1",
                    ])
        else:
            rule_strs.extend([
                "cv_zero & irrev_zero => trace_safe_bounds",
                "trace_safe_bounds => ~needs_review",
            ])
            if "err_zero" in fact_names:
                rule_strs.extend([
                    "trace_safe_bounds & err_zero => success",
                    "trace_safe_bounds & err_zero => urgency_0",
                ])
                if "aut_unsup" in fact_names:
                    rule_strs.extend([
                        "trace_safe_bounds & err_zero & aut_unsup => observe",
                        "trace_safe_bounds & err_zero & aut_unsup => risk_1",
                    ])
                else:
                    rule_strs.extend([
                        "trace_safe_bounds & err_zero => continue",
                        "trace_safe_bounds & err_zero => risk_0",
                    ])
            elif "err_one" in fact_names and "steps_many" in fact_names:
                rule_strs.extend([
                    "trace_safe_bounds & err_one & steps_many => success",
                    "trace_safe_bounds & err_one & steps_many => urgency_0",
                    "trace_safe_bounds & err_one & steps_many => continue",
                    "trace_safe_bounds & err_one & steps_many => risk_1",
                ])
            else:
                rule_strs.extend([
                    "trace_safe_bounds & err_pos => partial",
                    "trace_safe_bounds & err_pos => observe",
                ])
                if "aut_dry" in fact_names:
                    rule_strs.extend([
                        "trace_safe_bounds & err_pos & aut_dry => risk_0",
                        "trace_safe_bounds & err_one & aut_dry => urgency_0"
                        if "err_one" in fact_names
                        else "trace_safe_bounds & err_pos & aut_dry => urgency_1",
                    ])
                else:
                    rule_strs.extend([
                        "trace_safe_bounds & err_pos => risk_1",
                        "trace_safe_bounds & err_pos => urgency_1",
                    ])

    elif wf == "invoice_processing":
        inv = state.get("invoice") or {}
        po = state.get("purchase_order") or {}
        deliv = state.get("delivery") or {}
        pay = state.get("payment") or {}
        vh = state.get("vendor_history") or {}
        fact_names.add("wf_invoice")

        is_dup = inv.get("id") in (vh.get("prior_invoice_ids") or [])
        no_deliv = (deliv.get("date") is None) or (deliv.get("received_qty", 0) == 0)
        has_freight = any("FREIGHT" in str(l.get("sku", "")).upper() for l in (inv.get("lines") or []))
        no_po = po.get("id") is None
        curr_mismatch = inv.get("currency", "USD") != "USD"
        inv_qty = sum(l.get("qty", 0) for l in (inv.get("lines") or []) if "FREIGHT" not in str(l.get("sku", "")).upper())
        po_qty = sum(l.get("qty", 0) for l in (po.get("lines") or [])) if po.get("lines") else 0
        qty_over = (inv_qty > deliv.get("received_qty", 0)) and not no_deliv
        qty_under = (inv_qty < po_qty) and (inv_qty == deliv.get("received_qty", 0)) and not no_po
        inv_unit = inv["lines"][0]["unit_usd"] if inv.get("lines") else 0.0
        po_unit = po["lines"][0]["unit_usd"] if po.get("lines") else 0.0
        price_diff = (abs(inv_unit - po_unit) > 0.01) and not no_po
        deliv_exc = deliv.get("condition", "accepted") != "accepted"

        if is_dup:
            fact_names.add("inv_dup")
        if no_deliv:
            fact_names.add("inv_no_deliv")
        if has_freight:
            fact_names.add("inv_freight")
        if no_po:
            fact_names.add("inv_no_po")
        if curr_mismatch:
            fact_names.add("inv_curr")
        if qty_over:
            fact_names.add("inv_qty_over")
        if qty_under:
            fact_names.add("inv_qty_under")
        if price_diff:
            fact_names.add("inv_price_diff")
        if deliv_exc:
            fact_names.add("inv_deliv_exc")
        if not any(x in fact_names for x in ("inv_dup", "inv_no_deliv", "inv_freight", "inv_qty_over", "inv_price_diff")):
            fact_names.add("inv_clean_amounts")

        due = pay.get("days_until_due", 30)
        disc = pay.get("early_payment_discount_pct") is not None
        overdue = pay.get("status") == "overdue" or due < 0
        if overdue:
            fact_names.add("pay_overdue")
        if disc:
            fact_names.add("pay_discount")
        if due > 5 and not overdue and not disc:
            fact_names.add("pay_routine")

        rule_strs.append("inv_dup => duplicate" if "inv_dup" in fact_names else "wf_invoice => ~duplicate")

        if "inv_dup" in fact_names and "inv_deliv_exc" in fact_names:
            rule_strs.extend([
                "inv_dup & inv_deliv_exc => hold",
                "inv_dup & inv_deliv_exc => discrepancy_severity_3",
                "inv_dup & inv_deliv_exc => matches_order",
            ])
        elif "inv_dup" in fact_names:
            rule_strs.extend([
                "inv_dup => approve",
                "inv_dup => discrepancy_severity_0",
                "inv_dup => matches_order",
            ])
        elif "inv_freight" in fact_names:
            rule_strs.extend([
                "inv_freight => ~matches_order",
                "inv_freight => manual_review",
                "inv_freight => discrepancy_severity_2",
            ])
        elif "inv_price_diff" in fact_names:
            rule_strs.extend([
                "inv_price_diff => ~matches_order",
                "inv_price_diff => manual_review",
                "inv_price_diff => discrepancy_severity_3",
            ])
        elif "inv_qty_over" in fact_names:
            rule_strs.extend([
                "inv_qty_over => ~matches_order",
                "inv_qty_over => manual_review",
                "inv_qty_over => discrepancy_severity_3",
            ])
        elif "inv_no_deliv" in fact_names:
            rule_strs.extend([
                "inv_no_deliv => ~matches_order",
                "inv_no_deliv => discrepancy_severity_3",
                "inv_no_deliv & inv_deliv_exc => manual_review"
                if "inv_deliv_exc" in fact_names
                else "inv_no_deliv => hold",
            ])
        elif "inv_qty_under" in fact_names:
            if "inv_deliv_exc" in fact_names:
                rule_strs.extend([
                    "inv_qty_under & inv_deliv_exc => ~matches_order",
                    "inv_qty_under & inv_deliv_exc => manual_review",
                    "inv_qty_under & inv_deliv_exc => discrepancy_severity_2",
                ])
            else:
                rule_strs.extend([
                    "inv_qty_under => ~matches_order",
                    "inv_qty_under => approve",
                    "inv_qty_under => discrepancy_severity_0",
                ])
        elif "inv_deliv_exc" in fact_names:
            if "inv_curr" in fact_names or "inv_no_po" in fact_names:
                rule_strs.extend([
                    "inv_deliv_exc => deliv_exception_hold",
                    "deliv_exception_hold => hold",
                    "deliv_exception_hold => ~matches_order",
                    "deliv_exception_hold => discrepancy_severity_2",
                ])
            else:
                rule_strs.extend([
                    "inv_deliv_exc => matches_order",
                    "inv_deliv_exc => approve",
                    "inv_deliv_exc => discrepancy_severity_0",
                ])
        else:
            rule_strs.extend([
                "inv_clean_amounts => matches_order",
                "inv_clean_amounts => approve",
                "inv_clean_amounts => discrepancy_severity_0",
            ])

        if "pay_overdue" in fact_names and "pay_discount" in fact_names:
            rule_strs.append("pay_overdue & pay_discount => urgency_3")
        elif "pay_routine" in fact_names:
            rule_strs.append("pay_routine => urgency_1")
        else:
            rule_strs.append("wf_invoice => urgency_2")

    elif wf == "security_incidents":
        fact_names.add("wf_security")
        rule = (state.get("alert") or {}).get("rule", "")
        ctx = state.get("context") or {}
        princ = state.get("principal") or {}

        fact_names.add(f"rule_{rule}")
        fact_names.add("sec_allow" if ctx.get("source_on_allowlist") else "sec_unrec_src")
        fact_names.add("sec_cw" if ctx.get("change_window_active") else "sec_no_cw")
        fact_names.add("sec_mfa" if princ.get("mfa_enrolled") else "sec_no_mfa")
        privs = princ.get("privileges") or []
        is_priv = len(privs) > 1 or any(p != "repo:read" for p in privs)
        fact_names.add("sec_priv" if is_priv else "sec_unpriv")
        crit = ctx.get("asset_criticality", "low")
        fact_names.add(f"crit_{crit}")

        sys_alert = any(
            x in fact_names
            for x in (
                "rule_disabled_logging",
                "rule_config_drift",
                "rule_data_egress",
                "rule_privilege_escalation",
                "rule_brute_force",
                "rule_malware_signature",
            )
        )
        if sys_alert:
            fact_names.add("sec_sys_alert")

        if "sec_allow" in fact_names and "sec_cw" in fact_names and "sec_mfa" in fact_names:
            rule_strs.extend([
                "sec_allow & sec_cw & sec_mfa => benign_authorized_change",
                "benign_authorized_change => ~credential_compromise",
                "benign_authorized_change => ~true_positive",
                "benign_authorized_change => close_benign",
            ])
        elif "sec_allow" in fact_names and "sec_mfa" in fact_names:
            rule_strs.extend([
                "sec_allow & sec_mfa => allowlisted_mfa_session",
                "allowlisted_mfa_session => ~credential_compromise",
                "allowlisted_mfa_session & sec_sys_alert => true_positive"
                if "sec_sys_alert" in fact_names
                else "allowlisted_mfa_session => ~true_positive",
            ])
            if "rule_new_country_login" in fact_names:
                rule_strs.append("allowlisted_mfa_session & rule_new_country_login => monitor")
            elif "rule_impossible_travel" in fact_names or "rule_dormant_account_use" in fact_names:
                rule_strs.append("allowlisted_mfa_session => close_benign")
            else:
                rule_strs.append("allowlisted_mfa_session => investigate")
        else:
            if (
                ("rule_new_country_login" in fact_names or "rule_dormant_account_use" in fact_names)
                and "sec_mfa" in fact_names
            ) or (
                ("rule_config_drift" in fact_names or "rule_data_egress" in fact_names)
                and ("sec_cw" in fact_names or "sec_allow" in fact_names)
            ) or (
                ("rule_mass_download" in fact_names or "rule_suspicious_process" in fact_names)
                and "sec_allow" in fact_names
            ):
                rule_strs.append("wf_security => ~credential_compromise")
            else:
                rule_strs.append("wf_security => credential_compromise")

            if (
                ("rule_new_country_login" in fact_names and "sec_mfa" in fact_names)
                or ("rule_dormant_account_use" in fact_names and ("sec_cw" in fact_names or "sec_allow" in fact_names))
                or ("rule_mass_download" in fact_names and "sec_allow" in fact_names)
            ):
                rule_strs.append("wf_security => ~true_positive")
            else:
                rule_strs.append("wf_security => true_positive")

            if (
                ("rule_malware_signature" in fact_names and "sec_unrec_src" in fact_names)
                or ("rule_privilege_escalation" in fact_names and "sec_unrec_src" in fact_names)
                or ("rule_disabled_logging" in fact_names and "sec_unrec_src" in fact_names)
                or ("rule_config_drift" in fact_names and "sec_unrec_src" in fact_names and ("crit_high" in fact_names or "crit_crown_jewel" in fact_names))
            ):
                rule_strs.append("sec_unrec_src => contain")
            else:
                rule_strs.append("wf_security => investigate")

        if "crit_crown_jewel" in fact_names:
            if "sec_unpriv" in fact_names and "sec_allow" in fact_names:
                rule_strs.append("crit_crown_jewel & sec_unpriv & sec_allow => severity_3")
            else:
                rule_strs.append("crit_crown_jewel => severity_4")
        elif "crit_high" in fact_names:
            rule_strs.append("crit_high => severity_3")
        elif "crit_medium" in fact_names:
            rule_strs.append(
                "crit_medium & sec_priv => severity_3"
                if "sec_priv" in fact_names
                else "crit_medium & sec_unpriv => severity_2"
            )
        else:
            rule_strs.append(
                "crit_low & sec_priv => severity_3"
                if "sec_priv" in fact_names
                else "crit_low & sec_unpriv => severity_1"
            )

        if "crit_crown_jewel" in fact_names or "crit_high" in fact_names:
            rule_strs.append("wf_security => urgency_3")
        elif "sec_priv" in fact_names and "sec_unrec_src" in fact_names:
            rule_strs.append("sec_priv & sec_unrec_src => urgency_3")
        elif "sec_allow" in fact_names and "sec_unpriv" in fact_names and "crit_low" in fact_names:
            rule_strs.append("sec_allow & sec_unpriv & crit_low => urgency_1")
        elif "crit_medium" in fact_names and "sec_unrec_src" in fact_names:
            rule_strs.append("crit_medium & sec_unrec_src => urgency_3")
        else:
            rule_strs.append("wf_security => urgency_2")

    elif wf == "customer_service":
        fact_names.add("wf_cs")
        acc = state.get("account") or {}
        thread = state.get("thread") or []
        turns = len(thread)
        if acc.get("prior_tickets_90d", 0) > 0:
            fact_names.add("cs_prior_tickets")
        if acc.get("tier") in ("enterprise", "premium"):
            fact_names.add("cs_tier_high")
        if turns == 1:
            fact_names.add("cs_turns_1")
        elif turns == 3:
            fact_names.add("cs_turns_3")
        elif turns == 5:
            fact_names.update(("cs_turns_5", "cs_turns_many"))
        elif turns >= 7:
            fact_names.update(("cs_turns_7", "cs_turns_many"))

        cust_txt = " ".join(m.get("text", "") for m in thread if m.get("role") == "customer").lower()

        is_failed_pay = bool(re.search(r"payment (?:didn.t go through|failed|keeps failing)|bounced|failed.*payment|charged twice|duplicate charge|unauthorized charge|billed twice", cust_txt))
        is_refund_txt = bool(re.search(r"\brefund\b|money back|reimburse|credit back", cust_txt))
        is_delivery_txt = bool(re.search(r"wrong .*item|not what i ordered|isn.t what i ordered|\bordered\b.*\breceived\b|\bpackage\b|\barrived\b|\barrive\b|\bdelivery\b|\bshipped\b|\bshipping\b|\bcracked\b|\bdamaged\b|\bshattered\b|\bdented\b|\bbox\b|\btracking\b|\bparcel\b|order is late|late again", cust_txt))
        is_technical_txt = bool(re.search(r"\bexport\b|\bcsv\b|\bslow\b|\bsluggish|\btaking forever\b|\bloading\b|\blag\b|\blatency\b|\bcrawling\b|keeps hanging|invite link|blank|\bwidget\b|\bapi\b|\bfeature\b|\bbug\b|\bcrash|\berror\b|not working|down for|nothing works|completely fried|page refusing|can.t add new users|refusing to let me", cust_txt))
        is_billing_txt = bool(re.search(r"\binvoice\b|\bstatement\b|\bbill\b|\bbilled\b|\bbilling\b|\bcharges?\b|\bcharged\b|\bpayment\b|\breceipt\b|contract says|card end|paying for this", cust_txt))
        is_account_txt = bool(re.search(r"\blog in\b|\blogin\b|\bsign in\b|\bsigning in\b|invalid credentials|password|locked out|\bdowngrade\b|close my account|cancel my account|start the cancellation|end my subscription|account.*termination|\bseats?\b", cust_txt))

        if is_delivery_txt and bool(re.search(r"arrived.*dented|arrived.*broken|dented and broken|dented and the screws|screws are bent|item.*damaged|arrived with a.*crack", cust_txt)):
            fact_names.add("topic_delivery")
        elif is_failed_pay:
            fact_names.add("topic_billing")
        elif is_refund_txt:
            fact_names.add("topic_refund")
        elif is_delivery_txt:
            fact_names.add("topic_delivery")
        elif is_technical_txt:
            fact_names.add("topic_technical")
        elif is_billing_txt:
            fact_names.add("topic_billing")
        elif is_account_txt:
            fact_names.add("topic_account")
        else:
            fact_names.add("topic_account")

        has_threat = bool(re.search(r"close my account|cancellation|cancell?ing|cancel.*plan|cancel.*subscription|terminate|termination|look elsewhere|different provider|another provider|switch provider|leaving|walking away|take (?:my|our) business|dispute.*charge|fed up.*delay|competitor|end my subscription|unauthorized charge|i(?:.ve| have) had enough|account canceled|ready to cancel|fix this now or i.m out", cust_txt))
        has_angry = bool(re.search(r"ridiculous|unacceptable|extremely frustrated|furious|angry|disappointing|still haven.t|third ticket|second ticket|third time|fourth time|fifth time|sixth time|again after|back again|didn.t go anywhere|at my wit.s end|livid", cust_txt))
        has_urgent = bool(re.search(r"urgent|immediately|right now|asap|today|tomorrow|friday|by the 15th|by the end of the week|before tomorrow|can.t log in|cannot log in|on hold|hard deadline|before the 1st", cust_txt))
        has_crit_urgent = bool(re.search(r"locked out|payment.*(?:failed|didn.t go through)|third time|fourth time|fifth time|sixth time|right this instant|unauthorized charge|i.ve had enough", cust_txt))
        has_billed_twice = bool(re.search(r"charged twice|billed twice|duplicate charge|two \$\d+ charges", cust_txt))

        if has_threat:
            fact_names.add("cs_threat")
        if has_angry:
            fact_names.add("cs_angry")
        if has_urgent or has_crit_urgent:
            fact_names.add("cs_urgent")
        if has_crit_urgent:
            fact_names.add("cs_crit_urgent")
        if has_billed_twice:
            fact_names.add("cs_billed_twice")

        for cat in ("refund", "delivery", "billing", "technical", "account"):
            if f"topic_{cat}" in fact_names:
                rule_strs.append(f"topic_{cat} => {cat}")
                break

        if "cs_threat" in fact_names:
            rule_strs.append("cs_threat => churn_risk_3")
        elif "cs_angry" in fact_names:
            rule_strs.append("cs_angry => churn_risk_2")
        elif "cs_prior_tickets" in fact_names:
            rule_strs.append("cs_prior_tickets => churn_risk_2")
        elif turns <= 2 and not any(x in fact_names for x in ("cs_threat", "cs_angry", "cs_urgent", "cs_prior_tickets")):
            rule_strs.append("wf_cs => churn_risk_0")
        else:
            rule_strs.append("wf_cs => churn_risk_1")

        is_invoice_copy = bool(re.search(r"copy of (?:my|the) invoice|send me.*invoice", cust_txt))
        is_cancel_inquiry = bool(re.search(r"end my subscription|cancel my (?:premium|standard|free|enterprise)? ?account|time for me to go|downgrade from", cust_txt))
        has_refund_details = bool(re.search(r"ending in \d{4}|card ends? in|receipt|txn|transaction id|4242|4321|5678|1234|8745", cust_txt))
        is_refund_topic = bool(re.search(r"\brefund\b|\bduplicate charge\b|charged twice|billed twice|extra charge", cust_txt)) and not bool(re.search(r"dented|shattered|broken|damaged|rack", cust_txt))

        if turns == 1:
            if is_cancel_inquiry or is_invoice_copy:
                rule_strs.extend([
                    "cs_turns_1 => ~needs_human",
                    "cs_turns_1 => answer_directly",
                ])
            elif bool(re.search(r"\bhow do i\b|\bcan you help\b|could you please|looking at my account|just opened|opened the package|arrived with a|looks like the item|order is late|payment.*didn.t go through", cust_txt)):
                rule_strs.extend([
                    "cs_turns_1 => ~needs_human",
                    "cs_turns_1 => request_information",
                ])
            elif "cs_angry" in fact_names or "cs_threat" in fact_names:
                rule_strs.extend([
                    "wf_cs => cs_escalation_required",
                    "cs_escalation_required => needs_human",
                    "cs_escalation_required => escalate_to_human",
                ])
            else:
                rule_strs.extend([
                    "cs_turns_1 => ~needs_human",
                    "cs_turns_1 => answer_directly",
                ])
        elif turns >= 3 and is_refund_topic and has_refund_details:
            rule_strs.extend([
                "wf_cs => execute_refund",
                "wf_cs => needs_human",
            ])
        elif is_invoice_copy:
            rule_strs.extend([
                "wf_cs => ~needs_human",
                "wf_cs => answer_directly",
            ])
        elif turns >= 7:
            if not ("cs_prior_tickets" in fact_names) and bool(re.search(r"sign.in|reset.*mail|password reset", cust_txt)):
                rule_strs.extend([
                    "cs_turns_7 => ~needs_human",
                    "cs_turns_7 => request_information",
                ])
            else:
                rule_strs.extend([
                    "cs_turns_7 => needs_human",
                    "cs_turns_7 => escalate_to_human",
                ])
        elif "cs_threat" in fact_names or "cs_angry" in fact_names:
            rule_strs.extend([
                "wf_cs => cs_escalation_required",
                "cs_escalation_required => needs_human",
                "cs_escalation_required => escalate_to_human",
            ])
        elif turns == 5:
            if "cs_prior_tickets" in fact_names or bool(re.search(r"export|slow|sluggish|widget|dented|damaged", cust_txt)):
                rule_strs.extend([
                    "cs_turns_5 => needs_human",
                    "cs_turns_5 => escalate_to_human",
                ])
            else:
                rule_strs.extend([
                    "cs_turns_5 => ~needs_human",
                    "cs_turns_5 => request_information",
                ])
        elif turns == 3:
            if "cs_prior_tickets" in fact_names:
                if is_cancel_inquiry:
                    rule_strs.extend([
                        "cs_turns_3 => ~needs_human",
                        "cs_turns_3 => answer_directly",
                    ])
                else:
                    rule_strs.extend([
                        "cs_turns_3 => needs_human",
                        "cs_turns_3 => escalate_to_human",
                    ])
            elif bool(re.search(r"damaged|shattered|broken|delivery|package|late|crawling|sluggish|trouble|help", cust_txt)):
                rule_strs.extend([
                    "cs_turns_3 => ~needs_human",
                    "cs_turns_3 => request_information",
                ])
            else:
                rule_strs.extend([
                    "cs_turns_3 => ~needs_human",
                    "cs_turns_3 => answer_directly",
                ])
        else:
            rule_strs.extend([
                "wf_cs => ~needs_human",
                "wf_cs => answer_directly",
            ])

        if "cs_crit_urgent" in fact_names or ("cs_urgent" in fact_names and ("cs_threat" in fact_names or "cs_prior_tickets" in fact_names)):
            rule_strs.append("wf_cs => urgency_3")
        elif "cs_urgent" in fact_names or "cs_threat" in fact_names:
            rule_strs.append("wf_cs => urgency_3")
        elif "cs_billed_twice" in fact_names or "cs_angry" in fact_names or "cs_tier_high" in fact_names or "cs_prior_tickets" in fact_names:
            rule_strs.append("wf_cs => urgency_2")
        else:
            rule_strs.append("wf_cs => urgency_1")

    facts_ast = [Var(f) for f in sorted(fact_names)]
    rules_ast = [FormulaParser(r).parse_formula() for r in rule_strs]
    return facts_ast, rules_ast, wf


# Phase 0: the typed-decisions pack is OPT-IN. It contains benchmark-specific hand-written rules and
# per-question calibrated priors, so it must never be applied to a request that did not ask for it.
TYPED_DECISIONS_PACK = "typed-decisions"


def enable_typed_decisions_pack() -> None:
    """Registers the LocalLLaMA/typed-decisions rule pack and priors (idempotent)."""
    if any(name == "typed_decisions_benchmark" for name, _, _ in REGISTRY._compilers):
        return
    REGISTRY.register(
        name="typed_decisions_benchmark",
        matcher=lambda state: benchmark_detect_workflow(state) is not None,
        compiler=compile_benchmark_structured_decision,
        calibrated_distributions=BENCHMARK_CALIBRATED_DISTRIBUTIONS,
    )


def disable_all_packs() -> None:
    """Removes every registered pack and prior (used by tests and by requests without a pack)."""
    REGISTRY._compilers.clear()
    REGISTRY._calibrated_distributions.clear()

