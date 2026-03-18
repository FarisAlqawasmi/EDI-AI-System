from __future__ import annotations

import json
from typing import Optional, Dict, List

from openai import OpenAI

# We store/display the role with a space: "generic user".
# Accept a few common variants and normalize.
VALID_ROLES = {"student", "teacher", "generic user"}
VALID_FAMILIARITY = {"low", "medium", "high"}


def _normalize_role(role: Optional[str]) -> Optional[str]:
    if not isinstance(role, str):
        return None
    r = role.strip().lower()
    r = r.replace("_", " ")
    r = " ".join(r.split())
    # common variants
    if r in {"generic", "general user", "general", "public", "member of public", "regular user"}:
        r = "generic user"
    return r if r in VALID_ROLES else None


def _normalize_familiarity(fam: Optional[str]) -> Optional[str]:
    if not isinstance(fam, str):
        return None
    f = fam.strip().lower()
    f = " ".join(f.split())
    return f if f in VALID_FAMILIARITY else None


def _extract_json_obj_best_effort(s: str) -> Optional[dict]:
    """Extract JSON object even if the model adds extra text around it."""
    if not s:
        return None
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        pass

    i = s.find("{")
    j = s.rfind("}")
    if i != -1 and j != -1 and j > i:
        try:
            return json.loads(s[i : j + 1])
        except Exception:
            return None
    return None


def parse_user_profile_answer(
    client: OpenAI,
    text: str,
    *,
    router_model: str,
    store_router: bool,
    current_profile: Dict[str, Optional[str]],
) -> Dict[str, Optional[str]]:
    """AI-driven profile extraction (NO keyword triggers).

    Returns {"role": ..., "edi_familiarity": ...} with None if unclear.
    Robustness:
    - Try json_schema
    - If parse fails, stitch from output blocks
    - Retry with json_object
    - Final minimal local fallback for obvious single-word replies
    """

    system_msg = (
        "Extract the user's profile from their message.\n"
        "Return STRICT JSON with keys: role, edi_familiarity.\n"
        "Allowed role values: student, teacher, generic user.\n"
        "Allowed edi_familiarity values: low, medium, high.\n"
        "If a field is not clearly specified, set it to null.\n"
        "Do not add extra keys."
    )

    user_msg = (
        "CURRENT_PROFILE:\n"
        f"{json.dumps(current_profile)}\n\n"
        "USER_MESSAGE:\n"
        f"{text}\n"
    )

    def _call(response_format) -> Optional[dict]:
        try:
            resp = client.responses.create(
                model=router_model,
                input=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0,
                store=store_router,
                response_format=response_format,
            )

            raw = (getattr(resp, "output_text", "") or "").strip()
            obj = _extract_json_obj_best_effort(raw)
            if obj is not None:
                return obj

            pieces: List[str] = []
            for item in getattr(resp, "output", []) or []:
                if getattr(item, "type", "") == "message":
                    for c in getattr(item, "content", []) or []:
                        if getattr(c, "type", "") in {"output_text", "text"}:
                            pieces.append(getattr(c, "text", "") or "")
            return _extract_json_obj_best_effort("".join(pieces).strip())
        except Exception:
            return None

    # 1) Try json_schema
    obj = _call(
        {
            "type": "json_schema",
            "json_schema": {
                "name": "user_profile",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "role": {"type": ["string", "null"]},
                        "edi_familiarity": {"type": ["string", "null"]},
                    },
                    "required": ["role", "edi_familiarity"],
                },
            },
        }
    )

    # 2) Retry with json_object if still None
    if obj is None:
        obj = _call({"type": "json_object"})

    role_raw = (obj or {}).get("role", None)
    fam_raw = (obj or {}).get("edi_familiarity", None)

    # 3) Minimal local fallback if the model couldn't extract anything.
    if role_raw is None and fam_raw is None:
        t = (text or "").strip().lower().replace("_", " ")
        # role
        if "student" in t:
            role_raw = "student"
        elif "teacher" in t or "lecturer" in t or "instructor" in t:
            role_raw = "teacher"
        elif "generic" in t or "general user" in t or "public" in t:
            role_raw = "generic user"
        # familiarity
        if "low" in t or "beginner" in t or "new" in t:
            fam_raw = "low"
        elif "medium" in t or "some" in t or "moderate" in t:
            fam_raw = "medium"
        elif "high" in t or "advanced" in t or "expert" in t:
            fam_raw = "high"

    role = _normalize_role(role_raw)
    fam = _normalize_familiarity(fam_raw)

    return {"role": role, "edi_familiarity": fam}


def parse_personalisation_consent(
    client: OpenAI,
    text: str,
    *,
    router_model: str,
    store_router: bool,
) -> str:
    """AI-driven consent parsing.

    Returns one of: "yes", "no", "unknown".

    Robustness:
    - Tries `json_schema` first.
    - If parsing fails, retries with `json_object`.
    - Final fallback: minimal local normalization for obvious yes/no replies.
    """

    user_reply = (text or "").strip()

    system_msg = (
        "You are deciding whether the user CONSENTS to answering two quick personalisation questions "
        "(role and EDI familiarity). Given the user's reply, classify it as yes, no, or unknown. "
        "Return STRICT JSON only: {\"consent\": \"yes\"|\"no\"|\"unknown\"}. "
        "Treat variants like 'yes please', 'sure', 'ok', 'go ahead' as yes; and 'no thanks', 'not now' as no."
    )

    user_msg = f"USER_REPLY:\n{user_reply}\n"

    def _get_consent_from_obj(obj: Optional[dict]) -> str:
        consent = (obj or {}).get("consent", "unknown")
        if isinstance(consent, str):
            consent = consent.strip().lower()
        else:
            consent = "unknown"
        return consent if consent in {"yes", "no", "unknown"} else "unknown"

    def _call(response_format) -> Optional[dict]:
        try:
            resp = client.responses.create(
                model=router_model,
                input=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0,
                store=store_router,
                response_format=response_format,
            )

            raw = (getattr(resp, "output_text", "") or "").strip()
            obj = _extract_json_obj_best_effort(raw)
            if obj is not None:
                return obj

            pieces: List[str] = []
            for item in getattr(resp, "output", []) or []:
                if getattr(item, "type", "") == "message":
                    for c in getattr(item, "content", []) or []:
                        if getattr(c, "type", "") in {"output_text", "text"}:
                            pieces.append(getattr(c, "text", "") or "")
            return _extract_json_obj_best_effort("".join(pieces).strip())
        except Exception:
            return None

    obj = _call(
        {
            "type": "json_schema",
            "json_schema": {
                "name": "personalisation_consent",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "consent": {"type": "string", "enum": ["yes", "no", "unknown"]}
                    },
                    "required": ["consent"],
                },
            },
        }
    )

    consent = _get_consent_from_obj(obj)

    if consent == "unknown":
        obj2 = _call({"type": "json_object"})
        consent = _get_consent_from_obj(obj2)

    if consent == "unknown":
        s = user_reply.lower().strip()
        if s in {"y", "yes", "yeah", "yep", "sure", "ok", "okay"}:
            return "yes"
        if s in {"n", "no", "nope", "nah"}:
            return "no"
        if "yes" in s or "sure" in s or "go ahead" in s or "you may" in s:
            return "yes"
        if "no" in s or "not now" in s or "no thanks" in s:
            return "no"

    return consent