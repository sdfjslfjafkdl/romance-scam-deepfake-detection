SCAM_JSON_SCHEMA_DESC = """
Return ONLY valid JSON with this structure:

{
  "risk_level": "SAFE"|"CAUTION"|"HIGH",
  "risk_score": 0..100,
  "scam_type_guess": ["romance"|"phishing"|"impersonation"|"job"|"loan"|"crypto"|"other"],
  "crime_script": [
    {"stage":"rapport_building"|"isolation_or_offplatform"|"request_money_or_credentials"|"unknown",
     "evidence_quotes":["..."], "why":"..."}
  ],
  "red_flags": [
    {"flag":"...", "severity":1..5, "evidence_quotes":["..."]}
  ],
  "scammer_next_message_simulations": [
    {"simulated_message":"...", "why_this_is_common":"..."}
  ],
  "recommended_actions": ["..."],
  "need_more_context": ["..."]
}
"""