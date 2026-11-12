# CrewAI vs LangGraph: what is the same, what differs

Same: tools (all go through the gateway), tool JSON schemas, system prompt and task prompt text, step limit, trace format, validators.

Switched off in CrewAI so it cannot bias results: tool result cache, agent retries, argument validation (gateway does it),
and BOTH repeat guards (identical call twice in a row; 5 identical calls force-ends the run).

Differences to state in the report:
1. CrewAI wraps our task in its own prompt template (ReAct text format for models without native tool calling).
2. CrewAI has its own provider retry for rate limits; those retries are not logged as retry events.
3. CrewAI may ask the model for a forced final answer at its own iteration cap (set to step limit + 5, never reached in practice).
4. Tests use a scripted fake model on the ReAct path. The native function calling path (real Gemini/Groq) is only exercised with real keys.
