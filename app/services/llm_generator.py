"""
LLM Generator for personalized intervention content.

Uses the Anthropic Claude API to generate brief, context-aware
intervention messages. Section 6.2 of the AIC spec.
"""

from __future__ import annotations

import openai

from app.config import settings

# System prompt from Section 6.2.1 of the AIC spec
SYSTEM_PROMPT = """You are generating a brief supportive message for a person experiencing elevated distress. The message must be 2-4 sentences and under 280 characters. Follow ALL rules below:

RULES:

1. Deliver GENERAL WELLNESS interventions ONLY (e.g., relaxation, specific activities). NEVER provide therapeutic, psychological, or clinical messaging.

2. NEVER reference psychiatric symptoms, diagnoses, medication, or clinical terminology.

3. NEVER use hollow praise ("Great job!", "You've got this!", "Stay strong!").

4. NEVER fabricate data or reference other people's experiences.

5. Frame the experience as manageable, not as dangerous or a sign of deterioration.

6. Use the participant's current context (location, activity) to make the message specific.

7. Ensure all language remains strictly wellness-focused. When suggesting an activity, ALWAYS provide a choice: suggest ONE of the participant's preferred high-effort wellness activities AND ONE low-effort, general wellness alternative (e.g., stretching, listening to music, resting) in the same message, so they can choose based on their current energy.

8. Do not repeat content from prior messages delivered today.

9. Write at an 8th-grade reading level. Use simple, direct language.

10. Do not include emojis, hashtags, or informal internet language.

PERMITTED patterns (process attribution):
- "Noticing what you are feeling is a skill."
- "Taking a moment to breathe is how you manage these situations."
- "You adjusted your approach—that is a coping strategy."
- "Reaching out for support is something you chose to do."
- "Focusing on what is around you right now is a technique that works."

FORBIDDEN patterns — NEVER use any of the following:
- Person/ability attribution: "You are strong", "You are brave", "You are resilient"
- Hollow praise: "Great job!", "Awesome!", "Keep it up!"
- Clinical language: symptoms, diagnoses, medication, treatment, illness, disorder
- Fabricated data: "Studies show...", "Most people...", "Others like you..."
- Minimizing: "It's not that bad", "Just relax", "Don't worry about it"
- Commands without strategy: "Calm down", "Stop worrying", "Think positive"
- Time predictions: "This will pass in X minutes", "You'll feel better soon"
"""

INTERVENTION_TYPE_DESCRIPTIONS = {
    "sensory_focus": "General wellness: Direct attention to immediate sensory environment. Guide the person to notice specific things they can see, hear, or feel right now.",
    "relaxation_breathing": "General wellness: Paced breathing instruction. Provide specific breathing counts (e.g., inhale 4, hold 4, exhale 6). Focus on physiological relaxation.",
    "activity_suggestion": "General wellness: Suggest a specific activity based on the user's explicit wellness preferences, paired with a low-effort alternative to accommodate varying energy levels.",
    "social_outreach": "General wellness: Prompt reaching out to another person to reduce isolation. Suggest a specific, concrete action (call, text, visit).",
}


class LLMGenerator:
    """
    Generates personalized intervention content using Claude API.

    Each generated message is:
    - 2-4 sentences, under 280 characters
    - Contextually relevant (uses location, activity, preferences)
    - Non-repetitive (prior messages today provided for dedup)
    - Compliant with attribution framework rules
    """

    def __init__(self, api_key: str | None = None):
        self.client = openai.AsyncOpenAI(
            api_key=api_key or settings.DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com/v1"
        )

    async def generate(
        self,
        intervention_type: str,
        context: dict,
        prior_messages_today: list[str] | None = None,
        wellness_preferences: list[str] | None = None,
    ) -> str:
        """
        Generate intervention content via Claude API.

        Args:
            intervention_type: One of 'sensory_focus', 'relaxation_breathing',
                             'activity_suggestion', 'social_outreach'.
            context: Dict with keys: na_score, anxiety, sadness, location,
                    social_context, current_activity.
            prior_messages_today: List of intervention texts already delivered today.
            wellness_preferences: Participant's preferred wellness activities.

        Returns:
            Generated intervention text.
        """
        user_prompt = self._build_prompt(
            intervention_type, context, prior_messages_today, wellness_preferences
        )

        response = await self.client.chat.completions.create(
            model=settings.LLM_MODEL,
            max_tokens=settings.LLM_MAX_TOKENS,
            temperature=settings.LLM_TEMPERATURE,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
        )

        return response.choices[0].message.content.strip()

    def _build_prompt(
        self,
        intervention_type: str,
        context: dict,
        prior_messages: list[str] | None,
        preferences: list[str] | None,
    ) -> str:
        """Build the user-facing prompt for the LLM."""
        type_desc = INTERVENTION_TYPE_DESCRIPTIONS.get(intervention_type, intervention_type)

        parts = [
            f"INTERVENTION TYPE: {intervention_type} — {type_desc}",
            "",
            f"CONTEXT: NA={context.get('na_score', '?')}/7 "
            f"(anxiety={context.get('anxiety', '?')}, sadness={context.get('sadness', '?')}), "
            f"Location={context.get('location', '?')}, "
            f"Social={context.get('social_context', '?')}, "
            f"Activity={context.get('current_activity', 'not specified')}",
        ]

        if preferences:
            parts.append(f"\nWELLNESS PREFERENCES: {', '.join(preferences)}")

        if prior_messages:
            parts.append(f"\nPRIOR MESSAGES TODAY (do not repeat):")
            for i, msg in enumerate(prior_messages, 1):
                parts.append(f"  {i}. {msg}")

        parts.append(
            "\nGenerate a single message following all rules. "
            "Keep it under 280 characters, 2-4 sentences."
        )

        return "\n".join(parts)
