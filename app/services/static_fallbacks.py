"""
Static fallback messages for when LLM generation fails.

These pre-written messages are matched to the selected arm type
and delivered when all LLM retries are exhausted.
"""

from __future__ import annotations

import random

# Fallback messages by intervention type
STATIC_FALLBACKS: dict[str, list[str]] = {
    "sensory_focus": [
        "Take a moment. Notice 5 things you can see around you right now. Focus on their colors and shapes. Let your attention rest on what is directly in front of you.",
        "Pause and listen. What sounds can you hear right now? Try to notice three different sounds, near and far. Let each one hold your attention for a few seconds.",
        "Look around you. Pick one object and really notice it. What color is it? What texture does it have? Spend a moment just observing it closely.",
    ],
    "relaxation_breathing": [
        "Try this: breathe in slowly for 4 counts, hold for 4, breathe out for 6. Repeat 3 times. Focusing on a slow exhale helps your body shift out of alarm mode.",
        "Place one hand on your chest. Breathe in through your nose for 4 counts, then out through your mouth for 6. Notice your hand rising and falling with each breath.",
        "Take three slow breaths right now. In for 4 counts, out for 6. With each exhale, let your shoulders drop a little. This gives your body a signal to ease up.",
    ],
    "activity_suggestion": [
        "Try standing up and stretching for just 2 minutes. Reach your arms overhead, roll your shoulders back. Even small movement can shift how you feel right now.",
        "Step outside for a brief walk, even just around the block. Or if staying in feels better, put on a song you like and listen to the whole thing.",
        "Get a glass of water and drink it slowly. Or, if you have the energy, take a 5-minute walk. Sometimes changing what your body is doing changes how you feel.",
    ],
    "social_outreach": [
        "Being around someone can change how this moment feels. Is there one person you could call or text right now? Even a brief conversation can help.",
        "Think of someone you trust. Send them a short message or give them a quick call. Connecting with another person, even briefly, is a way to take care of yourself.",
    ],
}


def get_fallback(intervention_type: str) -> str:
    """
    Get a random static fallback message for the given intervention type.

    Args:
        intervention_type: One of 'sensory_focus', 'relaxation_breathing',
                          'activity_suggestion', 'social_outreach'.

    Returns:
        A pre-written intervention message.
    """
    messages = STATIC_FALLBACKS.get(intervention_type, [])
    if not messages:
        # Ultimate fallback — should never happen
        return "Take a moment to pause. Notice your breathing. Let each exhale be a little longer than the inhale. This small adjustment can help right now."
    return random.choice(messages)
