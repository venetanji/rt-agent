"""``decision-bundle/v1`` — the frozen question bundle the robot asks once per utterance.

Everything the model knows about a question is the text in ``instructions`` and
``criteria``: the question ids below are ours and are never sent. The wording is
therefore load-bearing, and because a System One answer shifts when its *siblings*
change, the bundle is frozen and versioned as a whole. Changing any word here means
bumping :data:`BUNDLE_VERSION` and re-recording the fixtures.

Design rules followed here (from the live scouting pass):

* one judgment per question — independent dimensions are separate questions and are
  combined in visible code, never inside one overloaded question;
* several labels may be true at once => one ``noul`` each, not one ``choice``;
* every ``choice`` carries an escape hatch (``unclear`` / ``other``) so the model is
  never forced to name a winner;
* ``score`` levels are self-contained sentences, not "low / medium / high".
"""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import Field, model_validator

from rt_agent.contracts.base import FrozenModel

__all__ = [
    "BUNDLE_VERSION",
    "MEMORY_FAITHFUL_QUESTION",
    "MEMORY_FAITHFUL_VERSION",
    "QUESTION_BUNDLE_V1",
    "Q_ADDRESSED_TO_ROBOT",
    "Q_ADDRESSEE",
    "Q_EMOTION",
    "Q_EMOTION_INTENSITY",
    "Q_INTELLIGIBLE_COMPLETE",
    "Q_INVITES_RESPONSE_NOW",
    "Q_MEMORY_FAITHFUL",
    "Q_MEMORY_KIND",
    "Q_SENSITIVE_PERSONAL",
    "Q_STAY_QUIET_SAFETY",
    "Q_WORTH_REMEMBERING",
    "ChoiceQuestion",
    "NoulQuestion",
    "QuestionBundle",
    "ScoreQuestion",
]

#: Bump this whenever any word of the bundle changes.
BUNDLE_VERSION = "decision-bundle/v1"

#: The second, off-critical-path call that gates a written memory.
MEMORY_FAITHFUL_VERSION = "memory-faithful/v1"

Q_INTELLIGIBLE_COMPLETE = "intelligible_complete"
Q_ADDRESSED_TO_ROBOT = "addressed_to_robot"
Q_INVITES_RESPONSE_NOW = "invites_response_now"
Q_STAY_QUIET_SAFETY = "robot_should_stay_quiet_safety"
Q_ADDRESSEE = "addressee"
Q_EMOTION = "emotion"
Q_EMOTION_INTENSITY = "emotion_intensity"
Q_WORTH_REMEMBERING = "worth_remembering"
Q_SENSITIVE_PERSONAL = "sensitive_personal"
Q_MEMORY_KIND = "memory_kind"
Q_MEMORY_FAITHFUL = "memory_faithful"


class NoulQuestion(FrozenModel):
    """A yes/no question answered with a calibrated probability."""

    id: str = Field(min_length=1, max_length=64)
    instructions: str = Field(min_length=1)
    when_true: str = Field(min_length=1)
    when_false: str = Field(min_length=1)

    @property
    def type(self) -> Literal["noul"]:
        """Wire type."""
        return "noul"

    def to_wire(self) -> dict[str, Any]:
        """Render this question as the server expects it."""
        return {
            "type": "noul",
            "instructions": self.instructions,
            "criteria": {"true": self.when_true, "false": self.when_false},
        }


class ChoiceQuestion(FrozenModel):
    """Pick exactly one option; the full distribution comes back as well."""

    id: str = Field(min_length=1, max_length=64)
    instructions: str = Field(min_length=1)
    options: tuple[tuple[str, str], ...]

    @model_validator(mode="after")
    def _check(self) -> Self:
        if len(self.options) < 2:
            raise ValueError("a choice needs at least two options")
        names = [name for name, _ in self.options]
        if len(set(names)) != len(names):
            raise ValueError("choice option names must be unique")
        return self

    @property
    def type(self) -> Literal["choice"]:
        """Wire type."""
        return "choice"

    @property
    def option_names(self) -> tuple[str, ...]:
        """The frozen option set, in presentation order."""
        return tuple(name for name, _ in self.options)

    def to_wire(self) -> dict[str, Any]:
        """Render this question as the server expects it."""
        return {
            "type": "choice",
            "instructions": self.instructions,
            "criteria": {name: description for name, description in self.options},
        }


class ScoreQuestion(FrozenModel):
    """Place the utterance on an ordered rubric; position 0 is the bottom level."""

    id: str = Field(min_length=1, max_length=64)
    instructions: str = Field(min_length=1)
    levels: tuple[str, ...]

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not 2 <= len(self.levels) <= 10:
            raise ValueError("a score rubric needs between 2 and 10 levels")
        return self

    @property
    def type(self) -> Literal["score"]:
        """Wire type."""
        return "score"

    @property
    def max_level(self) -> int:
        """Index of the top level."""
        return len(self.levels) - 1

    def to_wire(self) -> dict[str, Any]:
        """Render this question as the server expects it."""
        return {
            "type": "score",
            "instructions": self.instructions,
            "criteria": list(self.levels),
        }


Question = NoulQuestion | ChoiceQuestion | ScoreQuestion


class QuestionBundle(FrozenModel):
    """An ordered, frozen set of questions asked in a single request."""

    version: str = Field(min_length=1, max_length=64)
    questions: tuple[Question, ...]

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not self.questions:
            raise ValueError("a bundle needs at least one question")
        ids = [question.id for question in self.questions]
        if len(set(ids)) != len(ids):
            raise ValueError("question ids must be unique inside a bundle")
        return self

    @property
    def ids(self) -> tuple[str, ...]:
        """Question ids in wire order."""
        return tuple(question.id for question in self.questions)

    def get(self, question_id: str) -> Question:
        """Look one question up by id."""
        for question in self.questions:
            if question.id == question_id:
                return question
        raise KeyError(question_id)

    def to_wire(self) -> dict[str, dict[str, Any]]:
        """The ``questions`` object of a ``POST /v1/systemone`` body."""
        return {question.id: question.to_wire() for question in self.questions}


# --------------------------------------------------------------------------------------
# The bundle itself. Read this as prose: it is what the model reads.
# --------------------------------------------------------------------------------------

_ADMISSION = (
    NoulQuestion(
        id=Q_INTELLIGIBLE_COMPLETE,
        instructions=(
            "Is the current utterance something intelligible that the speaker finished saying?"
        ),
        when_true=(
            "The words make sense as a sentence, question, greeting or remark, and the "
            "speaker reached the end of the thought. Ordinary hesitations, repeated words "
            "and small speech-recognition slips are fine as long as the meaning is clear."
        ),
        when_false=(
            "The text is speech-recognition garbage, an isolated word or noise with no "
            "meaning here, or the speaker was cut off in the middle of a thought and did "
            "not finish it."
        ),
    ),
    NoulQuestion(
        id=Q_ADDRESSED_TO_ROBOT,
        instructions=(
            "Is the current utterance spoken to the robot named at the top of the state, "
            "rather than to another person in the room?"
        ),
        when_true=(
            "The speaker is talking to the robot: they call it by name, they answer or "
            "follow up on something the robot just said, or they give it an instruction or "
            "a question that only the robot could be meant to handle."
        ),
        when_false=(
            "The speaker is talking to another person, to the room in general, or to "
            "themselves; or this is background conversation that happens to be audible. "
            "Talking *about* the robot in the third person to someone else is not "
            "addressing it; starting a sentence with the robot's name and then telling "
            "it something is."
        ),
    ),
    NoulQuestion(
        id=Q_INVITES_RESPONSE_NOW,
        instructions="Does the current utterance invite the robot to reply right now?",
        when_true=(
            "It is a question, a request or a remark that leaves the floor open and waits "
            "for the robot to answer immediately."
        ),
        when_false=(
            "No answer is expected, or not yet: the speaker is still going, the floor was "
            "handed to another person, the reply is owed by someone else, or the utterance "
            "is a closing remark that needs nothing back."
        ),
    ),
    NoulQuestion(
        id=Q_STAY_QUIET_SAFETY,
        instructions=(
            "Should the robot stay quiet right now because of what is happening in the "
            "room, even if it was addressed?"
        ),
        when_true=(
            "Someone is distressed, frightened, crying, hurt, or dealing with an emergency, "
            "or people are in the middle of an urgent or private exchange where a remark "
            "from a machine would intrude or make things worse."
        ),
        when_false=(
            "Nothing in the room makes speaking harmful or intrusive; this is an ordinary "
            "conversation, including one about a difficult topic discussed calmly."
        ),
    ),
)

_ADDRESSEE = ChoiceQuestion(
    id=Q_ADDRESSEE,
    instructions="Who is the current utterance directed at?",
    options=(
        (
            "robot",
            "The robot named at the top of the state, whether or not the speaker used its name.",
        ),
        (
            "other_human",
            "One specific other person in the room, for example the previous speaker.",
        ),
        (
            "self_or_group",
            "Nobody in particular: the speaker is thinking out loud, reading aloud, or "
            "talking to the room as a whole.",
        ),
        (
            "unclear",
            "The words do not say who is being addressed and the recent turns do not settle "
            "it either.",
        ),
    ),
)

_EMOTION = ChoiceQuestion(
    id=Q_EMOTION,
    instructions=(
        "Which expression should the robot's face show while it listens to the current "
        "utterance? Choose the expression it would be right for the robot to wear, not a "
        "diagnosis of what the speaker is feeling."
    ),
    options=(
        (
            "neutral",
            "Nothing here calls for an expression: a calm, unremarkable resting face.",
        ),
        (
            "attentive",
            "Plain listening. The robot is following along and shows quiet engagement and "
            "nothing more.",
        ),
        (
            "warm",
            "The utterance is friendly, personal or kind, and the robot should look warm "
            "towards the speaker.",
        ),
        (
            "happy",
            "The speaker shares good news or something they are pleased about, and the "
            "robot should look pleased with them.",
        ),
        (
            "amused",
            "The utterance is a joke, a tease or something playful, and the robot should "
            "look amused.",
        ),
        (
            "curious",
            "The utterance raises something new, odd or intriguing, and the robot should "
            "look interested and inquisitive.",
        ),
        (
            "surprised",
            "The utterance is unexpected or startling, and the robot should look taken aback.",
        ),
        (
            "concerned",
            "Something is wrong, risky or worrying, and the robot should look concerned.",
        ),
        (
            "sympathetic",
            "The speaker is having a hard time, and the robot should look gentle and "
            "sympathetic rather than alarmed.",
        ),
        (
            "sad",
            "The utterance is about a loss or genuinely sad news, and sadness is the honest "
            "expression for the robot to show.",
        ),
    ),
)

_EMOTION_INTENSITY = ScoreQuestion(
    id=Q_EMOTION_INTENSITY,
    instructions=(
        "How strongly should that expression be shown on the robot's face while it listens "
        "to the current utterance?"
    ),
    levels=(
        "barely perceptible, almost a neutral face",
        "clearly visible but restrained",
        "moderate and unmistakable",
        "strong and hard to miss",
        "as strong as the face can show",
    ),
)

_MEMORY = (
    NoulQuestion(
        id=Q_WORTH_REMEMBERING,
        instructions=(
            "Does the current utterance state something about a person that the robot "
            "should still know next week?"
        ),
        when_true=(
            "It states a durable fact: a name, a relationship, a job, where someone lives, "
            "a lasting preference, an allergy or condition, or a dated plan or occasion "
            "that will still matter in the coming days."
        ),
        when_false=(
            "Small talk, a passing mood, a comment on what is happening right now, or a "
            "request that is finished once it has been answered. Anything that stops "
            "mattering when this conversation ends."
        ),
    ),
    NoulQuestion(
        id=Q_SENSITIVE_PERSONAL,
        instructions=(
            "Is the personal information in the current utterance sensitive, so that "
            "storing it or repeating it aloud in front of other people could embarrass or "
            "harm the speaker?"
        ),
        when_true=(
            "Health, medical appointments, diagnoses or medication; money, debt or legal "
            "trouble; religion, politics or sexuality; family conflict; or anything else "
            "the speaker would clearly not want said back in front of others."
        ),
        when_false=(
            "Everyday information the speaker would not mind the robot repeating in front "
            "of other people, or nothing personal at all."
        ),
    ),
)

_MEMORY_KIND = ChoiceQuestion(
    id=Q_MEMORY_KIND,
    instructions=(
        "If the current utterance holds something durable worth remembering, which kind of "
        "fact is it?"
    ),
    options=(
        (
            "personal_fact",
            "A stable fact about a person: their name, where they live or work, their job, "
            "a language they speak.",
        ),
        ("preference", "Something a person likes, dislikes, wants or avoids."),
        (
            "event",
            "Something that happened, or a recurring or dated occasion such as a birthday "
            "or an anniversary.",
        ),
        ("plan", "Something a person intends to do at a future time."),
        (
            "relationship",
            "How two people are connected: family, partner, colleague, friend, or a pet and "
            "its owner.",
        ),
        (
            "health",
            "Health or medical information: a condition, an allergy, medication, or an "
            "appointment.",
        ),
        (
            "other",
            "A durable fact that none of the other kinds fits, or nothing durable in the "
            "utterance at all.",
        ),
    ),
)

#: The frozen bundle. One request per finalized utterance answers all of it in parallel.
QUESTION_BUNDLE_V1 = QuestionBundle(
    version=BUNDLE_VERSION,
    questions=(
        *_ADMISSION,
        _ADDRESSEE,
        _EMOTION,
        _EMOTION_INTENSITY,
        *_MEMORY,
        _MEMORY_KIND,
    ),
)

#: Asked in a separate call, off the critical path, once a ChatLLM has written a memory
#: sentence. The candidate sentence is appended to the state under "Proposed memory:".
MEMORY_FAITHFUL_QUESTION = NoulQuestion(
    id=Q_MEMORY_FAITHFUL,
    instructions=("Is the proposed memory statement fully supported by the turns shown above it?"),
    when_true=(
        "Everything the statement says was said, or is unmistakably implied, in those "
        "turns, it is attributed to the right speaker, and it adds nothing that was not "
        "said."
    ),
    when_false=(
        "The statement adds, changes, exaggerates or guesses at any detail the turns do not "
        "actually contain, or it attributes it to the wrong person."
    ),
)
