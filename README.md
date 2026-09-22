# rt-agent

`rt-agent` is a hardware-free "listening robot" harness. It consumes VAD-segmented, diarized
transcript events and, for each finalized utterance, asks a System One model (hosted Jev at
`api.typesafe.ai` or a local Kev server, same wire protocol) a single frozen bundle of typed
questions. From the calibrated answers it decides speak or wait, picks the robot's facial
emotion preset, and decides whether the utterance is worth remembering — with a fail-closed
policy that waits whenever the backend is slow, unavailable or ambiguous. Speaking and memory
writing delegate to a larger remote LLM; the face receives `affect-cue/v1` vectors and
alice-compatible `speech-clause/v1` units.

See docs/ARCHITECTURE.md.
