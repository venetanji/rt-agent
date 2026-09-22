# Audio fixtures

## `librispeech-1272-128104-0000.wav`

- **Source**: LibriSpeech ASR corpus, `dev-clean` utterance `1272-128104-0000`
  (speaker 1272, chapter 128104), obtained through the Hugging Face dataset
  [`hf-internal-testing/librispeech_asr_dummy`](https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy),
  config `clean`, split `validation`, row 0.
- **Licence**: CC BY 4.0 (<https://creativecommons.org/licenses/by/4.0/>).
  LibriSpeech is derived from public-domain LibriVox recordings; see
  Panayotov, Chen, Povey and Khudanpur, *LibriSpeech: an ASR corpus based on public
  domain audio books*, ICASSP 2015, <https://www.openslr.org/12>.
- **Attribution**: Vassil Panayotov, Guoguo Chen, Daniel Povey, Sanjeev Khudanpur.
- **Format**: 16 kHz mono PCM16 WAV, 5.855 s, 187 404 bytes.
- **Reference transcript**: `MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND
  WE ARE GLAD TO WELCOME HIS GOSPEL`
- **No modifications** were made to the audio.

It is committed (under the 1 MB fixture budget) so the end-to-end audio test runs
offline. `tests/audio/conftest.py::ensure_speech_clip` re-downloads it from the
Hugging Face datasets-server rows API if it is ever missing.
