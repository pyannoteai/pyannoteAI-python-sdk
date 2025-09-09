<p align="center">
  <a href="https://pyannote.ai/" target="blank"><img src="https://avatars.githubusercontent.com/u/162698670" width="64" /></a>
</p>

<div align="center">
    <h1>Official pyannoteAI Python SDK </h1>
</div>

## Installation

```bash
$ pip install pyannoteai-sdk
```

Then head over to [`dashboard.pyannote.ai`](https://dashboard.pyannote.ai) and create an API key.

## Speaker diarization

```python
# instantiate client
from pyannoteai.sdk import Client
client = Client("your-api-key")

# upload conversation file
media_url = client.upload('/path/to/conversation.wav')

# submit a diarization job
job_id = client.diarize(media_url)

# retrieve diarization
diarization = client.retrieve(job_id)
# diarization['output']['diarization'] contains diarization output
```

Use `help(client.diarize)` to learn about options.

## Speaker identification

```python
# create a voiceprint from a sample of Lex's voice
lex_url = client.upload('/path/to/lex.wav')
job_id = client.voiceprint(lex_url)
lex = client.retrieve(job_id)
# lex['output']['voiceprint'] contains Lex's voiceprint

# create a voiceprint from a sample of Mark's voice
mark_url = client.upload('/path/to/mark.wav')
job_id = client.voiceprint(mark_url)
mark = client.retrieve(job_id)
# mark['output']['voiceprint'] contains Mark's voiceprint

# use those voiceprints to track Lex and Mark in the conversation
voiceprints = {'lex': lex['output']['voiceprint'], 'mark': mark['output']['voiceprint']}
job_id = client.identify('/path/to/conversation.wav', voiceprints)
identification = client.retrieve(job_id)
# identification['output']['identification'] contains identification output
```

Use `help(client.identify)` to learn about options.
