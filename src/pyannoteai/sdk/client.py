# MIT License

# Copyright (c) 2025 pyannoteAI

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import hashlib
import importlib.metadata
import io
import os
import time
import warnings
from pathlib import Path
from typing import Callable, Optional, Union

import requests
from requests import Response
from requests.exceptions import ConnectionError, HTTPError

__version__ = importlib.metadata.version("pyannoteai-sdk")


class PyannoteAIFailedJob(Exception):
    """Raised when a job failed on the pyannoteAI web API"""

    pass


class PyannoteAICanceledJob(Exception):
    """Raised when a job was canceled on the pyannoteAI web API"""

    pass


class _UploadingCallbackBytesIO(io.BytesIO):
    """BytesIO subclass that calls a callback during the upload process

    Parameters
    ----------
    callback : Callable
        Callback called during upload as `callback(total_in_bytes, completed_in_bytes)`
    total_size : int
        Total size to upload (in bytes)
    initial_bytes : bytes
        Initial bytes to upload
    """

    def __init__(
        self,
        callback: Callable,
        total_size: int,
        initial_bytes: bytes,
    ):
        self.total_size = total_size
        self._completed_size = 0
        self._callback = callback
        super().__init__(initial_bytes)

    def read(self, size=-1) -> bytes:
        data = super().read(size)
        self._completed_size += len(data)
        if self._callback:
            self._callback(
                total=self.total_size,
                completed=self._completed_size,
            )
        return data


class Client:
    """Official client for pyannoteAI web API

    Parameters
    ----------
    token : str, optional
        pyannoteAI API key created from https://dashboard.pyannote.ai.
        Defaults to using `PYANNOTEAI_API_TOKEN` environment variable.

    Usage
    -----

    # instantiate client for pyannoteAI web API
    >>> from pyannoteai.sdk import Client
    >>> client = Client(token="{PYANNOTEAI_API_KEY}")

    # upload your audio file to the pyannoteAI web API
    # and store it for a few hours for later re-use.
    >>> media_url = client.upload("/path/to/your/audio.wav")

    # initiate a diarization job on the pyannoteAI web API
    >>> job_id = client.diarize(media_url)

    # retrieve prediction from pyannoteAI web API
    >>> prediction = client.retrieve(job_id)
    """

    API_URL = "https://api.pyannote.ai/v1"

    def __init__(self, token: Optional[str] = None, **kwargs):
        self.token = token
        self.api_key = token or os.environ.get("PYANNOTEAI_API_KEY", "")

    def _raise_for_status(self, response: Response):
        """Raise an exception if the response status code is not 2xx"""

        if response.status_code == 401:
            raise HTTPError(
                """
Failed to authenticate to pyannoteAI API. Please create an API key on https://dashboard.pyannote.ai/ and
provide it either via `PYANNOTEAI_API_TOKEN` environment variable or with `token` parameter."""
            )

        # TODO: add support for other status code when
        # they are documented on docs.pyannote.ai

        response.raise_for_status()

    def _authenticated_get(self, route: str) -> Response:
        """Send GET authenticated request to pyannoteAI API

        Parameters
        ----------
        route : str
            API route to send the GET request to.

        Returns
        -------
        response : Response

        Raises
        ------
        ConnectionError
        HTTPError
        """

        try:
            response = requests.get(f"{self.API_URL}{route}", headers=self._headers)
        except ConnectionError:
            raise ConnectionError(
                """
Failed to connect to pyannoteAI web API. Please check your internet connection
or visit https://pyannote.openstatus.dev/ to check the status of the pyannoteAI web API."""
            )

        self._raise_for_status(response)

        return response

    def _authenticated_post(self, route: str, json: Optional[dict] = None) -> Response:
        """Send POST authenticated request to pyannoteAI web API

        Parameters
        ----------
        route : str
            API route to send the GET request to.
        json : dict, optional
            Request body to send with the POST request.

        Returns
        -------
        response : Response

        Raises
        ------
        ConnectionError
        HTTPError
        """

        try:
            response = requests.post(
                f"{self.API_URL}{route}", json=json, headers=self._headers
            )
        except ConnectionError:
            raise ConnectionError(
                """
Failed to connect to pyannoteAI web API. Please check your internet connection
or visit https://pyannote.openstatus.dev/ to check the status of the pyannoteAI web API."""
            )

        self._raise_for_status(response)

        return response

    @property
    def api_key(self):
        return self._api_key

    @api_key.setter
    def api_key(self, api_key: str) -> None:
        if not api_key:
            raise ValueError(
                """
Failed to authenticate to pyannoteAI web API. Please create an API key on https://dashboard.pyannote.ai/ and
provide it either via `PYANNOTEAI_API_TOKEN` environment variable or with `token` parameter."""
            )

        # store the API key and prepare the headers
        self._api_key = api_key
        self._headers = {
            "Authorization": f"Bearer {self._api_key}",
            "User-Agent": f"pyannoteAI-sdk-python/{__version__}",
            "Content-Type": "application/json",
        }
        # test authentication
        self._authenticated_get("/test")

    def _create_presigned_url(self, media_url: str) -> str:
        """Create a presigned URL to upload audio file to pyannoteAI platform

        Parameters
        ----------
        media_url : str
            Unique identifier used to retrieve the uploaded audio file on the pyannoteAI platform.
            Any combination of letters (a-z, A-Z), digits (0-9), and the characters -./  prefixed
            with media:// is allowed. One would usually use a string akin to a path on filesystem
            (e.g. "media://path/to/audio.wav").

        Returns
        -------
        url : str
            Presigned URL to upload audio file to pyannoteAI platform
        """

        response = self._authenticated_post("/media/input", json={"url": media_url})
        return response.json()["url"]

    def _hash_md5(self, file: Union[str, Path]) -> str:
        """Compute MD5 hash of a file (used for media_url when not provided)"""
        # source: https://stackoverflow.com/questions/3431825/how-to-generate-an-md5-checksum-of-a-file
        hash_md5 = hashlib.md5()
        with open(file, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def upload(
        self,
        audio: str | Path | dict[str, str|Path],
        media_url: Optional[str] = None,
        callback: Optional[Callable] = None,
    ) -> str:
        """Upload audio file to pyannoteAI platform

        Parameters
        ----------
        audio : str or Path or dict
            Path to audio file to be uploaded. Can be a "str" or "Path" instance, or a dict with an
            "audio" key (e.g. {"audio": "/path/to/audio.wav"}).
        media_url : str, optional
            Unique identifier used to retrieve the uploaded audio file on the pyannoteAI platform.
            Any combination of letters {a-z, A-Z}, digits {0-9}, and {-./} characters prefixed
            with 'media://' is allowed. One would usually use a string akin to a path on filesystem
            (e.g. "media://path/to/audio.wav"). Defaults to media://{md5-hash-of-audio-file}.
        callback : Callable, optional
            When provided, `callback` is called during the uploading process with the following signature:
                callback(total=...,     # number of bytes to upload
                         completed=...) # number of bytes uploaded)

        Returns
        -------
        media_url : str
            Same as the input `media_url` parameter when provided,
            or "media://{md5-hash-of-audio-file}" otherwise.
        """

        if isinstance(audio, dict):
            if "audio" not in audio:
                raise ValueError(
                    "When `audio` is a dict, it must provide the path to the audio file in 'audio' key."
                )
            audio = audio["audio"]

        # get the total size of the file to upload
        # to provide progress information to the hook
        total_size = os.path.getsize(audio)

        if media_url is None:
            media_url = f"media://{self._hash_md5(audio)}"

        # for now, only str and Path audio instances are supported
        with open(audio, "rb") as f:
            # wrap the file object in a _UploadingCallbackBytesIO instance
            # to allow calling the hook during the upload process
            data = _UploadingCallbackBytesIO(callback, total_size, f.read())

        if not (isinstance(media_url, str) and media_url.startswith("media://")):
            raise ValueError(
                f"""
Invalid media URI: {media_url}. Any combination of letters {{a-z, A-Z}}, digits {{0-9}},
and {{-./}} characters prefixed with 'media://' is allowed."""
            )

        # created the presigned URL to upload the audio file
        presigned_url = self._create_presigned_url(media_url)

        # upload the audio file to the presigned URL
        try:
            response = requests.put(presigned_url, data=data)
        except ConnectionError:
            raise ConnectionError(
                f"""
Failed to upload audio to presigned URL {presigned_url}.
Please check your internet connection or visit https://pyannote.openstatus.dev/ to check the status of the pyannoteAI API."""
            )

        # TODO: handle HTTPError returned by the API
        response.raise_for_status()

        warnings.warn("""
You are using pyannoteAI's temporary storage solution. Your file will be permanently deleted from our servers within 24hs. 
If you are running in production, we highly recommend to use your own storage to reduce network latency and obtain results faster. 
Please check our documentation at https://docs.pyannote.ai/ for more information.""")

        return media_url

    def diarize(
        self,
        media_url: str,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        confidence: bool = False,
        turn_level_confidence: bool = False,
        exclusive: bool = False,
        model: str = "precision-2",
        **kwargs,
    ) -> str:
        """Initiate a diarization job on the pyannoteAI web API

        Parameters
        ----------
        media_url : str
            media://{...} URL created with the `upload` method or
            any other public URL pointing to an audio file.
        num_speakers : int, optional
            Force number of speakers to diarize. If not provided, the
            number of speakers will be determined automatically.
        min_speakers : int, optional
            Minimum number of speakers.
        max_speakers : int, optional
            Maximum number of speakers.
        confidence : bool, optional
            Enable confidence scores.
        turn_level_confidence: bool, optional
            Enable turn-based confidence scores.
        exclusive: bool, optional
            Enable exclusive speaker diarization.
        model : str, optional
            Defaults to "precision-2"
        **kwargs : optional
            Extra arguments to send in the body of the request.

        Returns
        -------
        job_id: str

        Raises
        ------
        HTTPError
            If something else went wrong
        """

        json = {
            "url": media_url,
            "model": model,
            "numSpeakers": num_speakers,
            "minSpeakers": min_speakers,
            "maxSpeakers": max_speakers,
            "confidence": confidence,
            "turnLevelConfidence": turn_level_confidence,
            "exclusive": exclusive,
        }
        # add extra arguments to the request body
        json.update(kwargs)

        response = self._authenticated_post("/diarize", json=json)
        data = response.json()
        return data["jobId"]

    def voiceprint(
        self,
        media_url: str,
        model: str = "precision-2",
        **kwargs,
    ) -> str:
        """Initiate a voiceprint job on the pyannoteAI web API

        Parameters
        ----------
        media_url : str
            media://{...} URL created with the `upload` method or
            any other public URL pointing to an audio file.
        model : str, optional
            Defaults to "precision-2".
        **kwargs : optional
            Extra arguments to send in the body of the request.

        Returns
        -------
        job_id: str

        Raises
        ------
        HTTPError
            If something else went wrong
        """

        json = {"url": media_url, "model": model}
        # add extra arguments to the request body
        json.update(kwargs)

        response = self._authenticated_post("/voiceprint", json=json)
        data = response.json()
        return data["jobId"]

    def identify(
        self,
        media_url: str,
        voiceprints: dict[str, str],
        exclusive_matching: bool = True,
        matching_threshold: float = 0.0,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        confidence: bool = False,
        turn_level_confidence: bool = False,
        exclusive: bool = False,
        model: str = "precision-2",
        **kwargs,
    ) -> str:
        """Initiate an identification job on the pyannoteAI web API

        Parameters
        ----------
        media_url : str
            media://{...} URL created with the `upload` method or
            any other public URL pointing to an audio file.
        voiceprints : dict
            Voiceprints.
        exclusive_matching : bool, optional
            Prevent multiple speakers from being matched to the same voiceprint.
            Defaults to True.
        matching_threshold : float, optional
            Prevent matching if confidence score is below this threshold.
            Value is between 0 and 100. Default is 0, meaning all voiceprints are matched.
        num_speakers : int, optional
            Force number of speakers to diarize. If not provided, the
            number of speakers will be determined automatically.
        min_speakers : int, optional
            Minimum number of speakers.
        max_speakers : int, optional
            Maximum number of speakers.
        confidence : bool, optional
            Enable confidence scores.
        turn_level_confidence: bool, optional
            Enable turn-based confidence scores.
        exclusive: bool, optional
            Enable exclusive speaker diarization.
        model : str, optional
            Defaults to "precision-2"
        **kwargs : optional
            Extra arguments to send in the body of the request.

        Returns
        -------
        job_id: str

        Raises
        ------
        HTTPError
            If something else went wrong
        """

        json = {
            "url": media_url,
            "model": model,
            "numSpeakers": num_speakers,
            "minSpeakers": min_speakers,
            "maxSpeakers": max_speakers,
            "confidence": confidence,
            "turnLevelConfidence": turn_level_confidence,
            "exclusive": exclusive,
            "voiceprints": [
                {"label": speaker, "voiceprint": voiceprint}
                for speaker, voiceprint in voiceprints.items()
            ],
            "matching": {
                "exclusive": exclusive_matching,
                "threshold": matching_threshold,
            },
        }
        # add extra arguments to the request body
        json.update(kwargs)

        response = self._authenticated_post("/identify", json=json)
        data = response.json()
        return data["jobId"]

    def retrieve(self, job_id: str, every_seconds: int = 10) -> dict:
        """Retrieve output of a job (once completed) from pyannoteAI web API

        Parameters
        ----------
        job_id : str
            Job ID.

        Returns
        -------
        job_output : dict
            Job output

        Raises
        ------
        PyannoteAIFailedJob
            If the job failed
        PyannoteAICanceledJob
            If the job was canceled
        HTTPError
            If something else went wrong
        """

        job_status = None

        while True:
            job = self._authenticated_get(f"/jobs/{job_id}").json()
            job_status = job["status"]

            if job_status not in ["succeeded", "canceled", "failed"]:
                time.sleep(every_seconds)
                continue

            break

        if job_status == "failed":
            error = job.get("output", dict()).get("error", "Please contact support.")
            raise PyannoteAIFailedJob(error, job_id)

        if job_status == "canceled":
            error = job.get("output", dict()).get("error", "Please contact support.")
            raise PyannoteAICanceledJob(error, job_id)

        warnings.warn("""
You are using periodic polling to retrieve results. 
If you are running in production, we highly recommend to setup a webhook server to obtain results faster, as soon as they are available. 
Please check our documentation at https://docs.pyannote.ai/ for more information.""")

        return job
