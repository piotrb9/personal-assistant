import hashlib
import json
import logging
import os
import signal
import sys
import time
import wave
from multiprocessing import Pipe, Process, Queue, active_children
from multiprocessing.connection import Connection
from threading import Thread
from typing import Optional, Sequence

import openai
import pvporcupine
from pvrecorder import PvRecorder
from pvspeaker import PvSpeaker

import requests
from pydub import AudioSegment
import io

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class Commands:
    START = 'start'
    CLOSE = 'close'
    PROCESS = 'process'
    SYNTHESIZE = 'synthesize'
    SPEAK = 'speak'
    FLUSH = 'flush'
    INTERRUPT = 'interrupt'


class RTFProfiler:
    def __init__(self, sample_rate: int) -> None:
        self._sample_rate = sample_rate
        self._compute_sec = 0.
        self._audio_sec = 0.
        self._tick_sec = 0.

    def tick(self) -> None:
        self._tick_sec = time.perf_counter()

    def tock(self, audio: Optional[Sequence[int]] = None) -> None:
        self._compute_sec += time.perf_counter() - self._tick_sec
        self._audio_sec += (len(audio) / self._sample_rate) if audio is not None else 0.

    def rtf(self) -> float:
        if self._audio_sec > 0:
            rtf = self._compute_sec / self._audio_sec
        else:
            rtf = 0
        self._compute_sec = 0.
        self._audio_sec = 0.
        return rtf

    def reset(self) -> None:
        self._compute_sec = 0.
        self._audio_sec = 0.
        self._tick_sec = 0.


class TPSProfiler(object):
    """
    Used to measure tokens per second. Useful if you want to see how fast a streaming
    generation is proceeding. Every time we receive "new tokens" from the stream,
    we call tock().
    """

    def __init__(self) -> None:
        self._num_tokens = 0
        self._start_sec = 0.

    def tock(self) -> None:
        if self._start_sec == 0.:
            self._start_sec = time.perf_counter()
        else:
            self._num_tokens += 1

    def tps(self) -> float:
        tps = self._num_tokens / (time.perf_counter() - self._start_sec) if self._start_sec > 0 else 0
        self._num_tokens = 0
        self._start_sec = 0.
        return tps

    def reset(self) -> None:
        self._num_tokens = 0
        self._start_sec = 0.


class CompletionText(object):
    """
    A helper to accumulate text from partial streaming responses and check
    if any stop phrases are encountered. This logic is carried over from
    the original code so that you can optionally intercept known EOS tokens
    (e.g. for different model families) if desired.
    """

    def __init__(self, stop_phrases: list) -> None:
        self.stop_phrases = stop_phrases
        self.start: int = 0
        self.text: str = ''
        self.new_tokens: str = ''
        logging.info("CompletionText initialized with stop phrases: %s", stop_phrases)

    def reset(self):
        logging.info("Resetting CompletionText state.")
        self.start = 0
        self.text = ''
        self.new_tokens = ''

    def append(self, text: str) -> None:
        if text is None:
            logging.debug("No text to append.")
            return
        logging.debug("Appending text: %s", text)
        self.text += text
        end = len(self.text)

        for stop_phrase in self.stop_phrases:
            if stop_phrase in self.text:
                contains = self.text.index(stop_phrase)
                if end > contains:
                    end = contains
                logging.info("Stop phrase detected: %s", stop_phrase)
            # partial overlap check
            for i in range(len(stop_phrase) - 1, 0, -1):
                if self.text.endswith(stop_phrase[:i]):
                    ends = len(self.text) - i
                    if end > ends:
                        end = ends
                    logging.debug("Partial overlap detected with stop phrase: %s", stop_phrase[:i])
                    break

        start = self.start
        self.start = end
        self.new_tokens = self.text[start:end]
        logging.debug("New tokens extracted: %s", self.new_tokens)

    def get_new_tokens(self) -> str:
        logging.debug("Retrieving new tokens: %s", self.new_tokens)
        return self.new_tokens


class Speaker:
    def __init__(
            self,
            speaker: PvSpeaker,
            config):
        self.speaker = speaker
        self.config = config
        self.speaker_warmup = self.speaker.sample_rate * self.config['speaker_warmup_sec']
        self.started = False
        self.speaking = False
        self.flushing = False
        self.pcmBuffer = []
        self.future = None
        logging.info("Speaker initialized.")

    def close(self):
        logging.info("Closing speaker.")
        self.interrupt()

    def start(self):
        logging.info("Starting speaker.")
        self.started = True

    def process(self, pcm: Optional[Sequence[int]]):
        if self.started and pcm is not None:
            logging.debug(f"Processing PCM data of length: {len(pcm)}.")
            self.pcmBuffer.extend(pcm)

    def flush(self):
        logging.info("Flushing speaker.")
        self.flushing = True

    def interrupt(self):
        logging.info("Interrupting speaker.")
        self.started = False
        if self.speaking:
            logging.info("Stopping speaker and clearing buffer.")
            self.speaking = False
            self.flushing = False
            self.pcmBuffer.clear()
            self.speaker.stop()

    def tick(self):
        def stop():
            logging.info("Stopping speaker after flush.")
            self.speaker.flush()
            self.speaker.stop()
            ppn_prompt = self.config['ppn_prompt']
            logging.info(f"Prompt user to say: {ppn_prompt}.")
            print(f'$ Say {ppn_prompt} ...', flush=True)

        if not self.speaking and len(self.pcmBuffer) > self.speaker_warmup:
            logging.info("Starting speaker playback.")
            self.speaking = True
            self.speaker.start()

        if self.speaking and len(self.pcmBuffer) > 0:
            written = self.speaker.write(self.pcmBuffer)
            logging.debug(f"Written {written} samples to speaker.")
            if written > 0:
                del self.pcmBuffer[:written]

        elif self.speaking and self.flushing and len(self.pcmBuffer) == 0:
            logging.info("Speaker buffer empty, stopping playback.")
            self.started = False
            self.speaking = False
            self.flushing = False
            Thread(target=stop).start()


class Synthesizer:
    """
    Synthesizer using OpenAI TTS endpoint.
    """

    def __init__(self, speaker: Speaker, tts_connection, tts_process, config):
        self.speaker = speaker
        self.tts_connection = tts_connection
        self.tts_process = tts_process
        self.config = config
        logging.info("Synthesizer initialized.")

    def close(self):
        try:
            logging.info("Closing synthesizer.")
            self.tts_connection.send({'command': Commands.CLOSE})
            self.tts_process.join(1.0)
        except Exception as e:
            logging.error(f"Error while closing synthesizer: {e}")
            self.tts_process.kill()

    def start(self, utterance_end_sec):
        logging.info(f"Starting synthesizer with utterance_end_sec={utterance_end_sec}.")
        self.speaker.start()
        self.tts_connection.send({'command': Commands.START, 'utterance_end_sec': utterance_end_sec})

    def process(self, text: str):
        logging.info(f"Processing text for synthesis: {text}")
        self.tts_connection.send({'command': Commands.PROCESS, 'text': text})

    def flush(self):
        logging.info("Flushing synthesizer.")
        self.tts_connection.send({'command': Commands.FLUSH})

    def interrupt(self):
        try:
            logging.info("Interrupting synthesizer.")
            self.tts_connection.send({'command': Commands.INTERRUPT})
            self.speaker.interrupt()
        except Exception as e:
            logging.error(f"Error while interrupting synthesizer: {e}")

    def tick(self):
        while self.tts_connection.poll():
            message = self.tts_connection.recv()
            if message['command'] == Commands.SPEAK:
                logging.info("Received SPEAK command from TTS worker.")
                self.speaker.process(message['pcm'])
            elif message['command'] == Commands.FLUSH:
                logging.info("Received FLUSH command from TTS worker.")
                self.speaker.flush()

    @staticmethod
    def create_worker(config):
        logging.info("Creating TTS worker process.")
        main_connection, process_connection = Pipe()
        process = Process(target=Synthesizer.worker, args=(process_connection, config))
        process.start()
        logging.info("TTS worker process started.")
        return main_connection, process

    @staticmethod
    def worker(connection: Connection, config):
        logging.info("TTS worker process initialized.")

        def handler(_, __):
            pass

        signal.signal(signal.SIGINT, handler)
        sample_rate = 24000
        connection.send(sample_rate)
        connection.send({'version': 'ElevenLabs TTS'})
        close = False
        text_queue = Queue()
        text_buffer = ""
        sentence_delimiters = ".!?"
        last_synthesized_text = ""

        white_characters = ['', '.', ' ', '\n', ' .', '. ', ',', ' ,', ', ', '!', '?', '? ', ' ?', ' !', '! ', ' .',
                            '.']

        while not close:
            time.sleep(0.02)

            while connection.poll():
                message = connection.recv()
                if message['command'] == Commands.CLOSE:
                    close = True
                    break
                elif message['command'] == Commands.PROCESS:
                    logging.info(f"TTS worker received text chunk: {message['text']}")
                    text_queue.put(message['text'])
                elif message['command'] == Commands.INTERRUPT:
                    logging.info("TTS worker interrupted.")
                    while not text_queue.empty():
                        text_queue.get()
                    text_buffer = ""

            while not text_queue.empty():
                chunk = text_queue.get()
                text_buffer += chunk

                should_synthesize = False
                for delimiter in sentence_delimiters:
                    if delimiter in text_buffer:
                        should_synthesize = True
                        break

                if should_synthesize and len(
                        text_buffer.strip()) > 0:
                    sentence_to_synthesize = text_buffer.strip()
                    text_buffer = ""

                    # Do this to avoid sending empty strings to the TTS
                    if sentence_to_synthesize in white_characters:
                        continue

                    logging.info(f"TTS worker synthesizing buffered text: {sentence_to_synthesize}")
                    try:
                        for pcm_samples in synthesize_with_elevenlabs_streaming_tts(
                                text=sentence_to_synthesize,
                                api_key=config['elevenlabs_api_key'],
                                voice_id=config.get('elevenlabs_voice_id'),

                                model_id=config.get('elevenlabs_model_id'),
                                sample_rate=sample_rate,
                                previous_text_input=last_synthesized_text
                        ):
                            last_synthesized_text = sentence_to_synthesize
                            connection.send({'command': Commands.SPEAK, 'pcm': pcm_samples})
                    except Exception as e:
                        logging.error(f"Error during synthesis: {e}")

            if close and len(text_buffer.strip()) > 0:
                sentence_to_synthesize = text_buffer.strip()

                # Do this to avoid sending empty strings to the TTS
                if sentence_to_synthesize in white_characters:
                    continue

                text_buffer = ""
                logging.info(f"TTS worker synthesizing remaining buffered text: {sentence_to_synthesize}")
                try:
                    for pcm_samples in synthesize_with_elevenlabs_streaming_tts(
                            text=sentence_to_synthesize,
                            api_key=config['elevenlabs_api_key'],
                            voice_id=config.get('elevenlabs_voice_id'),
                            model_id=config.get('elevenlabs_model_id'),
                            sample_rate=sample_rate,
                            previous_text_input=last_synthesized_text):
                        connection.send({'command': Commands.SPEAK, 'pcm': pcm_samples})

                    last_synthesized_text = sentence_to_synthesize
                except Exception as e:
                    logging.error(f"Error during synthesis: {e}")


def synthesize_with_elevenlabs_streaming_tts(
        text,
        api_key,
        voice_id,
        model_id='eleven_flash_v2_5',
        sample_rate=24000,
        cache_dir='tts_cache',
        previous_text_input=None):
    # Ensure the cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    # Generate a unique filename based on the text
    text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
    cache_path = os.path.join(cache_dir, f"{text_hash}.mp3")

    # Check if the audio file already exists in the cache
    if os.path.exists(cache_path):
        logging.info(f"Using cached audio for text: {text}")
        with open(cache_path, 'rb') as f:
            buffer = f.read()
    else:
        logging.info(f"Fetching audio from ElevenLabs API for text: {text}")
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
        headers = {
            "xi-api-key": api_key,
        }
        json_payload = {
            "text": text,
            "model_id": model_id,
            "optimize_streaming_latency ": 1,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.7
            }
        }

        if previous_text_input:
            json_payload["previous_text"] = previous_text_input

        try:
            # Make the API request
            with requests.post(url, json=json_payload, headers=headers, stream=True) as r:
                r.raise_for_status()
                buffer = b"".join(r.iter_content(chunk_size=4096))
            logging.info("Audio fetched successfully from ElevenLabs API.")

            # Save the response to the cache
            with open(cache_path, 'wb') as f:
                f.write(buffer)
            logging.info(f"Audio saved to cache: {cache_path}")
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching audio from ElevenLabs API: {e}")
            raise

    # Decode the audio and yield PCM samples
    try:
        segment = AudioSegment.from_file(io.BytesIO(buffer), format="mp3")
        segment = segment.set_frame_rate(sample_rate).set_channels(1).set_sample_width(2)
        raw_pcm = segment.raw_data

        pcm_samples = [
            int.from_bytes(raw_pcm[i: i + 2], "little", signed=True)
            for i in range(0, len(raw_pcm), 2)
        ]

        # Log successful decoding
        logging.info("Audio decoded successfully.")
    except Exception as e:
        logging.error(f"Error decoding audio: {e}")
        raise

    # Yield PCM samples in chunks
    pcm_chunk_size = 2048
    for i in range(0, len(pcm_samples), pcm_chunk_size):
        yield pcm_samples[i: i + pcm_chunk_size]


class Generator:
    """
    This is the class responsible for calling the OpenAI API instead of picoLLM.
    It uses a separate process and streams partial tokens back to the parent.
    """

    def __init__(
            self,
            synthesizer: Synthesizer,
            llm_connection: Connection,
            llm_process: Process,
            config):
        self.synthesizer = synthesizer
        self.llm_connection = llm_connection
        self.llm_process = llm_process
        self.config = config

    def close(self):
        try:
            self.llm_connection.send({'command': Commands.CLOSE})
            self.llm_process.join(1.0)
        except Exception as e:
            sys.stderr.write(str(e))
            self.llm_process.kill()

    def process(self, text: str, utterance_end_sec):
        ppn_prompt = self.config['ppn_prompt']
        print(f'LLM (say {ppn_prompt} to interrupt) > ', end='', flush=True)

        self.synthesizer.start(utterance_end_sec)
        self.llm_connection.send({'command': Commands.PROCESS, 'text': text})

    def interrupt(self):
        self.llm_connection.send({'command': Commands.INTERRUPT})
        self.synthesizer.interrupt()

    def tick(self):
        while self.llm_connection.poll():
            message = self.llm_connection.recv()
            if message['command'] == Commands.SYNTHESIZE:
                print(message['text'], end='', flush=True)
                self.synthesizer.process(message['text'])
            elif message['command'] == Commands.FLUSH:
                print('', flush=True)
                if self.config['profile']:
                    tps = message['profile']
                    print(f'[OpenAI TPS: {round(tps, 2)}]')
                self.synthesizer.flush()

    @staticmethod
    def create_worker(config):
        """
        Creates the worker process that calls OpenAI's ChatCompletion in a loop
        for streaming tokens. We communicate with the parent over pipe.
        """
        main_connection, process_connection = Pipe()
        process = Process(target=Generator.worker, args=(process_connection, config))
        process.start()
        return main_connection, process

    @staticmethod
    def worker(connection: Connection, config):
        """
        The worker function that handles:
          - Listening for commands (CLOSE, INTERRUPT, PROCESS)
          - On PROCESS, it calls OpenAI's ChatCompletion, streaming partial tokens
          - Each partial token is sent to parent with command SYNTHESIZE
          - When done, we send command FLUSH
        """

        client = openai.OpenAI(api_key=config.get('openai_api_key'))

        def handler(_, __) -> None:
            pass

        signal.signal(signal.SIGINT, handler)

        # Configure OpenAI
        if 'openai_api_key' in config:
            openai.api_key = config['openai_api_key']
        else:
            print("Missing 'openai_api_key' in config.", file=sys.stderr)
            sys.exit(1)

        # Send startup message to the parent
        openai_info = {
            'version': 'OpenAI API (Python client)',
            'model': config.get('openai_model_name', 'gpt-3.5-turbo')
        }
        connection.send(openai_info)

        # Profiler for tokens
        openai_profiler = TPSProfiler()

        # Stop phrases
        stop_phrases = [
            '</s>', '<end_of_turn>', '<|endoftext|>', '<|eot_id|>', '<|end|>', '<|user|>', '<|assistant|>'
        ]
        completion = CompletionText(stop_phrases)

        close_flag = [False]
        prompt = [None]
        interrupt_requested = [False]

        # A separate thread to handle inbound commands
        def event_manager():
            while not close_flag[0]:
                message = connection.recv()
                if message['command'] == Commands.CLOSE:
                    close_flag[0] = True
                    interrupt_requested[0] = True
                    return
                elif message['command'] == Commands.INTERRUPT:
                    interrupt_requested[0] = True
                elif message['command'] == Commands.PROCESS:
                    prompt[0] = message['text']

        Thread(target=event_manager).start()

        try:
            while not close_flag[0]:
                if prompt[0] is not None:
                    user_text = prompt[0]
                    prompt[0] = None

                    # Add short answers instruction if enabled
                    short_answers_instruction = \
                        "You are a voice assistant. Your answers must be very brief but informative. Respond in only 1 short sentence."
                    if config['short_answers']:
                        user_text = f"{short_answers_instruction} {user_text}"

                    completion.reset()
                    openai_profiler.reset()
                    interrupt_requested[0] = False

                    # Build messages for ChatCompletion
                    messages = []
                    if config.get('openai_system_prompt'):
                        messages.append({"role": "system", "content": config['openai_system_prompt']})
                    messages.append({"role": "user", "content": user_text})

                    try:
                        # Make chat completion request with streaming
                        response = client.chat.completions.create(
                            model=config.get('openai_model_name'),
                            messages=messages,
                            max_tokens=config.get('openai_max_tokens', 256),
                            temperature=config.get('openai_temperature', 0.7),
                            stream=True
                        )

                        # Stream partial chunks
                        for chunk in response:
                            if interrupt_requested[0]:
                                raise KeyboardInterrupt
                            delta = chunk.choices[0].delta
                            if hasattr(delta, 'content'):
                                token_text = delta.content
                                openai_profiler.tock()
                                completion.append(token_text)
                                new_tokens = completion.get_new_tokens()
                                if len(new_tokens) > 0:
                                    connection.send({'command': Commands.SYNTHESIZE, 'text': new_tokens})

                        # Send FLUSH command after completion
                        connection.send({'command': Commands.FLUSH, 'profile': openai_profiler.tps()})

                    except KeyboardInterrupt:
                        pass

                else:
                    time.sleep(0.1)
        finally:
            pass


def pcm_bytes_to_wav(pcm_bytes: bytes, sample_rate: int = 16000) -> io.BytesIO:
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    buf.seek(0)
    return buf


class Listener:
    def __init__(self, generator, porcupine: pvporcupine.Porcupine, config):
        self.generator = generator
        self.porcupine = porcupine
        self.config = config
        self.recording = False
        self.audio_buffer = bytearray()
        self.silence_threshold = config.get('silence_threshold')
        self.silence_duration = config.get('silence_duration_sec')
        self.initial_silence_duration = config.get('initial_silence_duration_sec')  # seconds to start speaking
        self.last_voice_time = 0.0
        self.speech_started = False
        logging.info(
            f"Listener initialized with silence_threshold={self.silence_threshold}, silence_duration={self.silence_duration}")

    def process(self, pcm: Sequence[int]):
        if not self.recording:
            if self.porcupine.process(pcm) == 0:
                print("\n$ Wake word detected! Recording now …")
                self.recording = True
                self.last_voice_time = time.perf_counter()
                self.audio_buffer.clear()
                self.speech_started = False  # Reset speech detection
                logging.info("Started recording after wake word detection")
            return

        now = time.perf_counter()
        # Compute peak amplitude and log it
        peak = max(abs(sample) for sample in pcm) if pcm else 0
        logging.debug(f"Current peak amplitude: {peak}")

        if peak > self.silence_threshold:
            self.last_voice_time = now
            self.speech_started = True
            logging.debug("Voice detected, reset silence timer")

        # Append raw pcm to buffer
        for sample in pcm:
            self.audio_buffer += int(sample).to_bytes(2, 'little', signed=True)

        silence_time = now - self.last_voice_time
        current_silence_threshold = self.silence_duration if self.speech_started else self.initial_silence_duration
        logging.debug(f"Current silence duration: {silence_time:.2f}s")

        if silence_time >= current_silence_threshold:
            # Only end utterance if speech started or initial silence timeout reached
            if self.speech_started or silence_time >= self.initial_silence_duration:
                logging.info(f"Silence detected for {silence_time:.2f}s - ending utterance")
                self._end_utterance()

    def _end_utterance(self):
        self.recording = False
        logging.info(f"Processing utterance of {len(self.audio_buffer)} bytes")

        wav = pcm_bytes_to_wav(
            bytes(self.audio_buffer),
            sample_rate=self.config['wav_sample_rate']
        )
        self.audio_buffer.clear()

        try:
            openai.api_key = self.config['openai_api_key']

            # Convert to MP3
            audio = AudioSegment.from_wav(wav)
            mp3_buffer = io.BytesIO()
            audio.export(mp3_buffer, format='mp3', bitrate='32k')
            mp3_buffer.seek(0)

            # Create a proper file object for OpenAI API
            mp3_file = io.BytesIO(mp3_buffer.getvalue())
            mp3_file.name = 'audio.mp3'

            logging.info("Sending compressed audio to OpenAI API")
            resp = openai.audio.transcriptions.create(
                file=mp3_file,  # Use file object with name
                model="whisper-1",
                response_format="text"
            )

            text = resp.strip()
            logging.info(f"Transcription received: {text!r}")
            self.generator.process(text, utterance_end_sec=None)

        except Exception as e:
            logging.error(f"Error during transcription: {e}")


class Recorder:
    def __init__(
            self,
            listener: Listener,
            recorder: PvRecorder):
        self.listener = listener
        self.recorder = recorder
        self.recording = False

    def close(self):
        if self.recording:
            self.recorder.stop()

    def tick(self):
        if not self.recording:
            self.recording = True
            self.recorder.start()
        pcm = self.recorder.read()
        self.listener.process(pcm)


def main(config):
    stop = [False]

    def handler(_, __) -> None:
        stop[0] = True

    signal.signal(signal.SIGINT, handler)

    # Create the LLM worker (OpenAI)
    llm_connection, llm_process = Generator.create_worker(config)

    # Create TTS worker
    tts_connection, tts_process = Synthesizer.create_worker(config)

    # Setup Porcupine
    if 'keyword_model_path' not in config:
        porcupine = pvporcupine.create(
            access_key=config['access_key'],
            keywords=['picovoice'],
            sensitivities=[config['porcupine_sensitivity']])
        config['ppn_prompt'] = '`Picovoice`'
    else:
        porcupine = pvporcupine.create(
            access_key=config['access_key'],
            keyword_paths=[config['keyword_model_path']],
            sensitivities=[config['porcupine_sensitivity']])
        config['ppn_prompt'] = 'the wake word'

    print(f"→ Porcupine v{porcupine.version}")

    # Setup Audio In/Out
    pv_recorder = PvRecorder(frame_length=porcupine.frame_length)
    pv_speaker = PvSpeaker(sample_rate=int(tts_connection.recv()), bits_per_sample=16, buffer_size_secs=1)

    # Retrieve info from the LLM worker
    llm_info = llm_connection.recv()
    print(f"→ OpenAI LLM: {llm_info['version']} <{llm_info['model']}>")

    # Retrieve info from the TTS worker
    tts_info = tts_connection.recv()
    print(f"→ tts v{tts_info['version']}")

    speaker = Speaker(pv_speaker, config)
    synthesizer = Synthesizer(speaker, tts_connection, tts_process, config)
    generator = Generator(synthesizer, llm_connection, llm_process, config)
    listener = Listener(generator, porcupine, config)
    recorder = Recorder(listener, pv_recorder)

    ppn_prompt = config['ppn_prompt']
    print(f'$ Say {ppn_prompt} ...', flush=True)

    try:
        while not stop[0]:
            if not llm_process.is_alive() or not tts_process.is_alive():
                break

            recorder.tick()
            generator.tick()
            synthesizer.tick()
            speaker.tick()
    finally:
        recorder.close()
        listener.close()
        generator.close()
        synthesizer.close()
        speaker.close()

        for child in active_children():
            child.kill()

        porcupine.delete()
        pv_recorder.delete()
        pv_speaker.delete()


if __name__ == '__main__':
    config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.json')
    with open(config_path, 'r') as fd:
        config = json.load(fd)

    main(config)
