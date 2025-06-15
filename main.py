import hashlib
import json
import logging
import os
import re
import struct
import sys
import time
import wave
import csv
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
import signal
import asyncio

from langflow_datastax_endpoint import query_hosted_langflow

LOG_FILE = "interaction_log.csv"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_interaction(question, answer):
    with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as logf:
        writer = csv.writer(logf, delimiter='\t')
        writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), question, answer])

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

class TPSProfiler:
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

    async def tick(self):
        await asyncio.sleep(0)
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

    async def tick(self):
        await asyncio.sleep(0)
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
        model_id='eleven_monolingual_v1',
        sample_rate=24000,
        cache_dir='tts_cache',
        previous_text_input=None):

    os.makedirs(cache_dir, exist_ok=True)

    text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
    cache_path = os.path.join(cache_dir, f"{text_hash}.pcm")

    if os.path.exists(cache_path):
        logging.info(f"Using cached PCM audio for text: {text}")
        with open(cache_path, 'rb') as f:
            raw_pcm = f.read()
    else:
        logging.info(f"Fetching PCM audio from ElevenLabs API for text: {text}")
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream?output_format=pcm_{sample_rate}"
        headers = {
            "xi-api-key": api_key,
        }
        json_payload = {
            "text": text,
            "model_id": model_id,
            "optimize_streaming_latency": 1,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.7
            }
        }

        if previous_text_input:
            json_payload["previous_text"] = previous_text_input

        try:
            with requests.post(url, json=json_payload, headers=headers, stream=True) as r:
                r.raise_for_status()
                raw_pcm = b''.join(r.iter_content(chunk_size=4096))
            logging.info("PCM audio fetched successfully.")
            with open(cache_path, 'wb') as f:
                f.write(raw_pcm)
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching audio from ElevenLabs API: {e}")
            raise

    # Convert raw PCM bytes to list of 16-bit samples
    try:
        pcm_samples = struct.unpack('<' + 'h' * (len(raw_pcm) // 2), raw_pcm)
        logging.info("PCM audio decoded successfully.")
    except Exception as e:
        logging.error(f"Error decoding PCM audio: {e}")
        raise

    pcm_chunk_size = 2048
    for i in range(0, len(pcm_samples), pcm_chunk_size):
        yield pcm_samples[i:i + pcm_chunk_size]

class Generator:
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
        self._last_user_question = None
        self._last_answer_accum = []
        self._last_question_time = None

    def close(self):
        try:
            self.llm_connection.send({'command': Commands.CLOSE})
            self.llm_process.join(1.0)
        except Exception as e:
            sys.stderr.write(str(e))
            self.llm_process.kill()

    async def langflow_orchestrate(self, text: str):
        orchestrated_response = await self.send_to_langflow(text)
        return orchestrated_response

    async def send_to_langflow(self, text: str):
        return f"Orchestrated response for: {text}"

    async def process(self, text: str, utterance_end_sec):
        ppn_prompt = self.config['ppn_prompt']
        print(f'LLM (say {ppn_prompt} to interrupt) > ', end='', flush=True)

        self._last_answer_accum = []
        orchestrated_response = await self.langflow_orchestrate(text)

        self.synthesizer.start(utterance_end_sec)
        self.llm_connection.send({'command': Commands.PROCESS, 'text': orchestrated_response})

    def interrupt(self):
        self.llm_connection.send({'command': Commands.INTERRUPT})
        self.synthesizer.interrupt()

    async def tick(self):
        await asyncio.sleep(0)
        while self.llm_connection.poll():
            message = self.llm_connection.recv()
            if message['command'] == Commands.SYNTHESIZE:
                print(message['text'], end='', flush=True)
                self._last_answer_accum.append(message['text'])
                self.synthesizer.process(message['text'])
            elif message['command'] == Commands.FLUSH:
                print('', flush=True)
                if self.config['profile']:
                    tps = message['profile']
                    print(f'[LLM TPS: {round(tps, 2)}]')
                self.synthesizer.flush()
                try:
                    if self._last_user_question and self._last_answer_accum:
                        log_interaction(
                            self._last_user_question,
                            ''.join(self._last_answer_accum)
                        )
                except Exception as e:
                    logging.error(f"Failed to log interaction: {e}")

    @staticmethod
    def create_worker(config):
        main_connection, process_connection = Pipe()
        process = Process(target=Generator.worker, args=(process_connection, config))
        process.start()
        return main_connection, process

    @staticmethod
    def worker(connection: Connection, config):

        def handler(_, __) -> None:
            pass

        signal.signal(signal.SIGINT, handler)

        llm_info = {
            'version': 'Local Langflow'
        }
        connection.send(llm_info)

        openai_profiler = TPSProfiler()

        close_flag = [False]
        prompt = [None]
        interrupt_requested = [False]

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

                    openai_profiler.reset()
                    interrupt_requested[0] = False

                    try:
                        response = query_hosted_langflow(user_text)

                        sentences = re.split(r'(?<=[.!?]) +', response)
                        for sentence in sentences:
                            if interrupt_requested[0]:
                                raise KeyboardInterrupt
                            openai_profiler.tock()
                            connection.send({'command': Commands.SYNTHESIZE, 'text': sentence})

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
        self.initial_silence_duration = config.get('initial_silence_duration_sec')
        self.last_voice_time = 0.0
        self.speech_started = False
        self.recording_start_time = 0.0
        self.max_recording_duration = config.get('max_recording_time')
        logging.info(
            f"Listener initialized with silence_threshold={self.silence_threshold}, "
            f"silence_duration={self.silence_duration}, max_duration={self.max_recording_duration}s"
        )

    async def process(self, pcm: Sequence[int]):
        if not self.recording:
            if self.porcupine.process(pcm) == 0:
                print("\n$ Wake word detected! Recording now … (max 30 seconds)")
                self.recording = True
                self.recording_start_time = time.perf_counter()
                self.last_voice_time = self.recording_start_time
                self.audio_buffer.clear()
                self.speech_started = False
                logging.info("Started recording after wake word detection")
            return

        now = time.perf_counter()

        if now - self.recording_start_time >= self.max_recording_duration:
            logging.warning("Maximum recording duration (30s) exceeded - ending utterance")
            await self._end_utterance()
            return

        peak = max(abs(sample) for sample in pcm) if pcm else 0
        logging.debug(f"Current peak amplitude: {peak}")

        if peak > self.silence_threshold:
            self.last_voice_time = now
            self.speech_started = True
            logging.debug("Voice detected, reset silence timer")

        for sample in pcm:
            self.audio_buffer += int(sample).to_bytes(2, 'little', signed=True)

        silence_time = now - self.last_voice_time
        current_silence_threshold = self.silence_duration if self.speech_started else self.initial_silence_duration
        logging.debug(f"Current silence duration: {silence_time:.2f}s")

        if silence_time >= current_silence_threshold:
            if self.speech_started or silence_time >= self.initial_silence_duration:
                logging.info(f"Silence detected for {silence_time:.2f}s - ending utterance")
                await self._end_utterance()

    async def _end_utterance(self):
        self.recording = False
        logging.info(f"Processing utterance of {len(self.audio_buffer)} bytes")

        wav = pcm_bytes_to_wav(
            bytes(self.audio_buffer),
            sample_rate=self.config['wav_sample_rate']
        )
        self.audio_buffer.clear()

        try:
            openai.api_key = self.config['openai_api_key']

            wav.name = 'audio.wav'  # Whisper API expects file-like with .name
            resp = openai.audio.transcriptions.create(
                file=wav,
                model="whisper-1",
                response_format="text"
            )

            text = resp.strip()
            logging.info(f"Transcription received: {text!r}")
            self.generator._last_user_question = text
            self.generator._last_question_time = time.strftime('%Y-%m-%d %H:%M:%S')
            await self.generator.process(text, utterance_end_sec=None)

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

    async def tick(self):
        await asyncio.sleep(0)
        if not self.recording:
            self.recording = True
            self.recorder.start()
        pcm = self.recorder.read()
        await self.listener.process(pcm)

async def run_agents(config):
    stop = [False]

    def handler(_, __) -> None:
        stop[0] = True

    signal.signal(signal.SIGINT, handler)

    llm_connection, llm_process = Generator.create_worker(config)

    tts_connection, tts_process = Synthesizer.create_worker(config)

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

    pv_recorder = PvRecorder(frame_length=porcupine.frame_length)
    pv_speaker = PvSpeaker(sample_rate=int(tts_connection.recv()), bits_per_sample=16, buffer_size_secs=1)

    llm_info = llm_connection.recv()
    print(f"→ Langflow LLM: {llm_info['version']}")

    tts_info = tts_connection.recv()
    print(f"→ tts {tts_info['version']}")

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

            await asyncio.gather(
                recorder.tick(),
                generator.tick(),
                synthesizer.tick(),
                speaker.tick()
            )
    finally:
        recorder.close()
        if hasattr(listener, "close"):
            await listener.close()
        await generator.close()
        await synthesizer.close()
        await speaker.close()

        for child in active_children():
            child.kill()

        porcupine.delete()
        pv_recorder.delete()
        pv_speaker.delete()

if __name__ == '__main__':
    config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.json')
    with open(config_path, 'r') as fd:
        config = json.load(fd)
    asyncio.run(run_agents(config))
