import asyncio
import io
import json
import time
import zipfile
import requests
import statistics
import os
import httpx
import random
import logging
import math
import secrets
from itertools import count

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
API_KEY = "Dummy"
TO_MS = 1000
DEFAULT_BASE_URL = "http://127.1.1.1:8000/v1"
DEFAULT_MAX_TOKENS = 128
DEFAULT_INPUT_TOKENS = 128
DEFAULT_FORCE_OUTPUT_LENGTH = True
MAX_INPUT_TOKENS = 131072
MAX_OUTPUT_TOKENS = 32768
OUTPUT_HINT = "Your answer must contain at least 128 tokens."
FALLBACK_CHARS_PER_TOKEN = 4.0
DEFAULT_CONCURRENCY_LEVELS = [1, 2, 4, 8]
MEASURED_ROUNDS = 10
WARMUP_ROUNDS = 1
DEFAULT_TTFT_SLO = 1000
DEFAULT_TPS_SLO = 20
DEFAULT_THINK_TIME = 45
OUTPUT_DIR = "out"
PROMPTS_FILE = "prompts.txt"
LOGO_PATH = "images/CORDATUS_LOGO.png"
REQUEST_TIMEOUT = 50
MAX_RETRIES = 0
DEFAULT_SEED = 0
TAG_CHARS = len("[000000-00000] ")
TOKEN_STEP = 128

def get_models_payload(url: str, api_key: str = API_KEY) -> Optional[Dict]:
    """
    Fetch the raw /models JSON payload from an OpenAI compatible server.

    Args:
        url: API base URL
        api_key: API key for authentication

    Returns:
        Raw response dict (e.g. {"object": "list", "data": [...]}),
        or None if the request failed or response was not valid JSON.
    """
    try:
        response = requests.get(
            f"{url}/models",
            timeout=REQUEST_TIMEOUT,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {api_key}"
            },
            verify=False
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.Timeout:
        logger.warning(f"Server timed out ({REQUEST_TIMEOUT}s)")
        return None
    except requests.exceptions.ConnectionError:
        logger.warning("Could not connect to server. Check URL.")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"API Error: {str(e)}")
        return None
    except ValueError as e:
        logger.error(f"Error parsing response: {str(e)}")
        return None


def get_model_list(url: str, api_key: str = API_KEY) -> List[str]:
    """
    Returns a list of model IDs from an OpenAI compatible server.
    The model list is dynamic and not cached.

    Args:
        url: API base URL
        api_key: API key for authentication

    Returns:
        List of model IDs
    """
    payload = get_models_payload(url, api_key=api_key)
    if not payload:
        return []

    model_ids = [model['id'] for model in payload.get('data', [])]

    if not model_ids:
        logger.warning("Could not retrieve model list from server.")

    return model_ids



class LLMBenchmark:
    """
    A class for benchmarking LLM models with various performance metrics.
    """

    def __init__(self, base_url: str, api_key: str, model: str,
                 force_output_length: bool = False):
        """
        Initialize the benchmark client.

        Args:
            base_url: The base URL of the API server
            api_key: The API key for authentication
            model: The model name to benchmark
            force_output_length: Ask the server to generate exactly max_tokens
                (ignore_eos + min_tokens; supported by vLLM and SGLang)
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.force_output_length = force_output_length
        self.target_chars = None
        self.run_id = ""

        self._build_client(read_timeout=300.0)

    def _build_client(self, read_timeout: float,
                      max_concurrency: int = 64) -> None:
        """
        (Re)build the OpenAI client with a given read timeout.

        Uses a generous initial read timeout so that the single-request
        calibration phase can complete before we tighten the value based
        on observed latency.

        The connection pool is sized from the highest concurrency level under
        test: httpx defaults to max 100 connections, which would queue requests
        at concurrency 128/256 and corrupt TTFT/latency measurements.

        The inter-chunk patience (chunk_timeout) also scales with the read
        timeout so that slow streams at high concurrency are not truncated
        early (truncation undercounts tokens and inflates TPS).

        MAX_RETRIES is 0: a silently retried request would carry the failed
        attempt and the retry back-off in its TTFT, so a failure is counted
        as an error instead.
        """
        old_http_client = getattr(self, "_http_client", None)
        if old_http_client is not None:
            old_http_client.close()

        self.chunk_timeout = max(100.0, read_timeout / 2.0)
        self._http_client = httpx.Client(
            verify=False,
            limits=httpx.Limits(
                max_connections=max_concurrency + 32,
                max_keepalive_connections=max(20, max_concurrency),
            ),
        )

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key if self.api_key else None,
            timeout=httpx.Timeout(60.0, connect=10.0, read=read_timeout, write=10.0),
            max_retries=MAX_RETRIES,
            http_client=self._http_client,
        )

    def measure_single(self, prompt: str, max_tokens: int) -> Dict:
        """
        Run a completion request and measure timing metrics.

        Args:
            prompt: The prompt to send to the model
            max_tokens: Maximum number of tokens to generate

        Returns:
            Dictionary containing performance metrics
        """
        # Monotonic clock: a wall-clock adjustment cannot corrupt a sample
        start_time = time.perf_counter()
        first_token_time = None
        last_token_time = None
        tokens = 0
        total_tokens = 0
        prompt_tokens = 0
        total_content = []
        chunk_count = 0

        try:
            # Streaming completion with explicit timeout
            logger.info(f"Starting completion for model: {self.model}")

            # Only sent when requested, so default requests stay unchanged
            extra = {}
            if self.force_output_length:
                extra["extra_body"] = {"ignore_eos": True, "min_tokens": max_tokens}

            # Create stream with timeout
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                stream=True,
                stream_options={"include_usage": True},
                **extra
            )

            # Process stream with chunk timeout protection
            last_chunk_time = time.perf_counter()
            chunk_timeout = self.chunk_timeout

            for event in stream:
                current_time = time.perf_counter()

                # A stalled stream is cut short; counting it as valid would
                # record too few tokens, so it is an error
                if current_time - last_chunk_time > chunk_timeout:
                    logger.error("Chunk timeout reached")
                    return {"error": "Chunk timeout - stream stalled"}

                last_chunk_time = current_time

                # Process the event
                if event.choices:
                    delta = event.choices[0].delta

                    # Check for content (both regular and reasoning)
                    content = None
                    if hasattr(delta, 'content') and delta.content:
                        content = delta.content
                    elif hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                        content = delta.reasoning_content
                    elif hasattr(delta, 'reasoning') and delta.reasoning:
                        content = delta.reasoning
                    if content:
                        chunk_count += 1
                        now = time.perf_counter()

                        if first_token_time is None:
                            first_token_time = now
                            logger.info(f"First token received after {(first_token_time - start_time)*1000:.2f}ms")

                        last_token_time = now
                        total_content.append(content)

                # Get usage information if available
                if hasattr(event, 'usage') and event.usage is not None:
                    total_tokens = event.usage.completion_tokens
                    prompt_tokens = event.usage.prompt_tokens or 0
                    logger.debug(f"Token count: {total_tokens}")

            logger.info(f"Stream completed. Chunks: {chunk_count}, Tokens: {total_tokens}")

        except httpx.ReadTimeout:
            logger.error("Read timeout occurred")
            return {"error": "Read timeout - server took too long to respond"}
        except httpx.WriteTimeout:
            logger.error("Write timeout occurred")
            return {"error": "Write timeout - failed to send request"}
        except httpx.ConnectTimeout:
            logger.error("Connection timeout occurred")
            return {"error": "Connection timeout - could not connect to server"}
        except Exception as e:
            logger.error(f"Error during completion: {type(e).__name__}: {e}")
            return {"error": f"{type(e).__name__}: {str(e)}"}

        end_time = time.perf_counter()

        # Calculate metrics
        if not total_content:
            logger.warning("No content received")
            return {"error": "No content received"}

        # Use actual token count if available, otherwise estimate from chunks
        tokens = total_tokens if total_tokens > 0 else chunk_count

        # Calculate timing metrics
        ttft = (first_token_time - start_time) * TO_MS if first_token_time else 0
        total_time = end_time - start_time
        latency = total_time
        itl = ((last_token_time - first_token_time) * TO_MS / max(tokens - 1, 1)) if tokens > 1 else 0
        tps = tokens / total_time if total_time > 0 else 0

        content = "".join(total_content)

        result = {
            "ttft": ttft,
            "itl": itl,
            "tps": tps,
            "latency": latency,
            "tokens": tokens,
            "prompt_tokens": prompt_tokens,
            "content": content,
            "start": start_time,
        }

        logger.info(f"Request completed - TTFT: {ttft:.2f}ms, TPS: {tps:.2f}, Tokens: {tokens}")
        return result

    async def run_concurrent(self, prompts: List[str], max_tokens: int, 
                           concurrency: int, progress_callback,
                           batch_timeout: float = 300.0) -> List[Dict]:
        """
        Run benchmark with concurrent requests.

        Args:
            prompts: List of prompts to test
            max_tokens: Maximum tokens per response
            concurrency: Number of concurrent requests
            progress_callback: Callback function to update progress
            batch_timeout: Total wall-clock budget for this batch, in seconds.

        Returns:
            List of benchmark results
        """
        loop = asyncio.get_running_loop()
        results = []
        executor = ThreadPoolExecutor(max_workers=concurrency)

        def task(index: int, prompt: str) -> Dict:
            result = self.measure_single(prompt, max_tokens)
            result["index"] = index
            return result

        # Add timeout to concurrent execution
        try:
            async with asyncio.timeout(batch_timeout):
                futures = [
                    loop.run_in_executor(executor, task, index, prompt)
                    for index, prompt in enumerate(prompts)
                ]

                completed_count = 0
                for coro in asyncio.as_completed(futures):
                    try:
                        result = await coro
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Future failed: {e}")
                        results.append({"error": str(e)})
                    finally:
                        completed_count += 1
                        progress_callback(completed_count, len(prompts))
        except TimeoutError:
            # Discard semantics: partial results are thrown away because some
            # requests completed abnormally (under the timeout's shadow).
            logger.warning(
                f"Batch timed out after {batch_timeout:.0f}s; discarding all "
                f"{len(results)} partial results for this concurrency level."
            )
            st.info(
                f"Batch timeout ({batch_timeout:.0f}s) reached — discarding "
                f"partial results and draining in-flight requests "
                f"(the UI pauses until the server is idle)..."
            )
            raise
        finally:
            # Cancels queued-but-not-yet-started requests and waits for the
            # in-flight ones to finish, so that leftover requests from a
            # discarded level cannot pollute the next level's measurement.
            executor.shutdown(wait=True, cancel_futures=True)

        return results

    def calculate_metrics(self, results: List[Dict]) -> Dict:
        """
        Calculate statistical metrics from benchmark results.

        Args:
            results: List of benchmark results

        Returns:
            Dictionary containing calculated metrics
        """
        valid_results = [
            r for r in results 
            if 'error' not in r and r['ttft'] is not None
        ]

        if not valid_results:
            return {"error": "No valid results"}

        # Extract metric values
        ttfts = [r['ttft'] for r in valid_results]
        itls = [r['itl'] for r in valid_results]
        tps_list = [r['tps'] for r in valid_results]
        latencies = [r['latency'] for r in valid_results]

        # Calculate throughput
        throughput = len(valid_results) / sum(latencies) if latencies else 0

        def calculate_percentiles(data: List[float]) -> Dict[str, float]:
            """Calculate mean, p50, and p90 percentiles."""
            if not data:
                return {"mean": 0, "p50": 0, "p90": 0}

            sorted_data = sorted(data)
            return {
                "mean": statistics.mean(data),
                "p50": statistics.median(data),
                "p90": sorted_data[min(int(math.ceil(len(sorted_data) * 0.9)) - 1, len(sorted_data) - 1)] if sorted_data else 0
            }

        error_count = len(results) - len(valid_results)

        # Generation speed without the wait before the first token: 1000 / mean
        # ITL, the same definition the published TPS values were revised to
        itl_mean = statistics.mean(itls)
        itl_p50 = statistics.median(itls)

        # Token counts actually sent/received (prompt_tokens is 0 when the
        # server does not report usage)
        prompt_counts = [r['prompt_tokens'] for r in valid_results if r.get('prompt_tokens')]

        return {
            "TTFT": calculate_percentiles(ttfts),
            "ITL": calculate_percentiles(itls),
            "TPS": calculate_percentiles(tps_list),
            "Latency": calculate_percentiles(latencies),
            "Throughput (RPS)": throughput,
            "Decode TPS": {
                "mean": TO_MS / itl_mean if itl_mean > 0 else 0,
                "p50": TO_MS / itl_p50 if itl_p50 > 0 else 0,
            },
            "Requests": {
                "valid": len(valid_results),
                "errors": error_count,
                "total": len(results),
                # Percentage of failed requests at this concurrency level.
                "error_rate": (error_count / len(results) * 100.0) if results else 0.0,
            },
            "Tokens": {
                "input_mean": statistics.mean(prompt_counts) if prompt_counts else None,
                "output_mean": statistics.mean(r['tokens'] for r in valid_results),
            },
        }

    def plot_results(self, all_metrics: Dict) -> Dict[str, go.Figure]:
        """
        Create visualization plots for benchmark results.

        Args:
            all_metrics: Dictionary of metrics for each concurrency level

        Returns:
            Dictionary of Plotly figures
        """
        figures = {}
        metrics_to_plot = ["TTFT", "ITL", "TPS", "Latency"]
        concurrency_levels = list(all_metrics.keys())

        for metric in metrics_to_plot:
            means = [all_metrics[conc][metric]["mean"] for conc in concurrency_levels]
            p90s = [all_metrics[conc][metric]["p90"] for conc in concurrency_levels]

            trace_mean = go.Bar(
                x=[str(conc) for conc in concurrency_levels],
                y=means,
                name="Mean",
                marker_color="skyblue",
                width=0.3
            )
            trace_p90 = go.Bar(
                x=[str(conc) for conc in concurrency_levels],
                y=p90s,
                name="P90",
                marker_color="salmon",
                width=0.3
            )

            fig = go.Figure(data=[trace_mean, trace_p90])

            # Set appropriate units
            metric_units = {
                "TPS": "tokens/s",
                "TTFT": "ms",
                "ITL": "ms",
                "Latency": "s"
            }
            metric_unit = metric_units.get(metric, "")

            fig.update_layout(
                title=f"{metric} Across Concurrency Levels",
                xaxis_title="Concurrency Level",
                yaxis_title=f"{metric} ({metric_unit})",
                barmode="group",
                template="plotly_white"
            )

            figures[metric] = fig

        # Plot throughput separately
        throughputs = [all_metrics[conc]["Throughput (RPS)"] for conc in concurrency_levels]
        fig = go.Figure(data=[
            go.Bar(
                x=[str(conc) for conc in concurrency_levels],
                y=throughputs,
                name="Throughput",
                marker_color="lightblue",
                width=0.25
            )
        ])
        fig.update_layout(
            title="Throughput Across Concurrency Levels",
            xaxis_title="Concurrency Level",
            yaxis_title="Throughput (RPS)",
            template="plotly_white"
        )
        figures['Throughput'] = fig

        decode = [all_metrics[conc]["Decode TPS"]["mean"] for conc in concurrency_levels]
        fig = go.Figure(data=[
            go.Bar(
                x=[str(conc) for conc in concurrency_levels],
                y=decode,
                name="Decode TPS",
                marker_color="skyblue",
                width=0.25
            )
        ])
        fig.update_layout(
            title="Decode TPS (TPS without TTFT) Across Concurrency Levels",
            xaxis_title="Concurrency Level",
            yaxis_title="Decode TPS (tokens/s)",
            template="plotly_white"
        )
        figures['Decode TPS'] = fig

        return figures

    def size_input(self, prompt: str, filler_pool: List[str],
                   input_tokens: int) -> Optional[int]:
        """
        Find the prompt length in characters that gives input_tokens prompt
        tokens on this server (its tokenizer and chat template).

        Two short requests, the bare prompt and a padded one, give a
        straight-line fit of usage.prompt_tokens against characters.

        Returns:
            Target prompt length in characters, or None if the server does
            not report prompt_tokens
        """
        guess = max(int(input_tokens * FALLBACK_CHARS_PER_TOKEN), 2 * len(prompt))
        padded = pad_prompt(prompt, filler_pool, guess, random.Random(0))

        short_tokens = self.measure_single(prompt, 16).get("prompt_tokens", 0)
        long_tokens = self.measure_single(padded, 16).get("prompt_tokens", 0)
        if not short_tokens or long_tokens <= short_tokens:
            return None

        chars_per_token = (len(padded) - len(prompt)) / (long_tokens - short_tokens)
        return int(len(prompt) + (input_tokens - short_tokens) * chars_per_token)

    async def run(self, prompts: List[str], max_tokens: int,
                  concurrency_levels: List[int],
                  input_tokens: int = DEFAULT_INPUT_TOKENS,
                  seed: int = DEFAULT_SEED) -> Dict:
        """
        Run the complete benchmark suite.

        Args:
            prompts: List of prompts to test
            max_tokens: Maximum tokens per response
            concurrency_levels: List of concurrency levels to test
            input_tokens: Target prompt tokens per request; the default sends
                prompts.txt unchanged, larger values pad every prompt
            seed: Seeds the filler order of padded prompts

        Returns:
            Dictionary containing all benchmark results
        """
        concurrency_levels = sorted(concurrency_levels)
        max_conc = max(concurrency_levels) if concurrency_levels else 1

        st.write("Warm Up Started ... ")
        self.measure_single("What is warmup?", 32)
        st.write("Warm Up Ended ...")

        # Workload: at the default 128/128 the prompts go out unchanged
        filler_pool = [p.replace(OUTPUT_HINT, "").strip() for p in prompts]
        if max_tokens != DEFAULT_MAX_TOKENS:
            prompts = [set_output_hint(p, max_tokens) for p in prompts]

        target_chars = None
        if input_tokens != DEFAULT_INPUT_TOKENS and prompts:
            st.write(f"Sizing prompts to {input_tokens} input tokens ...")
            target_chars = self.size_input(prompts[0], filler_pool, input_tokens)
            if target_chars is None:
                target_chars = int(input_tokens * FALLBACK_CHARS_PER_TOKEN)
                st.warning(
                    f"Server did not report prompt_tokens; assuming "
                    f"{FALLBACK_CHARS_PER_TOKEN:.0f} characters per token."
                )
            st.write(f"Prompts padded to {target_chars} characters")
        self.target_chars = target_chars

        rng = random.Random(seed)
        self.requests = []

        # Every request starts with a tag unique to this run and request, so
        # the server's prefix cache cannot serve a prompt it has seen before
        # (earlier in the run, or in an earlier run) without prefill
        self.run_id = secrets.token_hex(3)
        request_no = count()

        def tagged(text: str) -> str:
            return f"[{self.run_id}-{next(request_no):05d}] {text}"

        def make_batch(n: int) -> List[str]:
            batch = take_prompts(prompts, n)
            if target_chars is None:
                return [tagged(p) for p in batch]
            return [tagged(pad_prompt(p, filler_pool, target_chars - TAG_CHARS, rng))
                    for p in batch]

        st.write("Calibrating with a single representative request ...")
        calib = self.measure_single(make_batch(1)[0], max_tokens) if prompts else {"error": "no prompts"}
        if "error" in calib:
            calib_latency = 10.0
            calib_ttft_s = 5.0
            st.warning(
                f"Calibration failed ({calib['error']}). "
                f"Using fallback latency={calib_latency:.1f}s, ttft={calib_ttft_s:.1f}s."
            )
        else:
            calib_latency = calib["latency"]
            calib_ttft_s = calib["ttft"] / TO_MS

        read_timeout = max(60.0, calib_ttft_s * max_conc * 0.4 + 30.0)
        self._build_client(read_timeout, max_concurrency=max_conc)
        st.write(
            f"Calibration: latency={calib_latency:.2f}s, ttft={calib_ttft_s:.2f}s "
            f"=> read_timeout={read_timeout:.0f}s"
        )

        all_metrics = {}
        total_tasks = sum(MEASURED_ROUNDS * conc for conc in concurrency_levels)
        progress_bar = st.progress(0)
        completed_tasks = 0

        def update_progress(current: int, total: int) -> None:
            nonlocal completed_tasks
            completed_tasks += 1
            progress = min(completed_tasks / total_tasks, 1.0)
            progress_bar.progress(progress)

        prev_latency = calib_latency
        for conc in concurrency_levels:
            # Wall-clock budget per batch grows with concurrency: near
            # saturation latency rises superlinearly, so a fixed 3x slack
            # would time out and discard whole levels (e.g. 128 -> 256).
            slack = 3.0 + conc / 128.0

            warmup_batch = make_batch(WARMUP_ROUNDS * conc)

            warmup_timeout = max(300.0, WARMUP_ROUNDS * prev_latency * slack)
            logger.info(
                f"Warmup for concurrency {conc}: {len(warmup_batch)} requests "
                f"(timeout={warmup_timeout:.0f}s)"
            )
            try:
                await self.run_concurrent(
                    warmup_batch, max_tokens, conc,
                    lambda c, t: None, warmup_timeout
                )
            except Exception as e:
                logger.warning(
                    f"Warmup failed for concurrency {conc}: {e}. "
                    f"Proceeding with measurement."
                )

            batch = make_batch(MEASURED_ROUNDS * conc)

            batch_timeout = max(300.0, MEASURED_ROUNDS * prev_latency * slack)
            st.write(
                f"Running with concurrency level: {conc} "
                f"({len(batch)} prompts, {len(batch)//conc} rounds) "
                f"| batch_timeout={batch_timeout:.0f}s (prev_latency={prev_latency:.1f}s)"
            )

            try:
                results = await self.run_concurrent(
                    batch, max_tokens, conc, update_progress, batch_timeout
                )
                metrics = self.calculate_metrics(results)
                self.requests.extend(request_rows(conc, results))

                if "error" in metrics:
                    st.error(f"Error in metrics calculation: {metrics['error']}")
                    continue

                all_metrics[conc] = metrics
                prev_latency = metrics["Latency"]["mean"]
                st.write(f"Metrics for concurrency level {conc}:")
                st.json(metrics)

            except TimeoutError:
                logger.error(
                    f"Concurrency {conc} DISCARDED: batch timeout "
                    f"({batch_timeout:.0f}s) reached; partial results thrown "
                    f"away and in-flight requests drained."
                )
                st.error(
                    f"Concurrency {conc} DISCARDED: batch timeout "
                    f"({batch_timeout:.0f}s) reached. Partial results were "
                    f"thrown away; continuing with the next level."
                )
                continue
            except Exception as e:
                logger.error(f"Error running benchmark with concurrency {conc}: {e}")
                st.error(f"Error with concurrency {conc}: {e}")
                continue

        return all_metrics


def load_prompts(seed: int = DEFAULT_SEED) -> List[str]:
    """
    Load all prompts from file.

    Args:
        seed: Seeds the shuffle, so a run can be repeated with the same order

    Returns:
        List of prompt strings
    """
    try:
        with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
            prompts = f.readlines()
            prompts = [p.strip() for p in prompts if p.strip()]
            random.Random(seed).shuffle(prompts)
            return prompts

    except FileNotFoundError:
        st.error(f"Error: {PROMPTS_FILE} file not found!")
        return []
    except Exception as e:
        st.error(f"Error loading prompts: {e}")
        return []


def take_prompts(prompts: List[str], n: int) -> List[str]:
    """First n prompts, cycling through the list when n exceeds it."""
    if n <= len(prompts):
        return prompts[:n]
    return [prompts[i % len(prompts)] for i in range(n)]


def request_rows(concurrency: int, results: List[Dict]) -> List[Dict]:
    """One row per measured request for the per-request CSV."""
    starts = [r["start"] for r in results if "error" not in r]
    t0 = min(starts) if starts else 0.0
    rows = []
    for r in sorted(results, key=lambda r: r.get("index", 0)):
        ok = "error" not in r
        rows.append({
            "concurrency": concurrency,
            "request": r.get("index", ""),
            # seconds after the level's first request started
            "start_s": round(r["start"] - t0, 4) if ok else "",
            "ttft_ms": round(r["ttft"], 3) if ok else "",
            "itl_ms": round(r["itl"], 4) if ok else "",
            "tps": round(r["tps"], 3) if ok else "",
            "decode_tps": round(TO_MS / r["itl"], 3) if ok and r["itl"] > 0 else "",
            "latency_s": round(r["latency"], 4) if ok else "",
            "prompt_tokens": r.get("prompt_tokens", "") if ok else "",
            "output_tokens": r["tokens"] if ok else "",
            "error": r.get("error", ""),
        })
    return rows


def set_output_hint(prompt: str, output_tokens: int) -> str:
    """Point the prompt's length instruction at the chosen output length."""
    return prompt.replace(
        OUTPUT_HINT, f"Your answer must contain at least {output_tokens} tokens."
    )


def pad_prompt(prompt: str, filler_pool: List[str], target_chars: int,
               rng: random.Random) -> str:
    """
    Extend a prompt to target_chars with background text drawn from the
    other prompts.

    The question stays first and the filler order is drawn fresh on every
    call, so requests do not share a long prefix that a server's prefix
    cache could serve without prefill.
    """
    if len(prompt) >= target_chars:
        return prompt

    parts = [prompt, "", "Background notes (for reference only):"]
    length = sum(len(p) + 1 for p in parts)
    while length < target_chars:
        for item in rng.sample(filler_pool, len(filler_pool)):
            parts.append(item)
            length += len(item) + 1
            if length >= target_chars:
                break
    return "\n".join(parts)[:target_chars]


def build_zip(model_name: str, df: pd.DataFrame, figures: Dict[str, go.Figure],
              models_payload: Optional[Dict] = None,
              run_config: Optional[Dict] = None,
              requests: Optional[List[Dict]] = None) -> bytes:
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        zip_file.writestr(f"{model_name}-table.csv", csv_buffer.getvalue())
        
        for metric, fig in figures.items():
            try:
                png_buffer = io.BytesIO()
                fig.write_image(png_buffer, format='png')
                zip_file.writestr(f"{model_name}-{metric}.png", png_buffer.getvalue())
                
                html_buffer = io.StringIO()
                fig.write_html(html_buffer)
                zip_file.writestr(f"{model_name}-{metric}.html", html_buffer.getvalue())
            except Exception as e:
                logger.error(f"Error saving figure {metric}: {e}")

        if models_payload:
            try:
                models_buffer = io.StringIO()
                json.dump(models_payload, models_buffer, indent=2, ensure_ascii=False)
                zip_file.writestr("models.json", models_buffer.getvalue())
            except Exception as e:
                logger.error(f"Error saving models.json: {e}")

        if run_config:
            zip_file.writestr("config.json", json.dumps(run_config, indent=2))

        if requests:
            zip_file.writestr(f"{model_name}-requests.csv",
                              pd.DataFrame(requests).to_csv(index=False))
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def calculate_capacity(all_metrics: Dict, ttft_slo: int, tps_slo: int,
                       percentile_key: str, think_time: int) -> Dict:
    c_max = None
    slo_status = []

    for conc in sorted(all_metrics.keys()):
        metrics = all_metrics[conc]
        if "error" in metrics:
            continue

        ttft_val = metrics["TTFT"][percentile_key]
        tps_val = metrics["TPS"][percentile_key]

        ttft_ok = ttft_val <= ttft_slo
        tps_ok = tps_val >= tps_slo

        slo_status.append({
            "conc": conc,
            "ttft": ttft_val,
            "tps": tps_val,
            "ttft_ok": ttft_ok,
            "tps_ok": tps_ok,
        })

        if ttft_ok and tps_ok:
            c_max = conc

    latency_val = all_metrics[c_max]["Latency"][percentile_key] if c_max else 0
    total_users = int(c_max * (1 + think_time / latency_val)) if c_max else 0

    return {
        "c_max": c_max,
        "total_users": total_users,
        "latency": latency_val,
        "slo_status": slo_status
    }


def fmt_tokens(value: Optional[float]) -> str:
    """Token count for the results table; n/a when the server sent no usage."""
    return f"{value:.0f}" if value is not None else "n/a"


def build_run_config(model: str, input_tokens: int, output_tokens: int,
                     force_output_length: bool, concurrency_levels: List[int],
                     target_chars: Optional[int], all_metrics: Dict,
                     seed: int = DEFAULT_SEED, run_id: str = "") -> Dict:
    """Workload settings and measured token counts, saved as config.json."""
    return {
        "model": model,
        "input_tokens": input_tokens,
        "input_mode": "prompts.txt" if target_chars is None else "padded",
        "prompt_chars": target_chars,
        "output_tokens": output_tokens,
        "force_output_length": force_output_length,
        "seed": seed,
        "run_id": run_id,
        "max_retries": MAX_RETRIES,
        "concurrency_levels": sorted(concurrency_levels),
        "measured_rounds": MEASURED_ROUNDS,
        "warmup_rounds": WARMUP_ROUNDS,
        "measured_tokens": {
            str(conc): metrics["Tokens"] for conc, metrics in all_metrics.items()
        },
    }


def create_results_dataframe(all_metrics: Dict) -> pd.DataFrame:
    """
    Create a DataFrame from benchmark results.

    Args:
        all_metrics: Dictionary of metrics for each concurrency level

    Returns:
        DataFrame with formatted results
    """
    data = []
    for conc, metrics in all_metrics.items():
        if "error" not in metrics:
            data.append({
                "Concurrency": conc,
                "TTFT (Mean, ms)": f"{metrics['TTFT']['mean']:.2f}",
                "TTFT (p50, ms)": f"{metrics['TTFT']['p50']:.2f}",
                "TTFT (p90, ms)": f"{metrics['TTFT']['p90']:.2f}",
                "ITL (Mean, ms)": f"{metrics['ITL']['mean']:.2f}",
                "ITL (p50, ms)": f"{metrics['ITL']['p50']:.2f}",
                "ITL (p90, ms)": f"{metrics['ITL']['p90']:.2f}",
                "TPS (Mean, tokens/s)": f"{metrics['TPS']['mean']:.2f}",
                "TPS (p50, tokens/s)": f"{metrics['TPS']['p50']:.2f}",
                "TPS (p90, tokens/s)": f"{metrics['TPS']['p90']:.2f}",
                "Latency (Mean, s)": f"{metrics['Latency']['mean']:.2f}",
                "Latency (p50, s)": f"{metrics['Latency']['p50']:.2f}",
                "Latency (p90, s)": f"{metrics['Latency']['p90']:.2f}",
                "Throughput (RPS)": f"{metrics['Throughput (RPS)']:.2f}",
                "Valid Requests": f"{metrics['Requests']['valid']}/{metrics['Requests']['total']}",
                "Error Rate (%)": f"{metrics['Requests']['error_rate']:.2f}",
                "Input Tokens (Mean)": fmt_tokens(metrics['Tokens']['input_mean']),
                "Output Tokens (Mean)": fmt_tokens(metrics['Tokens']['output_mean']),
                "Decode TPS (Mean, tokens/s)": f"{metrics['Decode TPS']['mean']:.2f}",
            })
    return pd.DataFrame(data)


def snap_to_step(key: str) -> None:
    """Round a token-count input up to the next multiple of TOKEN_STEP."""
    st.session_state[key] = math.ceil(st.session_state[key] / TOKEN_STEP) * TOKEN_STEP


def reset_workload() -> None:
    """Put the workload settings back to 128 in / 128 out, forced."""
    st.session_state.input_tokens = DEFAULT_INPUT_TOKENS
    st.session_state.output_tokens = DEFAULT_MAX_TOKENS
    st.session_state.force_output_length = DEFAULT_FORCE_OUTPUT_LENGTH


def initialize_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if 'models' not in st.session_state:
        st.session_state.models = []
    if 'models_payload' not in st.session_state:
        st.session_state.models_payload = None
    if 'last_base_url' not in st.session_state:
        st.session_state.last_base_url = None
    if 'input_tokens' not in st.session_state:
        st.session_state.input_tokens = DEFAULT_INPUT_TOKENS
    if 'output_tokens' not in st.session_state:
        st.session_state.output_tokens = DEFAULT_MAX_TOKENS
    if 'force_output_length' not in st.session_state:
        st.session_state.force_output_length = DEFAULT_FORCE_OUTPUT_LENGTH


def configure_page():
    """Configure page layout and styles."""

    # Add CSS styles
    st.markdown(
        """
        <style>
            [data-testid="stSidebar"] {
                width: 400px !important;
            }
            [data-testid="stSidebar"] > div:first-child {
                width: 400px !important;
            }
            .main-header {
                font-size: 2.5rem;
                font-weight: bold;
                color: #1e3a8a;
                text-align: center;
                margin-bottom: 2rem;
            }
            .subtitle {
                font-size: 1.2rem;
                color: #6b7280;
                text-align: center;
                margin-bottom: 3rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Page configuration
    st.set_page_config(
        page_title="LLM Benchmark",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )



def main() -> None:
    """Main application function."""
    initialize_session_state()

    configure_page()

    # Sidebar setup
    if os.path.exists(LOGO_PATH):
        st.sidebar.image(LOGO_PATH)
    st.sidebar.title("LLM Benchmark Tool")
    st.sidebar.write("Enter the details below to benchmark an LLM model.")

    api_key = st.sidebar.text_input("API KEY ", value=API_KEY)
    # Input controls
    base_url = st.sidebar.text_input("Base URL", value=DEFAULT_BASE_URL)

    # Handle base URL changes or initial load
    if base_url != st.session_state.last_base_url:
        st.session_state.last_base_url = base_url
        payload = get_models_payload(base_url, api_key=api_key)
        st.session_state.models_payload = payload
        fetched = [m['id'] for m in payload.get('data', [])] if payload else []
        if fetched:
            st.session_state.models = fetched
        else:
            st.session_state.models = []
        st.rerun()

    # Refresh models button
    if st.sidebar.button(
        "🔄 Refresh Models",
        key="refresh_models",
        help="Refresh model list from server"
    ) and base_url:
        with st.spinner("Fetching model list..."):
            payload = get_models_payload(base_url, api_key=api_key)
            new_models = [m['id'] for m in payload.get('data', [])] if payload else []
            st.session_state.models_payload = payload

        if new_models:
            st.session_state.models = new_models
            st.success(f"Successfully loaded {len(new_models)} models!")
            st.rerun()
        else:
            st.error("Failed to fetch model list!")

    # Model selection
    model = st.sidebar.selectbox("Model Name:", st.session_state.models)

    # Benchmark parameters
    concurrency_levels = st.sidebar.multiselect(
        "Concurrency Levels", 
        options=[1, 2, 4, 8, 16, 32, 64, 128, 256], 
        default=DEFAULT_CONCURRENCY_LEVELS
    )

    with st.sidebar.expander("Workload (default: 128 in / 128 out)", expanded=False):
        input_tokens = st.number_input(
            "Input tokens", min_value=DEFAULT_INPUT_TOKENS,
            max_value=MAX_INPUT_TOKENS, step=TOKEN_STEP,
            key="input_tokens", on_change=snap_to_step, args=("input_tokens",),
            help="Rounded up to a multiple of 128. 128 sends the standard prompts.txt set "
                 "unchanged. Larger values extend every prompt with other "
                 "prompts in seeded random order, cut to that many prompt "
                 "tokens on the server's tokenizer."
        )
        output_tokens = st.number_input(
            "Output tokens", min_value=1, max_value=MAX_OUTPUT_TOKENS,
            step=TOKEN_STEP, key="output_tokens",
            help="Any value; max_tokens per request."
        )
        force_output_length = st.checkbox(
            "Force exact output length", key="force_output_length",
            help="Sends ignore_eos and min_tokens so every response is "
                 "exactly the output length. Supported by vLLM and SGLang; "
                 "untick it for a server that rejects the request."
        )
        st.button("Reset to defaults", on_click=reset_workload,
                  help="128 in / 128 out, exact output length forced.")

    ttft_slo = DEFAULT_TTFT_SLO
    tps_slo = DEFAULT_TPS_SLO
    percentile_key = "mean"
    think_time = DEFAULT_THINK_TIME

    if 'sonuc' in st.session_state:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Capacity Planning")
        ttft_slo = st.sidebar.number_input(
            "TTFT SLO (ms)", min_value=100, max_value=10000,
            value=DEFAULT_TTFT_SLO
        )
        tps_slo = st.sidebar.number_input(
            "TPS SLO (tok/s)", min_value=1, max_value=1000,
            value=DEFAULT_TPS_SLO
        )
        percentile_key = st.sidebar.radio(
            "Percentile", ["mean", "p90"], horizontal=True,
            format_func=lambda x: "P90" if x == "p90" else "Mean"
        )
        think_time = st.sidebar.number_input(
            "Think Time (s)", min_value=1, max_value=600,
            value=DEFAULT_THINK_TIME
        )

    if st.sidebar.button("Run Benchmark"):
        if not model:
            st.error("Please select a model first! Click '🔄 Refresh Models' to load the model list.")
        elif not concurrency_levels:
            st.error("Please select at least one concurrency level.")
        else:
            if 'sonuc' in st.session_state:
                del st.session_state.sonuc
            st.session_state.pop('zip_data', None)

            with st.spinner("Running benchmark..."):
                prompts = load_prompts(DEFAULT_SEED)
                if not prompts:
                    st.error("Could not load prompts. Please check the prompts.txt file.")
                else:
                    try:
                        benchmark = LLMBenchmark(base_url, api_key, model,
                                                 force_output_length)
                        all_metrics = asyncio.run(
                            benchmark.run(prompts, output_tokens, concurrency_levels,
                                          input_tokens, DEFAULT_SEED)
                        )

                        if all_metrics:
                            df = create_results_dataframe(all_metrics)
                            figures = benchmark.plot_results(all_metrics)
                            model_name = model.split('/')[-1] if model else "unknown"
                            # Default runs keep their file names
                            if (input_tokens, output_tokens) != (DEFAULT_INPUT_TOKENS, DEFAULT_MAX_TOKENS):
                                model_name += f"_in{input_tokens}_out{output_tokens}"
                            run_config = build_run_config(
                                model, input_tokens, output_tokens,
                                force_output_length, concurrency_levels,
                                benchmark.target_chars, all_metrics,
                                DEFAULT_SEED, benchmark.run_id
                            )
                            st.session_state.sonuc = (model_name, df, figures,
                                                      all_metrics, run_config,
                                                      benchmark.requests)
                        else:
                            st.error("No benchmark results were generated.")
                    except Exception as e:
                        logger.error(f"Benchmark failed: {e}")
                        st.error(f"Benchmark failed: {e}")

    if 'sonuc' in st.session_state:
        model_name, df, figures, all_metrics, run_config, requests = st.session_state.sonuc

        st.subheader("Benchmark Results")
        st.caption(
            f"Workload: {run_config['input_tokens']} input / "
            f"{run_config['output_tokens']} output tokens"
            + (" (forced)" if run_config['force_output_length'] else "")
        )
        st.dataframe(df, width='stretch')

        cap = calculate_capacity(all_metrics, ttft_slo, tps_slo, percentile_key, think_time)
        with st.expander("Capacity Planning", expanded=False):
            if cap["c_max"]:
                st.success(
                    f"Max Concurrency: {cap['c_max']} | "
                    f"Total Users: ~{cap['total_users']} | "
                    f"N = {cap['c_max']} x (1 + {think_time} / {cap['latency']:.2f})"
                )
            else:
                st.error("No concurrency level meets SLO thresholds!")

            cap_data = []
            for s in cap["slo_status"]:
                cap_data.append({
                    "Concurrency": s["conc"],
                    f"TTFT ({percentile_key}, ms)": f"{s['ttft']:.0f}",
                    "TTFT SLO": "OK" if s["ttft_ok"] else "X",
                    f"TPS ({percentile_key}, tok/s)": f"{s['tps']:.1f}",
                    "TPS SLO": "OK" if s["tps_ok"] else "X",
                })
            st.dataframe(pd.DataFrame(cap_data), width='stretch')

        st.subheader("Visualizations")
        for metric, fig in figures.items():
            st.plotly_chart(fig, use_container_width=True)

        # Built once per run; cleared when a new run starts
        if 'zip_data' not in st.session_state:
            st.session_state.zip_data = build_zip(
                model_name, df, figures,
                st.session_state.get('models_payload'), run_config, requests
            )

        st.sidebar.markdown("---")
        st.sidebar.download_button(
            label="📥 Download All Results (ZIP)",
            data=st.session_state.zip_data,
            file_name=f"{model_name}_results.zip",
            mime="application/zip"
        )

        if cap["c_max"] is None:
            st.sidebar.error("No concurrency level meets SLO thresholds!")
        else:
            st.sidebar.success(f"Max Concurrency: {cap['c_max']}")
            st.sidebar.info(f"Total Users: ~{cap['total_users']}")
            st.sidebar.caption(
                f"N = {cap['c_max']} x (1 + {think_time} / {cap['latency']:.2f})"
            )

        status_lines = []
        for s in cap["slo_status"]:
            ttft_icon = "OK" if s["ttft_ok"] else "X"
            tps_icon = "OK" if s["tps_ok"] else "X"
            status_lines.append(
                f"Conc {s['conc']}: TTFT={s['ttft']:.0f}ms [{ttft_icon}] "
                f"TPS={s['tps']:.1f} [{tps_icon}]"
            )
        st.sidebar.code("\n".join(status_lines))


if __name__ == "__main__":
    main()