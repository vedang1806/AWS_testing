#!/usr/bin/env python3
"""
AWS Transcribe & Comprehend PII Redaction Test Script

This script tests AWS Transcribe for transcription and AWS Comprehend for PII detection.
It accepts a .wav audio file as input and performs:
1. Audio transcription with speaker labels and sentiment analysis
2. PII redaction on the transcript using AWS Comprehend
3. Uploads original and redacted results to S3

Usage:
    python test_assemblyai_transcription.py <path_to_audio.wav>

Example:
    python test_assemblyai_transcription.py ./sample_call.wav
"""

import os
import sys
import json
import time
import asyncio
import argparse
import re
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError
import requests
from pydub import AudioSegment
from pydub.generators import Sine
import numpy as np
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Constants
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_S3_BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME", "bpo-box-dev")
AWS_S3_REGION = os.getenv("AWS_S3_REGION", "us-east-1")
ENVIRONMENT = os.getenv("ENVIRONMENT", "DEV")
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

# PII Entity Types to Detect (AWS Comprehend)
PII_ENTITY_TYPES = [
    "PERSON",
    "EMAIL",
    "PHONE",
    "ADDRESS",
    "CREDIT_CARD",
    "BANK_ACCOUNT",
    "BANK_ROUTING",
    "SSN",
    "DRIVER_ID",
    "PASSPORT",
    "DATE",
    "URL",
    "IP_ADDRESS",
    "MEDICAL_CONDITION",
    "MEDICAL_PROCEDURE",
    "MEDICATION",
]


class AWSTranscriber:
    """
    Handles transcription using AWS Transcribe
    """

    def __init__(self, region: str = AWS_S3_REGION):
        """Initialize AWS Transcribe and S3 clients"""
        self.transcribe_client = boto3.client(
            "transcribe",
            region_name=region,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        )
        self.s3_client = boto3.client(
            "s3",
            region_name=region,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        )
        self.comprehend_client = boto3.client(
            "comprehend",
            region_name=region,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        )

    def upload_audio_to_s3(self, audio_file_path: str, call_id: str) -> Optional[str]:
        """
        Upload audio file to S3 (required for AWS Transcribe)

        Args:
            audio_file_path: Path to the .wav audio file
            call_id: Unique call identifier

        Returns:
            S3 URI (s3://bucket/key) or None if failed
        """
        print(f"\n📤 Uploading audio file to S3: {audio_file_path}")

        if not os.path.exists(audio_file_path):
            print(f"❌ File not found: {audio_file_path}")
            return None

        try:
            s3_key = f"{ENVIRONMENT.lower()}/audio/{call_id}/original.wav"

            self.s3_client.upload_file(
                audio_file_path,
                AWS_S3_BUCKET_NAME,
                s3_key,
            )

            s3_uri = f"s3://{AWS_S3_BUCKET_NAME}/{s3_key}"
            print(f"✅ Audio uploaded successfully")
            print(f"   S3 URI: {s3_uri}")
            return s3_uri

        except ClientError as e:
            print(f"❌ S3 upload error: {e}")
            return None
        except Exception as e:
            print(f"❌ Error uploading audio: {e}")
            return None

    def submit_transcription(self, s3_uri: str, call_id: str) -> Optional[str]:
        """
        Submit audio for transcription to AWS Transcribe

        Args:
            s3_uri: S3 URI of the audio file
            call_id: Unique call identifier

        Returns:
            Job name or None if failed
        """
        print(f"\n📝 Submitting for transcription...")

        job_name = f"transcribe-{call_id}-{int(time.time())}"

        try:
            response = self.transcribe_client.start_transcription_job(
                TranscriptionJobName=job_name,
                Media={"MediaFileUri": s3_uri},
                MediaFormat="wav",
                LanguageCode="en-US",
                OutputBucketName=AWS_S3_BUCKET_NAME,
                OutputKey=f"{ENVIRONMENT.lower()}/transcripts/{call_id}/",
                Settings={
                    "ShowAlternatives": False,
                    "MaxSpeakerLabels": 2,
                    "ShowSpeakerLabels": True,
                    "VocabularyFilterMethod": "mask",
                },
            )

            print(f"✅ Transcription job submitted successfully")
            print(f"   Job Name: {job_name}")
            print(f"   Status: {response['TranscriptionJob']['TranscriptionJobStatus']}")
            return job_name

        except ClientError as e:
            print(f"❌ Submission failed: {e}")
            return None
        except Exception as e:
            print(f"❌ Error submitting transcription: {e}")
            return None

    def wait_for_completion(self, job_name: str, max_retries: int = 120) -> Optional[Dict[str, Any]]:
        """
        Poll AWS Transcribe API until transcription is complete

        Args:
            job_name: Name of the transcription job
            max_retries: Maximum number of polling attempts

        Returns:
            Transcription job details or None if failed
        """
        print(f"\n⏳ Waiting for transcription to complete (polling every 10 seconds)...")

        retry_count = 0

        while retry_count < max_retries:
            try:
                response = self.transcribe_client.get_transcription_job(
                    TranscriptionJobName=job_name
                )

                job = response["TranscriptionJob"]
                status = job["TranscriptionJobStatus"]

                if status == "COMPLETED":
                    print(f"✅ Transcription completed!")
                    return job
                elif status == "FAILED":
                    error = job.get("FailureReason", "Unknown error")
                    print(f"❌ Transcription failed: {error}")
                    return None
                else:
                    print(f"   Status: {status} ({retry_count * 10}s elapsed)")
                    time.sleep(10)
                    retry_count += 1

            except Exception as e:
                print(f"❌ Error polling status: {e}")
                return None

        print(f"❌ Transcription timeout after {max_retries * 10} seconds")
        return None

    def get_transcript_content(self, job: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Download and parse the transcription result JSON from S3

        Args:
            job: Transcription job details

        Returns:
            Parsed transcript content or None if failed
        """
        try:
            transcript_uri = job["Transcript"]["TranscriptFileUri"]
            print(f"\n📥 Downloading transcript from: {transcript_uri}")

            # Check if it's an HTTPS URL or S3 URI
            if transcript_uri.startswith("https://"):
                # Download from HTTPS URL
                response = requests.get(transcript_uri)
                response.raise_for_status()
                content = response.json()
            else:
                # Download from S3 URI (s3://bucket/key format)
                s3_parts = transcript_uri.replace("s3://", "").split("/", 1)
                bucket = s3_parts[0]
                key = s3_parts[1]

                # Download from S3
                response = self.s3_client.get_object(Bucket=bucket, Key=key)
                content = json.loads(response["Body"].read())

            print(f"✅ Transcript downloaded successfully")
            return content

        except Exception as e:
            print(f"❌ Error downloading transcript: {e}")
            return None

    def delete_transcription_job(self, job_name: str) -> bool:
        """
        Delete a completed transcription job from AWS Transcribe

        Args:
            job_name: Name of the transcription job to delete

        Returns:
            True if successful, False otherwise
        """
        print(f"\n🗑️  Deleting transcription job: {job_name}")

        try:
            self.transcribe_client.delete_transcription_job(
                TranscriptionJobName=job_name
            )
            print(f"✅ Transcription job deleted successfully")
            return True

        except ClientError as e:
            print(f"❌ Failed to delete transcription job: {e}")
            return False
        except Exception as e:
            print(f"❌ Error deleting transcription job: {e}")
            return False


class PIIRedactor:
    """
    Handles PII detection and redaction using AWS Comprehend
    """

    def __init__(self, region: str = AWS_S3_REGION):
        """Initialize AWS Comprehend client"""
        self.comprehend_client = boto3.client(
            "comprehend",
            region_name=region,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        )

    def detect_pii_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect PII entities in text using AWS Comprehend

        Args:
            text: Text to analyze

        Returns:
            List of PII entities found
        """
        try:
            response = self.comprehend_client.detect_pii_entities(
                Text=text,
                LanguageCode="en",
            )

            entities = response.get("Entities", [])
            print(f"\n🔒 PII Detection Summary:")
            print(f"   Total entities found: {len(entities)}")

            entity_types = {}
            for entity in entities:
                entity_type = entity.get("Type")
                entity_types[entity_type] = entity_types.get(entity_type, 0) + 1

            if entity_types:
                print(f"   Entity types detected:")
                for entity_type, count in sorted(entity_types.items()):
                    print(f"      - {entity_type}: {count}")
            else:
                print(f"   No sensitive information detected")

            return entities

        except Exception as e:
            print(f"❌ Error detecting PII: {e}")
            return []

    def redact_text(self, text: str, entities: List[Dict[str, Any]]) -> str:
        """
        Redact PII entities from text

        Args:
            text: Original text
            entities: List of PII entities

        Returns:
            Redacted text
        """
        if not entities:
            return text

        # Sort by character offset (descending) to avoid offset issues
        sorted_entities = sorted(
            entities,
            key=lambda x: x["BeginOffset"],
            reverse=True
        )

        redacted_text = text
        for entity in sorted_entities:
            entity_type = entity.get("Type", "UNKNOWN")
            begin = entity.get("BeginOffset")
            end = entity.get("EndOffset")

            if begin is not None and end is not None:
                placeholder = f"[{entity_type}]"
                redacted_text = redacted_text[:begin] + placeholder + redacted_text[end:]

        return redacted_text

    def extract_word_timings(self, transcript_content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract word-level timing information from AWS Transcribe results

        Args:
            transcript_content: Transcription job results from AWS Transcribe

        Returns:
            List of words with timing information
        """
        words = []
        try:
            results = transcript_content.get("results", {})
            items = results.get("items", [])

            for item in items:
                if item.get("type") == "pronunciation":
                    word_data = {
                        "word": item.get("alternatives", [{}])[0].get("content", ""),
                        "start_time": float(item.get("start_time", 0)),
                        "end_time": float(item.get("end_time", 0)),
                        "confidence": float(item.get("alternatives", [{}])[0].get("confidence", 1.0)),
                    }
                    words.append(word_data)
        except Exception as e:
            print(f"   Warning: Could not extract word timings: {e}")

        return words

    def map_character_to_time(self, text: str, words: List[Dict[str, Any]]) -> Optional[Dict[int, float]]:
        """
        Create a mapping of character offsets to audio timestamps using word-level timing

        Args:
            text: Full transcript text
            words: List of words with timing info

        Returns:
            Dictionary mapping character offset to timestamp (in ms)
        """
        char_to_time = {}

        if not words:
            return None

        current_char_pos = 0

        for word_info in words:
            word = word_info.get("word", "").lower()
            start_time_s = word_info.get("start_time", 0)
            end_time_s = word_info.get("end_time", 0)

            # Find word in remaining text (case-insensitive)
            remaining_text = text[current_char_pos:].lower()
            word_pos = remaining_text.find(word)

            if word_pos != -1:
                # Map character positions to timestamps
                word_start_char = current_char_pos + word_pos
                word_end_char = word_start_char + len(word)

                # Linear interpolation for characters within word
                for char_offset in range(word_start_char, word_end_char):
                    if word_end_char > word_start_char:
                        # Interpolate timestamp based on position within word
                        progress = (char_offset - word_start_char) / (word_end_char - word_start_char)
                        timestamp_s = start_time_s + progress * (end_time_s - start_time_s)
                        char_to_time[char_offset] = timestamp_s * 1000  # Convert to ms
                    else:
                        char_to_time[char_offset] = start_time_s * 1000

                current_char_pos = word_end_char

        return char_to_time if char_to_time else None

    def redact_audio(self, audio_file_path: str,
                     original_text: str,
                     entities: List[Dict[str, Any]],
                     transcript_content: Optional[Dict[str, Any]] = None,
                     redaction_mode: str = "silence",
                     beep_frequency: int = 1000,
                     beep_duration: int = 100) -> Optional[bytes]:
        """
        Redact audio by replacing PII segments with different methods

        Args:
            audio_file_path: Path to the original audio file
            original_text: Original transcript text
            entities: List of PII entities with character offsets
            transcript_content: Full AWS Transcribe response for word-level timing
            redaction_mode: Redaction method - 'silence', 'beep', 'white_noise', 'pink_noise', 'tone', 'mute'
            beep_frequency: Frequency of beep/tone in Hz (default 1000)
            beep_duration: Duration of beep in ms (default 100)

        Returns:
            Redacted audio bytes or None if failed
        """
        try:
            print(f"\n🔊 Redacting audio ({redaction_mode} mode) for {len(entities)} PII entities...")

            if not entities:
                print(f"   No PII entities to redact in audio")
                return None

            # Load audio file - auto-detect format
            print(f"   Loading audio file: {audio_file_path}")
            try:
                audio = AudioSegment.from_file(audio_file_path)
            except Exception as e:
                print(f"⚠️  Could not load audio file: {e}")
                print(f"   Trying alternative format detection...")
                # Try explicit format detection
                file_ext = Path(audio_file_path).suffix.lower().lstrip('.')
                if file_ext:
                    try:
                        audio = AudioSegment.from_file(audio_file_path, format=file_ext)
                    except Exception as e2:
                        print(f"❌ Failed to load audio: {e2}")
                        return None
                else:
                    return None

            audio_duration_ms = len(audio)

            print(f"   Audio duration: {audio_duration_ms / 1000:.2f} seconds")

            # Try to use word-level timing from AWS Transcribe
            char_to_time = None
            if transcript_content:
                words = self.extract_word_timings(transcript_content)
                if words:
                    print(f"   Using precise word-level timing ({len(words)} words detected)")
                    char_to_time = self.map_character_to_time(original_text, words)

            # Fallback to linear estimation if word timing unavailable
            if char_to_time is None:
                print(f"   Using linear character-to-time estimation (less accurate)")
                chars_per_second = len(original_text) / (audio_duration_ms / 1000)
                print(f"   Estimated speech rate: {chars_per_second:.1f} chars/second")

            # Sort entities by character offset (descending) to avoid offset shifting
            sorted_entities = sorted(
                entities,
                key=lambda x: x["BeginOffset"],
                reverse=True
            )

            # Replace PII segments based on mode
            for entity in sorted_entities:
                begin_char = entity.get("BeginOffset")
                end_char = entity.get("EndOffset")
                entity_type = entity.get("Type", "UNKNOWN")
                confidence = entity.get("Score", 1.0)

                if begin_char is not None and end_char is not None:
                    # Get timing information
                    if char_to_time:
                        # Use precise word-level timing
                        start_ms = char_to_time.get(begin_char)
                        end_ms = char_to_time.get(end_char)

                        # Fallback: find closest character times
                        if start_ms is None:
                            for offset in range(begin_char, end_char):
                                if offset in char_to_time:
                                    start_ms = char_to_time[offset]
                                    break
                        if end_ms is None:
                            for offset in range(end_char - 1, begin_char - 1, -1):
                                if offset in char_to_time:
                                    end_ms = char_to_time[offset]
                                    break

                        if start_ms is None or end_ms is None:
                            # Fallback to linear estimation
                            chars_per_second = len(original_text) / (audio_duration_ms / 1000)
                            start_ms = int((begin_char / chars_per_second) * 1000)
                            end_ms = int((end_char / chars_per_second) * 1000)
                    else:
                        # Linear estimation
                        chars_per_second = len(original_text) / (audio_duration_ms / 1000)
                        start_ms = int((begin_char / chars_per_second) * 1000)
                        end_ms = int((end_char / chars_per_second) * 1000)

                    # Ensure times are within audio bounds
                    start_ms = max(0, int(start_ms))
                    end_ms = min(audio_duration_ms, int(end_ms))
                    duration_ms = end_ms - start_ms

                    if duration_ms > 50:  # Only redact if segment is long enough
                        # Create redaction segment based on mode
                        if redaction_mode == "silence":
                            # Complete silence
                            redaction_segment = AudioSegment.silent(duration=duration_ms)

                        elif redaction_mode == "beep":
                            # Beep sounds
                            beep = Sine(beep_frequency).to_audio_segment(duration=beep_duration)
                            redaction_segment = AudioSegment.silent(duration=duration_ms)
                            num_beeps = max(1, duration_ms // (beep_duration + 100))
                            for i in range(num_beeps):
                                beep_pos = i * (duration_ms // num_beeps)
                                redaction_segment = redaction_segment.overlay(
                                    beep,
                                    position=min(beep_pos, duration_ms - beep_duration)
                                )

                        elif redaction_mode == "white_noise":
                            # White noise (random sound)
                            noise = self._generate_white_noise(duration_ms)
                            redaction_segment = noise - 12  # Reduce volume for subtlety

                        elif redaction_mode == "pink_noise":
                            # Pink noise (softer, more natural)
                            noise = self._generate_pink_noise(duration_ms)
                            redaction_segment = noise - 12  # Reduce volume

                        elif redaction_mode == "tone":
                            # Single continuous tone
                            tone = Sine(beep_frequency).to_audio_segment(duration=duration_ms)
                            redaction_segment = tone - 15  # Reduce volume

                        elif redaction_mode == "mute":
                            # Gradual fade to silence
                            redaction_segment = AudioSegment.silent(duration=duration_ms)
                            # Apply fade out effect
                            fade_duration = min(100, duration_ms // 3)
                            original_segment = audio[start_ms:end_ms]
                            if len(original_segment) > fade_duration * 2:
                                redaction_segment = (
                                    original_segment[:fade_duration].fade_out(fade_duration) +
                                    AudioSegment.silent(duration=max(0, duration_ms - fade_duration * 2)) +
                                    original_segment[-fade_duration:].fade_in(fade_duration)
                                )
                            else:
                                redaction_segment = original_segment.fade_out(duration_ms // 2)

                        else:
                            # Default to silence
                            redaction_segment = AudioSegment.silent(duration=duration_ms)

                        # Replace segment in audio
                        audio = audio[:start_ms] + redaction_segment + audio[end_ms:]

                        print(f"   ✓ Redacted {entity_type} ({confidence:.1%}) at {start_ms}ms-{end_ms}ms")

            # Convert to bytes
            audio_bytes = audio.export(format="wav").read()
            print(f"✅ Audio redaction completed")
            return audio_bytes

        except Exception as e:
            print(f"❌ Error redacting audio: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _generate_white_noise(self, duration_ms: int) -> AudioSegment:
        """Generate white noise for redaction"""
        sample_rate = 44100
        duration_s = duration_ms / 1000
        samples = np.random.randint(-32768, 32767, int(sample_rate * duration_s), dtype=np.int16)
        return AudioSegment(
            samples.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=1
        )

    def _generate_pink_noise(self, duration_ms: int) -> AudioSegment:
        """Generate pink noise (1/f noise) for redaction"""
        sample_rate = 44100
        duration_s = duration_ms / 1000
        num_samples = int(sample_rate * duration_s)

        # Generate pink noise using simple algorithm
        white = np.random.randn(num_samples)
        # Simple pink noise filter (not perfect but good enough)
        pink = np.convolve(white, [0.049922035, -0.095993537, 0.050612699, -0.004408786], mode='same')

        # Normalize to 16-bit range
        pink = pink / np.max(np.abs(pink)) * 32767
        pink = pink.astype(np.int16)

        return AudioSegment(
            pink.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=1
        )


class GeminiSentimentAnalyzer:
    """
    Handles sentiment analysis using Google Gemini API
    """

    def __init__(self, api_key: str):
        """Initialize Gemini API client"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')

    def format_time(self, seconds: float) -> str:
        """Convert seconds to MM:SS format"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    def extract_speaker_segments(self, transcript_content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract speaker segments from AWS Transcribe results
        Builds proper text with punctuation from items

        Args:
            transcript_content: Transcription job results from AWS Transcribe

        Returns:
            List of speaker segments with text and timing
        """
        segments = []

        try:
            results = transcript_content.get("results", {})
            items = results.get("items", [])
            speaker_labels = results.get("speaker_labels", {})
            segments_data = speaker_labels.get("segments", [])

            # Build speaker segments with proper text
            for segment in segments_data:
                speaker = segment.get("speaker_label", "Unknown")
                start_time = float(segment.get("start_time", 0))
                end_time = float(segment.get("end_time", 0))

                # Get items for this segment
                segment_items = segment.get("items", [])

                # Build text by finding items in the time range
                text_parts = []
                for i, item in enumerate(items):
                    item_type = item.get("type")

                    if item_type == "pronunciation":
                        item_start = float(item.get("start_time", 0))
                        item_end = float(item.get("end_time", 0))

                        # Check if this item belongs to this segment
                        if item_start >= start_time and item_end <= end_time:
                            word = item.get("alternatives", [{}])[0].get("content", "")
                            text_parts.append(word)

                            # Check if next item is punctuation
                            if i + 1 < len(items) and items[i + 1].get("type") == "punctuation":
                                punct = items[i + 1].get("alternatives", [{}])[0].get("content", "")
                                if punct and text_parts:
                                    text_parts[-1] = text_parts[-1] + punct

                text = " ".join(text_parts)

                if text.strip():
                    segments.append({
                        "speaker": speaker,
                        "text": text.strip(),
                        "start_time": start_time,
                        "end_time": end_time
                    })

        except Exception as e:
            print(f"❌ Error extracting speaker segments: {e}")
            import traceback
            traceback.print_exc()

        return segments

    def merge_consecutive_segments(self, segments: List[Dict[str, Any]], max_gap: float = 2.0) -> List[Dict[str, Any]]:
        """
        Merge consecutive segments from the same speaker that are close together
        Also fixes AWS Transcribe speaker diarization errors by detecting and merging
        suspicious short segments that are clearly part of the same utterance.

        Args:
            segments: List of speaker segments
            max_gap: Maximum gap in seconds between segments to merge

        Returns:
            List of merged segments with corrected speaker labels
        """
        if not segments:
            return segments

        # First pass: Fix obvious speaker diarization errors
        # Look for patterns where very short segments alternate speakers rapidly
        fixed_segments = []
        i = 0
        while i < len(segments):
            current = segments[i]

            # Look ahead for suspicious patterns (short segments with overlapping/close times)
            if i + 1 < len(segments):
                next_seg = segments[i + 1]

                # Check if we have suspicious alternating short segments
                # (less than 3 seconds duration, very close timestamps)
                current_duration = current["end_time"] - current["start_time"]
                next_duration = next_seg["end_time"] - next_seg["start_time"]
                time_gap = next_seg["start_time"] - current["end_time"]

                # If both segments are very short and very close together
                if (current_duration < 3.0 and next_duration < 3.0 and
                    time_gap <= 0.5 and current["speaker"] != next_seg["speaker"]):

                    # Look for a cluster of such segments
                    cluster = [current]
                    j = i + 1
                    while j < len(segments):
                        seg = segments[j]
                        seg_duration = seg["end_time"] - seg["start_time"]
                        gap = seg["start_time"] - cluster[-1]["end_time"]

                        if seg_duration < 3.0 and gap <= 0.5:
                            cluster.append(seg)
                            j += 1
                        else:
                            break

                    # If we found a cluster (3+ short segments), merge them
                    if len(cluster) >= 3:
                        # Determine the correct speaker based on context
                        # Use the speaker from the segment before the cluster
                        if i > 0:
                            correct_speaker = segments[i - 1]["speaker"]
                        elif i + len(cluster) < len(segments):
                            # Use the opposite of the next speaker
                            next_speaker = segments[i + len(cluster)]["speaker"]
                            correct_speaker = "spk_0" if next_speaker == "spk_1" else "spk_1"
                        else:
                            # Default to the first speaker in the cluster
                            correct_speaker = cluster[0]["speaker"]

                        # Merge the entire cluster
                        merged_text = " ".join(seg["text"] for seg in cluster)
                        merged_segment = {
                            "speaker": correct_speaker,
                            "text": merged_text.strip(),
                            "start_time": cluster[0]["start_time"],
                            "end_time": cluster[-1]["end_time"]
                        }

                        fixed_segments.append(merged_segment)
                        i = j  # Skip past the cluster
                        continue

            # No cluster found, add segment as-is
            fixed_segments.append(current)
            i += 1

        # Second pass: Merge consecutive segments from the same speaker
        if not fixed_segments:
            return []

        merged = []
        current_segment = fixed_segments[0].copy()

        for i in range(1, len(fixed_segments)):
            segment = fixed_segments[i]

            # Check if same speaker and within time gap
            if (segment["speaker"] == current_segment["speaker"] and
                segment["start_time"] - current_segment["end_time"] <= max_gap):
                # Merge with current segment
                current_segment["text"] += " " + segment["text"]
                current_segment["end_time"] = segment["end_time"]
            else:
                # Save current and start new
                merged.append(current_segment)
                current_segment = segment.copy()

        # Add the last segment
        merged.append(current_segment)

        return merged

    def analyze_sentiment_with_gemini(self, segments: List[Dict[str, Any]], full_transcript: str = "", pii_entities: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Analyze sentiment for each segment using Gemini API
        Uses original text for analysis but returns PII-redacted text in output

        Args:
            segments: List of speaker segments (with original text)
            full_transcript: Full transcript text for PII redaction mapping
            pii_entities: List of PII entities to redact from output text

        Returns:
            List of segments with sentiment analysis and PII-redacted text
        """
        print(f"\n🤖 Analyzing sentiment with Gemini AI for {len(segments)} segments...")

        analyzed_segments = []

        try:
            # Prepare the prompt for Gemini (using original text for better analysis)
            prompt = """You are a call quality monitoring expert analyzing a customer service conversation. Your job is to identify problems, compliance issues, and professionalism concerns.

IMPORTANT CONTEXT:
- Agent (spk_0): Customer service representative - should maintain professionalism, protect privacy, follow procedures
- Customer (spk_1): Caller seeking help - may show frustration when service is poor

For each segment, analyze:
1. sentiment: "positive", "negative", or "neutral"
2. confidence: float between 0.0-1.0
3. tone_note: Detailed description that identifies:
   - For AGENTS: Note any unprofessional behavior, privacy violations (mentioning other patients' info), procedural errors, dismissive attitudes, or security concerns
   - For CUSTOMERS: Note if frustration/anger is justified reaction to poor service, or if showing patience, concern, cooperation
   - Be specific about WHAT the problem is (e.g., "Sharing another patient's name - HIPAA violation", "Refusing to follow customer's explicit request", "Justifiably frustrated by agent's unprofessional conduct")

Examples of good tone_notes:
- Agent: "Extremely unprofessional - discussing another patient's medical procedure by name (privacy violation)"
- Agent: "Dismissive of customer's privacy concerns and security requests"
- Customer: "Angry and demanding accountability - justified response to multiple privacy breaches"
- Customer: "Cooperative and providing required information"
- Agent: "Professional and welcoming greeting"

Conversation segments:
"""

            # Add segments to prompt (using original text)
            for i, segment in enumerate(segments, 1):
                prompt += f"\n{i}. [{segment['speaker']}]: {segment['text']}"

            prompt += """

Return ONLY a valid JSON array in this exact format, with no additional text:
[
  {
    "sentiment": "neutral",
    "confidence": 0.9,
    "tone_note": "Professional and welcoming"
  },
  ...
]
"""

            # Call Gemini API
            print("   Sending request to Gemini API...")
            response = self.model.generate_content(prompt)

            # Parse response
            response_text = response.text.strip()

            # Extract JSON from response (handle markdown code blocks)
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()

            # Parse JSON
            sentiment_results = json.loads(response_text)

            # Combine segments with sentiment analysis
            for i, segment in enumerate(segments):
                # Redact PII from segment text if entities provided
                segment_text = segment["text"]
                if pii_entities and full_transcript:
                    segment_text = self._redact_segment_text(segment_text, full_transcript, pii_entities)

                # Map speaker labels to Agent/Customer
                speaker_label = segment["speaker"].replace("spk_", "Speaker ")
                if "0" in speaker_label or speaker_label == "Speaker 0":
                    speaker_label = "Agent"
                elif "1" in speaker_label or speaker_label == "Speaker 1":
                    speaker_label = "Customer"

                if i < len(sentiment_results):
                    sentiment_data = sentiment_results[i]

                    analyzed_segment = {
                        "order": i + 1,
                        "speaker": speaker_label,
                        "text": segment_text,
                        "start_time": self.format_time(segment["start_time"]),
                        "end_time": self.format_time(segment["end_time"]),
                        "sentiment": sentiment_data.get("sentiment", "neutral"),
                        "confidence": sentiment_data.get("confidence", 0.9),
                        "tone_note": sentiment_data.get("tone_note", "Neutral tone")
                    }

                    analyzed_segments.append(analyzed_segment)
                else:
                    # Fallback if Gemini doesn't return enough results
                    analyzed_segment = {
                        "order": i + 1,
                        "speaker": speaker_label,
                        "text": segment_text,
                        "start_time": self.format_time(segment["start_time"]),
                        "end_time": self.format_time(segment["end_time"]),
                        "sentiment": "neutral",
                        "confidence": 0.9,
                        "tone_note": "Neutral tone"
                    }

                    analyzed_segments.append(analyzed_segment)

            print(f"✅ Sentiment analysis completed for {len(analyzed_segments)} segments")

        except Exception as e:
            print(f"❌ Error analyzing sentiment with Gemini: {e}")
            import traceback
            traceback.print_exc()

            # Fallback: return segments with neutral sentiment
            for i, segment in enumerate(segments):
                # Redact PII from segment text if entities provided
                segment_text = segment["text"]
                if pii_entities and full_transcript:
                    segment_text = self._redact_segment_text(segment_text, full_transcript, pii_entities)

                # Map speaker labels to Agent/Customer
                speaker_label = segment["speaker"].replace("spk_", "Speaker ")
                if "0" in speaker_label or speaker_label == "Speaker 0":
                    speaker_label = "Agent"
                elif "1" in speaker_label or speaker_label == "Speaker 1":
                    speaker_label = "Customer"

                analyzed_segment = {
                    "order": i + 1,
                    "speaker": speaker_label,
                    "text": segment_text,
                    "start_time": self.format_time(segment["start_time"]),
                    "end_time": self.format_time(segment["end_time"]),
                    "sentiment": "neutral",
                    "confidence": 0.9,
                    "tone_note": "Neutral tone"
                }

                analyzed_segments.append(analyzed_segment)

        return analyzed_segments

    def _redact_segment_text(self, segment_text: str, full_transcript: str, pii_entities: List[Dict[str, Any]]) -> str:
        """
        Redact PII entities from text segment by finding segment position in full transcript

        Args:
            segment_text: Text from this specific segment
            full_transcript: Full transcript text
            pii_entities: List of PII entities from the full transcript

        Returns:
            Text with PII redacted using [ENTITY_TYPE] format
        """
        if not pii_entities or not segment_text:
            return segment_text

        # Find this segment's position in the full transcript (case-insensitive, handle spaces)
        normalized_full = " ".join(full_transcript.split())
        normalized_segment = " ".join(segment_text.split())

        segment_start = normalized_full.lower().find(normalized_segment.lower())
        if segment_start == -1:
            # Segment not found, return as-is
            return segment_text

        segment_end = segment_start + len(normalized_segment)

        # Collect entities that overlap with this segment
        overlapping_entities = []
        for entity in pii_entities:
            entity_begin = entity.get("BeginOffset", 0)
            entity_end = entity.get("EndOffset", 0)

            # Check if entity overlaps with this segment
            if entity_begin < segment_end and entity_end > segment_start:
                # Calculate relative position within segment
                rel_begin = max(0, entity_begin - segment_start)
                rel_end = min(len(normalized_segment), entity_end - segment_start)

                # Get entity text from full transcript
                entity_text = normalized_full[entity_begin:entity_end]
                entity_type = entity.get("Type", "UNKNOWN")

                overlapping_entities.append({
                    "text": entity_text,
                    "type": entity_type,
                    "rel_begin": rel_begin,
                    "rel_end": rel_end
                })

        # Sort by position (descending) to avoid offset issues
        overlapping_entities.sort(key=lambda x: x["rel_begin"], reverse=True)

        # Apply redactions
        redacted = normalized_segment
        for entity in overlapping_entities:
            entity_text = entity["text"]
            entity_type = entity["type"]
            placeholder = f"[{entity_type}]"

            # Replace entity text with placeholder (case-insensitive)
            # Find all occurrences in the segment
            import re
            pattern = re.escape(entity_text)
            redacted = re.sub(pattern, placeholder, redacted, count=1, flags=re.IGNORECASE)

        return redacted


class S3Manager:
    """
    Handles S3 bucket operations for storing redacted audio
    """

    def __init__(self,
                 access_key: str,
                 secret_key: str,
                 bucket_name: str,
                 region: str = "us-east-1"):
        """Initialize S3 client"""
        self.bucket_name = bucket_name
        self.region = region

        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )

    def upload_redacted_audio(self,
                              audio_bytes: bytes,
                              call_id: str,
                              audio_format: str = "wav") -> Optional[str]:
        """
        Upload redacted audio to S3

        Args:
            audio_bytes: Audio file bytes
            call_id: Unique call identifier
            audio_format: Audio format (wav, mp3, etc.)

        Returns:
            S3 streaming URL or None if failed
        """
        print(f"\n☁️  Uploading redacted audio to S3...")

        try:
            # Create S3 key path
            s3_key = f"{ENVIRONMENT.lower()}/transcriptions/{call_id}/redacted_audio.{audio_format}"

            # Determine content type
            content_type_map = {
                "wav": "audio/wav",
                "mp3": "audio/mpeg",
                "m4a": "audio/mp4",
            }
            content_type = content_type_map.get(audio_format, "audio/wav")

            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=audio_bytes,
                ContentType=content_type,
            )

            # Generate streaming URL
            s3_url = f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{s3_key}"

            print(f"✅ Redacted audio uploaded to S3")
            print(f"   S3 Key: {s3_key}")
            print(f"   Size: {len(audio_bytes) / (1024*1024):.2f} MB")

            return s3_url

        except ClientError as e:
            print(f"❌ S3 upload error: {e}")
            return None
        except Exception as e:
            print(f"❌ Error uploading to S3: {e}")
            return None


def print_transcription_summary(transcript_content: Dict[str, Any],
                              pii_entities: List[Dict[str, Any]],
                              redacted_text: str):
    """Print a formatted summary of transcription results"""

    print("\n" + "="*80)
    print("📊 TRANSCRIPTION SUMMARY")
    print("="*80)

    # Basic Info
    print(f"\n📋 Basic Information:")
    results = transcript_content.get("results", {})
    transcriptions = results.get("transcripts", [])

    if transcriptions:
        original_text = transcriptions[0].get("transcript", "N/A")
    else:
        original_text = "N/A"

    print(f"   Language: English (US)")

    # Transcription Text
    print(f"\n📝 Full Transcript (Original):")
    print(f"   {original_text[:300]}...")

    print(f"\n📝 Full Transcript (PII Redacted):")
    print(f"   {redacted_text[:300]}...")

    # PII Information
    if pii_entities:
        print(f"\n🔒 PII Redaction Summary:")
        print(f"   Total entities found and redacted: {len(pii_entities)}")

        entity_types = {}
        for entity in pii_entities:
            entity_type = entity.get("Type")
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1

        print(f"   Entity types detected:")
        for entity_type, count in sorted(entity_types.items()):
            print(f"      - {entity_type}: {count}")
    else:
        print(f"\n🔒 PII Redaction Summary:")
        print(f"   No sensitive information detected")

    # Speaker Information
    speaker_labels = results.get("speaker_labels", [])
    if speaker_labels:
        print(f"\n👥 Speaker Information:")
        unique_speakers = set(speaker_labels) if speaker_labels else set()
        print(f"   Total speakers: {len(unique_speakers)}")
        for speaker in sorted(unique_speakers):
            print(f"      - {speaker}")

    print("\n" + "="*80)


async def main():
    """Main execution function"""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Test AWS Transcribe and Comprehend PII Redaction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_assemblyai_transcription.py ./sample_call.wav
  python test_assemblyai_transcription.py ./recordings/call_001.wav --call-id call-001
        """
    )

    parser.add_argument(
        "audio_file",
        help="Path to the .wav audio file to transcribe",
    )

    parser.add_argument(
        "--call-id",
        default=None,
        help="Unique call ID (default: generated from timestamp)",
    )

    parser.add_argument(
        "--save-transcript",
        action="store_true",
        help="Save full transcript to JSON file",
    )

    parser.add_argument(
        "--save-audio",
        action="store_true",
        help="Save redacted transcript to text file",
    )

    args = parser.parse_args()

    # Set redaction mode to tone (fixed)
    args.redaction_mode = "tone"

    # Validate inputs
    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        print("❌ Error: AWS credentials not set in environment")
        sys.exit(1)

    if not GEMINI_API_KEY:
        print("❌ Error: GEMINI_API_KEY not set in environment")
        sys.exit(1)

    audio_file = args.audio_file
    if not os.path.exists(audio_file):
        print(f"❌ Error: Audio file not found: {audio_file}")
        sys.exit(1)

    # Generate call ID if not provided
    call_id = args.call_id or f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"\n{'='*80}")
    print(f"🎙️  AWS Transcribe & Comprehend PII Redaction Test")
    print(f"{'='*80}")
    print(f"Audio file: {audio_file}")
    print(f"Call ID: {call_id}")
    print(f"Environment: {ENVIRONMENT}")

    # Initialize AWS services
    transcriber = AWSTranscriber()
    pii_redactor = PIIRedactor()
    gemini_analyzer = GeminiSentimentAnalyzer(api_key=GEMINI_API_KEY)
    s3_manager = S3Manager(
        access_key=AWS_ACCESS_KEY_ID,
        secret_key=AWS_SECRET_ACCESS_KEY,
        bucket_name=AWS_S3_BUCKET_NAME,
        region=AWS_S3_REGION,
    )

    # Step 1: Upload audio to S3
    s3_uri = transcriber.upload_audio_to_s3(audio_file, call_id)
    if not s3_uri:
        print("\n❌ Failed to upload audio to S3. Exiting.")
        sys.exit(1)

    # Step 2: Submit for transcription
    job_name = transcriber.submit_transcription(s3_uri, call_id)
    if not job_name:
        print("\n❌ Failed to submit for transcription. Exiting.")
        sys.exit(1)

    # Step 3: Wait for completion
    job = transcriber.wait_for_completion(job_name)
    if not job:
        print("\n❌ Transcription failed or timed out. Exiting.")
        sys.exit(1)

    # Step 4: Get transcript content
    transcript_content = transcriber.get_transcript_content(job)
    if not transcript_content:
        print("\n❌ Failed to download transcript. Exiting.")
        sys.exit(1)

    # Extract transcript text
    results = transcript_content.get("results", {})
    transcriptions = results.get("transcripts", [])
    original_text = transcriptions[0].get("transcript", "") if transcriptions else ""

    # Step 5: Detect PII entities
    pii_entities = pii_redactor.detect_pii_entities(original_text)

    # Step 6: Redact PII from text
    redacted_text = pii_redactor.redact_text(original_text, pii_entities)

    # Step 6.5: Extract speaker segments and analyze sentiment with Gemini
    print("\n" + "="*80)
    print("🎭 SENTIMENT ANALYSIS WITH GEMINI")
    print("="*80)

    # Extract speaker segments (with original text)
    speaker_segments = gemini_analyzer.extract_speaker_segments(transcript_content)

    # Merge consecutive segments from same speaker for better context
    print(f"   Original segments: {len(speaker_segments)}")
    merged_segments = gemini_analyzer.merge_consecutive_segments(speaker_segments, max_gap=2.0)
    print(f"   Merged segments: {len(merged_segments)}")

    # Analyze sentiment using original text, but output will have PII redacted
    sentiment_analysis_results = gemini_analyzer.analyze_sentiment_with_gemini(
        merged_segments,
        full_transcript=original_text,
        pii_entities=pii_entities
    )

    # Save sentiment analysis results to JSON file
    sentiment_file = f"sentiment_analysis_{call_id}.json"
    with open(sentiment_file, "w") as f:
        json.dump(sentiment_analysis_results, f, indent=2)
    print(f"\n✅ Sentiment analysis saved to: {sentiment_file}")

    # Print preview of sentiment analysis
    print(f"\n📊 Sentiment Analysis Preview (first 3 segments):")
    for segment in sentiment_analysis_results[:3]:
        print(f"\n   Order: {segment['order']}")
        print(f"   Speaker: {segment['speaker']}")
        print(f"   Text: {segment['text'][:100]}...")
        print(f"   Time: {segment['start_time']} - {segment['end_time']}")
        print(f"   Sentiment: {segment['sentiment']} (confidence: {segment['confidence']:.2f})")
        print(f"   Tone: {segment['tone_note']}")

    # Step 7: Redact PII from audio
    redacted_audio_bytes = pii_redactor.redact_audio(
        audio_file,
        original_text,
        pii_entities,
        transcript_content=transcript_content,  # Pass transcript for word-level timing
        redaction_mode=args.redaction_mode  # Use user's chosen redaction method
    )

    # Step 8: Upload redacted audio to S3
    redacted_audio_s3_url = None
    if redacted_audio_bytes:
        redacted_audio_s3_url = s3_manager.upload_redacted_audio(
            redacted_audio_bytes,
            call_id,
            audio_format="wav"
        )

    # Print summary
    print_transcription_summary(transcript_content, pii_entities, redacted_text)

    # Step 7: Save results
    results_to_save = {
        "call_id": call_id,
        "job_name": job_name,
        "original_transcript": original_text,
        "redacted_transcript": redacted_text,
        "pii_entities": pii_entities,
        "redacted_audio_s3_url": redacted_audio_s3_url,
        "sentiment_analysis": sentiment_analysis_results,
        "full_transcript_content": transcript_content,
    }

    if args.save_transcript:
        transcript_file = f"transcript_{call_id}.json"
        with open(transcript_file, "w") as f:
            json.dump(results_to_save, f, indent=2)
        print(f"\n✅ Transcript saved to: {transcript_file}")

    if args.save_audio:
        redacted_file = f"redacted_transcript_{call_id}.txt"
        with open(redacted_file, "w") as f:
            f.write(redacted_text)
        print(f"✅ Redacted transcript saved to: {redacted_file}")

    # Step 9: Delete the transcription job
    transcriber.delete_transcription_job(job_name)

    # Final Results
    print(f"\n{'='*80}")
    print(f"✅ TEST COMPLETED SUCCESSFULLY")
    print(f"{'='*80}")
    print(f"\n📊 Results Summary:")
    print(f"   Call ID: {call_id}")
    print(f"   Job Name: {job_name}")
    print(f"   Original transcript length: {len(original_text)} characters")
    print(f"   Redacted transcript length: {len(redacted_text)} characters")
    print(f"   PII entities found: {len(pii_entities)}")
    print(f"   Speaker segments analyzed: {len(sentiment_analysis_results)}")
    print(f"   Sentiment analysis file: {sentiment_file}")
    print(f"   Original audio location: {s3_uri}")
    if redacted_audio_s3_url:
        print(f"   Redacted audio location: {redacted_audio_s3_url}")
    else:
        print(f"   Redacted audio location: Not uploaded (no PII found or error occurred)")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# def run_pipeline(audio_file_path: str, call_id: str):
#     """
#     Runs the full transcription + PII + sentiment pipeline
#     Returns results as a dictionary
#     """

#     transcriber = AWSTranscriber()
#     pii_redactor = PIIRedactor()
#     gemini_analyzer = GeminiSentimentAnalyzer(api_key=GEMINI_API_KEY)
#     s3_manager = S3Manager(
#         access_key=AWS_ACCESS_KEY_ID,
#         secret_key=AWS_SECRET_ACCESS_KEY,
#         bucket_name=AWS_S3_BUCKET_NAME,
#         region=AWS_S3_REGION,
#     )

#     # 1. Upload audio
#     s3_uri = transcriber.upload_audio_to_s3(audio_file_path, call_id)
#     if not s3_uri:
#         raise RuntimeError("Failed to upload audio")

#     # 2. Transcription
#     job_name = transcriber.submit_transcription(s3_uri, call_id)
#     job = transcriber.wait_for_completion(job_name)
#     transcript_content = transcriber.get_transcript_content(job)

#     results = transcript_content["results"]
#     original_text = results["transcripts"][0]["transcript"]

#     # 3. PII detection
#     pii_entities = pii_redactor.detect_pii_entities(original_text)
#     redacted_text = pii_redactor.redact_text(original_text, pii_entities)

#     # 4. Sentiment
#     segments = gemini_analyzer.extract_speaker_segments(transcript_content)
#     merged_segments = gemini_analyzer.merge_consecutive_segments(segments)

#     sentiment = gemini_analyzer.analyze_sentiment_with_gemini(
#         merged_segments,
#         full_transcript=original_text,
#         pii_entities=pii_entities
#     )

#     # 5. Audio redaction
#     redacted_audio = pii_redactor.redact_audio(
#         audio_file_path,
#         original_text,
#         pii_entities,
#         transcript_content=transcript_content,
#         redaction_mode="tone",
#     )

#     redacted_audio_url = None
#     if redacted_audio:
#         redacted_audio_url = s3_manager.upload_redacted_audio(
#             redacted_audio, call_id
#         )

#     transcriber.delete_transcription_job(job_name)

#     return {
#         "call_id": call_id,
#         "original_text": original_text,
#         "redacted_text": redacted_text,
#         "pii_entities": pii_entities,
#         "sentiment": sentiment,
#         "redacted_audio_url": redacted_audio_url,
#     }
