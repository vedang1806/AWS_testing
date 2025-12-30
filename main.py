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

# Load environment variables
load_dotenv()

# Constants
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_S3_BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME", "bpo-box-dev")
AWS_S3_REGION = os.getenv("AWS_S3_REGION", "us-east-1")
ENVIRONMENT = os.getenv("ENVIRONMENT", "DEV")

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

    def redact_audio(self, audio_file_path: str,
                     original_text: str,
                     entities: List[Dict[str, Any]],
                     beep_frequency: int = 1000,
                     beep_duration: int = 100) -> Optional[bytes]:
        """
        Redact audio by replacing PII segments with beep sounds

        Args:
            audio_file_path: Path to the original audio file
            original_text: Original transcript text
            entities: List of PII entities with character offsets
            beep_frequency: Frequency of beep in Hz (default 1000)
            beep_duration: Duration of beep in ms (default 100)

        Returns:
            Redacted audio bytes or None if failed
        """
        try:
            print(f"\n🔊 Redacting audio with beeps for {len(entities)} PII entities...")

            if not entities:
                print(f"   No PII entities to redact in audio")
                return None

            # Load audio file
            audio = AudioSegment.from_wav(audio_file_path)
            audio_duration_ms = len(audio)

            # Get word-level timing if available (estimate from character offsets)
            # Calculate character-to-time mapping based on speech rate
            chars_per_second = len(original_text) / (audio_duration_ms / 1000)

            print(f"   Audio duration: {audio_duration_ms / 1000:.2f} seconds")
            print(f"   Estimated speech rate: {chars_per_second:.1f} chars/second")

            # Create beep sound
            beep = Sine(beep_frequency).to_audio_segment(duration=beep_duration)

            # Sort entities by character offset (descending) to avoid offset shifting
            sorted_entities = sorted(
                entities,
                key=lambda x: x["BeginOffset"],
                reverse=True
            )

            # Replace PII segments with beeps
            for entity in sorted_entities:
                begin_char = entity.get("BeginOffset")
                end_char = entity.get("EndOffset")
                entity_type = entity.get("Type", "UNKNOWN")

                if begin_char is not None and end_char is not None:
                    # Estimate start and end times based on character positions
                    start_ms = int((begin_char / chars_per_second) * 1000)
                    end_ms = int((end_char / chars_per_second) * 1000)

                    # Ensure times are within audio bounds
                    start_ms = max(0, start_ms)
                    end_ms = min(audio_duration_ms, end_ms)
                    duration_ms = end_ms - start_ms

                    if duration_ms > 0:
                        # Create silence with beep duration
                        silence = AudioSegment.silent(duration=duration_ms)
                        # Overlay beeps throughout the segment
                        num_beeps = max(1, duration_ms // (beep_duration + 50))
                        for i in range(num_beeps):
                            beep_pos = i * (duration_ms // num_beeps)
                            silence = silence.overlay(beep, position=beep_pos)

                        # Replace segment in audio
                        audio = audio[:start_ms] + silence + audio[end_ms:]

                        print(f"   ✓ Redacted {entity_type} at {start_ms}ms-{end_ms}ms")

            # Convert to bytes
            audio_bytes = audio.export(format="wav").read()
            print(f"✅ Audio redaction completed")
            return audio_bytes

        except Exception as e:
            print(f"❌ Error redacting audio: {e}")
            return None


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

    # Validate inputs
    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        print("❌ Error: AWS credentials not set in environment")
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

    # Step 7: Redact PII from audio with beeps
    redacted_audio_bytes = pii_redactor.redact_audio(audio_file, original_text, pii_entities)

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
