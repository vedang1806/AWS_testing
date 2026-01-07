# AWS Transcribe + Gemini Sentiment Analysis

This script transcribes audio files using AWS Transcribe and performs sentiment analysis using Google Gemini AI.

## Features

- 🎙️ **Audio Transcription**: Uses AWS Transcribe with speaker diarization
- 🔒 **PII Detection & Redaction**: Detects and redacts sensitive information using AWS Comprehend
- 🎭 **Sentiment Analysis**: Analyzes conversation tone and sentiment using Google Gemini AI
- 🔊 **Audio Redaction**: Multiple redaction modes (silence, beep, noise, etc.)
- ☁️ **S3 Integration**: Automatic upload to AWS S3

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env` with your actual credentials:
- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
- GEMINI_API_KEY (Get from https://ai.google.dev/)

### 3. Get API Keys

**AWS Credentials:**
1. Log into AWS Console
2. Go to IAM → Users → Your User → Security Credentials
3. Create access key

**Gemini API Key:**
1. Visit https://ai.google.dev/
2. Click "Get API Key"
3. Create a new API key

## Usage

### Basic Usage

```bash
python main.py path/to/audio.wav
```

### With Custom Call ID

```bash
python main.py path/to/audio.wav --call-id my-call-123
```

### Save Transcript and Audio

```bash
python main.py path/to/audio.wav --save-transcript --save-audio
```

### Choose Redaction Mode

```bash
python main.py path/to/audio.wav --redaction-mode beep
```

Available redaction modes:
- `silence` (default)
- `beep`
- `white_noise`
- `pink_noise`
- `tone`
- `mute`

## Output Format

The script generates a sentiment analysis JSON file with the following structure:

```json
[
  {
    "order": 1,
    "speaker": "Speaker 0",
    "text": "Hello. Thank you for calling...",
    "start_time": "00:02",
    "end_time": "00:12",
    "sentiment": "neutral",
    "confidence": 0.9,
    "tone_note": "Professional and welcoming"
  },
  {
    "order": 2,
    "speaker": "Speaker 1",
    "text": "Yeah sure. My name is...",
    "start_time": "00:14",
    "end_time": "00:33",
    "sentiment": "neutral",
    "confidence": 0.9,
    "tone_note": "Cooperative and providing information"
  }
]
```

## Output Files

- `sentiment_analysis_<call_id>.json` - Sentiment analysis results
- `transcript_<call_id>.json` - Full transcript with metadata (if --save-transcript)
- `redacted_transcript_<call_id>.txt` - Redacted text transcript (if --save-audio)

## Example

```bash
python main.py sample_call.wav --call-id healthcare-001 --save-transcript

# Output:
# ✅ Sentiment analysis saved to: sentiment_analysis_healthcare-001.json
# ✅ Transcript saved to: transcript_healthcare-001.json
```

## Architecture

1. **AWS Transcribe**: Transcribes audio with speaker labels
2. **AWS Comprehend**: Detects PII entities in text
3. **Google Gemini**: Analyzes sentiment and tone for each speaker segment
4. **Audio Redaction**: Removes PII from audio using word-level timing
5. **S3 Upload**: Stores original and redacted files

## Error Handling

The script includes comprehensive error handling:
- Validates AWS and Gemini credentials
- Checks audio file existence
- Handles API failures gracefully
- Provides detailed error messages

## Notes

- Audio files must be in WAV format
- Maximum audio duration depends on AWS Transcribe limits
- Gemini API has rate limits (adjust delays if needed)
- PII redaction uses character-to-time mapping for accuracy

## Troubleshooting

**"GEMINI_API_KEY not set"**
- Make sure you've created a `.env` file with your API key

**"AWS credentials not set"**
- Verify your `.env` file has valid AWS credentials

**"Audio file not found"**
- Check the file path is correct and file exists

**Gemini API errors**
- Check your API key is valid
- Ensure you haven't exceeded rate limits
- Verify internet connectivity

## License

MIT License
