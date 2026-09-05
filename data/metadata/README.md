# ASVspoof 2021 Speech DeepFake Dataset

## Dataset source
Official ASVspoof Zenodo record.

## Downloaded files
ASVspoof2021_DF_eval_part00.tar.gz and DF-keys-full.tar.gz

## Directory structure
data/raw/ASVspoof2021_DF/flac/

## Audio format
FLAC, 16kHz, mono.

## Metadata structure
Space-separated txt file with: SPEAKER_ID AUDIO_FILE_NAME - SYSTEM_ID LABEL

## Label structure
'bonafide' for real speech, 'spoof' for synthetic speech.

## Dataset partitions
Contains 'eval' partition.

## Speaker information
Speaker IDs are present (e.g. LA_0039).

## Attack information
Attack system IDs are present (e.g. A01).

## Validation results
No missing audio files or corrupted data detected.

## Important observations
Data leakage must be prevented by splitting based on Speaker ID.

## Limitations
Only eval set is available in this mock download.

