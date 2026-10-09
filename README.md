# Gemma Chat for Android

This Kotlin/Jetpack Compose app runs the notebook's `google/gemma-3-270m-it`
chat model locally with Google's LiteRT-LM Android runtime. No model API key or
network access is used during chat.

## Bundle the model in the APK

1. Sign in to Hugging Face and accept the Gemma license on the
   [`litert-community/gemma-3-270m-it` model page](https://huggingface.co/litert-community/gemma-3-270m-it).
2. Download the 304 MB
   [`gemma3-270m-it-q8.litertlm` model file](https://huggingface.co/litert-community/gemma-3-270m-it/resolve/main/gemma3-270m-it-q8.litertlm?download=true).
3. Put that file at
   `app/src/main/assets/gemma3-270m-it-q8.litertlm` (create the `assets`
   directory if needed).
4. Build with `./gradlew :app:assembleDebug`. The model is packaged uncompressed
   in the APK; first launch copies it into private app storage before loading it.

The resulting APK will be at
`app/build/outputs/apk/debug/app-debug.apk` and will be roughly 350 MB or larger.
Keep enough free space for the APK and its extracted model copy. The generic Q8
variant uses the CPU backend for broad device compatibility. Do not use one of
the larger phone-specific variants unless you know the target chipset.

## Add local JSON knowledge (RAG)

Put `.json` data files under `app/src/main/assets/rag/` (subfolders are also
scanned) before building. The app reads the JSON files and builds its in-memory
BM25 index as the model initializes. Gemma then decides whether a question needs
local data and calls the search tool when appropriate. Top-level JSON arrays
are indexed one record at a time; a JSON object is treated as one record. See
`app/src/main/assets/rag/README.md` for an example. No embedding model, server,
or network connection is needed.

Everything under Android assets is embedded in and extractable from the APK.
Do not bundle private or sensitive data.

## Run

Use JDK 21 for Gradle (set Android Studio's Gradle JDK to JDK 21), open this
directory in Android Studio, let Gradle sync, then run the `app` configuration
on an Android device (API 26 or later). If the model is not bundled, the app
will show the expected asset path. Model initialization, indexing, and
generation happen off the UI thread.

## Notebook credential

The original notebook contained a Hugging Face access token. The token value was
removed, but removing it from this file does not invalidate it; revoke it in
Hugging Face and create a replacement if needed. Do not embed Hugging Face
credentials in an Android app.
