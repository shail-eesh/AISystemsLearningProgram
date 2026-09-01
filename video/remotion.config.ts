import { Config } from "@remotion/cli/config";

// Per-topic audio lives under video/topics/<id>/audio/, so `topics` is the
// static root: staticFile("p0.1/audio/s1-000.wav") resolves there.
Config.setPublicDir("./topics");
Config.setVideoImageFormat("jpeg");
Config.setCodec("h264");
Config.setConcurrency(2);
Config.setChromiumOpenGlRenderer("swiftshader");
Config.setOverwriteOutput(true);

// CRF 28 rather than the h264 default of 18. These episodes are flat colour,
// large type and slow motion — the kind of content h264 compresses extremely
// well — and CRF 18 spends ~490 kbps on it, which puts a ten-minute episode at
// ~38 MB and over the 30 MB chat-delivery limit. CRF 28 lands around 8 MB with
// no visible difference on this material. Found on Day 3, when t4-e1 came out
// at 37.6 MB and could not be delivered.
Config.setCrf(28);

// And the audio: Remotion defaults to a high AAC bitrate, which on a
// narration-only track was ~26 of t4-e1's 31 MB — five times the video. 64 kbps
// mono is transparent for a single speaking voice at 24 kHz.
Config.setAudioBitrate("64k");

// This sandbox cannot reach remotion.media to download Chrome Headless Shell,
// but a Playwright Chromium is preinstalled. FORGE_CHROMIUM overrides it.
const chromium =
  process.env.FORGE_CHROMIUM ??
  "/opt/pw-browsers/chromium_headless_shell-1194/chrome-linux/headless_shell";
Config.setBrowserExecutable(chromium);
