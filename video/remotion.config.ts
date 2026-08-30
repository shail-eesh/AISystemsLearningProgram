import { Config } from "@remotion/cli/config";

// Per-topic audio lives under video/topics/<id>/audio/, so `topics` is the
// static root: staticFile("p0.1/audio/s1-000.wav") resolves there.
Config.setPublicDir("./topics");
Config.setVideoImageFormat("jpeg");
Config.setCodec("h264");
Config.setConcurrency(2);
Config.setChromiumOpenGlRenderer("swiftshader");
Config.setOverwriteOutput(true);

// This sandbox cannot reach remotion.media to download Chrome Headless Shell,
// but a Playwright Chromium is preinstalled. FORGE_CHROMIUM overrides it.
const chromium =
  process.env.FORGE_CHROMIUM ??
  "/opt/pw-browsers/chromium_headless_shell-1194/chrome-linux/headless_shell";
Config.setBrowserExecutable(chromium);
