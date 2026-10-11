import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  testMatch: "traffic-vision.spec.ts",
  workers: 1,
  retries: 0,
  timeout: 180000,
  expect: { timeout: 30000 },
  use: { baseURL: "http://localhost:3000", viewport: { width: 1440, height: 1000 },
    browserName: "chromium", channel: "msedge", headless: true, acceptDownloads: true,
    screenshot: "only-on-failure", trace: "retain-on-failure" },
  reporter: [["list"]],
  // Use installed Edge on Windows; no browser/model downloads during tests.
  // Start the existing backend and frontend before running this integration suite.
});
