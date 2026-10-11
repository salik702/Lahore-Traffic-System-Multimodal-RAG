import { test, expect, type Page } from "@playwright/test";
import path from "node:path";
import fs from "node:fs/promises";
import type { VisionResult } from "../lib/vision";

const root = path.resolve(__dirname, "../..");
async function openDetection(page: Page) {
  await page.goto("/");
  const menu = page.getByRole("button", { name: "Open navigation", exact: true });
  if (await menu.isVisible()) await menu.click();
  await page.getByRole("button", { name: "AI Violation Detection", exact: true }).click();
  await expect(page.getByRole("heading", { name: "AI Traffic Violation Detection", exact: true })).toBeVisible();
}
async function analyze(page: Page, filename: string) {
  await page.getByLabel("Upload road photograph", { exact: true }).setInputFiles(path.join(root, "backend/storage/verification", filename));
  const response = page.waitForResponse(r => r.url().endsWith("/vision/analyze") && r.request().method() === "POST");
  await page.getByRole("button", { name: "Analyze Image", exact: true }).click();
  const actual = await response;
  expect(actual.status()).toBe(200);
  const result = await actual.json() as VisionResult;
  await expect(page.getByRole("heading", { name: "Inspect the evidence" })).toBeVisible();
  return result;
}

test("real bus inference, zero helmet findings, and server PDF download", async ({ page }) => {
  const failures: string[] = [];
  page.on("pageerror", e => failures.push(e.message));
  await openDetection(page);
  const result = await analyze(page, "bus.jpg");
  expect(result.vehicles.some(v => v.class === "bus")).toBeTruthy();
  expect(result.riders).toHaveLength(0);
  expect(result.potential_violations).toHaveLength(0);
  expect(result.summary.verified_potential_fine).toBe(0);
  const [download] = await Promise.all([
    page.waitForEvent("download"),
    page.getByRole("button", { name: "Download Violation Report (PDF)", exact: true }).click(),
  ]);
  const target = path.join(root,"backend/storage/verification/browser-bus-report.pdf");
  await download.saveAs(target);
  expect((await fs.readFile(target)).subarray(0,4).toString()).toBe("%PDF");
  await page.screenshot({ path: path.join(root,"backend/storage/verification/dashboard-bus.png"), fullPage: true });
  expect(failures).toEqual([]);
});

test("real mixed traffic, independent selection, and source-based result data", async ({ page }) => {
  await openDetection(page);
  const result = await analyze(page,"helmeted-couple.jpg");
  expect(result.vehicles.some(v => v.class === "car")).toBeTruthy();
  expect(result.vehicles.some(v => v.class === "motorcycle")).toBeTruthy();
  expect(result.potential_violations).toHaveLength(0);
  const motorcycle = page.getByLabel("Detected vehicles").getByRole("button").filter({ hasText: "motorcycle" });
  await motorcycle.click();
  await expect(motorcycle).toHaveAttribute("aria-pressed","true");
  await expect(page.getByText("Vehicle information",{ exact:true })).toBeVisible();
  await page.getByRole("button",{ name:"Scene",exact:true }).click();
  await page.getByRole("button",{ name:"Original",exact:true }).click();
  await expect(page.getByRole("img",{ name:"Original image at analysis resolution",exact:true })).toBeVisible();
  await page.screenshot({ path: path.join(root,"backend/storage/verification/dashboard-mixed.png"), fullPage:true });
});

test("real bare-head finding, uncertainty, verified potential fine and PDF", async ({ page }) => {
  await openDetection(page);
  const result = await analyze(page,"motorcycle-riders.jpg");
  expect(result.potential_violations.some(v => v.type === "helmet_violation")).toBeTruthy();
  const finding = result.potential_violations.find(v => v.vehicle_id === result.primary_vehicle_id)!;
  expect(finding).toBeTruthy();
  const selected = page.getByLabel("Detected vehicles").getByRole("button").nth(result.vehicle_results.findIndex(v => v.vehicle_id === finding.vehicle_id));
  await selected.click();
  await expect(page.getByRole("heading",{name:"Potential helmet violation",exact:true})).toBeVisible();
  const [download] = await Promise.all([page.waitForEvent("download"),page.getByRole("button",{name:"Download Violation Report (PDF)",exact:true}).click()]);
  await download.saveAs(path.join(root,"output/pdf/traffic-analysis-verification.pdf"));
  await page.screenshot({ path: path.join(root,"backend/storage/verification/dashboard-finding.png"),fullPage:true });
});

test("saved rider upload selects foreground bike and includes actual helmet fine in PDF", async ({ page }) => {
  await openDetection(page);
  await page.getByLabel("Upload road photograph",{exact:true}).setInputFiles(path.join(root,"backend/storage/uploads/bdd82670bec54f0eaab14c8ea13e7736.jpg"));
  const response=page.waitForResponse(r=>r.url().endsWith('/vision/analyze') && r.request().method()==='POST');
  await page.getByRole('button',{name:'Analyze Image',exact:true}).click();
  const actual=await response;expect(actual.status()).toBe(200);
  const result=await actual.json() as VisionResult;
  const foreground=result.vehicle_results.find(v=>v.vehicle_id===result.primary_vehicle_id)!;
  expect(foreground.vehicle_type).toBe('motorcycle');
  expect(foreground.summary.verified_potential_fine).toBe(2000);
  expect(foreground.plate?.text).toBe('SGK3695');
  expect(foreground.plate?.status).toBe('recognized');
  await expect(page.getByText('SGK3695',{exact:true})).toBeVisible();
  await expect(page.getByRole('heading',{name:'Potential helmet violation',exact:true})).toBeVisible();
  await expect(page.getByRole('img',{name:'Annotated evidence for the selected report vehicle',exact:true})).toBeVisible();
  const downloadResponse=page.waitForResponse(r=>r.url().includes('/vision/reports/') && r.url().includes(`vehicle_id=${encodeURIComponent(foreground.vehicle_id)}`));
  const [download]=await Promise.all([page.waitForEvent('download'),page.getByRole('button',{name:'Download Violation Report (PDF)',exact:true}).click()]);
  expect((await downloadResponse).status()).toBe(200);
  await download.saveAs(path.join(root,'output/pdf/foreground-rider-report.pdf'));
  await page.screenshot({path:path.join(root,'backend/storage/verification/dashboard-foreground-rider.png'),fullPage:true});
  const car=page.getByLabel('Detected vehicles').getByRole('button').filter({hasText:'car'});
  await car.click();
  await expect(page.getByRole('heading',{name:'Potential helmet violation',exact:true})).not.toBeVisible();
  await expect(page.getByRole('status').filter({hasText:'Report subject: car'})).toBeVisible();
});

test("unsupported formats are rejected before inference", async ({ page }) => {
  await openDetection(page);
  await page.getByLabel("Upload road photograph",{exact:true}).setInputFiles({ name:"invalid.gif",mimeType:"image/gif",buffer:Buffer.from("not an image") });
  await expect(page.getByRole("alert").filter({hasText:"JPEG, PNG, or WebP"})).toBeVisible();
  await expect(page.getByRole("button",{name:"Analyze Image",exact:true})).toBeDisabled();
});

test("mobile navigation and dashboard fit the screen", async ({ page }) => {
  await page.setViewportSize({ width:390,height:844 });
  await openDetection(page);
  await analyze(page,"bus.jpg");
  expect(await page.locator("body").evaluate(element => element.scrollWidth <= window.innerWidth)).toBeTruthy();
  await page.screenshot({ path:path.join(root,"backend/storage/verification/dashboard-mobile.png"),fullPage:true });
});
