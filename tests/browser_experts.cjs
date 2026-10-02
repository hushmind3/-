"use strict";
// Read-only live browser verification. Never invokes account/mode POST routes.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require("playwright");
(async () => {
  const folder = process.env.EXPERT_PROOF_DIR;
  fs.mkdirSync(folder, {recursive:true});
  const browser = await chromium.launch({channel:"msedge", headless:true});
  const page = await browser.newPage({viewport:{width:1440,height:1050}});
  const errors = [];
  const posts = [];
  const failedRequests = [];
  page.on("response", r => {if (r.status() >= 400) failedRequests.push({url:r.url(),status:r.status()});});
  page.on("pageerror", e => errors.push(e.message));
  page.on("console", e => {if (e.type() === "error") errors.push(e.text());});
  page.on("request", r => {if (r.method() === "POST") posts.push(r.url());});
  await page.goto("http://127.0.0.1:8766/#experts");
  await page.waitForSelector(".expert-card");
  assert.equal(await page.locator(".expert-card").count(), 14);
  assert.equal(await page.locator(".expert-card").filter({hasText:"가중치 고정 · frozen"}).count(),14);
  assert.equal(await page.locator(".expert-card").filter({hasText:"미적재 · unloaded"}).count(),14);
  const registryResponse = await page.request.get("http://127.0.0.1:8766/api/experts");
  const registry = await registryResponse.json();
  for (const expert of registry.experts) {
    const response = await page.request.get("http://127.0.0.1:8766/api/experts/output?id=" + encodeURIComponent(expert.id));
    assert.equal(response.ok(),true);
    const raw = await response.json();
    assert.deepEqual(raw.packet.output_shape, expert.output_shape);
    assert.equal(raw.packet.frozen,true);
    assert.equal(raw.packet.parameters,expert.parameters);
  }
  await page.route("**/api/experts", async route => {
    const fixture = structuredClone(registry);
    fixture.experts[0].name = "Registry supplied name";
    await route.fulfill({json:fixture});
  });
  await page.evaluate(() => refreshExperts());
  assert.equal(await page.locator('.expert-card [data-field="name"]').first().textContent(),"Registry supplied name");
  await page.unroute("**/api/experts");
  await page.evaluate(() => refreshExperts());
  await page.evaluate(() => { window.firstExpertCard = document.querySelector(".expert-card"); });
  await page.evaluate(() => refreshExperts());
  assert.equal(await page.evaluate(() => window.firstExpertCard === document.querySelector(".expert-card")), true);
  const timesfm = page.locator('[data-expert-id="timesfm"]');
  await timesfm.locator("summary").click();
  await timesfm.locator('[data-field="rawLoad"]').click();
  await page.waitForFunction(() => document.querySelector('[data-expert-id="timesfm"] [data-field="rawOrigin"]').textContent.includes("shape"));
  const raw = JSON.parse(await timesfm.locator('[data-field="raw"]').textContent());
  assert.equal(raw.length,2);
  assert.equal(raw[0][0].length,10);
  await page.screenshot({path:path.join(folder,"experts-raw-output.png"),fullPage:false});
  await page.evaluate(() => scrollTo(0,0));
  await page.screenshot({path:path.join(folder,"experts-desktop.png"),fullPage:false});
  for (const hash of ["control","markets","learning","promotionTrial","connection","system","experts"]) {
    await page.evaluate(hash => {location.hash = hash;}, hash);
    await page.waitForFunction(() => [...document.querySelectorAll("[data-view]")].filter(n=>!n.hidden).length === 1);
    await page.evaluate(() => readStatus());
  }
  await page.setViewportSize({width:390,height:844});
  await page.screenshot({path:path.join(folder,"experts-mobile.png"),fullPage:false});
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1),true);
  assert.deepEqual(errors,[],JSON.stringify(failedRequests));
  assert.deepEqual(posts,[]);
  const result = {cards:14,frozen:14,unloaded:14,stable_card_nodes:true,raw_timesfm_shape:[2,1,10],
    raw_outputs_verified:14,registry_names_dynamic:true,screens_checked:7,mobile_overflow:false,console_errors:errors,post_requests:posts};
  fs.writeFileSync(path.join(folder,"browser-result.json"),JSON.stringify(result,null,2));
  console.log(JSON.stringify(result));
  await browser.close();
})().catch(error => {console.error(error); process.exit(1);});
