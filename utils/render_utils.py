import os
import sys
import asyncio
import httpx
from utils.load_env import load_env_file
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

if sys.platform.startswith("win"):
    from asyncio import WindowsProactorEventLoopPolicy
    asyncio.set_event_loop_policy(WindowsProactorEventLoopPolicy())

async def deploy_to_render() -> str:
    # Load .env variables
    env_vars = load_env_file()
    email = env_vars.get("RENDER_EMAIL")
    password = env_vars.get("RENDER_PASSWORD")

    if not email or not password:
        raise ValueError("Missing Render credentials in environment variables.")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()

        # Step 1: Navigate to Render
        print("[🌐] Navigating to Render deploy page...")
        await page.goto("https://dashboard.render.com/web/new", wait_until="networkidle")

        # Step 2: Login
        if "login" in page.url:
            print("[🔐] Logging in to Render...")
            await page.fill('input[name="email"]', email)
            await page.fill('input[name="password"]', password)
            await page.click('button[type="submit"]')
            await page.wait_for_load_state("networkidle")
            print("[✅] Logged in!")

        # Step 3: Wait for GitHub repo section to load
        print("[📦] Waiting for GitHub repo list to load...")
        await page.wait_for_selector('h2:has-text("Git Provider")', timeout=60000)

        # Step 4: Click latest GitHub repo link (not Public Git)
        print("[📁] Looking for the latest GitHub repo link...")
        repo_links = await page.query_selector_all('a[href*="/web/new/from-repo"]')
        if repo_links:
            print(f"[📁] Found {len(repo_links)} repo links. Clicking the first one...")
            await repo_links[0].click()
            await page.wait_for_load_state("networkidle")
            print("[✅] Repo link clicked!")
        else:
            print("[⚠️] No GitHub repo links found.")
            return

        # Step 5: Fill deployment configuration
        await page.wait_for_selector('input[placeholder="Start command"]', timeout=30000)
        print("[💻] Setting start command...")
        await page.fill('input[placeholder="Start command"]', "streamlit run frontend.py")

        print("[⚙️] Selecting 'Free' instance type...")
        await page.wait_for_selector('label:has-text("Free")')
        await page.click('label:has-text("Free")')

        # Step 6: Set environment variables from local .env file
        print("[🧬] Adding environment variables from local .env...")
        for key, val in env_vars.items():
            await page.click('text="Add Environment Variable"')
            await page.fill('input[placeholder="Key"]', key)
            await page.fill('input[placeholder="Value"]', val)
            await page.press('input[placeholder="Value"]', "Enter")
            await page.wait_for_timeout(300)

        # Step 7: Deploy
        print("[🚀] Clicking 'Deploy Web Service'...")
        await page.click('text="Deploy Web Service"')

        # Step 8: Wait for deployment to go Live
        print("[⌛] Waiting for app to go live...")
        for i in range(5):
            print(f"[⌛] Checking deployment status... Attempt {i + 1}/5")
            await asyncio.sleep(120)
            await page.reload()
            live = await page.query_selector('text=Live')
            if live:
                print("[✅] App is now live!")
                break
        else:
            raise TimeoutError("App did not go live in 10 minutes.")

        # Step 9: Get live URL
        print("[🔗] Getting live URL...")
        link = await page.query_selector('a[href^="https://"]')
        live_url = await link.get_attribute("href") if link else None

        if not live_url:
            print("[⚠️] Unable to fetch live URL.")
            return "❌ Deployment completed, but live URL not found."

        # Step 10: Verify the live app is accessible
        print("[📡] Verifying the live app is reachable (HTTP 200)...")
        for _ in range(3):
            try:
                async with httpx.AsyncClient() as client:
                    res = await client.get(live_url, timeout=30)
                    if res.status_code == 200:
                        print(f"[🎉] Successfully Deployed: {live_url}")
                        break
            except:
                await asyncio.sleep(5)
        else:
            raise Exception("Deployment succeeded but app did not return 200 OK")

        await browser.close()

        # Return iframe embed
        embed_code = f"""<iframe src="{live_url}" width="100%" height="800px" frameborder="0"></iframe>"""
        return f"✅ Live at: {live_url}\n\n🔗 Embed:\n{embed_code}"
