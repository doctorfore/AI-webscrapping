import asyncio
from langchain_openai import ChatOpenAI
from langchain.chains import create_extraction_chain
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from util import load_config   # 你需要自建 util.py 并写 load_config

# 读取配置（包括 OpenAI Key）
config = load_config('config.yml')
OPENAI_KEY = config['open_ai']['key']

# -------------------------------
# 用 Playwright 打开网页并提取纯文本
# -------------------------------
async def run_playwright(site):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)  # 打开浏览器，headless=False 可见
        # browser = await p.firefox.launch(headless=False) # 也可切换 Firefox
        page = await browser.new_page()
        try:
            await page.goto(site, wait_until='networkidle')  # 等待页面加载完成
        except TimeoutError:
            print("Timeout reached during page load, proceeding with available content")

        # 获取网页源码
        page_source = await page.content()

        # 用 BeautifulSoup 解析 HTML，去掉 script/style
        soup = BeautifulSoup(page_source, "html.parser")
        for script in soup(["script", "style"]):
            script.extract()

        text = soup.get_text()

        # 整理成干净的文本行
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split(" "))
        data = '\n'.join(chunk for chunk in chunks if chunk)

        await browser.close()
        return data

# -------------------------------
# 设置模型
# -------------------------------
GPT_4 = 'gpt-4'
llm = ChatOpenAI(temperature=0, model=GPT_4, openai_api_key=OPENAI_KEY)

# -------------------------------
# 主程序：定义 schema，提取结构化信息
# -------------------------------
async def main():
    stock_website_info = {
        "url": "https://stockanalysis.com/stocks/googl/",
        "schema": {
            "properties": {
                "market_cap": {"type": "string"},
                "revenue": {"type": "string"},
                "net_income": {"type": "string"},
                "shares_out": {"type": "string"},
                "analyst_forecast_report": {"type": "string"},
            },
        }
    }

    bilibili_website_info = {
        "url": "https://space.bilibili.com/282739748/video",
        "schema": {
            "properties": {
                "title": {"type": "string"},
                "article_content": {"type": "string"},
            },
        }
    }

    # 可切换不同目标
    parsing_target = stock_website_info
    # parsing_target = bilibili_website_info
    # parsing_target = twitter_website_info
    # parsing_target = techcruch_website_info

    # 爬取网页内容
    output = await run_playwright(parsing_target['url'])

    # 用 LLM 把非结构化文本 -> JSON 结构
    json_result = create_extraction_chain(parsing_target['schema'], llm).invoke(output)

    print(json_result['text'])

# -------------------------------
# 程序入口
# -------------------------------
if __name__ == "__main__":
    asyncio.run(main())
