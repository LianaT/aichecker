# agent_web_scraper.py
"""
Agent for scraping internal links from a website using Playwright and LangChain WebBaseLoader.
Stores results in a JSON file.
"""
import sys
import asyncio

if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import json
import re
from urllib.parse import urlparse, urljoin
from typing import List, Optional

from playwright.async_api import async_playwright
from langchain_community.document_loaders import WebBaseLoader

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse, HTMLResponse

# Requirements:
# pip install langchain-openai
# You must set up OpenAI API key in .env file as OPENAI_API_KEY=your_key_here
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for dev; restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def root():
    """Root endpoint that provides API documentation"""
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <title>AI Checker API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
                h1 { color: #333; }
                .endpoint { background: #f4f4f4; padding: 15px; margin-bottom: 20px; border-radius: 5px; }
                pre { background: #eee; padding: 10px; border-radius: 5px; overflow-x: auto; }
                code { font-family: monospace; }
            </style>
        </head>
        <body>
            <h1>AI Checker API</h1>
            <p>Welcome to the AI Checker API. This service uses Google's Gemini LLM to analyze websites for EU AI Act compliance.</p>
            
            <div class="endpoint">
                <h2>POST /scrape</h2>
                <p>Scrapes a website and returns its content.</p>
                <h3>Request Body:</h3>
                <pre><code>{
    "url": "https://example.com",
    "save_to_file": false,
    "output_json": "scraped_data.json"
}</code></pre>
            </div>
            
            <div class="endpoint">
                <h2>POST /analyze_compliance</h2>
                <p>Analyzes a website for EU AI Act compliance using Gemini LLM.</p>
                <h3>Request Body:</h3>
                <pre><code>{
    "url": "https://example.com"
}</code></pre>
            </div>
            
            <p>To test these endpoints, use a tool like curl, Postman, or create a frontend application.</p>
        </body>
    </html>
    """


def is_internal_link(base_url, link):
    """Check if the link is internal (same domain as base_url)."""
    parsed_base = urlparse(base_url)
    parsed_link = urlparse(urljoin(base_url, link))
    return parsed_base.netloc == parsed_link.netloc

def filter_links(base_url, links):
    """Filter only internal and non-social links."""
    social_domains = [
        'facebook.com', 'twitter.com', 'linkedin.com', 'instagram.com', 'youtube.com', 't.me', 'wa.me', 'pinterest.com', 'reddit.com', 'discord.com'
    ]
    filtered = []
    for link in links:
        parsed = urlparse(urljoin(base_url, link))
        if any(domain in parsed.netloc for domain in social_domains):
            continue
        if is_internal_link(base_url, link):
            filtered.append(urljoin(base_url, link))
    return list(set(filtered))  # Remove duplicates

async def get_internal_links(url):
    """Use Playwright to get all internal links from the page."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url)
        # Get all hrefs from anchor tags
        hrefs = await page.eval_on_selector_all('a', 'elements => elements.map(e => e.href)')
        await browser.close()
    return filter_links(url, hrefs)

def scrape_with_langchain(urls):
    """Use LangChain WebBaseLoader to scrape content from each URL."""
    results = []
    for url in urls:
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            for doc in docs:
                results.append({
                    'url': url,
                    'content': doc.page_content
                })
        except Exception as e:
            print(f"Failed to scrape {url}: {e}")
    return results

async def agent_scrape_site(base_url: str) -> List[dict]:
    """Main agent function: gets internal links, scrapes, returns data."""
    internal_links = await get_internal_links(base_url)
    data = scrape_with_langchain(internal_links)
    return data

class ScrapeRequest(BaseModel):
    url: str
    save_to_file: Optional[bool] = False
    output_json: Optional[str] = 'scraped_data.json'

@app.post("/scrape")
async def scrape_endpoint(request: ScrapeRequest):
    try:
        data = await agent_scrape_site(request.url)
        if request.save_to_file:
            with open(request.output_json, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        return JSONResponse(content={"results": data})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ComplianceRequest(BaseModel):
    url: str

EU_PROMPT = '''You are an expert in EU regulatory compliance specifically trained on the European Union's Artificial Intelligence Act. Given this input JSON (which describes a website content), your task is to carefully assess which EU AI Act category the AI product/service falls into.

## EU AI Act Methodology

1. First, carefully analyze what the AI system actually does in practice, not just how it's marketed
2. Identify the specific use case(s) and domain(s) of application
3. Consider the potential impact on fundamental rights, health and safety
4. Check for exact matches against the criteria in Articles 5, 6, 69, 52 of the EU AI Act
5. When borderline, look at the level of autonomy, the consequences of decisions, and whether humans are meaningfully involved
6. Pay special attention to applications in regulated sectors (education, healthcare, employment, etc.)

## EU AI Act Categories:

**1. Prohibited AI Practices (Banned)**
- Social scoring by governments
- Real-time biometric identification in public spaces
- AI that exploits vulnerabilities (age, disability, economic situation)
- Subliminal techniques to manipulate behavior
- Emotion recognition in workplace/education (with exceptions)

**2. High-Risk AI Systems**
- Biometrics (face recognition, emotion detection)
- Employment (hiring, firing, performance evaluation)
- Education/Learning: Systems that:
  * Evaluate student performance or assess learning outcomes
  * Make or influence admission decisions 
  * Determine educational opportunities or pathways
  * Automatically grade or evaluate assessments
  * Generate or evaluate formal academic credentials
- Finance (credit scoring, insurance risk assessment)
- Law enforcement (criminal risk assessment)
- Healthcare (diagnostic systems)
- Critical infrastructure (traffic, utilities)
- Government services (benefits, immigration)

**3. Limited Risk AI Systems**
- Chatbots and conversational AI
- AI that interacts directly with humans
- Educational tools that only provide study assistance without formal assessment
- Educational content creation tools that support teaching but don't evaluate students
- Emotion recognition systems (outside prohibited contexts)
- Biometric categorization systems
- AI that generates/manipulates content (deepfakes)

**4. Minimal Risk AI Systems**
- AI-enabled video games
- Spam filters
- AI for inventory management
- Most other AI applications not in above categories

**5. General Purpose AI (GPAI) Models**
- Foundation models (like GPT, Claude)
- Large language models
- Multimodal AI models
- AI models that can be adapted for various tasks

## Assessment Framework (Based on Official EU AI Act):

**Legal Framework Assessment**
- Prohibited AI (Article 5): Check if the AI system clearly violates fundamental rights or utilizes specifically prohibited techniques
- High-Risk AI (Article 6 & Annex III): Evaluate if the AI system:
  * Is used in a sector listed in Annex III (education, employment, law enforcement, etc.)
  * Makes decisions with significant impact on individuals
  * Has limited human oversight
  * Affects vulnerable groups (e.g., children, disabled persons)
- Limited Risk AI (Article 52): Systems with transparency requirements but lower impact
- Minimal Risk AI: Systems not covered by specific provisions
- GPAI (Article 28b): Large-scale foundational models with broad capabilities

## Your Task:
1. Carefully examine the website content about their AI product/service
2. Apply the legal assessment framework above
3. Score likelihood (0-10) for each category based on the exact requirements in the EU AI Act
4. For highest scoring category, provide specific reasoning referencing relevant EU AI Act articles
5. Extract website URL and brief description
6. Output as JSON only

## Output Format (JSON only):
```json
{
 "website_url": "[URL]",
 "website_description": "[Brief 1-2 sentence description of what the company/product does]",
 "category_scores": {
 "prohibited_ai_practices": [0-10],
 "high_risk_ai_systems": [0-10],
 "limited_risk_ai_systems": [0-10],
 "minimal_risk_ai_systems": [0-10],
 "general_purpose_ai_models": [0-10]
 },
 "explanations": {
 "highest_score_explanation": "[1-3 sentences explaining why the highest scoring category was selected and what factors led to that score]"
 }
```

Respond with JSON only, no additional text.'''

def truncate_to_tokens(text: str, max_tokens: int = 100000) -> str:
    """Truncate text to approximately max_tokens (rough estimate: 1 token ≈ 4 chars)"""
    max_chars = max_tokens * 4
    return text[:max_chars] if len(text) > max_chars else text

@app.post("/analyze_compliance")
async def analyze_compliance(request: ComplianceRequest):
    try:
        # 1. Scrape site
        scraped_data = await agent_scrape_site(request.url)
        scraped_json = json.dumps(scraped_data, ensure_ascii=False)
        # Truncate to stay under token limit
        scraped_json = truncate_to_tokens(scraped_json)
        # 2. Prepare LLM prompt
        prompt = [
            {"role": "system", "content": EU_PROMPT},
            {"role": "user", "content": f"Input JSON: {scraped_json}"}
        ]
        # 3. Run LLM (OpenAI GPT-4o-mini)
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.4
        )
        llm_response = llm.invoke(prompt)
        # 4. Parse and return JSON only
        try:
            compliance_json = json.loads(llm_response.content.strip())
        except Exception:
            # Fallback: try to extract JSON from response
            import re
            match = re.search(r'\{.*\}', llm_response.content, re.DOTALL)
            if match:
                compliance_json = json.loads(match.group(0))
            else:
                raise HTTPException(status_code=500, detail="LLM did not return valid JSON.")
        return JSONResponse(content=compliance_json)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import sys
    import uvicorn
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        uvicorn.run("agent_web_scraper:app", host="0.0.0.0", port=8000, reload=True)
    else:
        if len(sys.argv) < 2:
            print("Usage: python agent_web_scraper.py <url> [output_json]")
            exit(1)
        url = sys.argv[1]
        output_json = sys.argv[2] if len(sys.argv) > 2 else 'scraped_data.json'
        data = asyncio.run(agent_scrape_site(url))
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Scraped data saved to {output_json}")
