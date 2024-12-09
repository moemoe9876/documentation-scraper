import os
import re
import json
import time
import logging
import signal
import tiktoken
import argparse
import sys
import asyncio
from datetime import datetime
from urllib.parse import urljoin, urlparse
import httpx 
from bs4 import BeautifulSoup
from openai import AsyncOpenAI
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import urllib3 

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure urllib3 connection pool
urllib3.PoolManager(maxsize=25, retries=3)

# Configure httpx limits
HTTPX_LIMITS = httpx.Limits(
    max_keepalive_connections=20,
    max_connections=100,
    keepalive_expiry=30.0
)

# Load environment variables from .env file
load_dotenv()

# Load environment variables and initialize OpenAI client
openai_client = AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

class GracefulTerminator:
    terminate_now = False
    save_in_progress = False
    
    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
        
    def exit_gracefully(self, signum, frame):
        if self.save_in_progress:
            logger.info("Please wait, saving current progress...")
            return
            
        if self.terminate_now:
            logger.info("\nForce quitting...")
            sys.exit(1)
            
        logger.info("\nReceived termination signal. Will save progress and exit after current page...")
        self.terminate_now = True

# Token counting and cost tracking
class TokenCounter:
    def __init__(self):
        self.total_tokens = 0
        self.total_cost = 0
        self.model_costs = {
            'gpt-4o-mini': {
                'input': 0.150 / 1000000,    # $0.150 per 1M input tokens
                'output': 0.600 / 1000000    # $0.600 per 1M output tokens
            }
        }
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text):
        return len(self.encoding.encode(text))
    
    def add_usage(self, model, input_tokens, output_tokens):
        """Add token usage and calculate cost."""
        self.total_tokens += input_tokens + output_tokens
        
        if model in self.model_costs:
            cost = (
                (input_tokens) * self.model_costs[model]['input'] +
                (output_tokens) * self.model_costs[model]['output']
            )
            self.total_cost += cost
            
            logger.info(f"API Call - Input tokens: {input_tokens:,} (${(input_tokens * self.model_costs[model]['input']):.4f})")
            logger.info(f"API Call - Output tokens: {output_tokens:,} (${(output_tokens * self.model_costs[model]['output']):.4f})")
            logger.info(f"Cost for this call: ${cost:.4f}")
    
    def get_summary(self):
        return {
            'total_tokens': self.total_tokens,
            'total_cost': self.total_cost
        }

class DocumentationScraper:
    def __init__(self, base_url, priorities=None, max_concurrent=5, use_selenium=True):
        """Initialize the documentation scraper."""
        self.base_url = base_url
        self.base_domain = urlparse(base_url).netloc
        self.visited_urls = set()
        self.processed_contents = []
        self.terminator = GracefulTerminator()
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.total_tokens = {'input': 0, 'output': 0}
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.use_selenium = use_selenium
        
        if use_selenium:
            chrome_options = Options()
            chrome_options.add_argument('--headless=new')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--disable-software-rasterizer')
            chrome_options.add_argument('--disable-extensions')
            chrome_options.add_argument('--disable-web-security')
            chrome_options.add_argument('--allow-running-insecure-content')
            chrome_options.add_argument('--ignore-certificate-errors')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument('user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
            
            self.driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()),
                options=chrome_options
            )
        
        # Default priorities if none provided
        self.priorities = priorities or {
            'essential': {
                'patterns': ['introduction', 'getting-started', 'overview'],
                'score': 100,
                'required': True
            },
            'core_concepts': {
                'patterns': ['concepts', 'architecture', 'lifecycle'],
                'score': 80,
                'required': True
            },
            'features': {
                'patterns': ['features', 'components', 'routing'],
                'score': 70,
                'required': False
            },
            'api': {
                'patterns': ['api', 'reference'],
                'score': 60,
                'required': False
            }
        }

    async def get_page_content(self, url, client):
        """Get page content using either Selenium or httpx."""
        if self.use_selenium:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._get_selenium_content, url)
        else:
            response = await client.get(url)
            response.raise_for_status()
            return response.text

    def _get_selenium_content(self, url):
        """Get page content using Selenium with better code block handling."""
        try:
            self.driver.get(url)
            
            # Wait for main content to load
            WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "main, article, .docs-content, [class*='documentation'], [class*='content']"))
            )
            
            # Wait for dynamic navigation to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "nav, .sidebar, [class*='navigation']"))
            )
            
            # Wait additional time for dynamic content
            time.sleep(3)
            
            return self.driver.page_source
        except Exception as e:
            logger.error(f"Error loading page with Selenium: {str(e)}")
            return None

    def extract_code_blocks(self, soup):
        """Enhanced code block extraction with language detection."""
        code_blocks = []
        
        # Common code block selectors for various documentation frameworks
        code_selectors = [
            'pre code',  # Standard HTML
            '.highlight',  # GitHub style
            '.prism-code',  # Prism.js
            '.hljs',  # highlight.js
            '[class*="language-"]',  # Common language class pattern
            '.code-block',  # Common documentation frameworks
            'div[class*="codeBlock"]',  # React/Next.js style
            '.code',  # OpenAI docs style
            '[class*="code"]'  # Generic code class pattern
        ]
        
        for selector in code_selectors:
            for element in soup.select(selector):
                # Try to detect the programming language
                language = ''
                classes = element.get('class', [])
                
                # Common language class patterns
                for class_name in classes:
                    if 'language-' in class_name:
                        language = class_name.replace('language-', '')
                        break
                    elif 'lang-' in class_name:
                        language = class_name.replace('lang-', '')
                        break
                
                code = element.get_text().strip()
                if code and len(code) > 10:  # Ignore very short snippets
                    code_blocks.append({
                        'language': language,
                        'code': code
                    })
        
        return code_blocks

    async def process_page(self, url, client):
        """Process a single documentation page asynchronously."""
        try:
            async with self.semaphore:
                html_content = await self.get_page_content(url, client)
                if not html_content:
                    logger.error(f"Failed to get content for {url}")
                    return None
                    
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Extract main content with more selectors for SPAs
                main_content = (
                    soup.find('main') or 
                    soup.find('article') or 
                    soup.find('div', class_='content') or
                    soup.find('div', class_=lambda x: x and ('docs' in x.lower() or 'documentation' in x.lower())) or
                    soup.find('div', class_=lambda x: x and 'main' in x.lower()) or
                    soup.find('div', class_=lambda x: x and 'content' in x.lower())
                )
                
                if not main_content:
                    logger.error(f"No main content found for {url}")
                    return None
                
                # Get page title
                title = (
                    soup.find('h1') or 
                    soup.find('title') or
                    soup.find(['h1', 'h2'], class_=lambda x: x and ('title' in x.lower() if x else False))
                )
                title = title.get_text().strip() if title else url.split('/')[-1].replace('-', ' ').title()
                
                # Extract code examples with enhanced detection
                code_examples = self.extract_code_blocks(main_content)
                
                # Extract links with improved selectors
                links = []
                for link in soup.find_all('a', href=True):
                    href = link.get('href')
                    if href and not href.startswith(('#', 'mailto:', 'tel:', 'javascript:')):
                        normalized_url = self.normalize_url(href)
                        if normalized_url and self.base_domain in normalized_url:
                            # Calculate priority score based on URL patterns
                            score = 0
                            link_type = 'other'
                            for section, info in self.priorities.items():
                                if any(pattern in normalized_url.lower() for pattern in info['patterns']):
                                    score = info['score']
                                    link_type = section
                                    break
                            
                            if score > 0:
                                links.append({
                                    'url': normalized_url,
                                    'priority': {
                                        'score': score,
                                        'type': link_type
                                    }
                                })
                
                # Prepare content for AI processing
                content_text = main_content.get_text()
                
                # Use OpenAI to convert to clean markdown
                response = await openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": """You are a technical documentation expert. Convert the provided documentation into clean, well-formatted markdown focusing on implementation details and practical usage. Your output should:

1. Focus on actual implementation details, code examples, and practical usage
2. Remove navigation elements, menus, and other UI components
3. Keep only the essential technical content that would be useful for implementation
4. Preserve all code examples and technical details
5. Use proper markdown formatting for headings, lists, and code blocks
6. Include configuration options and their explanations
7. Keep error handling and troubleshooting information
8. Remove any duplicate content or redundant navigation elements

Do NOT include:
- Table of contents or navigation menus
- Links to other pages
- UI elements or website navigation
- Marketing content
- Duplicate section headers"""},
                        {"role": "user", "content": f"Title: {title}\n\nContent: {content_text}"}
                    ],
                    temperature=0
                )
                
                # Update token counts
                self.total_tokens['input'] += response.usage.prompt_tokens
                self.total_tokens['output'] += response.usage.completion_tokens
                
                # Log token usage and cost
                input_cost = (response.usage.prompt_tokens / 1000) * 0.01
                output_cost = (response.usage.completion_tokens / 1000) * 0.03
                total_cost = input_cost + output_cost
                
                logger.info(f"API Call - Input tokens: {response.usage.prompt_tokens:,} (${input_cost:.4f})")
                logger.info(f"API Call - Output tokens: {response.usage.completion_tokens:,} (${output_cost:.4f})")
                logger.info(f"Cost for this call: ${total_cost:.4f}")
                
                return {
                    'title': title,
                    'content': response.choices[0].message.content,
                    'links': links,
                    'code_examples': code_examples
                }
                
        except Exception as e:
            logger.error(f"Error processing {url}: {str(e)}")
            return None

    def normalize_url(self, url):
        """Normalize URL to handle relative paths and ensure it's within the documentation."""
        normalized = urljoin(self.base_url, url)
        return normalized if self.base_domain in normalized else None

    async def crawl(self):
        """Enhanced crawling with concurrent processing."""
        urls_to_visit = [{'url': self.base_url, 'priority': {'score': 100, 'type': 'essential'}}]
        visited_count = 0
        total_tokens = 0
        MAX_TOKENS = 100000
        
        # Track required sections
        required_sections = {
            section: False 
            for section, info in self.priorities.items() 
            if info.get('required', False)
        }
        
        try:
            # Configure httpx client with proper limits
            async with httpx.AsyncClient(
                limits=HTTPX_LIMITS,
                timeout=30.0,
                follow_redirects=True
            ) as client:
                while urls_to_visit and total_tokens < MAX_TOKENS:
                    if self.terminator.terminate_now:
                        logger.info("Termination requested, saving progress...")
                        break
                    
                    # Process in smaller batches to prevent connection pool exhaustion
                    batch_size = min(self.max_concurrent, 3)
                    current_batch = []
                    while urls_to_visit and len(current_batch) < batch_size:
                        current = urls_to_visit.pop(0)
                        if self.normalize_url(current['url']) not in self.visited_urls:
                            current_batch.append(current)
                            self.visited_urls.add(self.normalize_url(current['url']))
                    
                    if not current_batch:
                        continue
                    
                    # Add delay between batches to prevent overwhelming the server
                    if visited_count > 0:
                        await asyncio.sleep(1)
                    
                    # Process batch concurrently
                    tasks = []
                    for item in current_batch:
                        logger.info(f"Processing page {visited_count + len(tasks) + 1} - {item['priority']['type']} ({item['priority']['score']} score): {item['url']}")
                        tasks.append(self.process_page(item['url'], client))
                    
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Process results
                    for result, item in zip(results, current_batch):
                        if isinstance(result, Exception):
                            logger.error(f"Error processing {item['url']}: {str(result)}")
                            continue
                        
                        if result:
                            # Update section tracking
                            section_type = item['priority']['type']
                            if section_type in required_sections:
                                required_sections[section_type] = True
                            
                            # Add to processed contents
                            self.processed_contents.append({
                                'title': result['title'],
                                'content': result['content'],
                                'type': section_type,
                                'priority_score': item['priority']['score'],
                                'code_examples': result.get('code_examples', [])
                            })
                            visited_count += 1
                            
                            # Update total tokens
                            total_tokens = sum(self.total_tokens.values())
                            
                            # Add new links to visit with rate limiting
                            if total_tokens < MAX_TOKENS:
                                new_links = result.get('links', [])
                                filtered_links = [
                                    link for link in new_links 
                                    if self.normalize_url(link['url']) not in self.visited_urls
                                ]
                                
                                # Prioritize links that might contain missing sections
                                for link in filtered_links:
                                    for section, found in required_sections.items():
                                        if not found and any(
                                            pattern in link['url'].lower() 
                                            for pattern in self.priorities[section]['patterns']
                                        ):
                                            link['priority']['score'] += 5
                                
                                urls_to_visit.extend(filtered_links)
                                urls_to_visit.sort(key=lambda x: x['priority']['score'], reverse=True)
                    
                    # Show progress
                    total_tokens = sum(self.total_tokens.values())
                    token_percentage = (total_tokens / MAX_TOKENS) * 100
                    remaining_tokens = MAX_TOKENS - total_tokens
                    remaining_urls = len(urls_to_visit)
                    logger.info(f"Progress: {visited_count} pages processed")
                    logger.info(f"Tokens: {total_tokens:,} / {MAX_TOKENS:,} ({token_percentage:.1f}%) - {remaining_tokens:,} tokens remaining")
                    logger.info(f"URLs: {remaining_urls} remaining in queue")
                    
                    # Add delay between batches
                    await asyncio.sleep(2)
            
            # Sort contents by priority before saving
            self.processed_contents.sort(key=lambda x: x['priority_score'], reverse=True)
            
            # Save final or partial results
            if not self.terminator.terminate_now:
                self.save_progress(final=True)
            else:
                self.save_progress(final=False)
                
        except Exception as e:
            logger.error(f"Error during crawl: {str(e)}")
            self.save_progress(final=False)
            raise

    def save_progress(self, final=False):
        """Save the current progress to a markdown file."""
        try:
            base_name = self.base_url.split('//')[1].split('/')[0].split('.')[0]
            filename = f"{base_name}_documentation.md"
            
            total_cost = self.calculate_cost()
            total_tokens = sum(self.total_tokens.values())
            
            logger.info(f"Saving progress to {filename}")
            
            with open(filename, 'w', encoding='utf-8') as f:
                # Write header with token usage and cost information
                f.write("# Documentation Summary\n\n")
                f.write("## Processing Information\n")
                f.write(f"- Total Input Tokens: {self.total_tokens['input']:,}\n")
                f.write(f"- Total Output Tokens: {self.total_tokens['output']:,}\n")
                f.write(f"- Total Tokens: {total_tokens:,}\n")
                f.write(f"- Estimated Total Cost: ${total_cost:.2f}\n\n")
                f.write("---\n\n")
                
                # Write table of contents
                f.write("## Table of Contents\n\n")
                for content in self.processed_contents:
                    f.write(f"- [{content['title']}](#{content['title'].lower().replace(' ', '-')})\n")
                f.write("\n---\n\n")
                
                # Write the actual content, organized by type
                current_type = None
                for content in self.processed_contents:
                    if content['type'] != current_type:
                        current_type = content['type']
                        f.write(f"\n## {current_type.title()}\n\n")
                    
                    f.write(f"### {content['title']}\n\n")
                    f.write(f"{content['content']}\n\n")
                    
                    if content.get('code_examples'):
                        f.write("#### Code Examples\n\n")
                        for example in content['code_examples']:
                            f.write(f"```{example.get('language', '')}\n{example['code']}\n```\n\n")
                    f.write("---\n\n")
            
            logger.info(f"Documentation saved to {filename}")
            logger.info(f"Total tokens used: {total_tokens:,}")
            logger.info(f"Estimated total cost: ${total_cost:.2f}")
            
        except Exception as e:
            logger.error(f"Error saving progress: {str(e)}")
            raise

    def calculate_cost(self):
        """Calculate the total cost based on token usage."""
        # GPT-4o-mini pricing: $0.150/1M input tokens, $0.600/1M output tokens
        input_cost = (self.total_tokens['input'] / 1000000) * 0.150
        output_cost = (self.total_tokens['output'] / 1000000) * 0.600
        return input_cost + output_cost

    def __del__(self):
        """Clean up Selenium driver."""
        if hasattr(self, 'driver'):
            self.driver.quit()

async def main_async():
    parser = argparse.ArgumentParser(description='Scrape documentation websites and convert to markdown.')
    parser.add_argument('urls', nargs='+', help='One or more documentation URLs to scrape')
    parser.add_argument('--output-dir', default='.',
                      help='Directory to save the output files (default: current directory)')
    parser.add_argument('--concurrent', type=int, default=5,
                      help='Number of pages to process concurrently (default: 5)')
    parser.add_argument('--no-selenium', action='store_true',
                      help='Disable Selenium and use static HTML parsing only')
    args = parser.parse_args()
    
    try:
        for url in args.urls:
            logger.info(f"Starting documentation scraping for: {url}")
            scraper = DocumentationScraper(
                url, 
                max_concurrent=args.concurrent,
                use_selenium=not args.no_selenium
            )
            await scraper.crawl()
            
            # Use the new token tracking system
            total_tokens = sum(scraper.total_tokens.values())
            total_cost = scraper.calculate_cost()
            
            logger.info(f"Total tokens used: {total_tokens:,}")
            logger.info(f"Estimated total cost: ${total_cost:.2f}")
            
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        sys.exit(1)

def main():
    asyncio.run(main_async())

if __name__ == '__main__':
    main() 


