import os
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
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure urllib3 connection pool
urllib3_pool = urllib3.PoolManager(
    maxsize=50,  # Increase from 25
    retries=urllib3.Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504]
    ),
    timeout=urllib3.Timeout(connect=5, read=30)
)

# Load environment variables from .env file
load_dotenv()

# Load environment variables and initialize OpenAI client
openai_client = AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# Configuration Constants
class TokenConfig:
    """Dynamic token configuration."""
    def __init__(self, max_tokens=20000):
        self.max_tokens = max_tokens
        # Dynamic thresholds based on max_tokens
        self.warning_threshold = 0.85  # Warn at 85% of max tokens
        self.batch_reduction_threshold = 0.70  # Reduce batch size at 70%
        self.min_tokens_per_page = 1000  # Minimum expected tokens per page
        
    def get_safe_batch_size(self, current_tokens, avg_tokens_per_page):
        """Calculate safe batch size based on remaining tokens."""
        remaining_tokens = self.max_tokens - current_tokens
        estimated_pages = remaining_tokens / max(avg_tokens_per_page, self.min_tokens_per_page)
        return max(1, min(3, int(estimated_pages)))
        
    def is_limit_reached(self, current_tokens):
        """Check if token limit is reached."""
        return current_tokens >= self.max_tokens
        
    def get_remaining_capacity(self, current_tokens):
        """Get remaining capacity percentage."""
        return 1 - (current_tokens / self.max_tokens)

# Network Configuration
HTTPX_LIMITS = httpx.Limits(
    max_keepalive_connections=50,  # Increase from 20
    max_connections=200,  # Increase from 100
    keepalive_expiry=60.0  # Increase from 30.0
)

# Cost Configuration
MODEL_COSTS = {
    'gpt-4o-mini': {
        'input': 0.075 / 1_000_000,
        'output': 0.300 / 1_000_000
    }
}

# Add this after MODEL_COSTS constant
SYSTEM_PROMPT = """You are a technical documentation expert creating beautiful, structured markdown documentation. Follow these strict guidelines:

STRUCTURE:
1. Each document section must follow this format:
   ```markdown
   ## Section Title
   Brief overview of the section (1-2 sentences)
   
   ### Key Concepts
   - Concept 1: Brief explanation
   - Concept 2: Brief explanation
   
   ### Implementation
   Step-by-step guide or explanation
   
   ### Examples
   Practical examples with explanations
   ```

CONTENT RULES:
1. Every code block must have:
   - Title describing purpose
   - Language specification
   - Brief explanation before code
   - Comments for complex parts
   - Expected output (if applicable)

2. Installation instructions must include:
   - Prerequisites list
   - Step-by-step commands
   - Expected output
   - Troubleshooting notes

3. API documentation must show:
   - Method signature
   - Parameter descriptions
   - Return value details
   - Usage example
   - Common errors/solutions

FORMATTING:
1. Headers:
   - H1 (#) for document title
   - H2 (##) for main sections
   - H3 (###) for subsections
   - H4 (####) for example titles

2. Code blocks:
   ```language
   # Title: Purpose of code
   code here
   ```

3. Notes and warnings:
   > **Note:** Important information here
   
   > ⚠️ **Warning:** Critical warning here

4. Lists:
   - Use bullet points for related items
   - Use numbered lists for sequential steps

QUALITY REQUIREMENTS:
1. Every section must have:
   - Clear purpose
   - Practical examples
   - Implementation details
   - Common pitfalls/solutions

2. All content must be:
   - Concise but complete
   - Technically accurate
   - Well-structured
   - Easy to follow

3. Code examples must be:
   - Practical and realistic
   - Well-commented
   - Error-handled
   - Production-ready

NEVER INCLUDE:
- Marketing language
- Redundant information
- Unexplained jargon
- Incomplete examples
- Non-English content
- Raw HTML (use markdown)

ALWAYS:
1. Start with an overview
2. Include prerequisites
3. Show practical examples
4. Explain error scenarios
5. Add troubleshooting tips
6. Link related sections
7. Use consistent terminology"""

# Set seed for consistent language detection
DetectorFactory.seed = 0

class LanguageFilter:
    """Handles language detection and filtering for content."""
    
    def __init__(self, primary_language='en'):
        self.primary_language = primary_language
        self.cache = {}  # Cache detection results
        
    def is_primary_language(self, text: str, min_length: int = 50) -> bool:
        """
        Check if text is in the primary language.
        
        Args:
            text: Text to check
            min_length: Minimum text length for reliable detection
            
        Returns:
            bool: True if text is in primary language
        """
        if not text or len(text.strip()) < min_length:
            return True  # Too short to reliably detect
            
        # Check cache first
        cache_key = hash(text[:1000])  # Use first 1000 chars for cache key
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        try:
            # Detect language from a sample of the text for efficiency
            sample = text[:1000] if len(text) > 1000 else text
            detected = detect(sample) == self.primary_language
            self.cache[cache_key] = detected
            return detected
        except LangDetectException:
            logger.warning("Language detection failed, defaulting to accept")
            return True

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

# Replace the TokenCounter class with this improved version
class TokenCounter:
    """Handles token counting and cost tracking with accurate pricing."""
    
    def __init__(self):
        self.usage = {
            'input_tokens': 0,
            'output_tokens': 0
        }
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            logger.error(f"Error counting tokens: {e}")
            return 0
            
    def add_api_call_usage(self, prompt_tokens: int, completion_tokens: int, model: str = 'gpt-4o-mini') -> None:
        """Track token usage from an API call."""
        if model not in MODEL_COSTS:
            logger.error(f"Unknown model: {model}")
            return
            
        self.usage['input_tokens'] += prompt_tokens
        self.usage['output_tokens'] += completion_tokens
        
        input_cost = prompt_tokens * MODEL_COSTS[model]['input']
        output_cost = completion_tokens * MODEL_COSTS[model]['output']
        total_cost = input_cost + output_cost
        
        logger.info(
            f"API Call Costs:\n"
            f"  Input:  {prompt_tokens:,} tokens = ${input_cost:.6f}\n"
            f"  Output: {completion_tokens:,} tokens = ${output_cost:.6f}\n"
            f"  Total:  ${total_cost:.6f}"
        )
        
    def get_total_cost(self, model: str = 'gpt-4o-mini') -> float:
        """Calculate total cost for all usage."""
        if model not in MODEL_COSTS:
            logger.error(f"Unknown model: {model}")
            return 0.0
            
        input_cost = self.usage['input_tokens'] * MODEL_COSTS[model]['input']
        output_cost = self.usage['output_tokens'] * MODEL_COSTS[model]['output']
        return input_cost + output_cost
        
    def get_usage_summary(self) -> dict:
        """Get summary of token usage and costs."""
        total_cost = self.get_total_cost()
        return {
            'input_tokens': self.usage['input_tokens'],
            'output_tokens': self.usage['output_tokens'],
            'total_tokens': sum(self.usage.values()),
            'total_cost': total_cost
        }

class MarkdownFormatter:
    """Handles consistent markdown formatting and structure."""
    
    @staticmethod
    def format_title(title: str) -> str:
        """Format main title with proper decoration."""
        return f"# {title}\n\n"
    
    @staticmethod
    def format_section(title: str, level: int = 2) -> str:
        """Format section header with proper level."""
        return f"{'#' * level} {title}\n\n"
    
    @staticmethod
    def format_code_block(code: str, language: str = '', title: str = None) -> str:
        """Format code block with optional title and language."""
        result = []
        if title:
            result.append(f"**{title}**\n")
        result.append(f"```{language}")
        result.append(code.strip())
        result.append("```\n")
        return "\n".join(result)
    
    @staticmethod
    def format_note(text: str) -> str:
        """Format a note or important information."""
        return f"> **Note:** {text}\n\n"
    
    @staticmethod
    def format_warning(text: str) -> str:
        """Format a warning message."""
        return f"> ⚠️ **Warning:** {text}\n\n"
    
    @staticmethod
    def format_installation(steps: list) -> str:
        """Format installation instructions."""
        result = ["### Installation\n\n"]
        for i, step in enumerate(steps, 1):
            result.append(f"{i}. {step}\n")
        return "\n".join(result) + "\n"

class DocumentationScraper:
    def __init__(self, base_url, priorities=None, max_concurrent=5, use_selenium=True, max_tokens=20000):
        """Initialize the documentation scraper."""
        self.base_url = base_url
        self.base_domain = urlparse(base_url).netloc
        self.visited_urls = set()
        self.processed_contents = []
        self.terminator = GracefulTerminator()
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.use_selenium = use_selenium
        
        # Initialize utilities
        self.token_counter = TokenCounter()
        self.language_filter = LanguageFilter(primary_language='en')
        self.markdown_formatter = MarkdownFormatter()  # Add this initialization
        
        # Add these new tracking sets/dicts for deduplication
        self.seen_titles = set()
        self.seen_content_hashes = set()
        self.seen_code_blocks = set()
        self.content_hash_map = {}  # Add this line to initialize the content hash map
        self.content_similarity_threshold = 0.85  # For fuzzy matching
        
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
        
        self.token_config = TokenConfig(max_tokens)

    def extract_links(self, soup) -> list:
        """Extract and filter links from the page."""
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
        return links

    def normalize_url(self, url: str) -> str:
        """Normalize URL to handle relative paths and ensure it's within the documentation."""
        try:
            # Handle fragment identifiers
            url = url.split('#')[0]
            
            # Skip empty URLs
            if not url:
                return None
                
            # Convert relative to absolute URL
            normalized = urljoin(self.base_url, url)
            
            # Ensure URL is within the documentation domain
            if self.base_domain in urlparse(normalized).netloc:
                return normalized
            return None
        except Exception as e:
            logger.error(f"Error normalizing URL {url}: {e}")
            return None

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
        """Enhanced code block extraction with language filtering and deduplication."""
        code_blocks = []
        
        # Define priority languages with expanded variations
        priority_languages = {
            'python': ['python', 'py', 'python3', 'pycon'],
            'javascript': [
                'javascript', 'js', 'jsx', 'typescript', 'ts', 
                'tsx', 'react', 'reactjs', 'react-jsx', 'react-tsx'
            ],
            'nodejs': ['node', 'nodejs', 'node.js', 'npm', 'yarn']
        }
        
        # Language-specific patterns to identify code type
        language_patterns = {
            'tsx': [
                'React', 'jsx', 'tsx', '<Component', '<div', 'interface Props',
                'React.FC', 'useState', 'useEffect', '<>'
            ],
            'jsx': ['React.', 'jsx', '<div', 'className=', 'props.', 'render()'],
            'typescript': ['interface', 'type ', ': string', ': number', ': boolean'],
        }
        
        # Flatten language variations for easier checking
        allowed_languages = sum(priority_languages.values(), [])
        
        # Common code block selectors
        code_selectors = [
            'pre code',
            '.highlight',
            '.prism-code',
            '.hljs',
            '[class*="language-"]',
            '.code-block',
            'div[class*="codeBlock"]',
            '.code',
            '[class*="code"]'
        ]
        
        for selector in code_selectors:
            for element in soup.select(selector):
                # Get the code content first for language detection
                code = element.get_text().strip()
                if not code or len(code) < 10:  # Ignore very short snippets
                    continue
                
                # Try to detect the programming language
                language = ''
                classes = element.get('class', [])
                
                # Check for language in class names
                for class_name in classes:
                    if 'language-' in class_name:
                        language = class_name.replace('language-', '')
                        break
                    elif 'lang-' in class_name:
                        language = class_name.replace('lang-', '')
                        break
                
                # Normalize language name
                language = language.lower()
                
                # Smart language detection for React/TypeScript code
                if any(pattern in code for pattern in language_patterns['tsx']):
                    language = 'tsx'
                elif any(pattern in code for pattern in language_patterns['jsx']):
                    language = 'jsx'
                elif any(pattern in code for pattern in language_patterns['typescript']):
                    language = 'typescript'
                
                # Map similar language names
                for main_lang, variations in priority_languages.items():
                    if language in variations:
                        language = main_lang
                        break
                
                # Skip if not a priority language
                if language not in ['python', 'javascript', 'nodejs', 'tsx']:
                    continue
                
                code_blocks.append({
                    'language': language,
                    'code': code
                })
        
        # Add deduplication check
        unique_blocks = []
        for block in code_blocks:
            if not self.is_duplicate_code(block['code']):
                unique_blocks.append(block)
        
        return unique_blocks

    def is_duplicate_content(self, content: str) -> bool:
        """Check if content is duplicate using fuzzy matching."""
        content_hash = hash(content.strip().lower())
        if content_hash in self.seen_content_hashes:
            return True
            
        # Check for similar content using fuzzy matching
        content_words = set(content.lower().split())
        for existing_hash in self.seen_content_hashes:
            existing_content = self.content_hash_map.get(existing_hash, '')
            existing_words = set(existing_content.lower().split())
            similarity = len(content_words & existing_words) / len(content_words | existing_words)
            if similarity > self.content_similarity_threshold:
                return True
                
        self.seen_content_hashes.add(content_hash)
        self.content_hash_map[content_hash] = content
        return False

    def is_duplicate_code(self, code: str) -> bool:
        """Check if code block is duplicate, with language-specific normalization."""
        # Normalize code by removing whitespace and comments
        normalized_code = []
        for line in code.splitlines():
            line = line.strip()
            # Skip empty lines and comments in various languages
            if line and not any(line.startswith(c) for c in [
                '//', '#', '/*', '*', '*/', '<!--', '-->'
            ]):
                # Remove language-specific string literals and variable names
                normalized_line = (
                    line.replace('var ', '')
                        .replace('let ', '')
                        .replace('const ', '')
                        .replace('def ', '')
                        .replace('async ', '')
                        .replace('await ', '')
                )
                normalized_code.append(normalized_line)
        
        normalized_code = '\n'.join(normalized_code)
        code_hash = hash(normalized_code)
        
        if code_hash in self.seen_code_blocks:
            return True
        self.seen_code_blocks.add(code_hash)
        return False

    def extract_content(self, main_content) -> tuple[str, bool]:
        """
        Extract and validate content from main content element.
        
        Returns:
            tuple: (processed_text, is_valid_language)
        """
        content_text = main_content.get_text(separator=' ', strip=True)
        is_english = self.language_filter.is_primary_language(content_text)
        
        if not is_english:
            logger.info("Skipping non-English content")
            return content_text, False
            
        return content_text, True

    async def process_page(self, url, client):
        """Process a single documentation page with deduplication."""
        retries = 3
        while retries > 0:
            try:
                async with self.semaphore:
                    # Add delay between requests to prevent overwhelming the server
                    await asyncio.sleep(0.5)
                    
                    html_content = await self.get_page_content(url, client)
                    if not html_content:
                        logger.error(f"Failed to get content for {url}")
                        return None
                    
                    soup = BeautifulSoup(html_content, 'html.parser')
                    
                    # Extract main content
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
                    
                    # Extract and validate content
                    content_text, is_english = self.extract_content(main_content)
                    if not is_english:
                        return None
                    
                    # Get page title
                    title = (
                        soup.find('h1') or 
                        soup.find('title') or
                        soup.find(['h1', 'h2'], class_=lambda x: x and ('title' in x.lower() if x else False))
                    )
                    title = title.get_text().strip() if title else url.split('/')[-1].replace('-', ' ').title()
                    
                    # Validate title language
                    if not self.language_filter.is_primary_language(title, min_length=10):
                        logger.info(f"Skipping page with non-English title: {title}")
                        return None
                    
                    # Extract code examples with enhanced detection
                    code_examples = self.extract_code_blocks(main_content)
                    
                    # Extract and filter links
                    links = self.extract_links(soup)
                    
                    # Use OpenAI to convert to clean markdown
                    response = await openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": f"Title: {title}\n\nContent: {content_text}"}
                        ],
                        temperature=0
                    )
                    
                    # Update token tracking
                    self.token_counter.add_api_call_usage(
                        response.usage.prompt_tokens,
                        response.usage.completion_tokens
                    )
                    
                    # Validate generated content
                    generated_content = response.choices[0].message.content
                    if not self.language_filter.is_primary_language(generated_content):
                        logger.warning("Generated content appears to be non-English, skipping")
                        return None
                    
                    # Check for duplicate title
                    if title.lower().strip() in self.seen_titles:
                        logger.info(f"Skipping duplicate title: {title}")
                        return None
                    self.seen_titles.add(title.lower().strip())
                    
                    # Check for duplicate content
                    if self.is_duplicate_content(content_text):
                        logger.info(f"Skipping duplicate content for: {url}")
                        return None
                    
                    # Deduplicate generated content
                    if self.is_duplicate_content(generated_content):
                        logger.info(f"Skipping duplicate generated content for: {url}")
                        return None
                    
                    return {
                        'title': title,
                        'content': generated_content,
                        'links': links,
                        'code_examples': code_examples,
                        'language': 'en'
                    }
                    
            except httpx.TransportError as e:
                retries -= 1
                if retries > 0:
                    await asyncio.sleep(1)
                    continue
                logger.error(f"Transport error processing {url} after all retries: {str(e)}")
                return None
            except Exception as e:
                logger.error(f"Error processing {url}: {str(e)}")
                return None
            break  # Success, exit retry loop
        return None  # All retries failed

    async def crawl(self):
        """Enhanced crawling with dynamic token management."""
        # Initialize urls_to_visit and tracking variables
        urls_to_visit = [{'url': self.base_url, 'priority': {'score': 100, 'type': 'essential'}}]
        visited_count = 0
        required_sections = {
            section: False 
            for section, info in self.priorities.items() 
            if info.get('required', False)
        }
        
        try:
            async with httpx.AsyncClient(
                limits=HTTPX_LIMITS,
                timeout=httpx.Timeout(
                    connect=5.0,
                    read=30.0,
                    write=30.0,
                    pool=30.0
                ),
                follow_redirects=True,
                http2=True,  # Enable HTTP/2 for better connection management
                transport=httpx.AsyncHTTPTransport(
                    retries=3,
                )
            ) as client:
                while urls_to_visit:
                    usage_summary = self.token_counter.get_usage_summary()
                    total_tokens = usage_summary['total_tokens']
                    
                    # Check token limit using TokenConfig
                    if self.token_config.is_limit_reached(total_tokens):
                        logger.info(f"Reached token limit ({total_tokens:,}/{self.token_config.max_tokens:,}). Stopping crawl...")
                        break
                    
                    # Calculate batch size dynamically
                    avg_tokens_per_page = total_tokens / max(1, len(self.processed_contents))
                    batch_size = self.token_config.get_safe_batch_size(total_tokens, avg_tokens_per_page)
                    
                    # Warn if approaching limit
                    remaining_capacity = self.token_config.get_remaining_capacity(total_tokens)
                    if remaining_capacity <= self.token_config.warning_threshold:
                        logger.warning(f"Approaching token limit ({total_tokens:,}/{self.token_config.max_tokens:,})")
                    
                    # Process in smaller batches to prevent connection pool exhaustion
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
                            
                            # Fix token check here
                            if not self.token_config.is_limit_reached(total_tokens):
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
                    
                    # Show progress with correct token limit
                    logger.info(
                        f"Progress: {visited_count} pages processed\n"
                        f"Tokens: {total_tokens:,} / {self.token_config.max_tokens:,} "
                        f"({(total_tokens/self.token_config.max_tokens)*100:.1f}%) - "
                        f"{self.token_config.max_tokens - total_tokens:,} tokens remaining"
                    )
                    
                    # Check limit with token_config
                    if self.token_config.is_limit_reached(total_tokens):
                        logger.info("Token limit reached after processing batch. Stopping...")
                        break
            
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
        """Save progress with section deduplication."""
        try:
            # Create documentation directory if it doesn't exist
            docs_dir = Path('documentation')
            docs_dir.mkdir(exist_ok=True)
            
            base_name = self.base_url.split('//')[1].split('/')[0].split('.')[0]
            filename = docs_dir / f"{base_name}_documentation.md"
            
            usage_summary = self.token_counter.get_usage_summary()
            
            with open(filename, 'w', encoding='utf-8') as f:
                # Write beautiful header
                f.write("<!-- Auto-generated documentation -->\n\n")
                f.write(self.markdown_formatter.format_title("Documentation"))
                
                # Write metadata in a clean format
                f.write("## Document Information\n\n")
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")
                f.write(f"| Generated | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |\n")
                f.write(f"| Source | {self.base_url} |\n")
                f.write(f"| Pages | {len(self.processed_contents)} |\n")
                f.write(f"| Total Tokens | {usage_summary['total_tokens']:,} |\n")
                f.write(f"| Cost | ${usage_summary['total_cost']:.4f} |\n\n")
                
                # Write table of contents with proper formatting
                f.write("## Contents\n\n")
                for content in self.processed_contents:
                    clean_title = content['title'].replace('[', '').replace(']', '')
                    anchor = clean_title.lower().replace(' ', '-')
                    f.write(f"- [{clean_title}](#{anchor})\n")
                f.write("\n---\n\n")
                
                # Write content with proper section formatting
                current_type = None
                for content in self.processed_contents:
                    if content['type'] != current_type:
                        current_type = content['type']
                        f.write(self.markdown_formatter.format_section(current_type.title(), 2))
                    
                    f.write(self.markdown_formatter.format_section(content['title'], 3))
                    f.write(f"{content['content']}\n\n")
                    
                    if content.get('code_examples'):
                        for i, example in enumerate(content['code_examples'], 1):
                            title = f"Example {i}" if len(content['code_examples']) > 1 else "Example"
                            f.write(self.markdown_formatter.format_code_block(
                                example['code'],
                                example.get('language', ''),
                                title
                            ))
                    f.write("---\n\n")
            
            logger.info(f"Documentation saved to documentation/{filename.name}")
            logger.info(f"Total tokens used: {usage_summary['total_tokens']:,}")
            logger.info(f"Estimated total cost: ${usage_summary['total_cost']:.2f}")
            
            # Deduplicate sections while maintaining order
            seen_sections = set()
            unique_contents = []
            for content in self.processed_contents:
                section_key = (content['type'], content['title'].lower().strip())
                if section_key not in seen_sections:
                    seen_sections.add(section_key)
                    unique_contents.append(content)
            
            self.processed_contents = unique_contents
            
        except Exception as e:
            logger.error(f"Error saving progress: {str(e)}")
            raise

    def __del__(self):
        """Clean up Selenium driver."""
        if hasattr(self, 'driver'):
            self.driver.quit()

async def main_async():
    parser = argparse.ArgumentParser(description='Scrape documentation websites and convert to markdown.')
    parser.add_argument('urls', nargs='+', help='One or more documentation URLs to scrape')
    parser.add_argument('--max-tokens', type=int, default=20000,
                      help='Maximum number of tokens to process (default: 20000)')
    parser.add_argument('--output-dir', default='documentation',
                      help='Directory to save the output files (default: documentation)')
    parser.add_argument('--concurrent', type=int, default=3,
                      help='Number of pages to process concurrently (default: 3)')
    parser.add_argument('--no-selenium', action='store_true',
                      help='Disable Selenium and use static HTML parsing only')
    args = parser.parse_args()
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        for url in args.urls:
            logger.info(f"Starting documentation scraping for: {url}")
            scraper = DocumentationScraper(
                url, 
                max_concurrent=args.concurrent,
                use_selenium=not args.no_selenium,
                max_tokens=args.max_tokens  # Pass max_tokens to scraper
            )
            await scraper.crawl()
            
            # Use the new token tracking system
            usage_summary = scraper.token_counter.get_usage_summary()
            total_tokens = usage_summary['total_tokens']
            total_cost = usage_summary['total_cost']
            
            logger.info(f"Total tokens used: {total_tokens:,}")
            logger.info(f"Estimated total cost: ${total_cost:.2f}")
            
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        sys.exit(1)

def main():
    asyncio.run(main_async())

if __name__ == '__main__':
    main() 