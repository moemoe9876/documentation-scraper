# Documentation Scraper

A powerful documentation scraper that converts web-based documentation into well-structured markdown files with intelligent token management and language filtering.

## Features

- 🔄 Concurrent page processing
- 📝 Intelligent markdown formatting
- 🔍 Smart code block detection (Python, JavaScript, Node.js, TSX)
- 🌐 Language filtering (English-only content)
- 💰 Token usage tracking and cost estimation
- 🔄 Deduplication of content and code blocks
- 📊 Progress monitoring and reporting
- 🛑 Graceful termination handling

## Prerequisites

```bash
# Required Python version
Python 3.8+

# Create and activate virtual environment
## On Windows
python -m venv venv
.\venv\Scripts\activate

## On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

## Environment Setup

1. Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

2. Install Chrome/Chromium for Selenium support (optional)

## Usage

### Basic Usage

```bash
python documentation_scraper.py [URL]
```

### Advanced Options

```bash
python documentation_scraper.py [OPTIONS] URL

Options:
  --max-tokens INT    Maximum tokens to process (default: 20000)
  --concurrent INT    Number of concurrent pages to process (default: 3)
  --output-dir DIR   Output directory for documentation (default: documentation)
  --no-selenium      Disable Selenium for static HTML parsing only
```

### Examples

1. Basic scraping:
```bash
python documentation_scraper.py https://docs.example.com
```

2. High-throughput scraping:
```bash
python documentation_scraper.py --max-tokens 50000 --concurrent 3 https://docs.example.com
```

3. Multiple documentation sites:
```bash
python documentation_scraper.py https://docs.site1.com https://docs.site2.com
```

## Output

The scraper generates markdown files in the `documentation` directory:
- `site_documentation.md`: Main documentation file
- Includes:
  - Document information
  - Table of contents
  - Structured sections
  - Code examples
  - Cost and token usage summary

## Token Management

- Default limit: 20,000 tokens
- Adjustable via `--max-tokens`
- Cost tracking for:
  - Input tokens ($0.075/1M tokens)
  - Output tokens ($0.300/1M tokens)

## Code Block Support

Prioritizes and formats:
- Python
- JavaScript
- Node.js
- TSX/React
- TypeScript

## Error Handling

- Graceful termination with CTRL+C
- Automatic progress saving
- Connection retry logic
- Error logging

## Limitations

- English content only
- Requires JavaScript for dynamic sites
- Token usage based on OpenAI's pricing
- Rate limiting may apply

## Best Practices

1. Start with small token limits to test
2. Use concurrent=2 or 3 for optimal performance
3. Enable Selenium for JavaScript-heavy sites
4. Monitor token usage for cost control

## Troubleshooting

### Common Issues

1. Connection pool warnings:
```bash
# Reduce concurrent processing
python documentation_scraper.py --concurrent 2 [URL]
```

2. Token limit reached:
```bash
# Increase token limit
python documentation_scraper.py --max-tokens 50000 [URL]
```

3. Selenium issues:
```bash
# Try without Selenium
python documentation_scraper.py --no-selenium [URL]
```

### Debug Mode

Enable debug logging:
```bash
export PYTHONPATH=.
python -m logging -v DEBUG documentation_scraper.py [URL]
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License - See LICENSE file for details

## Acknowledgments

- OpenAI for API support
- Selenium for dynamic content handling
- Beautiful Soup for HTML parsing