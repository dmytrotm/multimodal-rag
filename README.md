# BatchScraper - Multimodal RAG system

A sophisticated web scraping and RAG (Retrieval-Augmented Generation) system designed for extracting and processing multimodal content from DeepLearning.AI's "The Batch" weekly issues archive. The system creates a comprehensive knowledge base by collecting both text and images, then processes them into a searchable multimodal RAG database with an intuitive web interface.

## 🌟 Key Features

- **Intelligent Content Extraction**: Automatically identifies and separates Andrew Ng's letters from news sections
- **Image Download & Management**: Downloads and organizes images with proper naming and metadata
- **Smart Content Filtering**: Filters out advertisements, sponsors, and low-quality content
- **Multimodal RAG Processing**: Handles both text content and images using Google's Gemini models
- **Interactive Web Interface**: Streamlit-based UI for easy querying and result visualization
- **Parallel Processing**: Multi-threaded processing with intelligent API key management
- **Rate Limiting**: Respectful scraping with configurable delays and retry mechanisms
- **Structured Output**: Creates clean, structured JSON data suitable for RAG applications

## 📋 Prerequisites

### System Requirements
- Python 3.8+
- Minimum 8GB RAM (16GB+ recommended for large datasets)
- SSD storage recommended for vector database performance
- Web browser for accessing the Streamlit interface

### Required Dependencies
```bash
# Core scraping and processing
pip install requirements.txt
```

### Environment Setup
Create a `.env` file with your Google API keys:
```env
GOOGLE_API_KEY_1=your_first_api_key_here
GOOGLE_API_KEY_2=your_second_api_key_here
GOOGLE_API_KEY_3=your_third_api_key_here
```

## 🚀 Quick Start

### 1. Basic Web Scraping

```python
from batch_scraper import BatchScraper

# Initialize scraper
scraper = BatchScraper()

# Process a single issue
result = scraper.process_single_issue('https://www.deeplearning.ai/the-batch/issue-123/')

# Auto-discover and process all issues
all_links = scraper.get_all_issue_links(max_articles=10)
results, failed = scraper.process_multiple_issues(all_links)
```

### 2. Multimodal RAG Processing

```python
from data_processor import DataProcessor

# Initialize the RAG processor
processor = DataProcessor(
    output_dir="./scraped_data",
    vectorstore_path="./chroma_db",
    docstore_path="./docstore",
    google_api_key="your_api_key"
)

# Process scraped data into RAG database
processor.process_data()

# Query the system programmatically
result = processor.query("What are the latest developments in AI?")
print(result['answer'])
```

### 3. Launch Interactive Web Interface

```bash
# Navigate to your project directory
cd /path/to/your/project/multimodal_rag

# Launch the Streamlit app
streamlit run ui.py
```

The web interface will be available at `http://localhost:8501`

### 4. Parallel Processing Pipeline

```python
from scraper import BatchScraper
from your_module import TrulyParallelRAGProcessor

# Initialize scraper
scraper = BatchScraper(
    download_images=False,
    base_images_dir="batch_images",
    max_retries=3
)

# Process with parallel workers
api_keys = ["key1", "key2", "key3"]
processor = TrulyParallelRAGProcessor(api_keys, max_workers=5)
```

## 🖥️ Web Interface Features

### Interactive Search
- **Real-time Query Processing**: Enter questions and get instant AI-powered answers
- **Multimodal Results**: View both text answers and related images
- **Source Attribution**: See references and citations for all answers
- **Adjustable Results**: Control the number of search results (1-10)

### Visual Elements
- **Image Gallery**: Related images displayed in a clean grid layout
- **Smart Descriptions**: AI-generated descriptions for all images
- **Expandable Sources**: Detailed source information with dates and URLs
- **Progress Indicators**: Real-time feedback during search operations

### Usage Examples
```python
# Example queries you can try in the web interface:
"What does Andrew Ng say about the future of AI?"
"Show me recent developments in computer vision"
"What are the latest AI safety concerns mentioned?"
"Explain the recent breakthroughs in large language models"
```

## ⚙️ Configuration

### BatchScraper Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_url` | str | `"https://www.deeplearning.ai"` | Base URL for the target website |
| `delay` | int | `5` | Delay in seconds between requests |
| `download_images` | bool | `True` | Whether to download images locally |
| `output_dir` | str | `"output"` | Directory for JSON output files |
| `base_images_dir` | str | `"batch_images"` | Directory for downloaded images |
| `max_retries` | int | `5` | Maximum retry attempts for failed requests |

### RAG Processor Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chunk_size` | int | `1000` | Text chunk size for splitting |
| `chunk_overlap` | int | `200` | Overlap between chunks |
| `use_vision_model` | bool | `True` | Enable vision analysis for images |
| `verbose` | bool | `True` | Enable detailed logging |

### Streamlit App Configuration

| Parameter | Description | Default | Customization |
|-----------|-------------|---------|---------------|
| `vectorstore_path` | Path to vector database | `"./chroma_db_final"` | Update in `streamlit_app.py` |
| `docstore_path` | Path to document store | `"./docstore_final"` | Update in `streamlit_app.py` |
| `use_vision_model` | Enable image processing | `True` | Set to `False` to disable |
| `num_results` | Default results count | `5` | Adjustable via slider |

### Parallel Processing Configuration

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `max_workers` | Number of parallel threads | `min(api_keys, 5)` | 3-8 |
| `max_retries` | Retry attempts per URL | 3 | 3-5 |
| `cooldown_time` | API key cooldown (seconds) | 30 | 30-60 |

## 📊 Data Structure

### Scraped JSON Output

```json
{
  "id": "issue-123",
  "date": "2024-01-15",
  "url": "https://www.deeplearning.ai/the-batch/issue-123/",
  "content": {
    "letter": {
      "text": "Andrew Ng's letter content with [links](https://example.com) preserved...",
      "images": [
        {
          "path": "batch_images/issue-123/abc123_image_description.jpg",
          "alt": "Image description"
        }
      ],
      "author": "Andrew Ng"
    },
    "news": [
      {
        "title": "AI News Headline",
        "text": "News content with [preserved links](https://example.com)...",
        "images": [
          {
            "path": "batch_images/issue-123/def456_news_image.png",
            "alt": "News image description"
          }
        ]
      }
    ]
  }
}
```

### RAG Query Results

```python
{
    'answer': 'Generated comprehensive answer',
    'sources': [
        {
            'ref_id': '[1]',
            'content_type': 'Letter Text',
            'date': '2024-01-15',
            'issue_id': 'batch_001',
            'source_url': 'https://...',
            'content_preview': 'Preview of content...',
            'news_title': 'Title if from news item'
        }
    ],
    'images': [
        {
            'path': '/path/to/image.jpg',
            'description': 'AI-generated image description',
            'ref_id': '[2]',
            'source_url': 'https://...',
            'date': '2024-01-15'
        }
    ],
    'num_results': 5
}
```

## 🔧 Core Components

### 1. BatchScraper
**Purpose**: Web scraping and content extraction
- Identifies different content sections (letters vs news)
- Downloads and organizes images
- Filters out advertisements and low-quality content
- Handles rate limiting and retry logic

### 2. DataProcessor
**Purpose**: Multimodal RAG database creation
- Processes both text and image content
- Uses Google's Gemini models for embeddings and vision analysis
- Stores summaries in vector database for search
- Maintains full content for context retrieval

### 3. SmartAPIKeyManager
**Purpose**: Intelligent API key rotation and failure handling
- Automatic failover between keys
- Cooldown management for rate-limited keys
- Load balancing based on usage patterns
- Comprehensive failure tracking

### 4. ParallelRAGProcessor
**Purpose**: High-performance parallel processing
- Thread-local processor management
- Independent processing pipelines per thread
- Optimal resource utilization
- Real-time progress monitoring

### 5. Streamlit Web Interface
**Purpose**: User-friendly query interface
- Real-time search and results display
- Image gallery with AI-generated descriptions
- Source attribution and reference tracking
- Responsive design for desktop and mobile

## 🔍 Content Processing Logic

### Section Identification
1. **Andrew Ng's Letter**: The main editorial content
2. **News Sections**: Individual AI news stories
3. **Filtered Content**: Advertisements, sponsors, and promotional content

### Image Assignment Strategy
- **Letter images**: Assigned to Andrew Ng's letter section
- **Section images**: Assigned to the corresponding news section
- **Orphaned images**: Assigned to the last relevant section

### Multimodal Analysis
- **Text Processing**: Generates searchable summaries using Gemini text models
- **Vision Analysis**: Uses Gemini Vision models to analyze and describe images
- **Content Categorization**: Automatically categorizes into content types:
  - `letter_text_summary` / `letter_text`
  - `letter_image_summary` / `letter_image`
  - `news_text_summary` / `news_text`
  - `news_image_summary` / `news_image`

## 📈 Monitoring and Statistics

### Collection Management
```python
# Get collection statistics
stats = processor.get_collection_stats()
print(f"Total documents: {stats['total_documents']}")
print(f"Content types: {stats['content_types']}")

# Clear collection if needed
processor.clear_collection()
```

### Web Interface Analytics
- **Query Response Times**: Monitor search performance
- **User Interaction Patterns**: Track popular queries and results
- **Error Rates**: Monitor API and processing failures
- **Resource Usage**: Memory and CPU utilization during searches

## 🛠️ API Reference

### BatchScraper Methods

#### `process_single_issue(url)`
Processes a single newsletter issue and returns structured data.
- **Parameters**: `url` (str) - The URL of the newsletter issue
- **Returns**: `dict` - Structured issue data or `None` if failed

#### `process_multiple_issues(urls)`
Processes multiple newsletter issues with rate limiting.
- **Parameters**: `urls` (list) - List of URLs to process
- **Returns**: `tuple` - (successful_results, failed_urls)

#### `get_all_issue_links(max_articles=None)`
Discovers all available newsletter issues from the archive.
- **Parameters**: `max_articles` (int, optional) - Maximum number of articles
- **Returns**: `list` - List of discovered issue URLs

### DataProcessor Methods

#### `process_data(json_data=None)`
Process data from files or direct input into RAG database.
- **Parameters**: `json_data` (optional) - Direct JSON input
- **Returns**: Processing statistics

#### `query(query, k=5, generate_answer=True)`
Query the RAG system for relevant content.
- **Parameters**: 
  - `query` (str) - Search query
  - `k` (int) - Number of results to retrieve
  - `generate_answer` (bool) - Whether to generate AI answer
- **Returns**: `dict` - Query results with answer and sources

#### `get_collection_stats()`
Get statistics about the current collection.
- **Returns**: `dict` - Collection statistics

### Streamlit App Functions

#### `load_rag_processor()`
Cached function to initialize and return the RAG processor.
- **Returns**: `DataProcessor` instance or `None` if failed

#### `main()`
Main application function that renders the UI and handles user interactions.

## 🚨 Error Handling

### Common Issues and Solutions

**API Rate Limits**
- System automatically switches between available API keys
- Implements cooldown periods for rate-limited keys
- Provides detailed error tracking and statistics

**Image Processing Failures**
- Falls back to text-based summaries using alt text and captions
- Supports both local files and HTTP URLs
- Continues processing if individual images fail

**Memory Issues**
- Reduce `max_workers` parameter
- Process data in smaller batches
- Monitor system resource usage

**Network Errors**
- Automatic retry with exponential backoff
- Configurable retry attempts
- Graceful degradation for failed requests

**Streamlit-Specific Issues**
- **Import Errors**: Ensure all dependencies are installed
- **Path Issues**: Verify vectorstore and docstore paths exist
- **API Key Issues**: Check environment variables and API quotas
- **Image Display Issues**: Verify image paths and file permissions

## 📁 File Organization

```
project/
├── .env                        # API keys and configuration
├── streamlit_app.py           # Web interface application
├── output/                     # JSON output files
│   ├── issue-123.json
│   └── issue-124.json
├── chroma_db_final/           # Vector database storage
├── docstore_final/            # Document storage
├── batch_images/              # Downloaded images
│   ├── issue-123/
│   │   ├── abc123_image1.jpg
│   │   └── def456_image2.png
│   └── issue-124/
│       └── ghi789_image3.webp
├── batch_scraper.py           # Main scraper code
├── data_processor.py          # RAG processing code
├── retriever.py      # Parallel processing engine
└── requirements.txt           # Python dependencies
```

## 🔧 Supported Formats

### Image Formats
- `.jpg` and `.jpeg`
- `.png`
- `.webp`

### Content Types
- Newsletter letters (Andrew Ng's editorial content)
- News articles and updates
- Images with alt text and captions
- Embedded links and references

### Models Used
**Text Models** (automatically rotated):
- `gemma-3-27b-it`
- `gemma-3-12b-it`
- `gemma-3-4b-it`
- `gemma-3-1b-it`

**Vision Models**:
- `gemini-2.0-flash-lite` (default)
- `gemini-2.0-flash`