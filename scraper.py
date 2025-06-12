import requests
from bs4 import BeautifulSoup
import time
import os
import json
import re
from urllib.parse import urljoin, urlparse, parse_qs, urlunparse
from pathlib import Path
from datetime import datetime
import hashlib
from dateutil import parser

class BatchScraper:
    def __init__(self, base_url="https://www.deeplearning.ai", delay=5, download_images=True, output_dir="output", base_images_dir="batch_images", max_retries=5):
        self.base_url = base_url
        self.delay = delay
        self.download_images = download_images
        self.output_dir = output_dir
        self.base_images_dir = base_images_dir
        self.max_retries = max_retries  # Додано
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        
        # Create base images directory if it doesn't exist
        if self.download_images:
            Path(self.base_images_dir).mkdir(exist_ok=True)

    def make_request_with_retry(self, url, **kwargs):
        """Make HTTP request with retry mechanism"""
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, **kwargs)
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                print(f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    print(f"Retrying in {self.delay} seconds...")
                    time.sleep(self.delay)
                else:
                    print(f"Max retries reached for {url}")
                    raise
    
    def get_issue_id_from_url(self, url):
        """Extract issue ID from URL for naming"""
        try:
            parsed = urlparse(url)
            path_parts = [part for part in parsed.path.split('/') if part]
            
            # Look for issue-related part
            for part in path_parts:
                if 'issue-' in part.lower():
                    return part
            
            # Fallback: use the last meaningful part of the path
            if path_parts:
                return path_parts[-1]
            
            # Final fallback: use hash of URL
            return hashlib.md5(url.encode()).hexdigest()[:12]
            
        except Exception as e:
            print(f"Error extracting issue ID from {url}: {e}")
            return hashlib.md5(url.encode()).hexdigest()[:12]
    
    def clean_url(self, url):
        """Remove tracking parameters and clean up URLs"""
        if not url:
            return url
            
        parsed = urlparse(url)
        
        # Parameters to remove (common tracking parameters)
        tracking_params = {
            'utm_campaign', 'utm_source', 'utm_medium', 'utm_content', 'utm_term',
            '_hsenc', '_hsmi', 'hsCtaTracking', 'ref', 'source', 'campaign_id',
            'fbclid', 'gclid', 'dclid', 'msclkid', 'twclid', 'li_fat_id'
        }
        
        query_params = parse_qs(parsed.query, keep_blank_values=True)
        clean_params = {k: v for k, v in query_params.items() 
                        if k not in tracking_params}
        
        clean_query = '&'.join([f"{k}={'&'.join(v)}" for k, v in clean_params.items()])
        
        clean_url = urlunparse((
            parsed.scheme, parsed.netloc, parsed.path, 
            parsed.params, clean_query, parsed.fragment
        ))
        
        return clean_url

    def download_image(self, img_url, article_url, img_element=None, issue_dir=None):
        """Download an image and return the relative path"""
        try:
            # Make image URL absolute
            if img_url.startswith('//'):
                img_url = 'https:' + img_url
            elif img_url.startswith('/'):
                img_url = urljoin(self.base_url, img_url)
            elif not img_url.startswith(('http://', 'https://')):
                img_url = urljoin(article_url, img_url)
            
            # Get image response
            img_response = self.make_request_with_retry(img_url, stream=True, timeout=30)
            img_response.raise_for_status()
            
            # Determine file extension
            content_type = img_response.headers.get('content-type', '').lower()
            if 'jpeg' in content_type or 'jpg' in content_type:
                ext = '.jpg'
            elif 'png' in content_type:
                ext = '.png'
            elif 'webp' in content_type:
                ext = '.webp'
            else:
                # Try to get extension from URL
                parsed_url = urlparse(img_url)
                path_ext = os.path.splitext(parsed_url.path)[1].lower()
                ext = path_ext if path_ext in ['.jpg', '.jpeg', '.png', '.webp'] else '.jpg'
            
            # Create filename
            url_hash = hashlib.md5(img_url.encode()).hexdigest()[:12]
            
            # Try to get alt text for better filename
            alt_text = ""
            if img_element:
                alt_text = img_element.get('alt', '')
                if alt_text:
                    alt_text = re.sub(r'[^\w\s-]', '', alt_text)[:50]
                    alt_text = re.sub(r'\s+', '_', alt_text.strip())
                    alt_text = f"_{alt_text}" if alt_text else ""
            
            filename = f"{url_hash}{alt_text}{ext}"
            filepath = os.path.join(issue_dir, filename)
            
            # Don't download if already exists
            if os.path.exists(filepath):
                print(f"Image already exists: {filename}")
                return os.path.relpath(filepath, os.getcwd())
            
            # Save image
            with open(filepath, 'wb') as f:
                for chunk in img_response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"Downloaded image: {filename}")
            return os.path.relpath(filepath, os.getcwd())
            
        except Exception as e:
            print(f"Error downloading image {img_url}: {e}")
            return None

    def extract_date_from_page(self, soup, url):
        """Extract publication date from page metadata or content"""
        try:
            # Try meta tags first
            date_selectors = [
                'meta[property="article:published_time"]',
                'meta[name="publish_date"]',
                'meta[name="date"]',
                'time[datetime]',
                '.post-date',
                '.publish-date'
            ]
            
            for selector in date_selectors:
                element = soup.select_one(selector)
                if element:
                    date_value = element.get('content') or element.get('datetime') or element.get_text(strip=True)
                    if date_value:
                        # Try to parse the date
                        try:
                            parsed_date = parser.parse(date_value)
                            return parsed_date.strftime('%Y-%m-%d')
                        except:
                            pass
            
            # Fallback: try to extract from URL
            url_match = re.search(r'/(\d{4})/(\d{2})/(\d{2})/', url)
            if url_match:
                year, month, day = url_match.groups()
                return f"{year}-{month}-{day}"
            
            # Final fallback: current date
            return datetime.now().strftime('%Y-%m-%d')
            
        except Exception as e:
            print(f"Error extracting date: {e}")
            return datetime.now().strftime('%Y-%m-%d')

    def should_ignore_section(self, header_text):
        """Check if a section should be ignored based on header content"""
        header_text = header_text.upper()
        ignore_patterns = [
            "A MESSAGE FROM",
            "ADVERTISEMENT", 
            "SPONSOR",
            "DEEPLEARNING.AI",
            "DATA POINTS",
            "HELP US"
        ]
        
        return any(pattern in header_text for pattern in ignore_patterns)

    def has_meaningful_text(self, text, min_words=3):
        """Check if text has meaningful content (not just whitespace/minimal text)"""
        if not text or not text.strip():
            return False
        
        # Remove common non-meaningful text patterns
        cleaned_text = re.sub(r'\s+', ' ', text.strip())
        
        # Split into words and filter out very short words
        words = [word for word in cleaned_text.split() if len(word) > 1]
        
        return len(words) >= min_words

    def extract_text_with_links(self, element):
        """Extract text from element while preserving links"""
        if not element:
            return ""
        
        parts = []
        # Iterate over contents of the element, not just descendants, to handle direct text
        for content in element.contents:
            if isinstance(content, str):
                text = content.strip()
                if text:
                    parts.append(text)
            elif content.name == 'a':
                link_text = content.get_text(strip=True)
                href = content.get('href', '')
                if href and link_text:
                    clean_href = self.clean_url(href)
                    # Make URL absolute if needed
                    if href.startswith('/'):
                        clean_href = self.base_url + href
                    parts.append(f'[{link_text}]({clean_href})')
                elif link_text:
                    parts.append(link_text)
            else: # Recursively process other tags like p, li, div etc.
                parts.append(self.extract_text_with_links(content))
        
        return ' '.join(parts).strip()

    def collect_images_info_from_section(self, section_element, article_url):
        """Collect image information from a section WITHOUT downloading"""
        images_info = []
        if not section_element:
            return images_info
        
        # Supported image formats
        supported_formats = {'.jpg', '.jpeg', '.png', '.webp'}
        
        img_elements = section_element.find_all('img')
        for img in img_elements:
            img_src = img.get('src') or img.get('data-src')
            if not img_src:
                continue
            
            # Make image URL absolute
            if img_src.startswith('//'):
                img_src = 'https:' + img_src
            elif img_src.startswith('/'):
                img_src = urljoin(self.base_url, img_src)
            elif not img_src.startswith(('http://', 'https://')):
                img_src = urljoin(article_url, img_src)
            
            # Check if image format is supported
            # Extract file extension from URL (handle query parameters)
            parsed_url = urlparse(img_src)
            path = parsed_url.path.lower()
            
            # Get extension from path
            _, ext = os.path.splitext(path)
            
            # If no extension in path, try to get it from query parameters
            if not ext and parsed_url.query:
                # Sometimes format is specified in query params like ?format=jpeg
                query_params = parse_qs(parsed_url.query)
                if 'format' in query_params:
                    ext = '.' + query_params['format'][0].lower()
                elif any(fmt in parsed_url.query.lower() for fmt in ['jpg', 'jpeg', 'png', 'webp']):
                    # Look for format hints in query string
                    for fmt in ['jpg', 'jpeg', 'png', 'webp']:
                        if fmt in parsed_url.query.lower():
                            ext = '.' + fmt
                            break
            
            if ext not in supported_formats:
                continue
            
            img_info = {
                'src': img_src,  # Keep original URL for later download
                'alt': img.get('alt', ''),
                'element': img,  # Keep reference to element for download
            }
            
            images_info.append(img_info)
        
        return images_info

    def download_images_from_final_structure(self, result_data, article_url, issue_dir):
        """Download only the images that are included in the final JSON structure"""
        if not self.download_images:
            # If downloading is disabled, just use the original URLs
            self.convert_image_info_to_urls_only(result_data)
            return result_data
        
        downloaded_count = 0
        
        # Download images from letter section
        for i, img_info in enumerate(result_data['content']['letter']['images']):
            local_path = self.download_image(
                img_info['src'], 
                article_url, 
                img_info.get('element'), 
                issue_dir
            )
            if local_path:
                result_data['content']['letter']['images'][i]['path'] = local_path
                downloaded_count += 1
            else:
                result_data['content']['letter']['images'][i]['path'] = img_info['src']
            
            # Clean up temporary data
            if 'src' in result_data['content']['letter']['images'][i]:
                del result_data['content']['letter']['images'][i]['src']
            if 'element' in result_data['content']['letter']['images'][i]:
                del result_data['content']['letter']['images'][i]['element']
        
        # Download images from news sections
        for news_idx, news_section in enumerate(result_data['content']['news']):
            for img_idx, img_info in enumerate(news_section['images']):
                local_path = self.download_image(
                    img_info['src'], 
                    article_url, 
                    img_info.get('element'), 
                    issue_dir
                )
                if local_path:
                    result_data['content']['news'][news_idx]['images'][img_idx]['path'] = local_path
                    downloaded_count += 1
                else:
                    result_data['content']['news'][news_idx]['images'][img_idx]['path'] = img_info['src']
                
                # Clean up temporary data
                if 'src' in result_data['content']['news'][news_idx]['images'][img_idx]:
                    del result_data['content']['news'][news_idx]['images'][img_idx]['src']
                if 'element' in result_data['content']['news'][news_idx]['images'][img_idx]:
                    del result_data['content']['news'][news_idx]['images'][img_idx]['element']
        
        print(f"Downloaded {downloaded_count} images that are included in final JSON")
        return result_data

    def convert_image_info_to_urls_only(self, result_data):
        """Convert image info to use original URLs when download is disabled"""
        # Process letter images
        for i, img_info in enumerate(result_data['content']['letter']['images']):
            result_data['content']['letter']['images'][i]['path'] = img_info['src']
            # Clean up temporary data
            if 'src' in result_data['content']['letter']['images'][i]:
                del result_data['content']['letter']['images'][i]['src']
            if 'element' in result_data['content']['letter']['images'][i]:
                del result_data['content']['letter']['images'][i]['element']
        
        # Process news images
        for news_idx, news_section in enumerate(result_data['content']['news']):
            for img_idx, img_info in enumerate(news_section['images']):
                result_data['content']['news'][news_idx]['images'][img_idx]['path'] = img_info['src']
                # Clean up temporary data
                if 'src' in result_data['content']['news'][news_idx]['images'][img_idx]:
                    del result_data['content']['news'][news_idx]['images'][img_idx]['src']
                if 'element' in result_data['content']['news'][news_idx]['images'][img_idx]:
                    del result_data['content']['news'][news_idx]['images'][img_idx]['element']

    def extract_issue_data(self, url):
        """Extract structured data from a single issue URL"""
        try:
            response = self.make_request_with_retry(url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract basic metadata
            issue_id = self.get_issue_id_from_url(url)
            date = self.extract_date_from_page(soup, url)
            
            # Create issue directory for images
            issue_dir = None
            if self.download_images:
                issue_dir = os.path.join(self.base_images_dir, issue_id)
                Path(issue_dir).mkdir(parents=True, exist_ok=True)
                print(f"Created/using directory: {issue_dir}")
            
            # Find main content area
            content_selectors = [
                '.prose--styled',
                '.post_postContent__wGZtc', 
                'article .container--boxed',
                'article',
                '.article-content',
                '.post-content',
                'main'
            ]
            
            main_content = None
            for selector in content_selectors:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            
            if not main_content:
                main_content = soup.find('body') # Fallback to body if no specific content area found
            
            # Remove unwanted elements
            unwanted_selectors = [
                'nav', 'header', 'footer', '.nav', '.navigation', '.menu',
                '.advertisement', '.ads', '.social-share', 'script', 'style', 
                'noscript', '[data-sentry-component="ShareIcons"]',
                '[data-sentry-component="MetaItem"]',
                '#elevenlabs-audionative-widget',
                'form', '.related-articles', '.post-navigation' # Added more common unwanted elements
            ]
            
            for selector in unwanted_selectors:
                for element in main_content.select(selector):
                    element.decompose()
            
            # Initialize result structure
            result = {
                'id': issue_id,
                'date': date,
                'url': url,
                'content': {
                    'letter': {
                        'text': '',
                        'images': [],
                        'author': 'Andrew Ng'
                    },
                    'news': []
                }
            }
            
            # Parse sections and collect image info (but don't download yet)
            sections = []
            current_section_elements = []
            current_section_title = "Andrew Ng's Letter" # Default for the first section

            # Iterate over all direct children of main_content
            for element in main_content.children:
                # Only process actual tags (not NavigableString for whitespace)
                if hasattr(element, 'name') and element.name:
                    if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                        header_text = element.get_text(strip=True)
                        
                        # Process the previous section if it exists
                        if current_section_elements:
                            section_container = soup.new_tag('div')
                            for el in current_section_elements:
                                section_container.append(BeautifulSoup(str(el), 'html.parser').find())

                            # Collect image info from current section (without downloading)
                            section_images_info = self.collect_images_info_from_section(section_container, url)
                            section_text = self.extract_text_with_links(section_container)
                            
                            sections.append({
                                'title': current_section_title,
                                'text': section_text,
                                'images_info': section_images_info,  # Changed from 'images' to 'images_info'
                                'is_letter': current_section_title == "Andrew Ng's Letter",
                                'should_ignore': self.should_ignore_section(current_section_title),
                                'has_content': self.has_meaningful_text(section_text)
                            })
                        
                        # Start a new section
                        current_section_title = header_text
                        current_section_elements = []
                    else:
                        current_section_elements.append(element)
            
            # Process the last collected section after the loop finishes
            if current_section_elements:
                section_container = soup.new_tag('div')
                for el in current_section_elements:
                    section_container.append(BeautifulSoup(str(el), 'html.parser').find())

                section_images_info = self.collect_images_info_from_section(section_container, url)
                section_text = self.extract_text_with_links(section_container)
                
                sections.append({
                    'title': current_section_title,
                    'text': section_text,
                    'images_info': section_images_info,  # Changed from 'images' to 'images_info'
                    'is_letter': current_section_title == "Andrew Ng's Letter",
                    'should_ignore': self.should_ignore_section(current_section_title),
                    'has_content': self.has_meaningful_text(section_text)
                })

            # If no headers found at all, treat everything as Andrew's letter
            if not sections:
                letter_images_info = self.collect_images_info_from_section(main_content, url)
                letter_text = self.extract_text_with_links(main_content)
                sections.append({
                    'title': "Andrew Ng's Letter",
                    'text': letter_text,
                    'images_info': letter_images_info,  # Changed from 'images' to 'images_info'
                    'is_letter': True,
                    'should_ignore': False,
                    'has_content': self.has_meaningful_text(letter_text)
                })
            
            # Now reassign image info: images from each section go to the NEXT news section
            # (except for the letter section and ignored sections)
            # FILTER OUT EMPTY SECTIONS HERE
            pending_images_info = []
            
            for i, section in enumerate(sections):
                if section['is_letter']:
                    # Letter keeps its own images (even if empty - we always want the letter section)
                    result['content']['letter']['text'] = section['text']
                    result['content']['letter']['images'] = section['images_info']
                elif section['should_ignore']:
                    # Ignored sections: ignore only the FIRST image, pass the rest to pending
                    images_to_ignore = len(section['images_info'])
                    if images_to_ignore > 0:
                        # Ignore first image, keep the rest for next section
                        images_to_keep = section['images_info'][1:]  # Skip first image
                        pending_images_info.extend(images_to_keep)
                        print(f"Ignoring section '{section['title']}': ignored 1 image, kept {len(images_to_keep)} images for next section")
                    else:
                        print(f"Ignoring section '{section['title']}': no images to process")
                elif not section['has_content']:
                    # FILTER: Skip sections without meaningful text content
                    print(f"Skipping empty section: {section['title']}")
                    # Add this section's images to pending for next non-empty section
                    pending_images_info.extend(section['images_info'])
                else:
                    # This is a news section with meaningful content
                    news_section = {
                        'title': section['title'],
                        'text': section['text'],
                        'images': pending_images_info.copy()  # Assign pending images from previous sections
                    }
                    result['content']['news'].append(news_section)
                    
                    # Current section's images become pending for the NEXT news section
                    pending_images_info = section['images_info'].copy()
            
            # If there are leftover pending images after processing all sections,
            # assign them to the last news section (if any exists)
            if pending_images_info and result['content']['news']:
                result['content']['news'][-1]['images'].extend(pending_images_info)
                print(f"Assigned {len(pending_images_info)} leftover images to last news section: {result['content']['news'][-1]['title']}")
            
            print(f"Processed {len(sections)} total sections, kept {len(result['content']['news'])} news sections")
            
            # NOW download only the images that are included in the final structure
            result = self.download_images_from_final_structure(result, url, issue_dir)
            
            return result
            
        except Exception as e:
            print(f"Error extracting data from {url}: {e}")
            return None

    def save_issue_json(self, data, filepath):
        """Save issue data as JSON file"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"Saved JSON: {filepath}")
            return True
            
        except Exception as e:
            print(f"Error saving JSON to {filepath}: {e}")
            return False

    def process_single_issue(self, url):
        output_dir = self.output_dir
        """Complete pipeline for processing a single issue"""
        print(f"Processing issue: {url}")
        
        # Extract data
        issue_data = self.extract_issue_data(url)
        if not issue_data:
            print(f"Failed to extract data from {url}")
            return None
        
        # Save JSON
        json_filename = f"{issue_data['id']}.json"
        json_filepath = os.path.join(output_dir, json_filename)
        
        if self.save_issue_json(issue_data, json_filepath):
            print(f"Successfully processed issue {issue_data['id']}")
            return issue_data
        else:
            print(f"Failed to save issue {issue_data['id']}")
            return None

    def process_multiple_issues(self, urls, output_dir="output", max_concurrent=3):
        """Process multiple issues with rate limiting"""
        results = []
        failed_urls = []
        
        print(f"Processing {len(urls)} issues...")
        
        for i, url in enumerate(urls, 1):
            print(f"\n--- Processing {i}/{len(urls)} ---")
            
            try:
                result = self.process_single_issue(url, output_dir)
                if result:
                    results.append(result)
                else:
                    failed_urls.append(url)
                    
            except Exception as e:
                print(f"Error processing {url}: {e}")
                failed_urls.append(url)
            
            # Rate limiting
            if i < len(urls):  # Don't sleep after the last URL
                print(f"Waiting {self.delay} seconds before next request...")
                time.sleep(self.delay)
        
        # Summary
        print(f"\n=== PROCESSING COMPLETE ===")
        print(f"Successfully processed: {len(results)}")
        print(f"Failed: {len(failed_urls)}")
        
        if failed_urls:
            print("Failed URLs:")
            for url in failed_urls:
                print(f"   - {url}")
        
        return results, failed_urls

    def get_all_issue_links(self, max_articles=None):
        """Get all issue links from the batch archive"""
        article_links = []
        page = 1

        while True:
            try:
                if max_articles and len(article_links) >= max_articles:
                    print(f"Reached limit of {max_articles} articles")
                    break
                
                page_url = f"{self.base_url}/the-batch/page/{page}/"
                print(f"Scanning page {page}: {page_url}")
                
                try:
                    response = self.make_request_with_retry(page_url, timeout=30)
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 404:
                        print(f"Page {page} not found (404), stopping scan")
                        break
                    else:
                        raise
                
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find article links
                # More specific selector for actual article links within the listing
                links = soup.select('a.PostGridItem_link__EwB6W') # Common class for deeplearning.ai batch links
                if not links:
                    links = soup.find_all('a', href=True) # Fallback to general links if specific class not found

                page_links = []
                
                for link in links:
                    href = link.get('href')
                    if (href and '/the-batch/' in href and 
                        href not in ['/the-batch/', '/the-batch/about/'] and
                        '/tag/' not in href and '/page/' not in href):
                        
                        # Build full URL
                        if href.startswith('/'):
                            full_url = self.base_url + href
                        else:
                            full_url = urljoin(page_url, href)
                        
                        clean_url = self.clean_url(full_url)
                        if clean_url not in article_links:
                            article_links.append(clean_url)
                            page_links.append(clean_url)
                
                print(f"Found {len(page_links)} new articles on page {page}")
                
                if not page_links and page > 1: # If no new links on a subsequent page, we've likely hit the end
                    print(f"No new articles on page {page}, stopping")
                    break
                
                if max_articles and len(article_links) >= max_articles:
                    break
                
                page += 1
                time.sleep(self.delay)
                
            except Exception as e:
                print(f"Error scanning page {page}: {e}")
                break
        
        # Trim to exact limit if exceeded
        if max_articles and len(article_links) > max_articles:
            article_links = article_links[:max_articles]
        
        print(f"Total collected: {len(article_links)} articles")
        return article_links
