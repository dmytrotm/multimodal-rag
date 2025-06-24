from scraper import BatchScraper
from data_processor import DataProcessor

from dotenv import load_dotenv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

from key_manager import SmartAPIKeyManager

# Load environment variables
load_dotenv()

# Load multiple API keys
api_keys = [
    os.getenv("GOOGLE_API_KEY_1"),
    os.getenv("GOOGLE_API_KEY_2"),
    os.getenv("GOOGLE_API_KEY_3"),
    os.getenv("GOOGLE_API_KEY_4"),
    os.getenv("GOOGLE_API_KEY_5"),
    # Add more keys as needed
]

# Filter out None values and ensure we have at least one key
api_keys = [key for key in api_keys if key is not None]
if not api_keys:
    raise ValueError("No valid API keys found! Please check your environment variables.")

print(f"Found {len(api_keys)} API keys for parallel processing")

class ParallelRAGProcessor:
    def __init__(self, api_keys, max_workers=None):
        self.api_key_manager = SmartAPIKeyManager(api_keys)
        self.max_workers = max_workers or min(len(api_keys), 5)
        
        # Only lock for stats, nothing else
        self.stats_lock = threading.Lock()
        self.processed_count = 0
        self.failed_count = 0
        
        # Thread-local storage for processors
        self.thread_local = threading.local()
    
    def get_thread_processor(self, api_key):
        """Get or create a processor for the current thread"""
        if not hasattr(self.thread_local, 'processor') or self.thread_local.api_key != api_key:
            print(f"[Thread {threading.current_thread().name}] Creating thread-local DataProcessor with API key: {api_key[:10]}...")
            self.thread_local.processor = DataProcessor(
                output_dir=None,
                vectorstore_path="./chroma_db_final",
                docstore_path="./docstore_final",
                use_vision_model=True,
                verbose=False,
                google_api_key=api_key
            )
            self.thread_local.api_key = api_key
        return self.thread_local.processor
    
    def process_single_issue_for_rag(self, scraper, url, issue_index, total_issues):
        """Process a single issue with thread-local processors"""
        thread_name = threading.current_thread().name
        max_retries = 3
        
        for attempt in range(max_retries):
            api_key = self.api_key_manager.get_best_api_key()
            
            try:
                print(f"\n[Thread {thread_name}] Processing issue {issue_index}/{total_issues} (attempt {attempt + 1})")
                print(f"[Thread {thread_name}] Using API key: {api_key[:10]}... for {url}")
                
                # Scrape the issue
                result = scraper.extract_issue_data(url)
                
                if result:
                    print(f"[Thread {thread_name}] ✅ Successfully scraped issue: {result['id']}")
                    
                    # Get thread-local processor
                    processor = self.get_thread_processor(api_key)
                    
                    # TRUE PARALLEL RAG PROCESSING - NO LOCKS!
                    print(f"[Thread {thread_name}] 📚 Adding issue {result['id']} to RAG system...")
                    processor.process_json_data(
                        json_data=result,
                        append_to_existing=True
                    )
                    print(f"[Thread {thread_name}] ✅ Successfully added issue {result['id']} to RAG system")
                    
                    # Only lock for stats
                    with self.stats_lock:
                        self.processed_count += 1
                    
                    self.api_key_manager.mark_key_success(api_key)
                    
                    return {
                        'success': True,
                        'issue_id': result['id'],
                        'url': url,
                        'thread': thread_name,
                        'api_key': api_key[:10] + "...",
                        'attempt': attempt + 1
                    }
                else:
                    print(f"[Thread {thread_name}] ❌ Failed to scrape issue from {url}")
                    if attempt < max_retries - 1:
                        continue
                    else:
                        with self.stats_lock:
                            self.failed_count += 1
                        return {
                            'success': False,
                            'url': url,
                            'error': 'Failed to scrape issue after all retries',
                            'thread': thread_name,
                            'api_key': api_key[:10] + "...",
                            'attempts': max_retries
                        }
                        
            except Exception as e:
                error_msg = str(e)
                print(f"[Thread {thread_name}] ❌ Error processing {url}: {error_msg}")
                
                if any(keyword in error_msg.lower() for keyword in ['api key', 'quota', 'limit', 'permission', 'auth']):
                    self.api_key_manager.mark_key_failed(api_key)
                
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                else:
                    with self.stats_lock:
                        self.failed_count += 1
                    return {
                        'success': False,
                        'url': url,
                        'error': error_msg,
                        'thread': thread_name,
                        'api_key': api_key[:10] + "...",
                        'attempts': max_retries
                    }
        
        with self.stats_lock:
            self.failed_count += 1
        return {
            'success': False,
            'url': url,
            'error': 'All retry attempts failed',
            'thread': thread_name,
            'attempts': max_retries
        }

def main():
    if __name__ == "__main__":
        # Initialize scraper with retry mechanism
        scraper = BatchScraper(
            download_images=False, 
            base_images_dir="batch_images",
            max_retries=3,
        )
        
        # Test with multiple issues
        print("=== TESTING TRULY PARALLEL PROCESSING ===")
        
        try:
            # Get all issue URLs
            urls = scraper.get_all_issue_links()  
            print(f"Found {len(urls)} issues to process")
            
            # Initialize DataProcessor for initial setup
            print("\n=== INITIALIZING RAG DATA PROCESSOR ===")
            main_processor = DataProcessor(
                output_dir=None,
                vectorstore_path="./chroma_db_final",
                docstore_path="./docstore_final",
                use_vision_model=True,
                verbose=True,
                google_api_key=api_keys[0]
            )
            
            # Clear collection for fresh start
            try:
                current_count = main_processor.vectorstore._collection.count()
                print(f"\n📊 Current collection has {current_count} documents")
                main_processor.clear_collection()
                print("🗑️ Collection cleared for fresh start")
            except Exception as e:
                print(f"Collection status check: {e}")
            
            parallel_processor = ParallelRAGProcessor(api_keys, max_workers=min(len(api_keys), len(urls)))
            
            max_workers = parallel_processor.max_workers
            print(f"\n🚀 Starting TRULY parallel processing with {max_workers} workers and {len(api_keys)} API keys...")
            start_time = time.time()
            
            # Process issues in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_url = {}
                for i, url in enumerate(urls):
                    future = executor.submit(
                        parallel_processor.process_single_issue_for_rag, 
                        scraper, url, i+1, len(urls)
                    )
                    future_to_url[future] = url
                
                # Process completed tasks
                results = []
                for future in as_completed(future_to_url):
                    url = future_to_url[future]
                    try:
                        result = future.result()
                        results.append(result)
                        
                        if result['success']:
                            print(f"✅ Completed: {result['issue_id']} (Thread: {result['thread']}, API: {result['api_key']}, Attempt: {result['attempt']})")
                        else:
                            print(f"❌ Failed: {url} - {result['error']} (Attempts: {result.get('attempts', 'N/A')})")
                            
                    except Exception as e:
                        print(f"❌ Future exception for {url}: {str(e)}")
                        results.append({
                            'success': False,
                            'url': url,
                            'error': f"Future exception: {str(e)}"
                        })
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Summary
            print(f"\n{'='*60}")
            print("TRULY PARALLEL PROCESSING SUMMARY")
            print(f"{'='*60}")
            print(f"✅ Successfully processed: {parallel_processor.processed_count} issues")
            print(f"❌ Failed to process: {parallel_processor.failed_count} issues")
            print(f"📊 Total attempted: {len(urls)} issues")
            print(f"⏱️  Total processing time: {processing_time:.2f} seconds")
            print(f"🚀 Average time per issue: {processing_time/len(urls):.2f} seconds")
            print(f"👥 Used {max_workers} parallel workers with {len(api_keys)} API keys")
            
            if parallel_processor.processed_count > 0:
                print(f"\n✅ Truly parallel RAG system processing complete!")
                print(f"📚 {parallel_processor.processed_count} issues successfully added to RAG system")
            else:
                print(f"\n❌ No issues were successfully processed.")
                
        except Exception as e:
            print(f"❌ Fatal error in main execution: {str(e)}")