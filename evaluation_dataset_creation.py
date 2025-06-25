from langchain_google_genai import ChatGoogleGenerativeAI
import os
import pickle
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque

load_dotenv()

class SmartAPIKeyManager:
    def __init__(self, api_keys, queries_per_minute=30, safety_buffer=1.2):
        self.api_keys = api_keys
        self.queries_per_minute = queries_per_minute
        self.min_interval = (60.0 / queries_per_minute) * safety_buffer  # Minimum seconds between requests
        
        # Track request times for each key (using deque for efficient operations)
        self.key_request_times = {key: deque() for key in api_keys}
        self.key_failures = {key: 0 for key in api_keys}
        self.key_cooldown = {key: 0 for key in api_keys}  # Failure-based cooldown
        self.lock = threading.Lock()
        
        self.failure_cooldown_time = 60  # 60 seconds cooldown after failure
        
        print(f"📊 Rate limiting: {queries_per_minute} queries/min per key (min interval: {self.min_interval:.2f}s)")
    
    def _clean_old_requests(self, key, current_time):
        """Remove request times older than 1 minute"""
        request_times = self.key_request_times[key]
        while request_times and current_time - request_times[0] > 60:
            request_times.popleft()
    
    def _can_make_request(self, key, current_time):
        """Check if we can make a request with this key based on rate limits"""
        # Clean old requests
        self._clean_old_requests(key, current_time)
        
        # Check failure cooldown
        if current_time - self.key_cooldown[key] < self.failure_cooldown_time:
            return False, "failure_cooldown"
        
        request_times = self.key_request_times[key]
        
        # Check if we've hit the per-minute limit
        if len(request_times) >= self.queries_per_minute:
            return False, "rate_limit"
        
        # Check minimum interval since last request
        if request_times and current_time - request_times[-1] < self.min_interval:
            return False, "min_interval"
        
        return True, "ready"
    
    def get_best_api_key(self):
        """Get the best available API key, waiting if necessary"""
        while True:
            with self.lock:
                current_time = time.time()
                
                # Find all keys and their availability status
                key_status = {}
                for key in self.api_keys:
                    can_use, reason = self._can_make_request(key, current_time)
                    key_status[key] = {
                        'can_use': can_use,
                        'reason': reason,
                        'failures': self.key_failures[key],
                        'requests_count': len(self.key_request_times[key]),
                        'last_request': self.key_request_times[key][-1] if self.key_request_times[key] else 0
                    }
                
                # Find available keys
                available_keys = [key for key, status in key_status.items() if status['can_use']]
                
                if available_keys:
                    # Sort by failures (ascending), then by request count (ascending), then by last request time (ascending)
                    best_key = min(available_keys, key=lambda k: (
                        key_status[k]['failures'],
                        key_status[k]['requests_count'],
                        key_status[k]['last_request']
                    ))
                    
                    # Record this request
                    self.key_request_times[best_key].append(current_time)
                    return best_key
                
                # No keys available - calculate wait time
                wait_times = []
                for key, status in key_status.items():
                    if status['reason'] == 'failure_cooldown':
                        wait_time = self.failure_cooldown_time - (current_time - self.key_cooldown[key])
                        wait_times.append(wait_time)
                    elif status['reason'] == 'min_interval':
                        wait_time = self.min_interval - (current_time - status['last_request'])
                        wait_times.append(wait_time)
                    elif status['reason'] == 'rate_limit':
                        # Wait until the oldest request expires
                        oldest_request = self.key_request_times[key][0]
                        wait_time = 60 - (current_time - oldest_request)
                        wait_times.append(wait_time)
                
                min_wait = min(wait_times) if wait_times else 1.0
                min_wait = max(0.1, min_wait)  # At least 0.1 seconds
            
            # Wait outside the lock
            print(f"⏳ All API keys busy, waiting {min_wait:.1f}s...")
            time.sleep(min_wait)
    
    def mark_key_failed(self, api_key):
        """Mark a key as failed"""
        with self.lock:
            self.key_failures[api_key] += 1
            self.key_cooldown[api_key] = time.time()
            print(f"🔑 API key {api_key[:10]}... failed (total failures: {self.key_failures[api_key]}, cooldown: {self.failure_cooldown_time}s)")
    
    def mark_key_success(self, api_key):
        """Mark a key as successful (reduces failure count)"""
        with self.lock:
            if self.key_failures[api_key] > 0:
                self.key_failures[api_key] = max(0, self.key_failures[api_key] - 1)
    
    def get_stats(self):
        """Get statistics for all API keys"""
        with self.lock:
            current_time = time.time()
            stats = {}
            
            for key in self.api_keys:
                self._clean_old_requests(key, current_time)
                can_use, reason = self._can_make_request(key, current_time)
                
                request_times = self.key_request_times[key]
                
                stats[key[:10] + "..."] = {
                    'status': '✅ READY' if can_use else f'⏳ {reason.upper()}',
                    'requests_last_minute': len(request_times),
                    'failures': self.key_failures[key],
                    'last_request_ago': f"{current_time - request_times[-1]:.1f}s" if request_times else "never",
                    'next_available_in': self._get_next_available_time(key, current_time)
                }
            
            return stats
    
    def _get_next_available_time(self, key, current_time):
        """Calculate when this key will next be available"""
        can_use, reason = self._can_make_request(key, current_time)
        if can_use:
            return "now"
        
        if reason == 'failure_cooldown':
            return f"{self.failure_cooldown_time - (current_time - self.key_cooldown[key]):.1f}s"
        elif reason == 'min_interval':
            last_request = self.key_request_times[key][-1]
            return f"{self.min_interval - (current_time - last_request):.1f}s"
        elif reason == 'rate_limit':
            oldest_request = self.key_request_times[key][0]
            return f"{60 - (current_time - oldest_request):.1f}s"
        
        return "unknown"

# QA generation prompt template
QA_generation_prompt = """
Your task is to write a factoid question and an answer given a context.
Your factoid question should be answerable with a specific, concise piece of factual information from the context.
Your factoid question should be formulated in the same style as questions users could ask in a search engine.
This means that your factoid question MUST NOT mention something like "according to the passage" or "context".

Provide your answer as follows:

Output:::
Factoid question: (your factoid question)
Answer: (your answer to the factoid question)

Now here is the context.

Context: {context}
Output:::"""

# Initialize API key manager with rate limiting
API_KEYS = [
    os.getenv("GOOGLE_API_KEY_1"),
    os.getenv("GOOGLE_API_KEY_2"),
    os.getenv("GOOGLE_API_KEY_3"),
    os.getenv("GOOGLE_API_KEY_4"),
    os.getenv("GOOGLE_API_KEY_5"),
]

# Filter out None values
API_KEYS = [key for key in API_KEYS if key is not None]

if not API_KEYS:
    raise ValueError("No API keys found! Please set GOOGLE_API_KEY_1, GOOGLE_API_KEY_2, etc. in your .env file")

# Configure rate limiting - adjust these values based on your API limits
QUERIES_PER_MINUTE = 30  # Adjust based on your Gemini API quota
SAFETY_BUFFER = 1.2      # 20% safety buffer

print(f"🔑 Initialized with {len(API_KEYS)} API keys")
api_key_manager = SmartAPIKeyManager(API_KEYS, QUERIES_PER_MINUTE, SAFETY_BUFFER)

def load_pickled_data_from_directory(directory_path):
    """Load pickled data from files in a directory"""
    import glob
    
    try:
        # Find all files in the directory (no extension)
        pickle_files = []
        
        # Look for files without extensions first
        for file_path in glob.glob(os.path.join(directory_path, "*")):
            if os.path.isfile(file_path) and "." not in os.path.basename(file_path):
                pickle_files.append(file_path)
        
        # If no files without extensions, look for .pkl files
        if not pickle_files:
            pickle_files = glob.glob(os.path.join(directory_path, "*.pkl"))
        
        # If still no files, get all files
        if not pickle_files:
            pickle_files = [f for f in glob.glob(os.path.join(directory_path, "*")) if os.path.isfile(f)]
        
        print(f"Found {len(pickle_files)} files in {directory_path}")
        for f in pickle_files:
            print(f"  - {os.path.basename(f)}")
        
        # Try to load each file
        all_data = {}
        for file_path in pickle_files:
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    filename = os.path.basename(file_path)
                    all_data[filename] = data
                    # print(f"✓ Successfully loaded {filename}")
            except Exception as e:
                print(f"✗ Failed to load {os.path.basename(file_path)}: {e}")
                continue
        
        return all_data
    except Exception as e:
        print(f"Error accessing directory {directory_path}: {e}")
        return None

def extract_documents_from_data(data_dict):
    """Extract document texts from loaded data dictionary"""
    documents = []
    
    print("Analyzing loaded data structure...")
    
    for filename, data in data_dict.items():
        print(f"\nProcessing {filename}:")
        print(f"  Type: {type(data)}")
        
        # Handle LangChain Document objects specifically
        if hasattr(data, 'page_content'):
            print(f"  Found LangChain Document")
            documents.append(data.page_content)
            continue
        
        # Handle different data structures
        if hasattr(data, 'docs'):
            # Docstore with docs attribute
            print(f"  Found docstore with {len(data.docs)} documents")
            for doc_id, doc in data.docs.items():
                if hasattr(doc, 'page_content'):
                    documents.append(doc.page_content)
                elif isinstance(doc, str):
                    documents.append(doc)
                    
        elif isinstance(data, dict):
            # Direct dictionary
            if 'docs' in data:
                print(f"  Found docs key with {len(data['docs'])} items")
                for key, value in data['docs'].items():
                    if hasattr(value, 'page_content'):
                        documents.append(value.page_content)
                    elif isinstance(value, str):
                        documents.append(value)
            else:
                print(f"  Dictionary with {len(data)} keys: {list(data.keys())[:5]}")
                for key, value in data.items():
                    if hasattr(value, 'page_content'):
                        documents.append(value.page_content)
                    elif isinstance(value, str) and len(value) > 50:  # Avoid IDs
                        documents.append(value)
                        
        elif isinstance(data, list):
            # List of documents
            print(f"  List with {len(data)} items")
            for doc in data:
                if hasattr(doc, 'page_content'):
                    documents.append(doc.page_content)
                elif isinstance(doc, str):
                    documents.append(doc)
                    
        elif hasattr(data, '_collection'):
            # ChromaDB collection
            try:
                print("  ChromaDB collection detected")
                all_docs = data._collection.get()
                if 'documents' in all_docs:
                    documents.extend(all_docs['documents'])
                    print(f"  Extracted {len(all_docs['documents'])} documents from ChromaDB")
            except Exception as e:
                print(f"  Error extracting from ChromaDB: {e}")
                
        else:
            print(f"  Unsupported data structure: {type(data)} - skipping")
    
    return documents

def generate_qa_pair_with_retry(doc_id, context, max_retries=3):
    """Generate a single QA pair with retry logic and rate-limited API key management"""
    for attempt in range(max_retries):
        # Get API key (this will handle all rate limiting and waiting)
        api_key = api_key_manager.get_best_api_key()
        
        try:
            # Create LLM client with current API key
            llm_client = ChatGoogleGenerativeAI(
                model="gemma-3-27b-it",
                temperature=0,
                google_api_key=api_key
            )
            
            prompt = QA_generation_prompt.format(context=context)
            response = llm_client.invoke(prompt)
            
            # Mark success
            api_key_manager.mark_key_success(api_key)
            
            # Parse response
            question, answer = parse_qa_response(response.content)
            
            if question and answer:
                return {
                    'document_id': doc_id,
                    'context': context,
                    'question': question,
                    'answer': answer
                }
            else:
                print(f"Failed to parse QA for document {doc_id}")
                return None
                
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for document {doc_id}: {e}")
            api_key_manager.mark_key_failed(api_key)
            
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"All attempts failed for document {doc_id}")
                return None

def parse_qa_response(response_text):
    """Parse the QA response to extract question and answer"""
    try:
        if "Output:::" in response_text:
            output_section = response_text.split("Output:::")[-1].strip()
        else:
            output_section = response_text
        
        lines = output_section.strip().split('\n')
        question = None
        answer = None
        
        for line in lines:
            line = line.strip()
            if line.startswith("Factoid question:"):
                question = line.replace("Factoid question:", "").strip()
            elif line.startswith("Answer:"):
                answer = line.replace("Answer:", "").strip()
        
        return question, answer
    except Exception as e:
        print(f"Error parsing response: {e}")
        return None, None

def create_qa_dataset_parallel(documents, max_documents=None, max_workers=None):
    """Create QA dataset from documents using parallel processing with rate limiting"""
    qa_pairs = []
    
    # Limit documents if specified
    if max_documents:
        documents = documents[:max_documents]
    
    # Automatically set max_workers based on API keys and rate limits
    if max_workers is None:
        # Conservative approach: use fewer workers to avoid overwhelming the rate limiter
        max_workers = min(len(API_KEYS) * 2, 8)
    
    print(f"🚀 Generating QA pairs for {len(documents)} documents using {max_workers} threads...")
    print(f"📊 Rate limiting: {QUERIES_PER_MINUTE} queries/min per key with {SAFETY_BUFFER}x safety buffer")
    
    # Prepare document tasks
    document_tasks = []
    for i, context in enumerate(documents):
        # Skip very short contexts
        if len(context.strip()) < 50:
            continue
            
        # Truncate very long contexts to avoid token limits
        if len(context) > 2000:
            context = context[:2000] + "..."
        
        document_tasks.append((i, context))
    
    print(f"📝 Processing {len(document_tasks)} valid documents...")
    
    # Process documents in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_doc = {
            executor.submit(generate_qa_pair_with_retry, doc_id, context): doc_id 
            for doc_id, context in document_tasks
        }
        
        # Collect results with progress bar
        with tqdm(total=len(future_to_doc), desc="Processing documents") as pbar:
            for future in as_completed(future_to_doc):
                doc_id = future_to_doc[future]
                try:
                    result = future.result()
                    if result:
                        qa_pairs.append(result)
                    pbar.update(1)
                    
                    # Print API key stats periodically
                    if len(qa_pairs) % 25 == 0 and len(qa_pairs) > 0:
                        stats = api_key_manager.get_stats()
                        print(f"\n📊 API Key Stats (after {len(qa_pairs)} QA pairs):")
                        for key, stat in stats.items():
                            print(f"  {key}: {stat['status']} | "
                                  f"Requests: {stat['requests_last_minute']}/{QUERIES_PER_MINUTE} | "
                                  f"Failures: {stat['failures']} | "
                                  f"Last: {stat['last_request_ago']} | "
                                  f"Next: {stat['next_available_in']}")
                        print()
                        
                except Exception as e:
                    print(f"Document {doc_id} generated an exception: {e}")
                    pbar.update(1)
    
    return qa_pairs

# Main execution
if __name__ == "__main__":
    # Load your pickled data from directories
    print("Loading pickled data from directories...")
    
    # Load from both directories
    docstore_data = load_pickled_data_from_directory("docstore_final")
    chroma_data = load_pickled_data_from_directory("chroma_db_final")
    
    # Combine all loaded data
    all_data = {}
    if docstore_data:
        all_data.update(docstore_data)
    if chroma_data:
        all_data.update(chroma_data)
    
    if not all_data:
        print("No data could be loaded from either directory!")
        exit()
    
    # Extract documents from all loaded data
    print("\nExtracting documents from loaded data...")
    documents = extract_documents_from_data(all_data)
    
    print(f"\nFound {len(documents)} documents total")
    
    if documents:
        # Show sample of first few documents
        print("\nSample documents:")
        for i, doc in enumerate(documents[:3]):
            print(f"Document {i+1}: {doc[:100]}...")
        
        # Generate QA pairs using parallel processing with rate limiting
        qa_dataset = create_qa_dataset_parallel(documents)
        
        # Convert to DataFrame with only required columns
        df = pd.DataFrame(qa_dataset)
        
        # Ensure we have the exact columns we want
        if len(df) > 0:
            df = df[['document_id', 'context', 'question', 'answer']]
        
        # Save the dataset
        df.to_csv("qa_dataset.csv", index=False)
        df.to_json("qa_dataset.json", orient="records", indent=2)
        
        print(f"\n✅ Generated {len(qa_dataset)} QA pairs")
        print(f"📄 Dataset saved as 'qa_dataset.csv' and 'qa_dataset.json'")
        
        # Display sample
        if len(df) > 0:
            print("\nSample QA pairs:")
            for i in range(min(3, len(df))):
                print(f"\n--- QA Pair {i+1} ---")
                print(f"Document ID: {df.iloc[i]['document_id']}")
                print(f"Question: {df.iloc[i]['question']}")
                print(f"Answer: {df.iloc[i]['answer']}")
                print(f"Context: {df.iloc[i]['context'][:100]}...")
    else:
        print("No documents found to process!")
        print("Please check the data structure in your directories.")