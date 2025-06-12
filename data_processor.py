import os
import json
import base64
import hashlib
import pickle
import shutil
import time
import warnings
from pathlib import Path
from typing import Dict, List

# LangChain components for retrieval and storage
from langchain_chroma import Chroma
from langchain.storage import LocalFileStore
from langchain.schema.document import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter

# LangChain components for Google Gemini models
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# For image processing via HTTP requests and local files
import requests

# Suppress common warnings for a cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning)


class DataProcessor:
    """
    Args:
        output_dir (str, optional): Directory containing the scraped JSON files. 
            If None, only direct JSON processing is available.
        vectorstore_path (str): Path to persist the Chroma vector database.
        docstore_path (str): Path for the local file document store.
        google_api_key (Optional[str]): Google API key. If None, it's read
            from the "GOOGLE_API_KEY" environment variable.
        chunk_size (int): Size of text chunks for splitting long documents.
        chunk_overlap (int): Overlap between text chunks.
        use_vision_model (bool): Whether to use the vision model for images.
        verbose (bool): If True, prints detailed logs during processing.
    """

    def __init__(self,
             output_dir=None,
             vectorstore_path="./chroma_db_final",
             docstore_path="./docstore_final",
             google_api_key=None,
             chunk_size=1000,
             chunk_overlap=200,
             use_vision_model=True,
             verbose=True):

        # --- Configuration and Paths ---
        self.output_dir = Path(output_dir) if output_dir else None
        self.vectorstore_path = Path(vectorstore_path)
        self.docstore_path = Path(docstore_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_vision_model = use_vision_model
        self.verbose = verbose
        self.id_key = "doc_id"
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.webp'}
        self.min_delay_seconds = 4  
        self._last_api_call_time = 0.0

        # --- Statistics Tracking ---
        self.stats = {
            'issues_loaded_successfully': 0,
            'issues_failed_loading': 0,
            'documents_processed': 0,
            'text_chunks_created': 0,
            'images_processed': 0,
            'images_skipped_format': 0,
            'summaries_generated': 0,
            'text_model_switches': 0,
            'vision_model_switches': 0,
            'errors': []
        }

        self._log("Initializing Data Processor...", "info")

        # --- API Key Setup ---
        if not google_api_key:
            google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY must be set in environment or passed as a parameter.")
        
        self.google_api_key = google_api_key  

        # Set up prompts first, then models
        self._setup_prompts_and_chains()
        self._setup_models()

        # --- LangChain Components Initialization ---
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        self.embedding_function = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=self.google_api_key  # Pass API key explicitly
        )
        self.vectorstore = Chroma(
            collection_name="the_batch_rag_collection",
            embedding_function=self.embedding_function,
            persist_directory=str(self.vectorstore_path)
        )
        self.docstore = LocalFileStore(root_path=str(self.docstore_path))
        self.retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore, docstore=self.docstore, id_key=self.id_key,
        )

        self._log("Data Processor initialized successfully.", "success")

    def _setup_models(self):
        """Initializes model configurations and tracking."""
        self.available_models = [
            {
                'name': 'models/gemma-3-27b-it',
                'requests_per_minute': 30,
                'tokens_per_minute': 15000,
                'requests_per_day': 14400
            },
            {
                'name': 'models/gemma-3-12b-it', 
                'requests_per_minute': 30,
                'tokens_per_minute': 15000,
                'requests_per_day': 14400
            },
            {
                'name': 'models/gemma-3-4b-it',
                'requests_per_minute': 30,
                'tokens_per_minute': 15000,
                'requests_per_day': 14400
            },
            {
                'name': 'models/gemma-3-1b-it',
                'requests_per_minute': 30,
                'tokens_per_minute': 15000,
                'requests_per_day': 14400
            }
        ]
        
        self.available_vision_models = [
            {'name': 'gemini-2.0-flash-lite', 'requests_per_minute': 30},
            {'name': 'gemini-2.0-flash', 'requests_per_minute': 15},
        ]
        
        # Model state tracking
        self.current_model_index = 0
        self.model_request_counts = {m['name']: 0 for m in self.available_models}
        self.model_last_reset = {m['name']: time.time() for m in self.available_models}
        self.failed_models = set()

        self.current_vision_model_index = 0
        self.vision_model_request_counts = {m['name']: 0 for m in self.available_vision_models}
        self.vision_model_last_reset = {m['name']: time.time() for m in self.available_vision_models}
        self.failed_vision_models = set()

        self.text_llm = None
        self.vision_llm = None
        self._initialize_llms()
        if self.use_vision_model:
            self._initialize_vision_llm()

    def _setup_prompts_and_chains(self):
        """Sets up all prompt templates and chains."""
        self.text_summary_prompt = ChatPromptTemplate.from_template(
            "Summarize the following text in 2-3 concise sentences, focusing on the main topic and key insights. "
            "Make the summary informative and easily searchable:\n\nText: {text}\n\nSummary:"
        )
        self.vision_prompt_template = ChatPromptTemplate.from_messages([
            ("human", [
                {"type": "text", "text": """Analyze this image and provide a clear, factual description. Focus on:
1.  **Main subject:** What is the primary focus?
2.  **Key elements:** Important objects, people, text, or data visualizations.
3.  **Context:** How might this relate to AI, technology, or machine learning?
Provide a practical description, not a list of potential AI applications."""},
                {"type": "image_url", "image_url": {"url": "data:{mime_type};base64,{image_data}"}}
            ])
        ])
        self.image_summary_prompt = ChatPromptTemplate.from_template(
            "Create a concise 1-2 sentence summary based on this context:\n"
            "Alt Text: {alt_text}\nCaption: {caption}\n"
            "Focus on the actual content and meaning. Make it informative for search."
        )
        # Initialize chains as None - they'll be set when models are ready
        self.text_summary_chain = None
        self.image_summary_chain = None

    # --- Core Processing Pipeline ---

    def process_data(self, json_data = None):
        """       
        Args:
            json_data: Optional JSON data to process directly. Can be a single dict 
                      or list of dicts. If None, loads data from files.
        """
        self._log("Starting data processing...", "info")
        
        if json_data is not None:
            scraped_data = self._validate_json_data(json_data)
            self._log(f"Processing direct JSON input with {len(scraped_data)} items.", "info")
        else:
            scraped_data = self._load_scraped_data()
        
        if not scraped_data:
            self._log("No valid data found to process.", "warning")
            return
            
        # Check if the collection is empty before processing
        try:
            if self.vectorstore._collection.count() > 0:
                self._log("Collection is not empty. Skipping data processing.", "info")
                self._log("To re-process, call clear_collection() first.", "info")
                return
        except ValueError:
            # Collection not initialized, which is fine - we'll process the data
            self._log("Collection not initialized, proceeding with data processing.", "info")

        self._process_and_add_to_retriever(scraped_data)
        self._print_processing_stats()

    def process_json_data(self, json_data, 
                         append_to_existing = False):
        """       
        Args:
            json_data: JSON data to process (single dict or list of dicts)
            append_to_existing: If True, adds to existing collection. 
                               If False, processes only if collection is empty.
        """
        self._log("Processing direct JSON data...", "info")
        scraped_data = self._validate_json_data(json_data)
        
        if not scraped_data:
            self._log("No valid data found to process.", "warning")
            return
        
        # Check collection state
        try:
            collection_count = self.vectorstore._collection.count()
            if collection_count > 0 and not append_to_existing:
                self._log("Collection is not empty and append_to_existing=False.", "warning")
                self._log("Set append_to_existing=True to add to existing data.", "info")
                return
        except ValueError:
            # Collection not initialized, which is fine
            pass

        self._process_and_add_to_retriever(scraped_data)
        self._print_processing_stats()

    def _validate_json_data(self, json_data):
        """        
        Args:
            json_data: Single dict or list of dicts to validate
            
        Returns:
            List of validated dictionaries
        """
        if isinstance(json_data, dict):
            json_data = [json_data]
        elif not isinstance(json_data, list):
            self._log("JSON data must be a dict or list of dicts.", "error")
            return []
        
        validated_data = []
        for i, item in enumerate(json_data):
            if not isinstance(item, dict):
                self._log(f"Item {i} is not a dictionary. Skipping.", "warning")
                self.stats['issues_failed_loading'] += 1
                continue
            
            # Basic validation - require at least id and content
            if 'id' not in item:
                self._log(f"Item {i} missing 'id' field. Skipping.", "warning")
                self.stats['issues_failed_loading'] += 1
                continue
            
            if 'content' not in item:
                self._log(f"Item {i} missing 'content' field. Skipping.", "warning")
                self.stats['issues_failed_loading'] += 1
                continue
            
            # Add default values for missing optional fields
            if 'url' not in item:
                item['url'] = f"direct_input_{item['id']}"
            if 'date' not in item:
                item['date'] = 'unknown_date'
            
            validated_data.append(item)
            self.stats['issues_loaded_successfully'] += 1
        
        self._log(f"JSON validation complete. Valid: {len(validated_data)}, Invalid: {len(json_data) - len(validated_data)}", "stats")
        return validated_data

    def _load_scraped_data(self):
        """
        Loads and validates all JSON files from the output directory.
        This method tracks and reports loading successes and failures.
        """
        if not self.output_dir:
            self._log("No output directory specified for file loading.", "error")
            return []
            
        all_data = []
        if not self.output_dir.exists():
            self._log(f"Output directory '{self.output_dir}' not found.", "error")
            return all_data

        json_files = list(self.output_dir.glob("*.json"))
        self._log(f"Found {len(json_files)} JSON files in '{self.output_dir}'.", "processing")

        for file in json_files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Basic validation
                if 'id' in data and 'url' in data and 'content' in data:
                    all_data.append(data)
                    self.stats['issues_loaded_successfully'] += 1
                else:
                    self._log(f"Invalid format in {file.name}. Skipping.", "warning")
                    self.stats['issues_failed_loading'] += 1
                    self.stats['errors'].append(f"InvalidDataFormat: {file.name}")

            except json.JSONDecodeError as e:
                self._log(f"Error decoding JSON from {file.name}: {e}", "error")
                self.stats['issues_failed_loading'] += 1
                self.stats['errors'].append(f"JSONDecodeError: {file.name}")
            except Exception as e:
                self._log(f"An unexpected error occurred loading {file.name}: {e}", "error")
                self.stats['issues_failed_loading'] += 1
                self.stats['errors'].append(f"FileLoadError: {file.name} - {e}")
        
        self._log(f"JSON Loading complete. Success: {self.stats['issues_loaded_successfully']}, Failed: {self.stats['issues_failed_loading']}", "stats")
        return all_data
        
    def _process_and_add_to_retriever(self, scraped_data):
        """Processes all documents and adds them to the retriever."""
        self._log(f"Processing {len(scraped_data)} issues for RAG system...", "processing")
        all_summary_docs: List[Document] = []
        all_parent_docs: Dict[str, Document] = {}

        for issue in reversed(scraped_data): # Process oldest first
            issue_id = issue.get('id', 'unknown_issue')
            issue_date = issue.get('date', 'unknown_date')
            issue_url = issue.get('url', '')

            # Process main letter content
            self._process_content_section(
                issue.get('content', {}).get('letter', {}), 'letter',
                issue_id, issue_date, issue_url,
                all_summary_docs, all_parent_docs
            )
            # Process news items
            for news_item in issue.get('content', {}).get('news', []):
                self._process_content_section(
                    news_item, 'news',
                    issue_id, issue_date, issue_url,
                    all_summary_docs, all_parent_docs,
                    news_title=news_item.get('title', 'Untitled News')
                )
            self.stats['documents_processed'] += 1
        
        self._add_documents_to_retriever(all_summary_docs, all_parent_docs)

    def _process_content_section(self, section, section_type, issue_id, date, url,
                             summary_docs, parent_docs, news_title = None):
        """Helper to process a section (letter or news) of an issue."""
        base_meta = {'type': section_type, 'issue_id': issue_id, 'date': date, 'source': url}
        if news_title:
            base_meta['news_title'] = news_title

        # Process text content
        text = section.get('text', '')
        if text:
            summary = self._generate_text_summary(text)
            doc_id = self._generate_deterministic_id(summary, {**base_meta, 'content_type': 'summary'})
            # FIX: Properly set content_type for text summaries
            summary_meta = {**base_meta, self.id_key: doc_id, 'content_type': f'{section_type}_text_summary'}
            
            summary_docs.append(Document(page_content=summary, metadata=summary_meta))
            # FIX: Set proper content_type for parent document
            parent_meta = {**base_meta, 'content_type': f'{section_type}_text'}
            parent_docs[doc_id] = Document(page_content=text, metadata=parent_meta)
            self.stats['text_chunks_created'] += 1

        # Process images
        for img_info in section.get('images', []):
            img_path = img_info.get('path')
            if not img_path or not Path(img_path).suffix.lower() in self.supported_formats:
                self.stats['images_skipped_format'] += 1
                continue
            
            alt = img_info.get('alt', '')
            cap = img_info.get('caption', '')
            
            img_summary = self._generate_image_summary(img_path, alt, cap)
            doc_id = self._generate_deterministic_id(img_summary, {**base_meta, 'content_type': 'image', 'path': img_path})
            # FIX: Properly set content_type for image summaries
            img_meta = {**base_meta, self.id_key: doc_id, 'content_type': f'{section_type}_image_summary', 'image_path': img_path}
            
            summary_docs.append(Document(page_content=img_summary, metadata=img_meta))
            # FIX: Set proper content_type for image parent document
            parent_img_meta = {**base_meta, 'content_type': f'{section_type}_image', 'image_path': img_path}
            parent_docs[doc_id] = Document(page_content=img_summary, metadata=parent_img_meta)
            self.stats['images_processed'] += 1

    def _add_documents_to_retriever(self, summary_docs, parent_docs):
        """Adds the processed documents to the vectorstore and docstore."""
        if not summary_docs:
            self._log("No new documents to add.", "warning")
            return
        
        try:
            self.retriever.vectorstore.add_documents(summary_docs)
            self._log(f"Added {len(summary_docs)} summary documents to vectorstore.", "success")
            
            # Serialize parent docs for storage
            serialized_parents = [(doc_id, pickle.dumps(doc)) for doc_id, doc in parent_docs.items()]
            self.retriever.docstore.mset(serialized_parents)
            self._log(f"Added {len(parent_docs)} parent documents to docstore.", "success")
            
        except Exception as e:
            self._log(f"Error adding documents to retriever: {e}", "error")
            self.stats['errors'].append(f"RetrieverAddError: {e}")

    # --- Utility Methods ---
    
    def _log(self, message, level = "info"):
        if self.verbose:
            emoji_map = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌", "processing": "🔄", "stats": "📊"}
            print(f"{emoji_map.get(level, '➡️')} {message}")

    def _generate_text_summary(self, text):
        """Generates a summary for a given text block."""
        if not text or len(text.strip()) < 50: return text.strip()
        try:
            summary = self._execute_api_call(
                lambda: self.text_summary_chain.invoke({"text": text}), 'text'
            )
            self.stats['summaries_generated'] += 1
            return summary.strip()
        except Exception as e:
            self._log(f"Could not generate text summary: {e}", "error")
            return text[:300].strip() + '...'

    def _generate_image_summary(self, image_path, alt_text, caption):
        """Generates a summary for an image, using Vision model if enabled."""
        if self.use_vision_model and self.vision_llm:
            try:
                base64_image = self._encode_image_to_base64(image_path)
                if not base64_image: 
                    raise ValueError("Failed to encode image")
                
                mime_type = f"image/{Path(image_path).suffix.lower().strip('.')}"
                prompt = self.vision_prompt_template.format_messages(mime_type=mime_type, image_data=base64_image)

                response = self._execute_api_call(lambda: self.vision_llm.invoke(prompt), 'vision')
                self.stats['summaries_generated'] += 1
                return f"Visual Analysis: {response.content}"
            except Exception as e:
                self._log(f"Vision summary failed for {Path(image_path).name}: {e}. Falling back to text.", "warning")
        
        # FIX: Improved fallback to text-based summary with better error handling
        try:
            fallback_summary = self._execute_api_call(
                lambda: self.image_summary_chain.invoke({"alt_text": alt_text or "N/A", "caption": caption or "N/A"}), 'text'
            )
            self.stats['summaries_generated'] += 1
            return fallback_summary
        except Exception as e:
            self._log(f"Text-based image summary also failed: {e}. Using basic description.", "warning")
            # Final fallback - create basic description from available text
            description_parts = []
            if alt_text and alt_text.strip():
                description_parts.append(f"Image description: {alt_text.strip()}")
            if caption and caption.strip():
                description_parts.append(f"Caption: {caption.strip()}")
            
            if description_parts:
                return ". ".join(description_parts)
            else:
                return f"Image file: {Path(image_path).name}"

    @staticmethod
    def _generate_deterministic_id(content, metadata):
        """Generates a stable MD5 hash ID to prevent content duplication."""
        unique_string = f"{content}|{metadata.get('issue_id')}|{metadata.get('type')}|{metadata.get('path')}"
        return hashlib.md5(unique_string.encode('utf-8')).hexdigest()

    def _print_processing_stats(self):
        """Prints final statistics after data processing is complete."""
        self._log("=== DATA PROCESSING COMPLETE ===", "stats")
        for key, value in self.stats.items():
            if 'issues' in key: continue # Already logged during loading
            formatted_key = key.replace('_', ' ').title()
            self._log(f"{formatted_key}: {value}", "stats")
        
        if self.stats['errors']:
            self._log(f"Total Errors Encountered: {len(self.stats['errors'])}", "warning")
        
        collection_stats = self.get_collection_stats()
        self._log("--- Collection Stats ---", "stats")
        self._log(f"Total Documents in Vectorstore: {collection_stats.get('total_documents', 'N/A')}", "stats")
        self._log(f"Parent Documents in Docstore: {collection_stats.get('docstore_size', 'N/A')}", "stats")
        if 'content_types' in collection_stats:
            self._log("Content Type Breakdown:", "stats")
            for c_type, count in collection_stats['content_types'].items():
                self._log(f"  - {c_type}: {count}", "stats")

    def get_collection_stats(self) -> Dict:
        """Retrieves statistics about the current state of the collection."""
        try:
            # More robust collection count method
            try:
                # Try the direct collection count first
                count = self.vectorstore._collection.count()
            except (ValueError, AttributeError):
                # Fallback: try to get documents and count them
                try:
                    all_docs = self.vectorstore.get()
                    count = len(all_docs.get('documents', []))
                except:
                    # Final fallback: similarity search with high limit
                    try:
                        docs = self.vectorstore.similarity_search("", k=1000)
                        count = len(docs)
                    except:
                        count = 0
            
            # Count docstore items more reliably
            try:
                docstore_items = list(self.docstore.ls())
                docstore_size = len(docstore_items)
            except:
                docstore_size = 0
            
            # Get content type breakdown
            content_types = {}
            if count > 0:
                try:
                    # Try to get all documents with metadata
                    all_data = self.vectorstore.get(include=["metadatas"])
                    metadatas = all_data.get('metadatas', [])
                    
                    for meta in metadatas:
                        c_type = meta.get('content_type', 'unknown')
                        content_types[c_type] = content_types.get(c_type, 0) + 1
                except Exception as e:
                    content_types = {"error": f"Could not retrieve content types: {e}"}
            
            return {
                'total_documents': count, 
                'docstore_size': docstore_size, 
                'content_types': content_types
            }
        except Exception as e:
            return {'error': str(e)}

    def clear_collection(self):
        """Deletes all data from the vectorstore and docstore."""
        try:
            # Clear Chroma by deleting the collection and recreating it
            self._log("Clearing Chroma vector store...", "processing")
            try:
                self.vectorstore.delete_collection()
            except Exception as e:
                self._log(f"Note: {e}. Creating fresh collection.", "info")
            
            # Re-initialize to create a fresh collection
            self.vectorstore = Chroma(
                collection_name="the_batch_rag_collection",
                embedding_function=self.embedding_function,
                persist_directory=str(self.vectorstore_path)
            )
            # Re-initialize the retriever with the new vectorstore
            self.retriever = MultiVectorRetriever(
                vectorstore=self.vectorstore,
                docstore=self.docstore,
                id_key=self.id_key,
            )
            self._log("Chroma vector store cleared and re-initialized.", "success")
            
            # Clear file docstore by deleting its directory
            if self.docstore_path.exists():
                self._log("Clearing file system docstore...", "processing")
                shutil.rmtree(self.docstore_path)
                self.docstore_path.mkdir(parents=True, exist_ok=True)
                # Re-initialize the docstore
                self.docstore = LocalFileStore(root_path=str(self.docstore_path))
                # Update the retriever with the new docstore
                self.retriever.docstore = self.docstore
                self._log("File system docstore cleared.", "success")
                
            self.stats = {k: 0 if isinstance(v, int) else [] for k, v in self.stats.items()}
        except Exception as e:
            self._log(f"Error clearing collection: {e}", "error")

    # --- Low-level API Handling Methods ---
    
    def _initialize_llms(self):
        model_name = self.available_models[self.current_model_index]['name']
        self.text_llm = ChatGoogleGenerativeAI(
            model=model_name, 
            temperature=0,
            google_api_key=self.google_api_key  # Pass API key explicitly
        )
        self._log(f"Text LLM initialized with: {model_name}", "info")
        self._update_chains()

    def _initialize_vision_llm(self):
        model_name = self.available_vision_models[self.current_vision_model_index]['name']
        self.vision_llm = ChatGoogleGenerativeAI(
            model=model_name, 
            temperature=0.1,
            google_api_key=self.google_api_key  # Pass API key explicitly
        )
        self._log(f"Vision LLM initialized with: {model_name}", "info")

    def _update_chains(self):
        if self.text_llm:
            self.text_summary_chain = self.text_summary_prompt | self.text_llm | StrOutputParser()
            self.image_summary_chain = self.image_summary_prompt | self.text_llm | StrOutputParser()

    def _execute_api_call(self, api_call_func, model_type):
        """A robust wrapper for API calls that handles rate limits and model fallback."""
        model_configs = {
            'text': (self.available_models, 'current_model_index', self.model_request_counts, self.model_last_reset, self.failed_models, self._initialize_llms),
            'vision': (self.available_vision_models, 'current_vision_model_index', self.vision_model_request_counts, self.vision_model_last_reset, self.failed_vision_models, self._initialize_vision_llm)
        }
        models, index_attr, counts, resets, failed, initializer = model_configs[model_type]
        max_retries = len(models)

        for attempt in range(max_retries):
            current_index = getattr(self, index_attr)
            model = models[current_index]
            
            # Rate limit check
            now = time.time()
            if now - resets[model['name']] > 60:
                counts[model['name']] = 0
                resets[model['name']] = now
            
            if counts[model['name']] >= model['requests_per_minute']:
                self._log(f"Rate limit for {model['name']} reached. Switching model.", "warning")
                if self._switch_model(model_type): continue
                else: time.sleep(60); continue

            try:
                # Enforce minimum delay between calls
                elapsed = time.time() - self._last_api_call_time
                if elapsed < self.min_delay_seconds:
                    time.sleep(self.min_delay_seconds - elapsed)
                
                self._last_api_call_time = time.time()
                counts[model['name']] += 1
                return api_call_func()
            
            except Exception as e:
                self._log(f"API Error with {model['name']}: {e}", "error")
                self.stats['errors'].append(f"APIError ({model['name']}): {e}")
                failed.add(model['name'])
                if not self._switch_model(model_type):
                    raise Exception(f"All {model_type} models have failed. Last error: {e}")
        
        raise Exception(f"All {model_type} models exhausted after multiple retries.")
    
    def _switch_model(self, model_type):
        """Switches to the next available, non-failed model."""
        configs = {
            'text': (self.available_models, 'current_model_index', self.failed_models, self._initialize_llms, 'text_model_switches'),
            'vision': (self.available_vision_models, 'current_vision_model_index', self.failed_vision_models, self._initialize_vision_llm, 'vision_model_switches')
        }
        models, index_attr, failed_set, initializer, stat_key = configs[model_type]
        start_index = getattr(self, index_attr)
        
        for i in range(1, len(models) + 1):
            next_index = (start_index + i) % len(models)
            if models[next_index]['name'] not in failed_set:
                setattr(self, index_attr, next_index)
                initializer()
                self.stats[stat_key] += 1
                return True
        return False

    def _encode_image_to_base64(self, image_path):
        try:
            if image_path.startswith(('http', 'https')):
                response = requests.get(image_path, timeout=10)
                response.raise_for_status()
                image_data = response.content
            else:
                with open(image_path, "rb") as image_file:
                    image_data = image_file.read()
            return base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            self._log(f"Error encoding image {image_path}: {e}", "error")
            return None

    def query(self, query: str, k: int = 5, generate_answer: bool = True):
        """
        Query the RAG system and generate a comprehensive answer.
        
        Args:
            query (str): The search query string
            k (int): Number of results to return (default: 5)
            generate_answer (bool): Whether to generate an AI answer from retrieved documents
        
        Returns:
            Dict containing:
                - 'answer': Generated answer from retrieved documents
                - 'sources': List of source information with metadata
                - 'images': List of related images with descriptions
                - 'documents': Raw document contents
                - 'query': The original query
                - 'num_results': Number of results returned
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        # Check if collection has data
        try:
            collection_stats = self.get_collection_stats()
            if collection_stats.get('total_documents', 0) == 0:
                raise ValueError("Collection is empty. Please process data first using process_data() or process_json_data()")
        except Exception as e:
            self._log(f"Warning: Could not verify collection state: {e}", "warning")
        
        try:
            self._log(f"Querying for: '{query}' (k={k})", "processing")
            
            # FIX: Use the helper method that properly handles k parameter
            relevant_docs = self._get_documents_safely(query, k)
            
            if not relevant_docs:
                self._log("No relevant documents found for the query", "warning")
                return {
                    'answer': "I couldn't find any relevant information to answer your query. Please try rephrasing your question or using different keywords.",
                    'sources': [],
                    'images': [],
                    'documents': [],
                    'query': query,
                    'num_results': 0
                }
            
            # Extract content and metadata
            documents = [doc.page_content for doc in relevant_docs]
            metadata_list = [doc.metadata for doc in relevant_docs]
            
            self._log(f"Found {len(relevant_docs)} relevant documents", "success")
            
            # Process sources and images
            sources = []
            images = []
            
            for i, (doc, meta) in enumerate(zip(relevant_docs, metadata_list)):
                # Create source entry
                content_type = meta.get('content_type', 'unknown')

                display_type = content_type
                if content_type.endswith('_text_summary'):
                    display_type = content_type.replace('_text_summary', ' Text').title()
                elif content_type.endswith('_image_summary'):
                    display_type = content_type.replace('_image_summary', ' Image').title()
                elif content_type.endswith('_text'):
                    display_type = content_type.replace('_text', ' Text').title()
                elif content_type.endswith('_image'):
                    display_type = content_type.replace('_image', ' Image').title()

                source_info = {
                    'ref_id': f"[{i+1}]",
                    'content_type': display_type,  # Use cleaned display type
                    'date': meta.get('date', 'unknown'),
                    'issue_id': meta.get('issue_id', ''),
                    'source_url': meta.get('source', ''),
                    'content_preview': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                }
                
                # Add news title if available
                if meta.get('news_title'):
                    source_info['news_title'] = meta['news_title']
                
                sources.append(source_info)
                
                # Extract image information if available
                if meta.get('content_type', '').endswith('_image') and meta.get('image_path'):
                    image_info = {
                        'path': meta['image_path'],
                        'description': doc.page_content,
                        'ref_id': f"[{i+1}]",
                        'source_url': meta.get('source', ''),
                        'date': meta.get('date', 'unknown')
                    }
                    images.append(image_info)
            
            # Generate answer if requested
            answer = ""
            if generate_answer and self.text_llm:
                try:
                    # Create context from retrieved documents
                    context_parts = []
                    for i, doc in enumerate(documents):
                        context_parts.append(f"[{i+1}] {doc}")
                    
                    context = "\n\n".join(context_parts)
                    
                    # Create answer generation prompt
                    answer_prompt = ChatPromptTemplate.from_template(
                        """Based on the following context from The Batch newsletter archives, provide a comprehensive and accurate answer to the user's question. 

    Use the information from the context to give specific details, examples, and insights. If the context contains multiple perspectives or developments over time, mention them. Always cite your sources using the reference numbers [1], [2], etc.

    If you cannot answer the question based on the provided context, say so clearly.

    Context:
    {context}

    Question: {question}

    Answer:"""
                    )
                    
                    # Generate answer
                    answer_chain = answer_prompt | self.text_llm | StrOutputParser()
                    answer = self._execute_api_call(
                        lambda: answer_chain.invoke({"context": context, "question": query}),
                        'text'
                    )
                    
                    self._log("Generated answer successfully", "success")
                    
                except Exception as e:
                    self._log(f"Error generating answer: {e}", "error")
                    answer = f"I found relevant information but couldn't generate a comprehensive answer due to an error: {str(e)}\n\nPlease check the sources below for the relevant information."
            else:
                # Provide a simple concatenated answer if not generating with AI
                answer = "Here's what I found from The Batch archives:\n\n"
                for i, doc in enumerate(documents[:3]):  # Limit to first 3 for readability
                    answer += f"[{i+1}] {doc}\n\n"
                if len(documents) > 3:
                    answer += f"... and {len(documents) - 3} more related results (see sources below)."
            
            # Log content type breakdown if verbose
            if self.verbose:
                content_types = {}
                for meta in metadata_list:
                    c_type = meta.get('content_type', 'unknown')
                    content_types[c_type] = content_types.get(c_type, 0) + 1
                
                if content_types:
                    self._log("Result content types:", "stats")
                    for c_type, count in content_types.items():
                        self._log(f"  - {c_type}: {count}", "stats")
            
            return {
                'answer': answer,
                'sources': sources,
                'images': images,
                'documents': documents,
                'query': query,
                'num_results': len(relevant_docs)
            }
            
        except Exception as e:
            self._log(f"Error during query execution: {e}", "error")
            return {
                'answer': f"An error occurred while processing your query: {str(e)}",
                'sources': [],
                'images': [],
                'documents': [],
                'query': query,
                'num_results': 0
            }

    def _get_documents_safely(self, query, k):
        """
        Safely retrieve documents with fallback mechanisms.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of Document objects
        """
        documents = []
        
        # Try multi-vector retriever first
        try:
            # FIX: Configure the retriever's search_kwargs properly
            # MultiVectorRetriever uses get_relevant_documents with search_kwargs
            self.retriever.search_kwargs = {"k": k}
            raw_docs = self.retriever.get_relevant_documents(query)
            
            for doc_data in raw_docs:
                doc = self._deserialize_document(doc_data)
                if doc:
                    documents.append(doc)
                    
        except Exception as e:
            self._log(f"Multi-vector retriever failed: {e}. Trying direct similarity search.", "warning")
            
            # Fallback to direct similarity search on vectorstore
            try:
                documents = self.vectorstore.similarity_search(query, k=k)
                self._log("Using similarity search results (summaries only)", "info")
            except Exception as e2:
                self._log(f"Similarity search also failed: {e2}", "error")
                raise Exception(f"Both retrieval methods failed. Multi-vector: {e}, Similarity: {e2}")
        
        # Ensure we don't return more than k documents
        return documents[:k]
    
    def _deserialize_document(self, doc_data):
            """
            Helper method to safely deserialize document data.
            
            Args:
                doc_data: Document data that might be bytes, Document, or other format
                
            Returns:
                Document object or None if deserialization fails
            """
            try:
                if isinstance(doc_data, bytes):
                    return pickle.loads(doc_data)
                elif hasattr(doc_data, 'page_content'):
                    return doc_data
                else:
                    self._log(f"Unknown document format: {type(doc_data)}", "warning")
                    return None
            except Exception as e:
                self._log(f"Error deserializing document: {e}", "error")
                return None