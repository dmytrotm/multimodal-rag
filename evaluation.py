import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import time
from typing import Dict, Any, Tuple
import re
from dotenv import load_dotenv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from key_manager import SmartAPIKeyManager

from data_processor import DataProcessor


# Load environment and setup
load_dotenv()
API_KEYS = [key for key in [os.getenv(f"GOOGLE_API_KEY_{i}") for i in range(1, 6)] if key]
if not API_KEYS:
    raise ValueError("No API keys found in .env file. Please set GOOGLE_API_KEY_1, GOOGLE_API_KEY_2, etc.")

# Create a global API key manager
api_key_manager = SmartAPIKeyManager(API_KEYS, queries_per_minute=12)

# Evaluation prompts
answer_correctness_prompt = """Compare the Generated Answer to the Ground Truth Answer for the given Question.

Rate 1-5:
5: Perfect match, complete and correct
4: Mostly correct, minor details missing
3: Partially correct, missing key info
2: Related but incorrect
1: Completely wrong

Question: {question}
Ground Truth: {ground_truth_answer}
Generated: {generated_answer}

Explanation: [brief reason]
Rating: [1-5]"""

retrieval_faithfulness_prompt = """Check if the Generated Answer is supported by the Retrieved Context. No hallucinations allowed.

Rate 1-5:
5: Fully supported by context
4: Mostly supported, minor inference
3: Partially supported
2: Some unsupported claims
1: Not supported/contradicts context

Context: {retrieved_context}
Answer: {generated_answer}

Explanation: [brief reason]
Rating: [1-5]"""

context_relevance_prompt = """Rate if the Retrieved Context is relevant for answering the Question.

Rate 1-5:
5: Highly relevant, complete info
4: Relevant, most info present
3: Partially relevant, missing key info
2: Slightly relevant
1: Irrelevant

Question: {question}
Context: {retrieved_context}

Explanation: [brief reason]
Rating: [1-5]"""

context_recall_prompt = """Check if the Ground Truth Context is present in the Retrieved Context.

Rate 1-5:
5: Excellent recall, all info found
4: Good recall, most info found
3: Partial recall, some info found
2: Poor recall, little info found
1: No recall, info not found

Ground Truth: {ground_truth_context}
Retrieved: {retrieved_context}

Explanation: [brief reason]
Rating: [1-5]"""

def parse_evaluation_response(response_text: str) -> Tuple[int, str]:
    """Robustly parses the LLM's response to extract rating and explanation."""
    try:
        rating_match = re.search(r'Rating:\s*([1-5])', response_text, re.IGNORECASE)
        rating = int(rating_match.group(1)) if rating_match else 1
        
        explanation_match = re.search(r'Explanation:\s*(.*?)(?=Rating:|$)', response_text, re.IGNORECASE | re.DOTALL)
        explanation = explanation_match.group(1).strip() if explanation_match else response_text[:100]
        
        return rating, explanation
    except Exception:
        return 1, response_text[:100]

def evaluate_single_metric(prompt: str, metric_name: str, thread_id: str) -> Tuple[int, str]:
    """Evaluate a single metric with proper error handling and retries."""
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            estimated_tokens = api_key_manager._estimate_tokens(prompt)
            api_key = api_key_manager.get_best_api_key(estimated_tokens, thread_id)
            
            eval_llm = ChatGoogleGenerativeAI(
                model="gemma-3-27b-it", 
                temperature=0, 
                google_api_key=api_key
            )
            
            response = eval_llm.invoke(prompt).content
            rating, explanation = parse_evaluation_response(response)
            
            api_key_manager.mark_key_success(api_key)
            return rating, explanation
            
        except Exception as e:
            retry_count += 1
            if 'api_key' in locals():
                api_key_manager.mark_key_failed(api_key, thread_id)
            
            if retry_count >= max_retries:
                error_msg = f"LLM Judge Failed after {max_retries} retries: {e}"
                print(f"❌ Thread {thread_id}: Failed to evaluate {metric_name}: {error_msg}")
                return 1, error_msg
            else:
                print(f"⚠️ Thread {thread_id}: Retry {retry_count}/{max_retries} for {metric_name}")
                time.sleep(2 ** retry_count)  # Exponential backoff
    
    return 1, "Max retries exceeded"

def evaluate_rag_row_parallel(data_processor: DataProcessor, index: int, row: pd.Series, thread_id: str) -> Dict[str, Any]:
    """Parallel evaluation function with thread-safe operations."""
    question = row['question']
    ground_truth_answer = row['ground_truth_answer']
    ground_truth_context = row['context']

    try:
        # Get RAG output
        rag_output = data_processor.query(question, generate_answer=True, k=3) 
        generated_answer = rag_output.get('answer', '')
        retrieved_context = "\n---\n".join(rag_output.get('documents', []))
        
        # Truncate long contexts to manage token usage
        if len(retrieved_context) > 2000:
            retrieved_context = retrieved_context[:2000] + "..."
        if len(ground_truth_context) > 1000:
            ground_truth_context = ground_truth_context[:1000] + "..."
            
    except Exception as e:
        error_msg = f"RAG Query Failed: {e}"
        error_results = {
            'index': index, 
            'generated_answer': error_msg, 
            'error': str(e),
            'thread_id': thread_id
        }
        for m in ['answer_correctness', 'faithfulness', 'context_relevance', 'context_recall']:
            error_results[f'{m}_rating'] = 1
            error_results[f'{m}_explanation'] = error_msg
        return error_results

    # Prepare prompts
    prompts = {
        'answer_correctness': answer_correctness_prompt.format(
            question=question, 
            ground_truth_answer=ground_truth_answer, 
            generated_answer=generated_answer
        ),
        'faithfulness': retrieval_faithfulness_prompt.format(
            retrieved_context=retrieved_context, 
            generated_answer=generated_answer
        ),
        'context_relevance': context_relevance_prompt.format(
            question=question, 
            retrieved_context=retrieved_context
        ),
        'context_recall': context_recall_prompt.format(
            ground_truth_context=ground_truth_context, 
            retrieved_context=retrieved_context
        )
    }
    
    results = {
        'index': index, 
        'generated_answer': generated_answer, 
        'error': None,
        'thread_id': thread_id
    }
    
    # Evaluate each metric
    for name, prompt in prompts.items():
        rating, explanation = evaluate_single_metric(prompt, name, thread_id)
        results[f'{name}_rating'] = rating
        results[f'{name}_explanation'] = explanation
    
    return results

def worker_function(args):
    """Worker function for ThreadPoolExecutor."""
    data_processor, index, row, thread_id = args
    return evaluate_rag_row_parallel(data_processor, index, row, thread_id)

if __name__ == "__main__":
    VECTORSTORE_PATH = "./chroma_db_final"
    DOCSTORE_PATH = "./docstore_final"
    EVALUATION_DATASET_PATH = "qa_dataset.csv"
    OUTPUT_EVALUATION_PATH = "rag_full_evaluation_results.csv"
    SAMPLE_SIZE = None  
    MAX_WORKERS = min(len(API_KEYS), 8)  

    print("--- 1. Connecting to Existing RAG System ---")
    try:
        # Create main data processor
        main_data_processor = DataProcessor(
            vectorstore_path=VECTORSTORE_PATH,
            docstore_path=DOCSTORE_PATH,
            google_api_key=API_KEYS[0],
            verbose=False
        )
        stats = main_data_processor.get_collection_stats()
        if stats.get('total_documents', 0) == 0:
            raise ValueError(f"The RAG database at '{VECTORSTORE_PATH}' is empty.")
        print(f"✅ Connected successfully. Found {stats['total_documents']} documents in the vectorstore.")
        print("-" * 35)
    except Exception as e:
        print(f"FATAL: Could not connect to the RAG database. Error: {e}")
        exit()

    print(f"--- 2. Loading Evaluation Dataset ---")
    try:
        df = pd.read_csv(EVALUATION_DATASET_PATH)
        df.rename(columns={'answer': 'ground_truth_answer'}, inplace=True)
        
        if SAMPLE_SIZE and SAMPLE_SIZE < len(df):
            df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
            print(f"📝 Using sample of {len(df)} rows for testing")
        else:
            print(f"📝 Loaded {len(df)} question-answer pairs for evaluation.")
        print("-" * 35)
    except Exception as e:
        print(f"FATAL: Could not load evaluation dataset at '{EVALUATION_DATASET_PATH}'. Error: {e}")
        exit()

    print(f"--- 3. Running End-to-End RAG Evaluation (Parallel) ---")
    print(f"🔄 Processing rows in parallel with {MAX_WORKERS} workers...")
    
    # Create data processor instances for each worker
    data_processors = []
    for i in range(MAX_WORKERS):
        dp = DataProcessor(
            vectorstore_path=VECTORSTORE_PATH,
            docstore_path=DOCSTORE_PATH,
            google_api_key=API_KEYS[i % len(API_KEYS)],  # Distribute API keys
            verbose=False
        )
        data_processors.append(dp)
    
    # Prepare arguments for parallel processing
    tasks = []
    for index, row in df.iterrows():
        # Assign data processor based on index to distribute load
        dp_index = index % len(data_processors)
        thread_id = f"T{dp_index:02d}"
        tasks.append((data_processors[dp_index], index, row, thread_id))
    
    results = []
    completed_count = 0
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_index = {executor.submit(worker_function, task): task[1] for task in tasks}
        
        # Process completed tasks with progress bar
        with tqdm(total=len(df), desc="Evaluating RAG") as pbar:
            for future in as_completed(future_to_index):
                try:
                    result = future.result()
                    results.append(result)
                    completed_count += 1
                    pbar.update(1)
                    
                    # Progress update every 10 rows
                    if completed_count % 10 == 0:
                        print(f"✅ Completed {completed_count}/{len(df)} evaluations")
                        
                except Exception as e:
                    index = future_to_index[future]
                    print(f"❌ Error processing row {index}: {e}")
                    # Add error result
                    error_result = {
                        'index': index, 
                        'generated_answer': f"Error: {e}", 
                        'error': str(e),
                        'thread_id': 'ERROR'
                    }
                    for m in ['answer_correctness', 'faithfulness', 'context_relevance', 'context_recall']:
                        error_result[f'{m}_rating'] = 1
                        error_result[f'{m}_explanation'] = f"Error: {e}"
                    results.append(error_result)
                    pbar.update(1)
    
    print("\n--- 4. Consolidating and Saving Results ---")
    if results:
        # Sort results by index to maintain order
        results.sort(key=lambda x: x['index'])
        results_df = pd.DataFrame(results).set_index('index')
        final_df = df.join(results_df)
        final_df.to_csv(OUTPUT_EVALUATION_PATH, index=False)
        print(f"✅ Evaluation complete. Full results saved to '{OUTPUT_EVALUATION_PATH}'")
    else:
        print("❌ No results to save.")
        exit()
    
    print("-" * 35)

    print("--- 5. Evaluation Summary ---")
    METRICS = ['answer_correctness', 'faithfulness', 'context_relevance', 'context_recall']
    
    for metric in METRICS:
        col = f'{metric}_rating'
        if col in final_df.columns:
            ratings = pd.to_numeric(final_df[col], errors='coerce')
            print(f"\n{metric.replace('_', ' ').title()} Score:")
            print(f"  - Average: {ratings.mean():.2f} / 5")
            print("  - Distribution:")
            dist = final_df[col].value_counts().sort_index()
            for rating, count in dist.items():
                print(f"    {rating}: {count} ({count/len(final_df)*100:.1f}%)")

    errors = final_df['error'].notna().sum() if 'error' in final_df.columns else 0
    if errors > 0:
        print(f"\n⚠️ Total errors during evaluation: {errors}")
    
    # Thread performance summary
    if 'thread_id' in final_df.columns:
        print(f"\n🧵 Thread Performance:")
        thread_counts = final_df['thread_id'].value_counts()
        for thread_id, count in thread_counts.items():
            print(f"  {thread_id}: {count} evaluations")
    
    print(f"\n🎉 Evaluation completed successfully!")
    print(f"📊 Processed {len(final_df)} rows with {MAX_WORKERS} parallel workers")
    print(f"💾 Results saved to: {OUTPUT_EVALUATION_PATH}")