import os
import json
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Please set GEMINI_API_KEY in your .env file.")

client = genai.Client(api_key=api_key)
MODEL_ID = 'gemini-2.5-flash' 

def generate_names_batch(count=50):
    """Generates a small batch of names."""
    prompt = f"""
    Generate a JSON array containing {count} authentic Cambodian names.
    The JSON schema must be an array of objects:
    [
      {{"last_kh": "ទិគ", "last_en": "TIT", "first_kh": "សំអុល", "first_en": "SAMOL", "gender": "ប្រុស"}},
      {{"last_kh": "សុខ", "last_en": "SOK", "first_kh": "ស្រីរ័ត្ន", "first_en": "SREYROTH", "gender": "ស្រី"}}
    ]
    """
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.8,
        )
    )
    return json.loads(response.text)

def generate_marks_batch(count=50):
    """Generates a small batch of marks."""
    prompt = f"""
    Generate a JSON array of strings containing {count} authentic Cambodian physical marks (ភិនភាគ).
    Example: ["សំលាកតូចចំ ០.៥សម ក្រោយក្រោមចុងចំព្នើមឆ្វេង", "ប្រជ្រុយលើបបូរមាត់ស្តាំ", "គ្មាន"]
    """
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.8,
        )
    )
    return json.loads(response.text)

def run_generator(target_count, batch_size, generator_func, description):
    results = []
    print(f"🚀 Starting generation for {description} (Target: {target_count})")
    
    while len(results) < target_count:
        remaining = target_count - len(results)
        current_batch_size = min(batch_size, remaining)
        
        print(f"--- Requesting {current_batch_size} items (Progress: {len(results)}/{target_count}) ---")
        try:
            batch = generator_func(current_batch_size)
            if isinstance(batch, list):
                results.extend(batch)
                print(f"✅ Received {len(batch)} items.")
            else:
                print("⚠️ Received non-list response, skipping...")
        except Exception as e:
            print(f"❌ Error in batch: {e}. Skipping and retrying...")
            time.sleep(2) # Small delay to avoid rate limits after error
            
    return results[:target_count]

if __name__ == "__main__":
    # 1. Generate Names
    # We use batch size of 50 to prevent JSON truncation
    all_names = run_generator(300, 50, generate_names_batch, "Names")
    with open("names.json", "w", encoding="utf-8") as f:
        json.dump(all_names, f, ensure_ascii=False, indent=2)
    print(f"🎉 Final: Saved {len(all_names)} names to names.json\n")

    # 2. Generate Marks
    all_marks = run_generator(200, 50, generate_marks_batch, "Marks")
    with open("marks.json", "w", encoding="utf-8") as f:
        json.dump(all_marks, f, ensure_ascii=False, indent=2)
    print(f"🎉 Final: Saved {len(all_marks)} marks to marks.json")