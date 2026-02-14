import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import asyncio
from app.infrastructure.gen_ai_client import MockGenAIClient

async def test_mock_selection():
    client = MockGenAIClient()
    
    test_cases = [
        ("Suspect is a male of European ancestry with blond hair.", "european", "male"),
        ("Suspect is a female of African ancestry with dark skin.", "african", "female"),
        ("Suspect is an Asian male.", "asian", "male"),
        ("Suspect is a Hispanic female.", "hispanic", "female")
    ]
    
    print("Testing MockGenAIClient selection logic...")
    for prompt, expected_cat, expected_gender in test_cases:
        result = await client.generate_suspect_visual(prompt)
        print(f"\nPrompt: {prompt}")
        print(f"Result URL: {result.image_url}")
        print(f"Metadata: {result.metadata}")
        
        # Verify category in metadata
        detected = result.metadata.get("detected_phenotype", "")
        if expected_cat in detected and expected_gender in detected:
            print("✅ PASS: Correctly detected category/gender")
        else:
            print(f"❌ FAIL: Expected {expected_cat}/{expected_gender}, got {detected}")

if __name__ == "__main__":
    asyncio.run(test_mock_selection())
