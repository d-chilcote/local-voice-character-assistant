from skills.registry import registry
import logging

# Setup basic logging to see registry output
logging.basicConfig(level=logging.INFO)

def verify_fix():
    print("Testing calculator with EXTRA arguments...")
    # This used to crash because 'search_query' and 'thought' were passed to execute()
    result = registry.execute_skill(
        "calculator", 
        expression="123 * 456", 
        search_query="confusing query", 
        thought="I should calculate this"
    )
    
    print(f"Result: {result}")
    assert result == str(123 * 456), f"Expected {123*456}, got {result}"
    print("✅ Success! Extra arguments were filtered correctly.")

    print("\nTesting google_search with EXTRA arguments...")
    # google_search only expects (query, api_key=None). 
    # We'll mock the config to avoid needing a real API key for the import phase, 
    # but the filter should happen before it even tries to call the function.
    result = registry.execute_skill(
        "google_search",
        query="test query",
        random_arg="should be filtered"
    )
    # It might return a 'No API key' error string, but it shouldn't be a TypeError
    print(f"Result: {result}")
    assert "TypeError" not in str(result), "TypeError found in result!"
    print("✅ Success! google_search didn't crash on extra args.")

if __name__ == "__main__":
    try:
        verify_fix()
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        exit(1)
