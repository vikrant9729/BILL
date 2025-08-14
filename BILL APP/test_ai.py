#!/usr/bin/env python3
"""
Test script to verify Gemini API key is working
"""

import requests

def test_gemini_api():
    """Test the Gemini API with the hardcoded key"""
    
    api_key = "AIzaSyCISRlocKiVnAlakm5GEllJu6VVnrBdP6s"
    url = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent"
    
    headers = {
        "Content-Type": "application/json",
    }
    
    data = {
        "contents": [{
            "parts": [{"text": "Hello! Can you help me with medical billing?"}]
        }]
    }
    
    try:
        print("Testing Gemini API...")
        response = requests.post(
            f"{url}?key={api_key}",
            headers=headers,
            json=data,
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            ai_response = result['candidates'][0]['content']['parts'][0]['text']
            print("✅ SUCCESS! AI is working.")
            print(f"AI Response: {ai_response}")
        else:
            print(f"❌ ERROR: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")

if __name__ == "__main__":
    test_gemini_api() 