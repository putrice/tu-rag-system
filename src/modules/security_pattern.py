import re

def is_prompt_injection(query: str) -> bool:
    
    # Simple regex to detect common prompt injection patterns
    injection_patterns = [
        r"ignore.*instructions",
        r"disregard the above",
        r"you are now",
        r"act as",
        r"your new instructions are",
        r"system prompt",
        r"reveal your prompt",
        r"confidential",
        r"jailbreak",
        r"hack"
    ]
    
    for pattern in injection_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return True
            
    return False