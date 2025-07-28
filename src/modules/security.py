from transformers import pipeline

moderation_classifier = pipeline (
    "text-classification",
    model="unitary/toxic-bert"
)

def is_content_toxic(text: str) -> bool:
   
    try:
        results = moderation_classifier(text)
        if results[0]['label'].lower() == 'toxic' and results[0]['score'] > 0.5:
            return True
    except Exception as e:
        print(f"Error during toxicity check: {e}")
        return True

    return False