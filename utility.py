import difflib

def is_similar_command(text, target, threshold=0.75):
    ratio = difflib.SequenceMatcher(None, text, target).ratio()
    return ratio >= threshold
