import re

# Step 1: Sliding Window
def sliding_window(text, window_size=100, overlap=30):
    chunks = []
    start = 0

    while start < len(text):
        chunk = text[start:start + window_size]
        chunks.append(chunk)
        start += (window_size - overlap)

    return chunks



# Step 2: Similarity
def similarity(query, text):
    query_words = set(re.findall(r'\w+', query.lower()))
    text_words = set(re.findall(r'\w+', text.lower()))

    if not text_words:
        return 0

    return len(query_words & text_words) / len(query_words)



# Step 3: Category Function
def categorize(text):
    text = text.lower()

    if "ai" in text or "machine learning" in text:
        return "Technology"
    elif "finance" in text or "money" in text:
        return "Finance"
    elif "legal" in text or "law" in text:
        return "Legal"
    else:
        return "General"



# Step 4: Build Knowledge Pyrami
def build_pyramid(chunks):
    pyramid = []

    for chunk in chunks:
        words = chunk.split()

        summary = " ".join(words[:10])
        keywords = list(set(words[:5]))
        category = categorize(chunk)

        entry = {
            "raw_text": chunk.strip(),
            "summary": summary,
            "keywords": keywords,
            "category": category
        }

        pyramid.append(entry)

    return pyramid



# Step 5: Retrieval
def retrieve(query, pyramid):
    best_score = 0
    best_source = ""

    for entry in pyramid:
        raw_score = similarity(query, entry["raw_text"])
        summary_score = similarity(query, entry["summary"])
        keyword_score = similarity(query, " ".join(entry["keywords"]))

        raw_score *= 1.2
        summary_score *= 1.1

        max_score = max(raw_score, summary_score, keyword_score)

        if max_score > best_score:
            best_score = max_score

            if max_score == raw_score:
                best_source = entry["raw_text"]
            elif max_score == summary_score:
                best_source = entry["summary"]
            else:
                best_source = entry["raw_text"]

    return best_source



# Step 6: Main Execution
if __name__ == "__main__":
    text = """AI models are trained using data. Machine learning involves algorithms.
    Finance systems handle money transactions. Legal systems handle laws."""

    chunks = sliding_window(text)
    pyramid = build_pyramid(chunks)

    # 🔥 PRINT ALL PYRAMID LEVELS
    print("\n--- Knowledge Pyramid ---\n")
    for i, entry in enumerate(pyramid):
        print(f"Chunk {i+1}:")
        print("Raw Text :", entry["raw_text"])
        print("Summary  :", entry["summary"])
        print("Category :", entry["category"])
        print("Keywords :", entry["keywords"])
        print("-" * 50)

    # Query
    query = "How are AI models trained?"

    result = retrieve(query, pyramid)

    # 🔥 FINAL OUTPUT
    print("\n--- Final Result ---\n")
    print("Query  :", query)
    print("Answer :", result)