from transformers import pipeline

# Load sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

# Input text
text = "The Generative AI workshop was extremely informative and useful."

# Predict sentiment
result = sentiment_analyzer(text)

# Print result
print(result)



# Load zero-shot classification pipeline
classifier = pipeline("zero-shot-classification")

# Input document
document = """
Artificial Intelligence and Machine Learning are transforming
industries through automation and intelligent decision-making.
"""

# Candidate labels
labels = ["Technology", "Sports", "Politics", "Entertainment"]

# Classify document
result = classifier(document, candidate_labels=labels)

# Print result
print(result)
