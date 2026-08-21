import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline


# ============================================================
# 1. CREATE THE KNOWLEDGE BASE
# ============================================================

documents = [
    """
    Generative Artificial Intelligence is a branch of AI that creates
    new content such as text, images, audio, video and computer programs.
    """,

    """
    Large Language Models are transformer-based models trained on massive
    text datasets. They are used for text generation, summarization,
    translation, question answering and conversational AI.
    """,

    """
    Retrieval-Augmented Generation combines information retrieval with
    text generation. It retrieves relevant documents from an external
    knowledge base and gives them to a language model as context.
    """,

    """
    Vector databases store high-dimensional embeddings and perform
    similarity searches. Examples of vector databases include FAISS,
    ChromaDB, Pinecone, Weaviate and Milvus.
    """,

    """
    Prompt engineering is the process of designing clear instructions
    that guide a language model to produce accurate and useful responses.
    Common techniques include zero-shot, few-shot and role-based prompting.
    """,

    """
    Fine-tuning adapts a pretrained language model to a specific domain
    or task by training it further using a smaller domain-specific dataset.
    """
]


# ============================================================
# 2. LOAD THE EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# ============================================================
# 3. CONVERT DOCUMENTS INTO EMBEDDINGS
# ============================================================

print("Creating document embeddings...")

document_embeddings = embedding_model.encode(
    documents,
    convert_to_numpy=True
)

# FAISS requires float32 vectors
document_embeddings = document_embeddings.astype("float32")


# ============================================================
# 4. NORMALIZE EMBEDDINGS
# ============================================================

faiss.normalize_L2(document_embeddings)


# ============================================================
# 5. CREATE THE FAISS VECTOR DATABASE
# ============================================================

embedding_dimension = document_embeddings.shape[1]

# Inner-product search on normalized vectors
# gives cosine similarity

vector_database = faiss.IndexFlatIP(embedding_dimension)

# Store document vectors in FAISS
vector_database.add(document_embeddings)


# ============================================================
# 6. LOAD THE PRETRAINED GENERATION MODEL
# ============================================================

print("Loading FLAN-T5 model...")

generator = pipeline(
    task="text2text-generation",
    model="google/flan-t5-base"
)


# ============================================================
# 7. DEFINE THE RETRIEVAL FUNCTION
# ============================================================

def retrieve_documents(query, top_k=2):
    """
    Retrieves the most relevant documents for a user query.
    """

    # Convert query into an embedding
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    # Normalize query vector
    faiss.normalize_L2(query_embedding)

    # Search for similar documents
    similarity_scores, document_indices = vector_database.search(
        query_embedding,
        top_k
    )

    retrieved_documents = []

    for index, score in zip(
        document_indices[0],
        similarity_scores[0]
    ):
        retrieved_documents.append({
            "document": documents[index].strip(),
            "score": float(score)
        })

    return retrieved_documents


# ============================================================
# 8. DEFINE THE ANSWER GENERATION FUNCTION
# ============================================================

def generate_answer(query, retrieved_documents):
    """
    Generates an answer using the retrieved context.
    """

    # Combine retrieved documents
    context = "\n\n".join(
        item["document"]
        for item in retrieved_documents
    )

    # Create the prompt
    prompt = f"""
Answer the question using only the information provided in the context.

Context:
{context}

Question:
{query}

Instructions:
1. Give a clear and concise answer.
2. Do not add information that is not present in the context.
3. If the answer is unavailable, state:
"The answer is not available in the knowledge base."

Answer:
"""

    # Generate answer
    result = generator(
        prompt,
        max_new_tokens=150,
        do_sample=False
    )

    return result[0]["generated_text"]


# ============================================================
# 9. EXECUTE THE RAG SYSTEM
# ============================================================

print("\nRETRIEVAL-AUGMENTED GENERATION SYSTEM")
print("=" * 55)

user_query = input("\nEnter your question: ")


# Retrieve relevant documents
retrieved_results = retrieve_documents(
    query=user_query,
    top_k=2
)


# Generate answer
answer = generate_answer(
    query=user_query,
    retrieved_documents=retrieved_results
)


# ============================================================
# 10. DISPLAY RETRIEVED DOCUMENTS
# ============================================================

print("\nRETRIEVED DOCUMENTS")
print("-" * 55)

for number, item in enumerate(
    retrieved_results,
    start=1
):
    print(f"\nDocument {number}:")
    print(item["document"])
    print(f"Similarity Score: {item['score']:.4f}")


# ============================================================
# 11. DISPLAY GENERATED ANSWER
# ============================================================

print("\nGENERATED ANSWER")
print("-" * 55)
print(answer)