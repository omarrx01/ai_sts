
from rag_engine import retrieve_chunks, generate_answer

question = "What is the price of NovaPhone S?"

print("Retrieving relevant information...")
chunks = retrieve_chunks(question)

print("Generating AI answer...")
answer = generate_answer(question, chunks)

print("\nQUESTION:")
print(question)

print("\nAI ANSWER:")
print(answer)
