
from ollama import chat

# Text we want to summarize
text = """
Artificial Intelligence is transforming many industries.
Machine learning allows computers to learn patterns from data.
Large Language Models can understand and generate human language.
These technologies are used in healthcare, education,
business, and software development.
"""

# Send the text to our local Ollama model
response = chat(
    model="llama3.2:3b",
    messages=[
        {
            "role": "system",
            "content": "You summarize text clearly and concisely."
        },
        {
            "role": "user",
            "content": f"Summarize this text in 2 sentences:\n\n{text}"
        }
    ],
    options={
        "temperature": 0.3
    }
)

print("SUMMARY:")
print(response.message.content)
