import litellm

response = litellm.completion(
    model="ollama/llama3.2",
    messages=[
        {
            "role": "user",
            "content": "Reply with: CrewAI LLM connection works"
        }
    ],
    api_base="http://localhost:11434"
)

print(response.choices[0].message.content)