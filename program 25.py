import openai

# Set your OpenAI GPT-3 API key
openai.api_key = 'YOUR_API_KEY'

def generate_text(prompt):
    response = openai.Completion.create(
        engine="text-davinci-002",  # Choose an appropriate engine
        prompt=prompt,
        max_tokens=100  # Adjust as needed
    )
    return response.choices[0].text.strip()

# Example usage
prompt = "Write a short story about a robot and a human"
generated_text = generate_text(prompt)
print("Generated Text:")
print(generated_text)
