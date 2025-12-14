
import os
import boto3
import json

from src.ingestion import search_resume

from dotenv import load_dotenv

load_dotenv()

os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
os.environ["AWS_DEFAULT_REGION"] = os.getenv("AWS_REGION")

# Create Bedrock Runtime client
bedrock = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION"))

print("Bedrock client created in region:", os.getenv("AWS_REGION"))

CLAUDE_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"  # or whichever Claude model you're using

def ask_claude_with_context(question: str) -> str:
    # 1. Retrieve most relevant chunks from the resume
    top_results = search_resume(question, top_k=4)
    context_parts = [c for _, c in top_results]
    context = "\n\n---\n\n".join(context_parts)

    # 2. System instruction (goes in top-level "system", NOT as a message role)
    system_instruction = (
        "You are a chatbot that answers questions only about MY RESUME. "
        "Use only the information in the 'resume context'. "
        "If the answer is not in the resume, say you don't know or it's not in the resume."
    )

    # 3. User message (normal chat content)
    user_message = (
        f"Resume context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer concisely and professionally. "
        "If relevant, refer to specific projects, skills, or experiences mentioned in the resume context."
    )

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 512,
        "temperature": 0.1,
        # ✅ system instruction here, NOT as a message
        "system": system_instruction,
        # ✅ only user/assistant messages in this list
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message}
                ],
            }
        ],
    }

    response = bedrock.invoke_model(
        modelId=CLAUDE_MODEL_ID,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json",
    )

    response_body = json.loads(response["body"].read())
    answer = response_body["content"][0]["text"]
    return answer

if __name__ == "__main__":

    while True:
        q = input("Ask about your resume (or type 'exit'): ")
        if q.lower().strip() in ("exit", "quit"):
            break
        ans = ask_claude_with_context(q)
        print("\n=== Answer ===\n")
        print(ans)
        print("\n==============\n")



