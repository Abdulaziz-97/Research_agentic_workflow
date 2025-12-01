import asyncio
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from agents.llm import DeepSeekChatOpenAI
from config.settings import settings

load_dotenv()

async def test_deepseek_reasoning():
    print("Initializing DeepSeekChatOpenAI...")
    llm = DeepSeekChatOpenAI(
        model=settings.openai_model,
        openai_api_key=settings.openai_api_key,
        openai_api_base=settings.openai_base_url,
        temperature=0.7
    )
    
    print(f"Model: {settings.openai_model}")
    print(f"Base URL: {settings.openai_base_url}")
    
    msg = HumanMessage(content="What is 2+2? Explain your reasoning.")
    
    print("\n--- Testing Invoke ---")
    try:
        # Force non-streaming for now to test _create_chat_result
        response = await llm.ainvoke([msg])
        print("Response type:", type(response))
        try:
            print("Content:", response.content.encode('utf-8', errors='replace').decode('utf-8'))
        except:
            print("Content: <encoding error>")
        print("Additional Kwargs:", response.additional_kwargs)
        
        if "reasoning_content" in response.additional_kwargs:
            print("SUCCESS: reasoning_content found in additional_kwargs")
        else:
            print("FAILURE: reasoning_content NOT found in additional_kwargs")
            
    except Exception as e:
        print(f"Error during invoke: {e}")

    print("\n--- Testing Stream (Overridden to Non-Streaming) ---")
    try:
        full_response = None
        async for chunk in llm.astream([msg]):
            if full_response is None:
                full_response = chunk
            else:
                full_response += chunk
                
        print("Streamed Response type:", type(full_response))
        try:
            print("Streamed Content:", full_response.content.encode('utf-8', errors='replace').decode('utf-8'))
        except:
            print("Streamed Content: <encoding error>")
        print("Streamed Additional Kwargs:", full_response.additional_kwargs)
        
        if "reasoning_content" in full_response.additional_kwargs:
            print("SUCCESS: reasoning_content found in streamed response")
        else:
            print("FAILURE: reasoning_content NOT found in streamed response")
            
    except Exception as e:
        print(f"Error during stream: {e}")

if __name__ == "__main__":
    asyncio.run(test_deepseek_reasoning())
