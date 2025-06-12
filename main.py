import os,random
import re
from typing import List
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from langchain_openai import ChatOpenAI
from openai import OpenAI
from fastapi.responses import FileResponse
from bs4 import BeautifulSoup
from urllib.parse import quote
import tempfile
import fitz
                        


load_dotenv()

LLMFOUNDRY_TOKEN = os.getenv("LLMFOUNDRY_TOKEN")
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID=os.getenv("GOOGLE_CSE_ID")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def get_home():
    return FileResponse("static/index.html")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextInput(BaseModel):
    text: str
    model: str = "text-embedding-3-large"

class PromptInput(BaseModel):
    text: str

class LLMTestInput(BaseModel):
    prompt: str
    model_id: str
    embedding_model: str

class AnalysisRequest(BaseModel):
    original_content: str
    llm_response: str

class SearchBasedRequest(BaseModel):
    prompt: str
    original_content: str

class SimilarityRequest(BaseModel):
    original_content: str
    llm_response: str
    search_based_answer: str


def get_openai_client():
    return OpenAI(api_key=f"{LLMFOUNDRY_TOKEN}:LLM-plagiarism-detector",
        base_url="https://llmfoundry.straive.com/openai/v1/"
    )

def get_chat_model(model_name="gpt-4o-mini"):
    return ChatOpenAI(openai_api_base="https://llmfoundry.straive.com/openai/v1/",
        openai_api_key=f"{LLMFOUNDRY_TOKEN}:LLM-plagiarism-detector",
        model=model_name,
    )

def get_embedding(text: str, model: str = "text-embedding-3-large") -> List[float]:
    try:
        client = get_openai_client()
        response = client.embeddings.create(
            input=text,
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        raise Exception(f"Error generating embedding: {str(e)}")

async def get_llm_response(prompt: str, model_id: str) -> str:
    try:
        response = requests.post(
            "https://llmfoundry.straive.com/openrouter/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {LLMFOUNDRY_TOKEN}",
                "Content-Type": "application/json"
            },
            json={"model": model_id,"messages": [{"role": "user", "content": prompt}]},
            verify=False
        )
        response.raise_for_status()
        data = response.json()
        if "choices" in data and len(data["choices"]) > 0:
            if "message" in data["choices"][0] and "content" in data["choices"][0]["message"]:
                return data["choices"][0]["message"]["content"]
        raise Exception("Unexpected response format from OpenRouter")
    except Exception as e:
        raise Exception(f"Error getting LLM response: {str(e)}")

@app.post("/generate-questions")
async def generate_questions(input_data: PromptInput):
    try:
        chat_model = get_chat_model()
        response = chat_model.invoke(
            [
                {"role": "user", "content": f"Create 5 unique, natural-sounding questions that subtly test if an AI knows a specific topic, without directly quoting it. Format your response as a JSON array with 5 question strings. Content:\n\n{input_data.text}"}
            ]
        )
        content = response.content.strip()
        potential_questions = [line.strip() for line in content.split('\n') if '?' in line]
        if potential_questions and len(potential_questions) >= 3:
            return {"questions": potential_questions[:5]}
            
    except Exception as e:
       raise HTTPException(status_code=500, detail=f"Error generating questions: {str(e)}")


@app.post("/test-llm")
async def test_llm(input_data: LLMTestInput):
    try:
        llm_response = await get_llm_response(input_data.prompt, input_data.model_id)
        return {
            "llm_response": llm_response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error testing LLM: {str(e)}")

@app.post("/analyze-response")
async def analyze_response(analysis_request: AnalysisRequest):
    try:
        chat_model = get_chat_model("gpt-4o-mini") 
        prompt = f"""
        ORIGINAL CONTENT:
        {analysis_request.original_content}
        
        LLM RESPONSE:
        {analysis_request.llm_response}
        
        TASK: How likely is it that the LLM's response was generated based on this original content versus from general knowledge? Provide your assessment and reasoning.
        """
        
        response = chat_model.invoke(prompt)
        analysis = response.content.strip()
        return {"analysis": analysis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing response: {str(e)}")

       
@app.post("/search-based-answer")
async def search_based_answer(request: SearchBasedRequest):
    """Generate an answer based on web search results"""
    try:
        clean_prompt = request.prompt.replace('"', '').replace("'", '')
        search_url = f"https://www.googleapis.com/customsearch/v1?key={GOOGLE_API_KEY}&cx={GOOGLE_CSE_ID}&q={quote(clean_prompt)}&num=10"
        
        search_data = requests.get(search_url).json()
        urls = [item['link'] for item in search_data.get('items', [])[:6] if 'link' in item]  
        
        all_content = []
        successful_extractions = 0
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0'
        ]
        
        total_content_size = 0
        max_content_size = 3 * 1024 * 1024  
        
        for url in urls:
            if successful_extractions >= 3:
                break
            
            try:
                headers = {
                    'User-Agent': random.choice(user_agents),
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Referer': 'https://www.google.com/'
                }
                
                if url.lower().endswith(".pdf"):
                    pdf_response = requests.get(url, headers=headers, timeout=15, verify=False)
                    if pdf_response.status_code == 200:
                        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
                        temp_file.write(pdf_response.content)
                        temp_file.close()
                        
                        try:
                            pdf_text = ""
                            with fitz.open(temp_file.name) as pdf_document:
                                for page_num in range(min(200, len(pdf_document))):
                                    pdf_text += pdf_document.load_page(page_num).get_text()
                            
                            pdf_text = re.sub(r'\s+', ' ', pdf_text).strip()
                            content_entry = f"From PDF {url}:\n{pdf_text}\n\n"
                            all_content.append(content_entry)
                            total_content_size += len(content_entry.encode('utf-8'))
                            successful_extractions += 1
                        finally:
                            try:
                                os.unlink(temp_file.name)
                            except:
                                pass
                else:
                    response = requests.get(url, headers=headers, timeout=10, verify=False, allow_redirects=True)
                    if response.status_code == 200 and 'text/html' in response.headers.get('Content-Type', '').lower():
                        soup = BeautifulSoup(response.text, 'html.parser')
                        for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
                            element.extract()
                        
                        main_content = None
                        for tag in ['main', 'article', 'div.content', 'div.main', 'div#content', 'div#main']:
                            main_elements = soup.select(tag)
                            if main_elements:
                                main_content = max(main_elements, key=lambda x: len(x.get_text()))
                                break
                        
                        text = (main_content or soup.body).get_text(separator=' ', strip=True) if main_content or soup.body else ""
                        text = re.sub(r'\s+', ' ', text).strip()
                        
                        if len(text) > 100:
                            successful_extractions += 1
                            content_entry = f"From {url}:\n{text}\n\n"
                            all_content.append(content_entry)
                            total_content_size += len(content_entry.encode('utf-8'))
            except Exception as e:
                continue
            
            if total_content_size >= max_content_size:
                break
            
        scraped_content = "\n".join(all_content)
        
        if len(scraped_content.encode('utf-8')) > max_content_size:
            avg_bytes_per_char = len(scraped_content.encode('utf-8')) / len(scraped_content)
            approx_chars = int(max_content_size / avg_bytes_per_char)
            scraped_content = scraped_content[:approx_chars]
        
        final_prompt = f"""Based ONLY on the following web content, answer this question: {request.prompt}
            
Use ONLY the information in the provided web content.
Web Content:
{scraped_content}"""

        search_based_answer = get_chat_model("gpt-4.1-nano").invoke(final_prompt).content.strip()
        
        return {
            "search_based_answer": search_based_answer,
            "scraped_content": scraped_content
        }
    except Exception as e:
        return f"Error: {str(e)}"
        
@app.post("/calculate-similarity")
async def calculate_similarity(request: SimilarityRequest):
    try:   
        original_embedding_arr = np.array(get_embedding(request.original_content)).reshape(1, -1)
        llm_response_embedding_arr = np.array(get_embedding(request.llm_response)).reshape(1, -1)
        search_based_embedding_arr = np.array(get_embedding(request.search_based_answer)).reshape(1, -1)
        
        original_vs_llm = float(cosine_similarity(original_embedding_arr, llm_response_embedding_arr)[0][0])
        original_vs_search = float(cosine_similarity(original_embedding_arr, search_based_embedding_arr)[0][0])
        llm_vs_search = float(cosine_similarity(llm_response_embedding_arr, search_based_embedding_arr)[0][0])
        
        return {
            "original_vs_llm": original_vs_llm,
            "original_vs_search": original_vs_search,
            "llm_vs_search": llm_vs_search
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating similarity: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)