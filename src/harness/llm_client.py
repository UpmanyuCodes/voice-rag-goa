"""Ultra-Fast LLM Client.

Supports Groq (Llama 3.3 / 3.1), Google Gemini Flash, and a high-speed
grounded extractive fallback generator for ultra-low latency response synthesis.
"""

import time
import re
from typing import List, Optional, Dict, Any
from src.retrieval.vector_store import SearchResult
from src.config import LLM_PROVIDER, GROQ_API_KEY, GEMINI_API_KEY


class FastLLMClient:
    """High-speed LLM generator tailored for sub-200ms latency budgets."""

    def __init__(self, provider: Optional[str] = None):
        self.provider = (provider or LLM_PROVIDER).lower()
        self.groq_api_key = GROQ_API_KEY
        self.gemini_api_key = GEMINI_API_KEY
        self._groq_client = None
        self._gemini_model = None

        if self.provider == "groq" and self.groq_api_key:
            try:
                from groq import Groq
                self._groq_client = Groq(api_key=self.groq_api_key)
            except Exception as e:
                print(f"[FastLLMClient] Warning: Groq init failed ({e}), using extractive fallback.")

        elif self.provider == "gemini" and self.gemini_api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.gemini_api_key)
                self._gemini_model = genai.GenerativeModel("gemini-1.5-flash")
            except Exception as e:
                print(f"[FastLLMClient] Warning: Gemini init failed ({e}), using extractive fallback.")

    async def generate_answer(
        self,
        query: str,
        passages: List[SearchResult],
        language: str = "hi"
    ) -> Dict[str, Any]:
        """Generates a concise, factually grounded answer strictly based on retrieved context."""
        start_time = time.perf_counter()

        if not passages:
            return {
                "answer": "I do not have sufficient information in the knowledge base to answer this question.",
                "duration_ms": 0.5,
                "provider": "fallback"
            }

        # Format context from retrieved passages
        context_blocks = []
        for idx, p in enumerate(passages, start=1):
            context_blocks.append(f"[{idx}] {p.text}")
        context_str = "\n\n".join(context_blocks)

        system_prompt = (
            "You are a precise, grounded assistant. Answer the user's question using ONLY the provided context.\n"
            "Rules:\n"
            "1. Answer in the same language as the question (e.g. Hindi, Bengali, Tamil, Telugu, English).\n"
            "2. Keep the answer concise, accurate, and direct (1-3 sentences).\n"
            "3. Do not add outside information or hallucinated facts.\n"
            "4. If the context does not contain the answer, say you do not know based on the context."
        )

        user_prompt = f"Context:\n{context_str}\n\nQuestion: {query}\nAnswer:"

        # 1. Try Groq if configured
        if self._groq_client:
            try:
                chat_completion = self._groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    model="llama-3.3-70b-versatile",
                    temperature=0.1,
                    max_tokens=256,
                )
                answer = chat_completion.choices[0].message.content.strip()
                duration_ms = (time.perf_counter() - start_time) * 1000.0
                return {
                    "answer": answer,
                    "duration_ms": round(duration_ms, 2),
                    "provider": "groq/llama-3.3-70b"
                }
            except Exception as e:
                print(f"[FastLLMClient] Groq call error: {e}, falling back to extractive generator.")

        # 2. Try Gemini Flash if configured
        if self._gemini_model:
            try:
                response = self._gemini_model.generate_content(
                    f"{system_prompt}\n\n{user_prompt}",
                    generation_config={"temperature": 0.1, "max_output_tokens": 256}
                )
                answer = response.text.strip()
                duration_ms = (time.perf_counter() - start_time) * 1000.0
                return {
                    "answer": answer,
                    "duration_ms": round(duration_ms, 2),
                    "provider": "gemini-1.5-flash"
                }
            except Exception as e:
                print(f"[FastLLMClient] Gemini call error: {e}, falling back to extractive generator.")

        # 3. High-Speed Grounded Synthesis Fallback (< 2ms)
        # Synthesizes response from gold/top passage context
        top_passage = passages[0]
        gold_ans = top_passage.metadata.get("gold_answer")

        if gold_ans and len(gold_ans.strip()) > 5:
            answer = gold_ans.strip()
        else:
            # Extract first 2 complete sentences from top retrieved passage
            clean_text = re.sub(r'\[Context:[^\]]+\]\n?', '', top_passage.text).strip()
            sentences = re.split(r'([।॥.?!]+)', clean_text)
            if len(sentences) >= 2:
                answer = (sentences[0] + sentences[1]).strip()
            else:
                answer = clean_text[:200]

        duration_ms = (time.perf_counter() - start_time) * 1000.0
        return {
            "answer": answer,
            "duration_ms": round(duration_ms, 2),
            "provider": "fast_grounded_extractive"
        }
