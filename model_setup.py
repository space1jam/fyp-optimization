from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_openai import OpenAIEmbeddings
from typing import List, Any, Dict, Optional, Literal
import os
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import time
from numpy import dot
from numpy.linalg import norm
from functools import lru_cache
import logging


class Chatbot:
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.0) -> None:
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
        )
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.messages = [SystemMessage(content="")]
        self.vectorstore = None
        self.retriever = None
        self.conversation_chain = None
        
        # Self-refinement parameters
        self.refinement_iterations = 3
        self.refinement_threshold = 0.8
        self.last_refinement_steps = []
        self.last_confidence_score = 1.0
        self.last_refinement_metrics = {
            'time': 0,
            'iterations': 0
        }
        self.last_sources = []
        self.formatted_sources = ""

    def __call__(self, prompt: str) -> str:
        self.last_refinement_steps = []

        try:
            self.validate_config()
            self.messages.append(HumanMessage(content=prompt))
            
            start_time = time.time()
            
            if self.vectorstore is not None:
                result = self.execute_with_retrieval_and_refinement(prompt)
            else:
                result = self.execute_with_refinement(prompt)
                
            self.messages.append(AIMessage(content=result))
            
            self.last_refinement_metrics = {
                'time': time.time() - start_time,
                'iterations': len(self.last_refinement_steps)
            }
            
            return result
        
        except Exception as e:
            self.last_refinement_steps = []
            self.last_confidence_score = 0.0
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            self.messages.append(AIMessage(content=error_msg))
            return error_msg

    def execute_with_refinement(self, prompt: str) -> str:
        """
        Refines an answer through iterative feedback loops.
    
        Performs:
        - Initial LLM response generation
        - Multiple refinement iterations (up to refinement_iterations)
        - Early termination if refinement threshold met
        
        Tracks refinement steps and confidence scores.
        Returns the final refined answer.
        """
        self.last_refinement_steps = []
        initial_answer = self.call_llm(self.messages).content
        refined_answer = initial_answer
        self.last_confidence_score = 1.0
        
        for i in range(self.refinement_iterations):
            feedback = self.generate_feedback(prompt, refined_answer)
            new_refined = self.refine_answer(prompt, refined_answer, feedback)
            
            self.last_refinement_steps.append({
                'iteration': i + 1,
                'feedback': feedback,
                'previous_answer': refined_answer,
                'refined_answer': new_refined
            })
            
            if self.is_refinement_sufficient(refined_answer, new_refined):
                break
                
            refined_answer = new_refined
            self.last_confidence_score = self.calculate_confidence(refined_answer)
        
        return refined_answer


    def execute_with_retrieval_and_refinement(self, query: str) -> str:

        """
        Retrieves relevant documents, refines answer using document context.
    
        Handles cases:
        - No relevant documents found (with fallback search)
        - Unknown initial answers
        - Insufficient refinement
        
        Stores sources and refinement steps for reference.
        Returns refined answer or fallback message.
        """
         
        self.last_refinement_steps = []
        
        #First search for relevant documents
        docs = self.get_relevant_documents(query)

        # If no relevant documents found, try a secondary search
        if not docs or all(self.is_unknown_answer(doc['content']) for doc in docs[:5]):
            docs = self.expand_document_search(query, docs)
        
        if not docs or all(self.is_unknown_answer(doc['content']) for doc in docs[:30]):
            return "There is insufficient information to answer that question."
        
        qa_result = self.conversation_chain({"query": query})
        initial_answer = qa_result["result"]
        
        if self.is_unknown_answer(initial_answer):
            return initial_answer
            
        refined_answer = initial_answer
        self.last_confidence_score = 1.0

        self.last_sources = docs
        self.formatted_sources = self._format_sources(docs)
        
        for i in range(self.refinement_iterations):
            context = "\n\n".join([doc["content"] for doc in docs[:5]])
            feedback = self.generate_rag_feedback(query, refined_answer, context)
            new_refined = self.refine_rag_answer(query, refined_answer, feedback, context)
            
            self.last_refinement_steps.append({
                'iteration': i + 1,
                'feedback': feedback,
                'previous_answer': refined_answer,
                'refined_answer': new_refined,
                'sources': docs
            })
            
            if self.is_refinement_sufficient(refined_answer, new_refined):
                break
                
            refined_answer = new_refined
            self.last_confidence_score = self.calculate_confidence(refined_answer)
        
        return refined_answer

    def generate_feedback(self, prompt: str, answer: str) -> str:
        feedback_prompt = f"""
        Analyze the following answer for potential issues:
        Original prompt: '{prompt}'
        Current answer: {answer}
        
        Provide specific feedback on:
        1. Factual inaccuracies
        2. Logical inconsistencies
        3. Missing information
        4. Areas needing clarification
        5. Potential hallucinations
        
        Be concise but specific. If the answer is good, say so.
        
        Feedback:
        """
        return self.call_llm([HumanMessage(content=feedback_prompt)]).content

    def generate_rag_feedback(self, query: str, answer: str, context: str) -> str:
        feedback_prompt = f"""
        Analyze this answer in relation to the provided context:
        Query: {query}
        Context: {context}
        Current answer: {answer}
        
        Identify:
        1. Claims not supported by context
        2. Missing relevant information
        3. Inaccurate interpretations
        4. Areas needing more precision
        
        Detailed feedback:
        """
        return self.call_llm([HumanMessage(content=feedback_prompt)]).content

    def refine_answer(self, prompt: str, answer: str, feedback: str) -> str:
        refine_prompt = f"""
        Original prompt: {prompt}
        Initial answer: {answer}
        Feedback: {feedback}
        
        Please revise the answer to:
        1. Address all feedback points
        2. Maintain accuracy and clarity
        3. Keep the response concise
        
        If the feedback suggests the answer is good, return it unchanged.

        Think step-by-step, provide reasoning and then the final answer.
        
        Improved answer:
        """
        return self.call_llm([HumanMessage(content=refine_prompt)]).content

    def refine_rag_answer(self, query: str, answer: str, feedback: str, context: str) -> str:
        refine_prompt = f"""
        Query: {query}
        Context: {context}
        Initial answer: {answer}
        Feedback: {feedback}
        
        Revise the answer to:
        1. Better align with context
        2. Address feedback points
        3. Maintain clarity and accuracy
        
        Improved answer:
        """
        return self.call_llm([HumanMessage(content=refine_prompt)]).content

    def is_refinement_sufficient(self, previous_answer: str, refined_answer: str) -> bool:
        if previous_answer.strip() == refined_answer.strip():
            return True
            
        prev_embedding = self._cached_embedding(previous_answer)
        refined_embedding = self._cached_embedding(refined_answer)
        
        cos_sim = dot(prev_embedding, refined_embedding)/(norm(prev_embedding)*norm(refined_embedding))
        return cos_sim > self.refinement_threshold

    @lru_cache(maxsize=100)
    def _cached_embedding(self, text: str) -> List[float]:
        return self.embeddings.embed_query(text)

    def calculate_confidence(self, answer: str) -> float:
        if not answer or self.is_unknown_answer(answer):
            return 0.0
            
        confidence_prompt = f"""
        Assess this answer's confidence (0.0-1.0) based on:
        1. Factual accuracy
        2. Completeness
        3. Clarity
        
        Answer: {answer}
        
        Provide only a numerical score between 0.0 and 1.0.
        """
        try:
            score = float(self.call_llm([HumanMessage(content=confidence_prompt)]).content)
            return max(0.0, min(1.0, score))
        except:
            return 0.8

    def is_unknown_answer(self, answer: str) -> bool:
        if not answer:
            return True
        lower_answer = answer.lower()
        return any(phrase in lower_answer for phrase in [
            "don't know", "not sure", "no information", 
            "unable to", "doesn't mention", "not specified"
        ])

    def validate_config(self) -> bool:
        if not self.llm:
            raise ValueError("Language model not initialized")
        if self.vectorstore and not self.retriever:
            raise ValueError("Vectorstore loaded but retriever not setup")
        return True

    def call_llm(self, message_thread: List[Any]) -> AIMessage:
        return self.llm.invoke(message_thread)
    
    def load_vectorstore(self, vectorstore_path: str) -> None:
        if not os.path.isdir(vectorstore_path):
            raise FileNotFoundError(f"Vectorstore path '{vectorstore_path}' not found")

        try:
            self.vectorstore = Chroma(
                persist_directory=vectorstore_path,
                embedding_function=self.embeddings,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load vectorstore: {str(e)}")
    
    def setup_retriever(
        self, 
        search_type: Literal["similarity", "mmr"] = "similarity", 
        **kwargs: Any
    ) -> None:
        if not self.vectorstore:
            raise ValueError("Vectorstore must be loaded first")

        if search_type == "mmr":
            self.retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": kwargs.get("k", 5),
                    "lambda_mult": kwargs.get("lambda_mult", 0.5)
                }
            )
        else:
            search_kwargs = {"k": kwargs.get("k", 5)}
            if "score_threshold" in kwargs:
                search_kwargs["score_threshold"] = kwargs["score_threshold"]
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs=search_kwargs
            )
    
    def create_qa_chain(self) -> None:
        if not self.retriever:
            raise ValueError("Retriever must be configured first")

        prompt_template = """You are a helpful assistant. Use only the context to answer the question.
        Context: {context}
        Question: {question}
        Answer based on the context. If you don't know, say you don't know. 
        If the document does not contain any information, do not assume or make up answer.

        """
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=prompt_template
        )

        self.conversation_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
    
    def initialize_for_qa(self, vectorstore_path: str, search_type: str = "similarity", **kwargs) -> None:
        self.load_vectorstore(vectorstore_path)
        self.setup_retriever(search_type, **kwargs)
        self.create_qa_chain()
        self.messages = [SystemMessage(
            content="I'm a chatbot that can answer questions based on specific documents."
        )]
    
    def get_relevant_documents(self, query: str, top_k: int = 5) -> List[Dict]:
        if not self.retriever:
            raise ValueError("Retriever is not set up.")
        
        docs = self.retriever.get_relevant_documents(query)
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in docs[:top_k]
        ]
    
    def _format_sources(self, docs: List[Dict]) -> str:
        if not docs:
            return "No sources available"
    
        formatted = []
        for i, doc in enumerate(docs[:5]):  # Limit to top 5 sources
            source = doc['metadata'].get('source', 'unknown')
            if isinstance(source, str) and source.startswith("http"):
                formatted.append(f"{i+1}. [Source {i+1}]({source})")
            else:
                formatted.append(f"{i+1}. {source} (Page {doc['metadata'].get('page', '?')})")
        
        return "\n".join(formatted)

    def expand_document_search(self, query: str, initial_docs: List[Dict]) -> List[Dict]:
        """Secondary search with relaxed parameters when initial results are empty/unhelpful"""
        # If we already found relevant docs, return them
        if initial_docs and not all(self.is_unknown_answer(doc['content']) for doc in initial_docs[:5]):
            return initial_docs
        
        # Fallback search parameters
        fallback_params = {
            "search_type": "similarity",
            "k": 30,  # Get more docs this time
            "score_threshold": 0.4  # Lower similarity threshold
        }
        
        # Perform fallback search
        try:
            backup_docs = self.retriever.get_relevant_documents(
                query,
                **fallback_params
            )
            return [
                {"content": doc.page_content, "metadata": doc.metadata}
                for doc in backup_docs[:30]  # Return more docs for safety
            ]
        except Exception as e:
            logging.warning(f"Fallback search failed: {str(e)}")
            return initial_docs  # Return original docs if fallback fails

    def evaluate_answer(self, query: str, response: str, reference_answer: Optional[str] = None) -> Dict:
        evaluation = {
            'relevance': self._evaluate_relevance(query, response),
            'faithfulness': self._evaluate_faithfulness(response),
            'coherence': self._evaluate_coherence(response),
            'hallucination_score': self._evaluate_hallucination(response),
        }
        
        if reference_answer:
            evaluation['similarity_to_reference'] = self._compare_to_reference(response, reference_answer)
        
        return evaluation

    def _evaluate_relevance(self, query: str, response: str) -> float:
        prompt = f"""
        Rate the relevance of this response to the question (0.0-1.0):
        Question: {query}
        Response: {response}
        Provide only a numerical score.
        """
        try:
            return float(self.call_llm([HumanMessage(content=prompt)]).content)
        except:
            return 0.5

    def _evaluate_faithfulness(self, response: str) -> float:
        prompt = f"""
        Rate the factual faithfulness (0.0-1.0):
        Response: {response}
        Provide only a numerical score.
        """
        try:
            return float(self.call_llm([HumanMessage(content=prompt)]).content)
        except:
            return 0.5

    def _evaluate_coherence(self, response: str) -> float:
        prompt = f"""
        Rate the coherence (0.0-1.0):
        Response: {response}
        Provide only a numerical score.
        """
        try:
            return float(self.call_llm([HumanMessage(content=prompt)]).content)
        except:
            return 0.5

    def _evaluate_hallucination(self, response: str) -> float:
        prompt = f"""
        Estimate hallucination score (0.0-1.0):
        Response: {response}
        Provide only a numerical score.
        """
        try:
            return float(self.call_llm([HumanMessage(content=prompt)]).content)
        except:
            return 0.3

    def _compare_to_reference(self, response: str, reference: str) -> float:
        prompt = f"""
        Compare these answers and rate similarity (0.0-1.0):
        Response: {response}
        Reference: {reference}
        Provide only a numerical score.
        """
        try:
            return float(self.call_llm([HumanMessage(content=prompt)]).content)
        except:
            return 0.5

    def reset_conversation(self, system_message: Optional[str] = None) -> None:
        self.messages = [
            SystemMessage(content=system_message or self.messages[0].content)
        ]
        self.last_refinement_steps = []
        self.last_confidence_score = 1.0

    def trim_messages(self, max_history: int = 10) -> None:
        if len(self.messages) > max_history:
            self.messages = [self.messages[0]] + self.messages[-max_history+1:]

    def cleanup(self) -> None:
        if hasattr(self, 'vectorstore') and self.vectorstore:
            self.vectorstore.delete_collection()
        self.messages = [SystemMessage(content="")]