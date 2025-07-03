# # chatbot_tab.py
# import streamlit as st
# import os
# from model_setup import Chatbot
# from document_processor_bot import DocumentProcessor
# from dotenv import load_dotenv

# load_dotenv('.env', override=True)

# def document_chat_tab():
#     if "messages" not in st.session_state:
#         st.session_state["messages"] = []

#     if "chatbot" not in st.session_state:
#         chatbot = Chatbot()
#         st.session_state["chatbot"] = chatbot
#     else:
#         chatbot = st.session_state["chatbot"]
        
#     if "document_processor" not in st.session_state:
#         document_processor = DocumentProcessor()
#         st.session_state["document_processor"] = document_processor
#     else:
#         document_processor = st.session_state["document_processor"]

#     st.title("💬 Document Q&A Chatbot")

#     with st.expander("Chatbot Settings"):
#         st.subheader("Document Management")

#         st.subheader("Process New Documents")
#         doc_dir = st.text_input("Document Directory Path", value="./documents")
#         store_name = st.text_input("Vector Store Name", value="my_vectorstore")

#         if st.button("Process Documents"):
#             with st.spinner("Processing documents..."):
#                 try:
#                     if not os.path.exists(doc_dir):
#                         st.error(f"Directory '{doc_dir}' does not exist!")
#                     else:
#                         vectorstore_path = document_processor.process_directory(doc_dir, store_name)
#                         st.success(f"Documents processed and stored as '{store_name}'")
#                 except Exception as e:
#                     st.error(f"Error processing documents: {str(e)}")

#         st.subheader("Select Vector Store")
#         selected_store = None
#         try:
#             available_stores = document_processor.list_available_vectorstores()
#             if available_stores:
#                 selected_store = st.selectbox("Available Vector Stores", options=available_stores)

#                 if st.button("Load Selected Vector Store"):
#                     with st.spinner("Loading vector store..."):
#                         store_path = os.path.join(document_processor.vector_db_path, selected_store)
#                         chatbot.initialize_for_qa(store_path)
#                         st.success(f"Vector store '{selected_store}' loaded successfully!")
#                         st.session_state["messages"] = []
#             else:
#                 st.info("No vector stores available. Process documents first.")
#         except Exception as e:
#             st.error(f"Error listing vector stores: {str(e)}")

#         st.subheader("Self-Refinement Settings")
#         chatbot.refinement_iterations = st.slider(
#             "Max Refinement Iterations", 1, 5, 3,
#             help="How many times the answer should be refined (max)"
#         )
#         chatbot.refinement_threshold = st.slider(
#             "Refinement Confidence Threshold", 0.1, 1.0, 0.8, 0.1,
#             help="When to stop refining (higher = more strict)"
#         )

#         st.subheader("Visualize Query Embedding")
#         query_input = st.text_input("Query to Visualize", value="Write query here")
#         if st.button("Visualize Embedding"):
#             if selected_store and query_input.strip():
#                 with st.spinner("Generating UMAP projection..."):
#                     try:
#                         document_processor.visualize_query_projection(
#                             store_name=selected_store,
#                             query=query_input
#                         )
#                         st.success("Visualization complete! Check the embedding_visualization folder.")
#                     except Exception as e:
#                         st.error(f"Visualization failed: {e}")
#             else:
#                 st.warning("Please select a vector store and enter a query.")

#         st.subheader("Debug Document Chunks")
#         if st.button("Preview Split Chunks"):
#             with st.spinner("Loading and splitting documents..."):
#                 try:
#                     docs = document_processor.load_documents(doc_dir)
#                     chunks = document_processor.split_documents(docs)

#                     st.write(f"📄 Total Chunks: {len(chunks)}")
#                     for i, chunk in enumerate(chunks[:5]):
#                         st.markdown(f"**Chunk {i+1}**")
#                         st.code(chunk.page_content[:500])
#                         st.text(f"Metadata: {chunk.metadata}")
#                 except Exception as e:
#                     st.error(f"Failed to preview chunks: {e}")

#         st.subheader("Evaluation Settings")
#         enable_evaluation = st.toggle("Show Evaluation Metrics", value=False)
#         if enable_evaluation:
#             reference_answer = st.text_area("Optional Reference Answer (for comparison)")
#         else:
#             reference_answer = ""

#     with st.container():
#         for message in st.session_state.messages[:]:
#             with st.chat_message(message["role"]):
#                 st.markdown(message["content"])

#                 if message.get("role") == "assistant" and message.get("refinement_steps"):
#                     with st.expander("View Refinement Process"):
#                         for i, step in enumerate(message["refinement_steps"]):
#                             st.markdown(f"**Step {i+1}:**")
#                             if "previous_answer" in step:
#                                 st.markdown (f"*Previous Answer:* {step['previous_answer']}")
#                             if "feedback" in step:
#                                 st.markdown (f"*Feedback:* {step['feedback']}")
#                             if "refined_answer" in step:
#                                 st.markdown(f"*Refined Answer:* {step['refined_answer']}")
#                             st.divider()

#                 if message.get("sources") or message.get("formatted_sources"):
#                     with st.expander("Sources"):
#                         if message.get("formatted_sources"):
#                             st.markdown(message["formatted_sources"])
#                         elif message.get("sources"):
#                             try:
#                                 st.json(message["sources"])
#                             except:
#                                 st.write(message["sources"])

#     if prompt := st.chat_input("Ask a question about your documents..."):
#         with st.chat_message("user"):
#             st.markdown(prompt)
#         st.session_state.messages.append({"role": "user", "content": prompt})

#         with st.chat_message("assistant"):
#             try:
#                 with st.spinner("Generating response..."):
#                     response = chatbot(prompt)
#                     if hasattr(chatbot, 'last_refinement_steps') and chatbot.last_refinement_steps:
#                         final_refined = chatbot.last_refinement_steps[-1].get("refined_answer")
#                         if final_refined:
#                             response = final_refined

#                     st.markdown(response)
#                     msg_data = {
#                         "role": "assistant",
#                         "content": response,
#                         "sources": chatbot.last_sources,
#                         "formatted_sources": chatbot.formatted_sources
#                     }
#                     if hasattr(chatbot, 'last_refinement_steps'):
#                         msg_data["refinement_steps"] = chatbot.last_refinement_steps

#                     st.session_state.messages.append(msg_data)

#                     if enable_evaluation:
#                         with st.expander("Evaluation Metrics", expanded=False):
#                             eval_kwargs = {"query": prompt, "response": response}
#                             if reference_answer.strip():
#                                 eval_kwargs["reference_answer"] = reference_answer

#                             evaluation = chatbot.evaluate_answer(**eval_kwargs)
#                             cols = st.columns(4)
#                             cols[0].metric("Relevance", f"{evaluation['relevance']:.2f}/1.00")
#                             cols[1].metric("Faithfulness", f"{evaluation['faithfulness']:.2f}/1.00")
#                             cols[2].metric("Coherence", f"{evaluation['coherence']:.2f}/1.00")
#                             cols[3].metric("Hallucination", f"{evaluation['hallucination_score']:.2f}/1.00")

#                             if "similarity_to_reference" in evaluation:
#                                 st.metric("Similarity to Reference", f"{evaluation['similarity_to_reference']:.2f}/1.00")

#             except Exception as e:
#                 error_msg = f"Error generating response: {str(e)}"
#                 st.error(error_msg)
#                 st.session_state.messages.append({
#                     "role": "assistant",
#                     "content": error_msg
#                 })
