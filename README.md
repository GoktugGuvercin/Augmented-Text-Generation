
## Retrieval Augmented Generation

- Large language models excel at maintaining the context of their extensive training data. When the relevant inquiries are prompted, they provide detailed answers with some examples and analogies that accurately reflect the subject matter. However, a notable limitation is their propensity to hallucinate in a way that they can give factually incorrect pieces of information.

- Many open-source and enterprise large Language models are built on a mixture of distinct, but mostly public data sources. This qualifies them to have a multifaceted perspective on many different topics, but they could not be exactly specialized in a specific domain. Consider a company's internal knowledge base, which contains a wealth of so many documents about legal matters, legislation, corporate policies, and technical details. Since these resources are confidential and not used in the models' training, LLMs have a significant knowledge gap in these areas, limiting their utility for specialized, internal applications.

- Retrieval-augmented generation (RAG) is a valid approach not only to solve hallucination problem of LLMs but also to enhance their specialization by incorporating external data sources. At this point, what RAG pipelines actually do is to allow the models to access and utilize supplementary information retrieved from external documents. In that way, it first mitigates hallucinations by grounding the model's responses in verifiable facts. Second, the model is enriched with domain-specific expertise on demand. This allows the LLM to process and respond to queries with a high degree of accuracy and specialization. Since the generated answer is grounded on specific data resources, making inline citations and finding the main source of the information in LLM's response become easier.

- Typical RAG application is actually constituted by ***indexing***, ***retrieval***, and ***generation***. 

## Advantages of RAG Pipelines:
- Verifiable Information
- Cited Information
- More Accurate and Detailed Information
- Information without Fabricated Facts

In the design and development of question-answering chatbots powered by LLMs, these profits are important assets to be used for complete and accurate flow of multi-turn conversation.

## LangChain

LangChain is an open-source framework for building applications that harness large language models (LLMs). These applications are founded at the top of a sequence of functions, which is called as "chain". At this point, LangChain provides the key modules and the configuration layer that we will use to weave those modules together. 

Let's assume that we will build an application with <ins>***retrieve-summarize-answer***</ins> workflow. LangChain streamlines this workflow by offering ready-made document loaders, text splitters, vector stores, and convenient access to both embedding models and also chat models, in which we bring them together to implement the entire pipeline. 

| Stage   | Purpose | Typical LangChain modules   |
|:-------:|---------|:---------------------------:|
| **Retrieve** | Pull raw content from a website | Document loaders, Text Splitters, Vector Stores |
| **Summarize** | Condense the content into a concise summary | Prompt + LLM |
| **Answer** | Use the summary as context for an LLM to answer user questions | Memory, Prompt, LLM|

In the summarization stage, a purpose-built prompt is defined to describe exactly how retrieved content should be summarized, and the large language model used in this stage takes the responsibility of this process with rspect to the instructions in the prompt.

In answer generation part, the summarized content and conversation history are preserved in a memory component, which is used to comprise the prompt itself. Large language model receives this prompt as input to generate an answer for the given question.

In both of these stages, we can use same or different language models, which is up to our own decision. When we look at the entire pipeline, we see that each step relies on the output of previous one, which adheres to a more sequential structure. While entire pipeline is a comprehensive chain of multiple modules, its sub stages can be also modelled as smaller chains. For example, summarization is a good candidate to be converted into a modular chain by means of `create_stuff_documents_chain()`. 

## LangGraph

LangGraph is a specialized library within LangChain ecosystem. As the logic of LLM-powered systems grows more sophisticated, their representastion as a linear pipeline converts into a more limiting factor, which actually demands dynamic control flow. LangGraph addresses this by allowing developers to define workflows as cyclical graphs, which is essential for advanced use cases like multi-agent systems. At this point, our AI agent is not a chain, instead it is structured as a graph operating like finite-state-machines. 

Let's imagine a versatile note-taking AI like Google's NotebookLM. After receiving a piece of text as input, it doesn't just do one thing; it offers a suite of potential actions:

- Polish the writing
- Summarize the document
- Create podcast from given content
- Generate a visual mind map

Each of those actions refers to the particular node in the graph. When one of these actions is initiated, the state is read and generally modified. The transitions between these nodes is controlled by the edges. If there is no edge between two nodes, this means that it is possible to execute one action right after the other one. For example, before summarizing the content, it may not be possible to generate a mind map. The edges can be structured as conditional or looped. For the implementation and realization of these actions, we can utilize both language and vision models.  The main graph structure helps to handle different user requests in any order. In that way, we have flexible, stateful agents.

The input text or the document that we provided forms the specific part of the state. When we generate a podcast or a mind map, it doesn’t discard the original content; instead, it enriches the state with audio and visual modalities. All nodes should be able to access and modify the state to enable truly stateful interactions. In chatbot applications, this state manager maintains the input, generated output, conversational history and context.

## LangChain vs LangGraph

|         Title          |      LangChain          |      LangGraph       |
|:----------------------:|:-----------------------:|----------------------|
|   **Primary Focus**    |  LLM Applications       | Multi-Agent Systems  |
|     **Structure**      |       Chain             |        Graph         |
| **Building Component** | Memory, Prompt, LLM     | Nodes, Edges, State  |
|  **State Management**  |       Limited           |        Robust        |
|     **Use Cases**      | Step-by-step operations | Finite-State Systems |

- *Primany Focus:* LangChain aims to chain language processing steps into end-to-end LLM applications by creating an abstraction over them, while LangGraph build and orchestrate entire multi-agent systems.

- *State Management:* In LangChain, there is limited amount of state management. We can pass it through the chain, but it does not maintain persistent state across multiple runs. In contrast, LangGraph offers quite robust state management system, because the state is considered as a major component on stateful systems in a way that all nodes can access and modify, promoting more complex and context aware behaviors.

## LangChain Embeddings Working Structure

LangChain provides a great variety of embeddings models through the integration modules with top technology companies. Each model is actually encapsulated within a dedicated class, such as `CohereEmbeddings`, `OpenAIEmbeddings`, and `ClovaEmbeddings`. These classes offer consistent and standardized way to generate embeddings, since they are actually built upon LangChain's base `Embeddings` [interface](https://python.langchain.com/api_reference/core/embeddings/langchain_core.embeddings.embeddings.Embeddings.html#langchain_core.embeddings.embeddings.Embeddings). This core interface creates an abstraction requiring every model to implement `embed_query()` and `embed_documents()` functions. Usually the query embedding is identical to the document embedding, but the abstraction allows treating them independently. Vectorstores are designed in a quite elegant way to choose the correct function for the relevant job. 

- To store documents, they use `embed_documents()` function.
- To search for information, they use `embed_query()` function.

This means you can easily swap embedding models without changing how your vector store works.

```python
vectorstore = InMemoryVectorStore(embedding)
chunk_ids = vectorstore.add_documents(chunks)  # calling embed_documents()
vectorstore.similarity_search("What is the capital of China ?")  # calling embed_query()
```