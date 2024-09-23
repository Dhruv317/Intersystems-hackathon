import os
import getpass
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_iris import IRISVector
import uuid

class IRISVectorStore:
    def __init__(self, collection_name, connection_string, embedding_type='openai'):
        """
        Initializes the VectorStore with the specified collection name and connection string.

        Args:
            collection_name (str): The name of the collection (or table) in the vector store.
            connection_string (str): The connection string for the IRIS database.
            embedding_type (str): The type of embeddings to use ('openai', 'huggingface', or 'fastembed').
        """
        load_dotenv(override=True)
        self.collection_name = collection_name
        self.connection_string = connection_string

        # Initialize the embedding function based on type
        if embedding_type == 'openai':
            self.embeddings = OpenAIEmbeddings()
        # Add other embeddings as needed
        # elif embedding_type == 'huggingface':
        #     self.embeddings = HuggingFaceEmbeddings()
        # elif embedding_type == 'fastembed':
        #     self.embeddings = FastEmbedEmbeddings()
        else:
            raise ValueError(f"Unsupported embedding type: {embedding_type}")

        # Initialize the vector store
        try:
            self.db = IRISVector(
                embedding_function=self.embeddings,
                dimension=1536,  # Ensure this matches the embedding dimension used
                collection_name=self.collection_name,
                connection_string=self.connection_string,
            )
        except TimeoutError:
            raise ConnectionError("Unable to connect to the IRIS database.")

    def add_documents(self, documents):
        """
        Adds documents to the vector store.

        Args:
            documents (list): A list of Document objects to add to the store.
        """
        self.db.add_documents(documents)

    def add_texts(self, texts,metadatas=None,ids=None):
        """
        Adds texts to the vector store.

        Args:
            texts (list): A list of strings to add to the store.
        """
        # metadatas = [{'t':"t"}]
        # ids = [str(uuid.uuid4())]
        self.db.add_texts(texts,metadatas=metadatas,ids=ids)
       
    def query(self, query_text):
        """
        Queries the vector store with the provided text and returns documents with scores.

        Args:
            query_text (str): The query string.

        Returns:
            list: A list of tuples containing documents and their corresponding scores.
        """
        return self.db.similarity_search_with_score(query_text)

    def get_retriever(self):
        """
        Returns the retriever object for the vector store.

        Returns:
            Retriever object.
        """
        return self.db.as_retriever()

    def document_count(self):
        """
        Returns the number of documents in the vector store.

        Returns:
            int: The number of documents.
        """
        return len(self.db.get()['ids'])


# Usage example
if __name__ == "__main__":
    # Load environment variables and get API key
    load_dotenv(override=True)
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")

    username = 'demo'
    password = 'demo'
    hostname = os.getenv('IRIS_HOSTNAME', 'localhost')
    port = '1972'
    namespace = 'USER'
    CONNECTION_STRING = f"iris://{username}:{password}@{hostname}:{port}/{namespace}"

    # Create an instance of VectorStore
    vector_store = IRISVectorStore(
        collection_name="state_of_the_union_test",
        connection_string=CONNECTION_STRING,
        embedding_type='openai'
    )

    # Load and split documents
    loader = TextLoader(
        "/Users/dhruvroongta/PycharmProjects/hackmit-2024/data/state_of_the_union.txt", encoding='utf-8')
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=20)
    docs = text_splitter.split_documents(documents)
    for doc in docs:
        print(docs)
        print("")
        print("")
    # Add documents to the vector store
    vector_store.add_documents(docs)
    print(f"Number of docs in vector store: {vector_store.document_count()}")

    # Query the vector store
    query = "Joint patrols to catch traffickers"
    docs_with_score = vector_store.query(query)
    for doc, score in docs_with_score:
        print("-" * 80)
        print("Score: ", score)
        print(doc.page_content)
        print("-" * 80)

    # Adding a new document
    vector_store.add_documents([Document(page_content="foo")])
    docs_with_score = vector_store.query("foo")
    print(docs_with_score[0])
