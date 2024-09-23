import os
import ast
import json
import networkx as nx
from openai import OpenAI
from dotenv import load_dotenv
from vectorstore import IRISVectorStore
import os
import getpass
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_iris import IRISVector
import uuid
load_dotenv(override=True)
username = 'demo'
password = 'demo'
hostname = os.getenv('IRIS_HOSTNAME', 'localhost')
port = '1972'
namespace = 'USER'
CONNECTION_STRING = f"iris://{username}:{password}@{hostname}:{port}/{namespace}"
vector_store = IRISVectorStore(
    collection_name="code_test",
    connection_string=CONNECTION_STRING,
    embedding_type='openai'
)
loader = TextLoader(
    "/Users/dhruvroongta/PycharmProjects/hackmit-2024/data/state_of_the_union.txt", encoding='utf-8')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=20)
docs = text_splitter.split_documents(documents)
# for doc in docs:
#     print(docs)
#     print("")
#     print("")
# Add documents to the vector store
vector_store.add_documents(docs)
print(f"Number of docs in vector store: {vector_store.document_count()}")
client = OpenAI()

class Document:
    def __init__(self, id, content = None, metadata = None):
        self.id = id
        self.page_content = "NOne"
        self.metadata = {"NODEID":id}
        
        
def create_docstring(node,nodeid):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages = [
            {
                "role": 'system',
                "content": 'Generate a description for the following code snippet'
            },
            {
                "role": 'user',
                "content": f"Create a description for the following {node['type']}:\n{node['content']}"
            }
        ],
        max_tokens=500
    )
    print(response)
    print(response.choices[0].message.content)
    # documents = [Document(
    #     nodeid, (node['content'], response.choices[0].message.content, "NODEID="+nodeid))]
    # print(documents)
    texts = [(response.choices[0].message.content)]
    ids = [nodeid]
    print('NODE ID',nodeid)
    ids = [str(uuid.uuid4())]
    metadatas = [{"node_id":nodeid,'node_content':node['content']}]
    print(texts)
    vector_store.add_texts(texts,metadatas,ids)
    # vector_store.add_documents(
    #     [{"id":nodeid,"content":(node['content'], response.choices[0].message.content, "NODEID="+nodeid)}])

    # vector_store.add_documents([(node['content'], response.choices[0].message.content,"NODEID="+nodeid)])
    
def parse_python_file(file_path, source):
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        print(f"Syntax error in file {file_path}: {str(e)}")
        return []

    chunks = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            try:
                chunk = ast.get_source_segment(source, node)
                chunks.append({
                    'type': type(node).__name__,
                    'name': node.name,
                    'content': chunk
                })
                # create_docstring(chunks[-1])
            except Exception as e:
                print(f"Error extracting chunk from {file_path}: {str(e)}")

    print(f"CHUNKS for {file_path}: {len(chunks)}")
    return chunks


def process_file(file_path):
    _, file_extension = os.path.splitext(file_path)

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return None, []

    file_node = {
        'type': 'file',
        'name': os.path.basename(file_path),
        'content': content
    }

    if file_extension.lower() == '.py':
        chunks = parse_python_file(file_path, content)
    else:
        chunks = []

    return file_node, chunks


def build_graph(directory_path):
    graph = nx.DiGraph()

    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_node, chunks = process_file(file_path)

            if file_node:
                file_node_id = f"{file_path}"
                graph.add_node(file_node_id, **file_node)
                create_docstring(file_node,file_node_id)

                for chunk in chunks:
                    chunk_node_id = f"{file_path}:{chunk['name']}"
                    graph.add_node(chunk_node_id, **chunk)
                    create_docstring(chunk, chunk_node_id)
                    graph.add_edge(file_node_id, chunk_node_id)
                    print("Added edge: ", file_node_id, " <-> ", chunk_node_id)

                # Add edges between chunks in the same file
                for i, chunk in enumerate(chunks):
                    for other_chunk in chunks[i+1:]:
                        graph.add_edge(
                            f"{file_path}:{chunk['name']}", f"{file_path}:{other_chunk['name']}")

    return graph

def get_node_by_id(graph, node_id):
    """
    Retrieve a node from the graph by its ID.
    
    :param graph: The NetworkX graph
    :param node_id: The ID of the node to retrieve
    :return: The node data if found, None otherwise
    """
    if node_id in graph.nodes:
        return graph.nodes[node_id]
    return None


def get_neighbors(graph, node_id):
    """
    Retrieve all neighbors of a node given its ID.
    
    :param graph: The NetworkX graph
    :param node_id: The ID of the node whose neighbors to retrieve
    :return: A list of neighboring node IDs
    """
    if node_id in graph.nodes:
        return list(graph.neighbors(node_id))
    return []


def main(directory_path):
    graph = build_graph(directory_path)

    # Convert graph to JSON for visualization or further processing
    graph_data = nx.node_link_data(graph)

    with open('directory_graph.json', 'w') as f:
        json.dump(graph_data, f, indent=2)

    print(f"Graph representation saved to directory_graph.json")
    print(f"Number of nodes: {graph.number_of_nodes()}")
    print(f"Number of edges: {graph.number_of_edges()}")

    # Example usage of new functions
    print("\nExample usage of new functions:")
    # Get a random node ID from the graph
    sample_node_id = list(graph.nodes)[5] if graph.nodes else None
    if sample_node_id:
        print(f"Sample node ID: {sample_node_id}")
        node_data = get_node_by_id(graph, sample_node_id)
        print(f"Node data: {node_data}")
        neighbors = get_neighbors(graph, sample_node_id)
        print(f"Neighbors of the node: {neighbors}")


if __name__ == "__main__":
    # Replace with the actual local directory path
    # local_directory = '/Users/dhruvroongta/PycharmProjects/basics/CS2340/CS2340Proj1'
    local_directory = '/Users/dhruvroongta/PycharmProjects/hackmit-2024/folder'
    main(local_directory)

# Replace with the actual repository URL
# repo_url = "https://github.com/Dhruv317/CS2340Proj1"
# repo =
# main(repo_url)
