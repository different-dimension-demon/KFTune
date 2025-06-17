from typing import List, Dict, Any, Optional
import uuid
import os
import pickle
import time
import numpy as np
from sklearn.cluster import KMeans
from langchain.text_splitter import SpacyTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# import sys
# sys.path.append('/root/KFTune/src')
from knowledge_handler.kf_Rooting import *
from knowledge_handler.kf_Growing import *


class ROOT:
    def __init__(self, name: str, content: str):
        self.name = name 
        self.grownode = None
        self.rootnode = RootNode(content)

    def __repr__(self):
        return f"ROOT({self.name})"

class RootNode:
    def __init__(self, content: str):
        self.type = "root"
        self.id = str(uuid.uuid4()) 
        self.content = content
        self.summary = None
        self.left: Optional['RootNode'] = None
        self.right: Optional['RootNode'] = None
        self.target_knobs = None
        # TODO Link 机制暂时还没有实现
        self.links: Dict[RootNode, str] = {}

    def add_link(self, target_node: 'RootNode', connection: str):
        self.links[target_node] = connection

    def __repr__(self):
        return f"RootNode(id={self.id}, summary={self.summary[:5]}...)"

class GrowNode: 
    def __init__(self, change, phenomenon):
        self.type = "grow"
        self.id = str(uuid.uuid4()) 
        self.change = change
        self.phenomenon = phenomenon
        self.analysis = None
        self.child: Optional['GrowNode'] = None

    def add_analysis(self, analysis: str):
        self.analysis = analysis


    def __repr__(self):
        return f"GrowNode(change={self.change[:20]}..., phenomenon={self.phenomenon[:20]}...)"


class KnowledgeForest:
    """Knowledge Forest built on LangChain and Chroma, supporting CRUD operations and vector-based retrieval."""
    STRUCT_FILE = "knowledge_forest.pkl"

    def __init__(self, openai_api_base, openai_api_key, root_dict: Dict[str, ROOT], persist_dir: str = "./chroma_db", rebuild: bool = False):
        self.trees = root_dict
        self.node_map = {}  # id -> RootNode mapping: fast identificaiton by vector search.
        self.text_splitter = SpacyTextSplitter(
            pipeline = "en_core_web_sm",
            chunk_size = 1, 
            chunk_overlap = 0
        )
        self.embedding_model = OpenAIEmbeddings(
            openai_api_base = openai_api_base,
            openai_api_key = openai_api_key,
            model = "text-embedding-ada-002",
            max_retries = 3
        )
        self.root_gpt = KF_Root(api_base=openai_api_base, api_key=openai_api_key, model="gpt-4o-mini")
        self.grow_gpt = KF_Grow(api_base=openai_api_base, api_key=openai_api_key, model="gpt-4o-mini")
        # Support for persistent storage in the vector database
        self.persist_directory = persist_dir
        os.makedirs(self.persist_directory, exist_ok=True)
        struct_path = os.path.join(self.persist_directory, self.STRUCT_FILE)
        # Load or build the knowledge forest
        if os.path.exists(struct_path) and not rebuild:
            self._load_structure(struct_path)
            print("Existing knowledge tree structure loaded. Number of nodes: ", len(self.node_map))
        else:
            self.init_forest()
            self._save_structure(struct_path)
            print("Knowledge tree structure has been built and saved. Number of nodes: ", len(self.node_map))  
        # Load or build the vector database
        self._init_vector_db(rebuild)

    def init_forest(self):
        for root in self.trees.values():
            knob_name = root.name
            docs = self.text_splitter.create_documents([root.rootnode.content])
            num_clusters = int(np.ceil(len(docs)**0.5))
            clusters = self.cluster_sentences(docs, num_clusters=num_clusters)
            root.rootnode = self.build_tree_from_clusters(knob_name, clusters)
            print(f"Local tree structure built for {root.name}")
        print("All knowledge trees have been built.")
        print("Starting to establish links between knowledge trees.")
        for root in self.trees.values():
            self.preorder(root.rootnode, root.name)
        print("Completed building links between knowledge trees.")

    def cosine_similarity(self, vec1, vec2):
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        if vec1.shape != vec2.shape:
            raise ValueError(f"Vector dimension mismatch: {vec1.shape} vs {vec2.shape}")
            
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        return np.dot(vec1, vec2) / (norm1 * norm2)

    def preorder(self, node, name):
        if node is None:
            return
        for relate_node in node.target_knobs:
            if relate_node not in self.trees:
                print(f"Warning: Node {relate_node} does not exist and has been skipped.")
                continue
            closest_node = None
            closest_score = -1
            def dfs(target_node, query):
                nonlocal closest_node, closest_score
                if target_node is None:
                    return
                score = self.cosine_similarity(target_node.embedding, query)
                if score > closest_score:
                    closest_score = score
                    closest_node = node
                dfs(target_node.left, query)
                dfs(target_node.right, query)
            dfs(self.trees[relate_node].rootnode, node.embedding)
            if closest_node is not None:
                relation = self.root_gpt.get_link_from_gpt(
                    name, 
                    relate_node, 
                    node.summary, 
                    closest_node.summary
                )
                node.links[closest_node.id] = relation["relation"]
            else:
                print(f"Warning: No child nodes found in the tree of node {relate_node}.")
        self.preorder(node.left, name)
        self.preorder(node.right, name)


    
    def _save_structure(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                "trees": self.trees,
                "node_map": self.node_map
            }, f)
    
    def _load_structure(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.trees = data["trees"]
            self.node_map = data["node_map"]

    def _init_vector_db(self, rebuild: bool):
        if rebuild or not os.path.isdir(self.persist_directory) or not os.listdir(self.persist_directory):
            self.build_vector_db()
        else:
            try:
                self.vectordb = Chroma(
                    embedding_function=self.embedding_model,
                    persist_directory=self.persist_directory,
                    collection_name="knowledge_forest"
                )
                print("Existing vector database loaded from path:", self.persist_directory)
            except Exception:
                print("Loading failed, rebuilding the vector database...")
                self.build_vector_db()

    def cluster_sentences(self, documents: List[Document], num_clusters: int = 4) -> Dict[int, List[Document]]:
        embeddings = self.embedding_model.embed_documents(
            [d.page_content for d in documents]
        )
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        clusters: Dict[int, List[Document]] = {i: [] for i in range(num_clusters)}
        for doc, lbl in zip(documents, labels):
            doc.metadata["cluster"] = lbl
            clusters[lbl].append(doc)
        return clusters

    def build_tree_from_clusters(self, knob_name, clusters: Dict[int, List[Document]]) -> RootNode:
        # 先并行生成叶子节点
        leaves: List[RootNode] = []
        for cid, docs in clusters.items():
            content = " ".join(d.page_content for d in docs)
            node = RootNode(content)
            answer = self.root_gpt.get_answer_from_gpt(knob_name, content)
            node.summary = answer["summary"]
            node.embedding = self.embedding_model.embed_query(answer["summary"])
            node.target_knobs = answer["related_knobs"]
            leaves.append(node)
            self.node_map[node.id] = node

        while len(leaves) > 1:
            temp: List[RootNode] = []
            for i in range(0, len(leaves), 2):
                if i + 1 < len(leaves):
                    left, right = leaves[i], leaves[i+1]
                    merged = RootNode(
                        left.content + right.content
                    )
                    merged.left, merged.right = left, right
                    answer = self.root_gpt.get_answer_from_gpt(knob_name, merged.content)
                    merged.summary = answer["summary"]
                    merged.embedding = self.embedding_model.embed_query(answer["summary"])
                    merged.target_knobs = answer["related_knobs"]
                    temp.append(merged)
                    self.node_map[merged.id] = merged
                else:
                    temp.append(leaves[i])
            leaves = temp
        return leaves[0]

    def build_vector_db(self):
        docs: List[Document] = []
        for node in self.node_map.values():
            docs.append(Document(page_content=node.summary, metadata={"node_id": node.id}))
        self.vectordb = Chroma.from_documents(
            docs,
            self.embedding_model,
            persist_directory=self.persist_directory,
            collection_name="knowledge_forest"
        )

        print("The vector database has been built and persisted to:", self.persist_directory)

    def add_node(self, node: RootNode):
        doc = Document(page_content=node.summary, metadata={"node_name": node.name})
        self.vectordb.add_documents([doc])


    def delete_node(self, node_name: str):
        self.vectordb.delete(filter={"node_name": node_name})


    def update_node(self, node: RootNode):
        self.delete_node(node.name)
        self.add_node(node)

    def query(self, query_text: str, k: int = 5) -> List[RootNode]:
        docs = self.vectordb.similarity_search(query_text, k=k)
        result_nodes: List[RootNode] = []
        result = []
        result_key = []
        for doc in docs:
            node_id = doc.metadata.get("node_id")
            if node_id and node_id in self.node_map:
                result_key.append(node_id)
                node = self.node_map[node_id]
                result.append(node)
                if node.type == "root":
                    result_nodes.append(self.node_map[node_id])
        loop_id = 0
        while len(result_nodes)>0 and loop_id < 2:
            link_nodes: List[RootNode] = []
            for node in result_nodes:
                for key, value in node.links.items():
                    if key and key in self.node_map and key not in result_key:
                        result_key.append(key)
                        link_nodes.append(self.node_map[key])
                        result.append(self.node_map[key])
            result_nodes = link_nodes
            loop_id = loop_id + 1
        
        return result

    def add_grow_node(self, root_name: str, change: str, phenomenon: str):
        grow_node = GrowNode(change, phenomenon)
        analysis = self.grow_gpt.get_analysis_from_gpt(root_name, change, phenomenon)['analysis']
        grow_node.add_analysis(analysis)
        root = self.trees.get(root_name)
        if root is None:
            print(f"ROOT node '{root_name}' not found")
            return
        self.node_map[grow_node.id] = grow_node
        if root.grownode is None:
            root.grownode = grow_node
        else:
            last = root.grownode
            while last.child and isinstance(last.child, GrowNode):
                last = last.child
            last.child = grow_node
        doc = Document(page_content=analysis, metadata={"node_id": grow_node.id})
        self.vectordb.add_documents([doc])
        print(f"GrowNode added to ROOT({root_name}) and its analysis indexed")

        self._save_structure(os.path.join(self.persist_directory, self.STRUCT_FILE))

def load_root_objects_from_directory(directory):
    loader = DirectoryLoader(
        directory,
        glob="**/*.txt",                     # 只加载 .txt 文件
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}  # 避免乱码
    )
    documents = loader.load()
    root_dict = {}
    for doc in documents:
        knob_name = os.path.splitext(os.path.basename(doc.metadata.get("source", "")))[0]
        root = ROOT(name=knob_name, content=doc.page_content)
        root_dict[knob_name] = root
    return root_dict
