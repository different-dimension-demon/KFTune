from knowledge_forest.knowledge_forest import *
if __name__ == "__main__":
    folder_path = "../../knowledge_collection/postgres/knowledge_forest/manual" # Replace with your own knobs related info
    persist_path = "../../my_chroma_db"
    openai_api_base=""
    openai_api_key=""

    rebuild = True

    print("=== Construction of the Knowledge Forest and Vector Database ===")
    root_dict = load_root_objects_from_directory(folder_path)
    forest = KnowledgeForest(openai_api_base, openai_api_key, root_dict, persist_dir=persist_path, rebuild=rebuild)
    print("=== Construction Complete ===")
