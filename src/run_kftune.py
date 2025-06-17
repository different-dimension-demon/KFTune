from configparser import ConfigParser
import argparse
import time
import os
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, CategoricalHyperparameter, UniformIntegerHyperparameter
import numpy as np
import os
os.chdir("/root/KFTune/src")
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'knowledge_forest'))


from dbms.postgres import PgDBMS
from dbms.mysql import  MysqlDBMS
from config_recommender.sample_workshop import Sample_Workshop
from config_recommender.recommender import Recommender
from knowledge_handler.kf_Knobs import KFKnobs
from knowledge_handler.kf_Summarization import KF_Sum
from base.util import ConfigEncoder


from knowledge_forest.knowledge_forest import *
import demo_tree


if __name__ == '__main__':
    # The construction of the Knowledge Forest and the preparation for knob tuning are placed in two separate files.
    parser = argparse.ArgumentParser()
    parser.add_argument("db", type=str, nargs='?', default="postgres")
    parser.add_argument("test", type=str, nargs='?', default="tpch")
    parser.add_argument("timeout", type=int, nargs='?', default=180)
    parser.add_argument("knob_num", type=int, default=10)
    parser.add_argument("-seed", type=int, default=50)
    
    args = parser.parse_args()
    print(f'Input arguments: {args}')
    time.sleep(2)
    config = ConfigParser()

    if args.db == 'postgres':
        config_path = "../configs/postgres.ini"
        config.read(config_path)
        dbms = PgDBMS.from_file(config)
    elif args.db == 'mysql':
        config_path = "../configs/mysql.ini"
        config.read(config_path)
        dbms = MysqlDBMS.from_file(config)
    else:
        raise ValueError("Illegal dbms!")
    dbms._connect("benchbase")
    
    openai_api_base = ""
    openai_api_key = ""
    persist_path = "../my_chroma_db"

    knowledge_forest = KnowledgeForest(openai_api_base, openai_api_key, {}, persist_dir = persist_path, rebuild = False)
    # Initial configurable space compression
    gpt_knobs = KFKnobs(openai_api_base, openai_api_key, args.db, args.test)
    target_knobs = gpt_knobs.knob_configurable_space
    config_encoder = ConfigEncoder(target_knobs)

    folder_path = "../../optimization_results/temp_results"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # GA-Driven Sample Workshop
    sample_workshop = Sample_Workshop(
        dbms = dbms, 
        config_encoder = config_encoder, 
        knowledge_forest = knowledge_forest,
        test=args.test, 
        timeout=args.timeout, 
        seed=args.seed,
    )

    sample_workshop.optimize()

    # Knob selection before Recommender
    query_text =  "Which database knobs contribute most to system performance under different workloads?"
    results = knowledge_forest.query(query_text, k=2)
    config_encoder.selected_knobs = gpt_knobs.selection(results, args.knob_num)['selected_knobs']

    # Recommender
    recommender = Recommender(
        dbms=dbms, 
        config_encoder=config_encoder,
        knowledge_forest=knowledge_forest,
        gpt_knobs = gpt_knobs,
        knob_num = args.knob_num,
        history=sample_workshop.get_history(),
        test=args.test, 
        timeout=args.timeout, 
        seed=args.seed,
    )

    recommender.optimize()

    # Knowledge summarizaiton
    sum_gpt = KF_Sum(api_base=openai_api_base, api_key=openai_api_key, model="gpt-4o-mini")
    for root in knowledge_forest.trees.values():
        name  = root.name
        analyses = []
        grownode = root.grownode
        analyses.append(grownode.analysis)
        while grownode.child is not None:
            grownode = grownode.child
            analyses.append(grownode.analysis)
        sum_gpt.summarize(analyses)