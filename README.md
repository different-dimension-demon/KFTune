# Harvesting Tuning Wisdom from a Growing Knowledge Forest
<div align="center">
  <img src="/overview/kftune.png" alt="KFTune overview" width="800">
</div>
<!-- <img src="/overview/kftune.png" alt="KFTune overview" width="800":> -->

**KFTune** is a knowledge-driven database tuning framework that builds a dynamic, knob-centric knowledge forest to guide efficient, workload-aware optimization. It integrates Bayesian optimization into genetic algorithms and uses Gumbel-Softmax for joint tuning of categorical and continuous knobs. 

## Quick Start
The following instructions have been tested on Ubuntu 20.04 and PostgreSQL v14.9:
### Step 1: Install PostgreSQL
```
sudo apt-get update
sudo apt-get install postgresql-14
```

### Step 2: Install [BenchBase](https://github.com/cmu-db/benchbase) with our script

```
cd ./scripts
sh install_benchbase.sh postgres
```

> **Note:** After executing `KFTune`, the `benchbase` and `optimization_results` directories should be located in the same folder. This folder (e.g., `DB_Tuner`) should also be the working directory where you run the following commands.


### Step 3: Install Benchmark with our script
> **Note:** modify `./benchbase/target/benchbase-postgres/config/postgres/sample_{your_target_benchmark}_config.xml` to customize your tuning setting first.

> **Note:** In our experiment `{your_target_benchmark}` refers to one of the following seven benchmarks: `tpch`, `tpcc`, `ycsb`, `chbenchmark`.

> **Note:** if you want to use `chbenchmark`, you need to modify the second line of the `build_benchmark.sh` script to `java -jar benchbase.jar -b tpcc,chbenchmark -c config/postgres/sample_chbenchmark_config.xml --create=true --load=true --execute=false`.

```
sh build_benchmark.sh postgres tpch
```

### Step 4: Install dependencies
```
pip install -r requirements.txt
```

### Step 5: Execute the KFTune to optimize your DBMS
> Note: modify `configs/postgres.ini` to determine the target DBMS first, the `restart` and `recover` commands depend on the environment

> Note: modify `./KFTune/src/run_kftune.py` and `./KFTune/src/knowledge_forest/run_builder.py` to set up your `api_base`, `api_key` and `model` first

> Note: modify `./KFTune/knowledge_collection/system_description.txt` to configure your `hardware` settings.

Construct the knowledge forest
```
python ./KFTune/src/knowledge_forest/run_builder.py
```

After completing the above preparations, you can run the following command to start optimizing your DBMS directly:
```
python ./KFTune/src/run_kftune.py
```
Besides, there are some optional settings you can modify in `run.py`:
- `--db` specifies the DBMS, we usually use postgres;
- `--test` specifies the workload to execute, the default is tpch;
- `--knob_num` specifies the number of knobs to be optimized by the Recommender module.



# Acknowledgments
This work makes use of [BenchBase](https://github.com/cmu-db/benchbase) and [GPTuner](https://github.com/SolidLao/GPTuner/tree/main). We are grateful for their contribution to the open-source community.
