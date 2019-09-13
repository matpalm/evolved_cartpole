see [blog post](http://matpalm.com/blog/evolving_cartpole_flat_buffers/)

run random left/right agent

```
./run_agent.py --agent random --trials 100
```

run random network agent

```
./run_agent.py --agent neural --trials 100
```

run CMA-ES

```
./run_neural_agent_cma_es.py \
 --popsize 20 \
 --fitness-log-file logs/cma.tsv \
 --weights-dir elites/cma
```

run simple GA

```
./run_neural_agent_simple_ga.py \
 --fitness-log-file logs/neural_simple_ga.tsv \
 --weights-dir elites/neural_simple_ga
```

run GA against flat buffers

```
./run_lite_neural_agent_simple_ga.py \
 --fitness-log-file logs/lite_neural_simple_ga.tsv \
 --weights-dir elites/lite_neural_simple_ga/
./run_agent.py --env-render --agent neural_lite \
 --lite-weights solutions/neural_agent_simple_ga/020.tflite
```
