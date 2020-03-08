<h1 align="center">
<p>Argument Mining and Retrieval using Transformers</p>
</h1>

This repository documents my approach to the Touché Shared Task on Argument Retrieval which is part of the CLEF conference. My efforts come in the context of a course in advanced information retrieval at the University of Leipzig.

## The data
```bash
./Data/
├── annotators.csv
├── args-me.json
├── arguments.csv
├── dataset.pkl
├── rankings.csv
├── tira-qrels
├── topics-automatic-runs-task-1.xml
└── topics.csv
```



## Docker

```bash
foo@bar:/informationretrieval docker run --gpus all -it --rm --network host -v $PWD:/tf/data -w /tf/data tensorflow/tensorflow:latest-gpu-py3-jupyter
foo@bar:/informationretrieval docker run --rm --network host -v $(pwd)/elastic/data:/usr/share/elasticsearch/data -e "discovery.type=single-node" docker.elastic.co/elasticsearch/elasticsearch:7.6.0

```

## Notebooks


## License
