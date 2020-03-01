#!/bin/bash
docker run --rm --network host -v $(pwd)/elastic/data:/usr/share/elasticsearch/data -e "discovery.type=single-node" docker.elastic.co/elasticsearch/elasticsearch:7.6.0

